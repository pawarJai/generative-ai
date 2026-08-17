"""POST /ingest — upload a file, start ingestion in the background, return job_id."""
import asyncio
import hashlib
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.jobs import create_job, update_job
from app.persistence import find_by_content_hash, register_file
from app.ingestion.universal import universal_ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# The actual cause of "stuck pending" was a catastrophic-backtracking regex
# in app/ingestion/docling_ingest.py (fixed separately) — sync BackgroundTasks
# already run off the event loop via Starlette's own run_in_threadpool, so
# this was never blocking the loop. This is defensive, not corrective: a
# dedicated, bounded pool for ingestion specifically, isolated from anyio's
# shared thread pool that every other run_in_threadpool call in this app
# (including /chat) also draws from — so a future slow step added to
# ingestion can degrade at most 2 concurrent ingestions, never starve
# unrelated requests of thread capacity.
# max_workers=1, not 2: two ingestions running at once means two threads
# writing to ChromaDB concurrently, and the binding segfaults under exactly
# that (see _SerializedVectorStore in app/config.py — nine threads queued on
# one mutex inside chromadb_rust_bindings, a tenth crashing on a null
# dereference). The lock in config.py already prevents the crash; keeping
# this at one worker also stops a second upload from simply blocking on that
# lock for the whole duration of the first one's embedding step.
_ingest_executor = ThreadPoolExecutor(max_workers=1)


def _content_hash(path: str) -> str:
    """Same 12-char sha256 prefix the ingestion log already records, so a
    hash here and a hash there refer to the same thing."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:12]


async def _run_ingest(path: str, fid: str, original_filename: str, job_id: str,
                      session_id: str = None) -> None:
    update_job(job_id, "running")
    loop = asyncio.get_running_loop()
    try:
        digest = await loop.run_in_executor(_ingest_executor,
                                            lambda: _content_hash(path))

        # Already ingested this exact content? Register the new file_id
        # against the SAME upload and skip re-embedding entirely. Confirmed
        # cause of a 203 GB vector store (see find_by_content_hash): the same
        # PDF was ingested 22 times in one day under 22 generated file_ids,
        # each writing another full copy of its chunks, and Chroma's HNSW
        # index never reclaims that space. Re-uploading is a completely
        # normal thing for a user to do — the storage cost of it was the bug.
        existing = find_by_content_hash(digest)
        if existing and existing["file_id"] != fid:
            def _reuse():
                from app.graph.agent import _restore_file_if_needed
                return _restore_file_if_needed(existing["file_id"])

            restore_error = await loop.run_in_executor(_ingest_executor, _reuse)
            if not restore_error:
                register_file(fid, original_filename, existing["path"],
                              existing["kind"], session_id, digest)
                result = {
                    "file_id": existing["file_id"],
                    "original_filename": original_filename,
                    "kind": existing["kind"],
                    "reused_existing": True,
                    "note": (f"This document was already indexed as "
                            f"'{existing['file_id']}' — reused its existing "
                            f"data instead of embedding a second copy."),
                }
                update_job(job_id, "done", result=result)
                return
            # Could not restore the previous copy (its upload is gone, or the
            # registry entry is stale) — fall through and ingest normally
            # rather than failing an upload that would otherwise work.

        result = await loop.run_in_executor(
            _ingest_executor, lambda: universal_ingest(path, fid))
        register_file(fid, original_filename, path, result["kind"], session_id,
                      digest)
        update_job(job_id, "done", result=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, "error", error=str(e))


@router.post("", status_code=202)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_id: str = None,
    session_id: str = None,
):
    """Upload + ingest a file. Returns immediately with job_id; poll GET /jobs/{job_id}.
    Supported: .pdf .png .jpg .jpeg .docx .pptx .xlsx .xls .csv .txt

    session_id ties the document to the conversation it was uploaded into, so
    that chat shows its own files and other chats do not."""
    fid = file_id or f"doc_{uuid.uuid4().hex[:10]}"
    dest_path = os.path.join(UPLOAD_DIR, f"{fid}_{file.filename}")
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    job_id = create_job(fid)
    background_tasks.add_task(_run_ingest, dest_path, fid, file.filename,
                              job_id, session_id)
    return {"job_id": job_id, "file_id": fid, "status": "pending"}
