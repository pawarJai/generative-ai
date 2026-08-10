"""POST /ingest — upload a file, start ingestion in the background, return job_id."""
import os
import shutil
import uuid
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.jobs import create_job, update_job
from app.persistence import register_file
from app.ingestion.universal import universal_ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _run_ingest(path: str, fid: str, original_filename: str, job_id: str) -> None:
    update_job(job_id, "running")
    try:
        result = universal_ingest(path, fid)
        register_file(fid, original_filename, path, result["kind"])
        update_job(job_id, "done", result=result)
    except Exception as e:
        update_job(job_id, "error", error=str(e))


@router.post("", status_code=202)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_id: str = None,
):
    """Upload + ingest a file. Returns immediately with job_id; poll GET /jobs/{job_id}.
    Supported: .pdf .png .jpg .jpeg .docx .pptx .xlsx .xls .csv .txt"""
    fid = file_id or f"doc_{uuid.uuid4().hex[:10]}"
    dest_path = os.path.join(UPLOAD_DIR, f"{fid}_{file.filename}")
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    job_id = create_job(fid)
    background_tasks.add_task(_run_ingest, dest_path, fid, file.filename, job_id)
    return {"job_id": job_id, "file_id": fid, "status": "pending"}
