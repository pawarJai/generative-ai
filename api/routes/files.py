"""GET endpoints for inspecting ingested files and downloading exports."""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List
from app import state
from app.config import OUTPUT_DIR
from app.models import FileSummary
from app.tables.helpers import get_all_real_tables
from app.logging_utils import load_interaction_log, detect_file_id_collisions

router = APIRouter(prefix="/files", tags=["files"])


@router.get("", response_model=List[FileSummary])
async def list_files(session_id: str = None):
    """Files the user has ingested, read from the durable registry.

    With a session_id, only the documents uploaded into that conversation.
    The UI always asks this way: showing every file ever ingested made the
    document strip unusable (14 chips and growing, none of them relevant to
    the chat on screen). Without one, the full list is returned, which is
    what the agent's own file-resolution needs.

    This used to read state.FILE_ORDER, which is pure in-memory and is wiped
    by every `uvicorn --reload` restart — so the UI showed zero files (and
    told users to re-upload) while the registry held the full list and Chroma
    held all their vectors. The registry is the source of truth; in-memory
    state only decides whether a file is loaded right now or restored lazily
    on first use.
    """
    from app.persistence import get_all_files

    out = []
    seen = set()
    for record in get_all_files(session_id):
        fid = record["file_id"]
        seen.add(fid)
        if fid in state.FILE_KIND:
            status = "loaded"
        elif os.path.exists(record["path"]):
            status = "on_disk"
        else:
            status = "missing"
        out.append(FileSummary(
            file_id=fid,
            original_filename=record["original_filename"],
            kind=record["kind"],
            summary=state.FILE_META.get(fid, {}).get("summary"),
            status=status,
        ))

    # Anything ingested in this process but not yet registered still shows up,
    # so a file can never disappear from the UI just because of registry lag.
    # Only when listing globally: an in-flight file has no session recorded
    # yet, and leaking it into a specific chat is the very sprawl this
    # endpoint's session filter exists to prevent.
    for fid in ([] if session_id else state.FILE_ORDER):
        if fid not in seen:
            out.append(FileSummary(
                file_id=fid,
                original_filename=state.FILE_ORIGINAL_NAME.get(fid, fid),
                kind=state.FILE_KIND.get(fid, "unknown"),
                summary=state.FILE_META.get(fid, {}).get("summary"),
                status="loaded",
            ))
    return out


@router.get("/{file_id}/tables")
async def file_tables(file_id: str):
    # Restore on demand rather than 404-ing a file that is registered and
    # present on disk but simply not loaded into this process yet.
    if file_id not in state.FILE_KIND:
        from app.graph.agent import _restore_file_if_needed
        err = _restore_file_if_needed(file_id)
        if err:
            raise HTTPException(status_code=404, detail=err)
    tables = get_all_real_tables(file_id)
    return [
        {"page": df.attrs.get("page"), "rows": df.shape[0], "cols": df.shape[1],
         "columns": list(df.columns), "source": df.attrs.get("source", "native")}
        for df in tables
    ]


@router.get("/{file_id}/extraction-report")
async def file_extraction_report(file_id: str):
    """Per-table extraction quality: which tables lost a column, which had
    their header recovered, and which still have no column names at all.
    Lets a user see where the data is untrustworthy before building a
    quotation on it, instead of discovering it in a wrong answer."""
    from app.tables.helpers import extraction_report
    if file_id not in state.FILE_KIND:
        from app.graph.agent import _restore_file_if_needed
        err = _restore_file_if_needed(file_id)
        if err:
            raise HTTPException(status_code=404, detail=err)
    return extraction_report(file_id)


@router.get("/collisions")
async def file_id_collisions():
    df = detect_file_id_collisions()
    return df.to_dict(orient="records") if not df.empty else []


@router.get("/download/{filename}")
async def download_export(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found in outputs/")
    return FileResponse(path, filename=filename)


@router.get("/logs")
async def interaction_logs(limit: int = 50):
    df = load_interaction_log()
    if df.empty:
        return []
    return df.tail(limit).to_dict(orient="records")
