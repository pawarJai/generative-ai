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
async def list_files():
    return [
        FileSummary(
            file_id=fid,
            original_filename=state.FILE_ORIGINAL_NAME.get(fid, fid),
            kind=state.FILE_KIND.get(fid, "unknown"),
            summary=state.FILE_META.get(fid, {}).get("summary"),
        )
        for fid in state.FILE_ORDER
    ]


@router.get("/{file_id}/tables")
async def file_tables(file_id: str):
    if file_id not in state.FILE_ORDER:
        raise HTTPException(status_code=404, detail="file_id not found")
    tables = get_all_real_tables(file_id)
    return [
        {"page": df.attrs.get("page"), "rows": df.shape[0], "cols": df.shape[1],
         "columns": list(df.columns), "source": df.attrs.get("source", "native")}
        for df in tables
    ]


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
