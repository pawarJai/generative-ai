"""Restore ingested file state and initialise DB tables on server boot."""
import os
from app.jobs import init_db as init_jobs_db
from app.persistence import init_db as init_registry_db, get_all_files


def restore_state() -> None:
    init_jobs_db()
    init_registry_db()

    files = get_all_files()
    restored, skipped, failed = 0, 0, 0
    for f in files:
        path = f["path"]
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            from app.ingestion.universal import universal_ingest
            universal_ingest(path, f["file_id"])
            restored += 1
        except Exception as e:
            print(f"[startup] Failed to restore {f['file_id']}: {e}")
            failed += 1

    print(f"[startup] Restored {restored} file(s), {skipped} missing, {failed} failed.")
