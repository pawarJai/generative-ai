"""SQLite-backed job tracker for async ingestion tasks."""
import json
import sqlite3
import time
import uuid
from typing import Optional

DB_PATH = "./jobs.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id    TEXT PRIMARY KEY,
                file_id   TEXT NOT NULL,
                status    TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                result    TEXT,
                error     TEXT
            )
        """)


def create_job(file_id: str) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?, 'pending', ?, ?, NULL, NULL)",
            (job_id, file_id, now, now),
        )
    return job_id


def update_job(job_id: str, status: str,
               result: Optional[dict] = None, error: Optional[str] = None) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, updated_at=?, result=?, error=? WHERE job_id=?",
            (status, time.time(), json.dumps(result) if result else None, error, job_id),
        )


def get_job(job_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("result"):
        d["result"] = json.loads(d["result"])
    return d
