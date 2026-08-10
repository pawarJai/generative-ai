"""File registry — SQLite store of ingested files so state can survive restarts."""
import sqlite3
import time
from typing import List, Optional

DB_PATH = "./file_registry.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_registry (
                file_id           TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                path              TEXT NOT NULL,
                kind              TEXT NOT NULL,
                ingested_at       REAL NOT NULL
            )
        """)


def register_file(file_id: str, original_filename: str, path: str, kind: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO file_registry VALUES (?, ?, ?, ?, ?)",
            (file_id, original_filename, path, kind, time.time()),
        )


def get_all_files() -> List[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM file_registry ORDER BY ingested_at").fetchall()
    return [dict(r) for r in rows]


def remove_file(file_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM file_registry WHERE file_id=?", (file_id,))
