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
        # A document belongs to the conversation it was uploaded into, so a
        # chat shows its own files instead of every file ever ingested.
        # Added after the table shipped, so existing registries are migrated
        # in place rather than rebuilt — rows from before this keep
        # session_id NULL and belong to no conversation.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(file_registry)")}
        if "session_id" not in cols:
            conn.execute("ALTER TABLE file_registry ADD COLUMN session_id TEXT")
        # Content hash, so re-uploading a document already ingested can reuse
        # its embeddings instead of writing a second full copy. Migrated in
        # place like session_id above; rows from before this keep NULL and
        # simply never match, so they cost nothing and break nothing.
        if "content_hash" not in cols:
            conn.execute("ALTER TABLE file_registry ADD COLUMN content_hash TEXT")


def register_file(file_id: str, original_filename: str, path: str, kind: str,
                  session_id: Optional[str] = None,
                  content_hash: Optional[str] = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO file_registry "
            "(file_id, original_filename, path, kind, ingested_at, session_id, "
            " content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_id, original_filename, path, kind, time.time(), session_id,
             content_hash),
        )


def find_by_content_hash(content_hash: str) -> Optional[dict]:
    """The most recently ingested file with this exact content, if any.

    Confirmed cause of a 203 GB vector store: the identical PDF was ingested
    22 times under 22 different file_ids in a single day (each upload mints a
    new id), and every one wrote a fresh 406-chunk copy into Chroma. Chroma's
    HNSW index never reclaims space on delete, so the store grew until the
    disk hit 97% full, writes began failing, the index corrupted, and every
    subsequent upload segfaulted the ingestion process — surfacing to the
    user as "upload stuck on pending forever".
    """
    if not content_hash:
        return None
    import os
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM file_registry WHERE content_hash = ? "
            "ORDER BY ingested_at DESC", (content_hash,)).fetchall()
    for row in rows:
        # Only reuse a file whose upload is still on disk — a registry row
        # pointing at a deleted file cannot be restored from later.
        if os.path.exists(row["path"]):
            return dict(row)
    return None


def get_all_files(session_id: Optional[str] = None) -> List[dict]:
    """Registered files, oldest first. With a session_id, only the documents
    uploaded into that conversation — the UI asks this way so one chat never
    displays another chat's documents."""
    with _conn() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM file_registry WHERE session_id=? ORDER BY ingested_at",
                (session_id,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM file_registry ORDER BY ingested_at").fetchall()
    return [dict(r) for r in rows]


def get_file(file_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM file_registry WHERE file_id=?", (file_id,)
        ).fetchone()
    return dict(row) if row else None


def remove_file(file_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM file_registry WHERE file_id=?", (file_id,))
