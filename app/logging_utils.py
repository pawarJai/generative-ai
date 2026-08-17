"""Append-only JSONL interaction/ingestion log — same format as the notebook,
so any analysis scripts you already have against interaction_log.jsonl keep
working unchanged."""
import os
import json
import time
import uuid
import hashlib
from typing import Optional, Dict, Any
import pandas as pd
from app import state
from app.config import LOG_PATH


def log_interaction(session_id, file_id, prompt, plan, response, success,
                     error=None, latency=None, tool_calls=None) -> str:
    full_response = response if isinstance(response, str) else str(response)
    system_note = f"routed intent={getattr(plan, 'intent', None)}"
    if getattr(plan, "sink", None):
        system_note += f", sink={plan.sink}"
    if getattr(plan, "filename", None):
        system_note += f", filename={plan.filename}"

    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "interaction",
        "session_id": session_id,
        "file_id": file_id,
        "intent": getattr(plan, "intent", None),
        "sink": getattr(plan, "sink", None),
        "filename": getattr(plan, "filename", None),
        "success": success,
        "error": error,
        "latency_sec": round(latency, 2) if latency is not None else None,
        "prompt": prompt,
        "response": full_response,
        # Every tool the agent actually called this turn, with the exact
        # arguments it passed and what the tool returned — alongside the
        # user's own prompt above. Confirmed need: every export bug in this
        # project's history ("called export_data without file_id on retry",
        # "the model never called modify_export at all") could only be
        # confirmed before this by re-running the live server and reading
        # the exported file back by hand; this puts that first check
        # directly in the log line itself.
        "tool_calls": tool_calls or [],
        "messages": [
            {"role": "system", "content": system_note},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response},
        ],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["id"]


def log_ingestion(file_id: str, path: str, kind: str, success: bool,
                   error: str = None, latency: float = None, **stats) -> str:
    try:
        content_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    except Exception:
        content_hash = None
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "ingestion",
        "file_id": file_id,
        "original_filename": os.path.basename(path),
        "extension": os.path.splitext(path)[1].lower(),
        "kind": kind,
        "content_hash": content_hash,
        "success": success,
        "error": error,
        "latency_sec": round(latency, 2) if latency is not None else None,
        **stats,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["id"]


def load_interaction_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(LOG_PATH) if line.strip()]
    return pd.DataFrame(rows)


def file_upload_log() -> pd.DataFrame:
    df = load_interaction_log()
    return df[df["type"] == "ingestion"] if not df.empty else df


def detect_file_id_collisions() -> pd.DataFrame:
    """Flags any file_id ingested more than once with a DIFFERENT content
    hash — i.e. the id silently pointing at different real files."""
    df = file_upload_log()
    if df.empty or "content_hash" not in df.columns:
        return df
    dupes = df.groupby("file_id")["content_hash"].nunique()
    collided_ids = dupes[dupes > 1].index.tolist()
    return df[df["file_id"].isin(collided_ids)].sort_values(["file_id", "timestamp"])


def log_feedback(interaction_id: str, rating: str, notes: str = "") -> None:
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "feedback",
        "interaction_id": interaction_id,
        "rating": rating,
        "notes": notes,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def rate_last(session_id: str = "default", rating: str = "bad", notes: str = "") -> None:
    iid = state.LAST_INTERACTION.get(session_id)
    if not iid:
        return
    log_feedback(iid, rating, notes)


def failure_report() -> pd.DataFrame:
    df = load_interaction_log()
    if df.empty:
        return df
    errored = df[(df["type"] == "interaction") & (df["success"] == False)]
    bad_ids = set(df[(df["type"] == "feedback") &
                      (df["rating"].isin(["bad", "wrong_intent"]))]["interaction_id"])
    flagged = df[(df["type"] == "interaction") & (df["id"].isin(bad_ids))]
    return pd.concat([errored, flagged]).drop_duplicates(subset="id")
