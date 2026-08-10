"""Plain-text (.txt) ingestion: read → embed. No table extraction."""
from app import state
from app.embedding import embed_text


def ingest_plain_text(path: str, file_id: str) -> None:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    state.FILE_KIND[file_id] = "text"
    state.FILE_META[file_id] = {"toc": []}
    embed_text(file_id, text)
