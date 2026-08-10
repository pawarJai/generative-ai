"""Vector-store indexing — isolated so it's the one place that touches Chroma."""
from app.config import vector_db


def embed_text(file_id: str, text: str) -> int:
    """Clears any existing chunks for this file_id, then re-embeds. Always
    clearing first (not 'skip if already indexed') is deliberate: if a
    file_id is ever reused for different content, stale chunks must never
    silently survive in the vector store.

    Returns 0 if text is empty (no embeddings needed)."""

    # Skip if no text to embed
    if not text or not text.strip():
        print(f"(skipped embedding for {file_id}: empty text)")
        return 0

    try:
        existing = vector_db.get(where={"file_id": file_id})
        if existing and existing.get("ids"):
            vector_db.delete(ids=existing["ids"])
    except Exception as e:
        print(f"(dedupe skipped: {e})")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200) \
            .create_documents([text], metadatas=[{"file_id": file_id}])
        if not chunks:
            print(f"(no chunks created for {file_id})")
            return 0
        vector_db.add_documents(chunks)
        return len(chunks)
    except Exception as e:
        print(f"(embedding failed for {file_id}: {e})")
        return 0
