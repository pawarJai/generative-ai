"""Vector-store indexing — isolated so it's the one place that touches Chroma."""
from app.config import vector_db

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

def embed_text(file_id: str, text: str) -> int:
    if not text or not text.strip():
        print(f"(no content to embed for '{file_id}')")
        return 0

    # Remove table-of-contents only chunks (lines of dots and dashes)
    import re
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Skip lines that are only dashes, dots, or pipe characters
        stripped = line.strip()
        if stripped and not re.match(r'^[\|\-\.\s]{5,}$', stripped):
            clean_lines.append(line)
    text = '\n'.join(clean_lines)

    if not text.strip():
        print(f"(all content was TOC/separator lines for '{file_id}')")
        return 0

    try:
        existing = vector_db.get(where={"file_id": file_id})
        if existing and existing.get("ids"):
            vector_db.delete(ids=existing["ids"])
    except Exception as e:
        print(f"(dedupe skipped: {e})")

    # Split by markdown headers first to preserve document structure
    headers_to_split_on = [
        ("#", "h1"), ("##", "h2"), ("###", "h3"),
    ]
    try:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        header_splits = md_splitter.split_text(text)
    except Exception:
        header_splits = []

    # Split remaining content — keep table rows together
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n|", "|\n", "\n", ". ", " "],
    )

    final_chunks = []
    if header_splits:
        for doc in header_splits:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            meta['file_id'] = file_id
            # Skip TOC-only sections
            if re.match(r'^[\|\-\.\s\n]{0,200}$', content.strip()):
                continue
            if len(content) > 1000:
                sub = char_splitter.create_documents([content], metadatas=[meta])
                final_chunks.extend(sub)
            elif content.strip():
                from langchain_core.documents import Document
                final_chunks.append(Document(page_content=content, metadata=meta))
    else:
        from langchain_core.documents import Document
        final_chunks = char_splitter.create_documents(
            [text], metadatas=[{'file_id': file_id}])

    if not final_chunks:
        print(f"(no chunks produced for '{file_id}')")
        return 0

    # Filter out chunks that are still mostly punctuation/separators
    good_chunks = []
    for chunk in final_chunks:
        content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        words = len(re.findall(r'\b[a-zA-Z]{2,}\b', content))
        if words >= 5:  # at least 5 real words
            good_chunks.append(chunk)

    if not good_chunks:
        print(f"(no meaningful chunks after filtering for '{file_id}')")
        return 0

    vector_db.add_documents(good_chunks)
    print(f"Embedded {len(good_chunks)} chunks for '{file_id}'")
    return len(good_chunks)
