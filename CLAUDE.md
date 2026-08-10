# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Document RAG — a FastAPI backend that processes uploaded documents (PDF, DOCX, XLSX, images, PPTX) and answers queries through a combination of semantic search and exact tabular lookups. Ingests structured data with tabular extraction and header detection, uses DuckDB for text-to-SQL queries against actual data, and exports results to CSV, Excel, Word, and PowerPoint.

## Setup & Run

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# System binaries (required for document processing)
# macOS:  brew install tesseract poppler
# Ubuntu: apt-get install tesseract-ocr poppler-utils

# Configure environment
cp .env.example .env
# Edit .env: fill in OPENROUTER_API_KEY (and optionally override LLM_MODEL, EMBED_MODEL, etc.)

# Run server
uvicorn app.main:app --reload
```

**API docs**: http://localhost:8000/docs (interactive Swagger UI)

## Architecture

### Directory structure

```
app/
  config.py              Centralized LLM/vector-store initialization
  state.py               In-memory session state (file ingestion cache, active tables)
  models.py              Request/response schemas (QueryPlan, ChatRequest, etc.)
  embedding.py           Vector-store indexing helpers
  summary.py             TOC extraction, document overview generation
  logging_utils.py       JSONL logging of all queries and ingestions
  main.py                FastAPI app setup, router registration

  ingestion/             File → text extraction & table detection
    docling_ingest.py    Docling processor for PDF/DOCX/PPTX/images
    tabular_ingest.py    XLSX/CSV: header detection, block splitting, schema inference
    ocr.py               Tesseract fallback for scanned pages
    universal.py         Router: picks the right ingester by file extension

  tables/                Table access & query context
    helpers.py           get_all_real_tables(), resolve_file_scope(), schema API

  query/                 Query understanding & execution
    planner.py           Intent routing: regex patterns first, LLM fallback
    data_query.py        DuckDB text-to-SQL for exact tabular facts
    dispatch.py          Orchestrator: parses QueryPlan, routes to embeddings or DuckDB

  export/                Result formatting
    exporters.py         CSV/Excel/DOCX/PPTX writers
    schema_map.py        Schema-driven result validation & LLM reshaping
    text_export.py       Route prose → file format

  agent/                 LangChain tool-calling agent for complex multi-step queries
    tools.py             Tool definitions

api/
  routes/
    ingest.py            POST /ingest — upload file, extract & index
    chat.py              POST /chat — query against indexed files
    files.py             GET /files, /files/{id}/tables, /files/download/{filename}, /files/logs
```

### Key data flow

1. **Upload** → `ingest.py` → routes to `universal.py` → language-specific ingester → `embedding.py` → Chroma
2. **Tabular upload** (XLSX/CSV) → `tabular_ingest.py` → DuckDB table created, schema cached in `state.py`
3. **Chat query** → `planner.py` (regex routing) → `dispatch.py` orchestrates:
   - **Exact match** (regex pattern found) → `data_query.py` generates SQL via LLM, executes on DuckDB → exact row/column result
   - **Semantic** (no match) → vector search in Chroma + LLM synthesis
4. **Export** → `exporters.py` + `schema_map.py` format result, write to `OUTPUT_DIR`, serve via `/files/download/{filename}`

### Critical architectural decisions

- **`state.py` is the only place holding mutable state** (ingested files, active tables, schemas). Every other module reads/writes through functions, never touches the dict directly. This makes swapping Redis/Postgres later a one-file change.
- **DuckDB + text-to-SQL replaces vector-search guessing** for exact facts. `planner.py` tries regex patterns first (fast, reliable); falls back to LLM SQL generation only for complex cases. This ensures "give me user_id = X" actually finds the row, not a 5-row embedding hallucination.
- **One responsibility per file.** `docling_ingest.py` doesn't know about exporters; `exporters.py` doesn't know about OCR. Bugs are locatable by filename.
- **All results written to `OUTPUT_DIR`**, served via API download endpoint. No stray files in working directory.

### Environment variables

```
OPENROUTER_API_KEY         API key for OpenRouter
OPENROUTER_BASE_URL        Base URL (default: https://openrouter.ai/api/v1)
LLM_MODEL                  Model name (default: qwen/qwen3-30b-a3b-instruct-2507)
EMBED_MODEL                Embedding model (default: BAAI/bge-m3)
EMBED_DEVICE               "cpu" or "cuda" (default: cpu)
DOCLING_CACHE_DIR          Cache directory for Docling (default: ./docling_cache)
CHROMA_DIR                 Chroma vector store directory (default: ./chroma_bge)
LOG_PATH                   JSONL log file (default: ./interaction_log.jsonl)
OUTPUT_DIR                 Export output directory (default: ./outputs)
```

## Common tasks

### Run single query
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "how many rows?", "file_id": "doc1"}'
```

### Upload and query tabular data
```bash
curl -X POST "http://localhost:8000/ingest?file_id=my_data" -F "file=@data.xlsx"
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "give me user_id = U0396", "file_id": "my_data"}'
```

### List uploaded files & their tables
```bash
curl http://localhost:8000/files
curl http://localhost:8000/files/my_data/tables
```

### View logs
```bash
curl http://localhost:8000/files/logs
tail -f interaction_log.jsonl  # or tail local log
```

### Export results
Once a query is executed, results are cached in `OUTPUT_DIR`. Download via:
```bash
curl http://localhost:8000/files/download/result_1.xlsx -o my_result.xlsx
```

## Known limitations (before production)

- **Single-process, in-memory state** (`state.py`). Restart loses all file ingestion. Swap for Redis (session/active-files) + Postgres/S3 (persistent metadata) before scaling.
- **No authentication.** Add API key or OAuth before exposing beyond localhost.
- **Synchronous request handling.** Long ingestion jobs block the request. Wrap in background tasks + webhooks for production.
- **Embedding device defaults to CPU.** Set `EMBED_DEVICE=cuda` for GPU inference.
- **DuckDB is in-memory per table.** For 100MB+ tables, consider direct S3/warehouse queries instead.

## Testing

The project includes logging to help debug issues:
- All queries and ingestions are logged to `LOG_PATH` in JSONL format
- View via `curl http://localhost:8000/files/logs` or `tail -f interaction_log.jsonl`
- Each log entry includes prompt, file_id, route taken (semantic vs. SQL), result, and latency
