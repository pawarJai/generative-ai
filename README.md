# Agentic Document RAG — FastAPI

## Project layout

```
app/
  main.py              FastAPI app + router registration
  config.py             All model/vector-store config (the ONLY place llm/vector_db are built)
  state.py              In-memory session state (swap for Redis/DB in production)
  models.py              QueryPlan + API request/response schemas
  embedding.py           Vector-store indexing
  summary.py             Overview generation, TOC extraction
  logging_utils.py       JSONL interaction/ingestion log
  ingestion/
    ocr.py                Tesseract OCR fallback for scanned pages
    docling_ingest.py      PDF/image/docx/pptx ingestion (Docling)
    tabular_ingest.py      XLSX/CSV ingestion (merge-unfill, header detection, block split)
    universal.py           Routes by extension -> the right ingester
  tables/
    helpers.py              Table access layer (get_all_real_tables, resolve_file_scope, etc.)
  query/
    planner.py               Intent routing (regex-first, LLM fallback)
    data_query.py            DuckDB text-to-SQL engine for exact tabular facts
    dispatch.py               Orchestrator: executes a QueryPlan
  export/
    exporters.py               csv/excel/docx/pptx/chart writers
    schema_map.py               Schema-driven reshape + validation/retry
    text_export.py              Prose -> file-format routing
  agent/
    tools.py                    LangChain tool-calling agent (for "complex" intent)

api/
  routes/
    ingest.py   POST /ingest
    chat.py     POST /chat
    files.py    GET /files, /files/{id}/tables, /files/download/{filename}, /files/logs
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY

# system binaries (OCR + PDF table extraction, if you use pytesseract elsewhere)
# macOS:  brew install tesseract poppler
# Ubuntu: apt-get install tesseract-ocr poppler-utils
```

## Run

```bash
uvicorn app.main:app --reload
```

Docs at http://localhost:8000/docs (Swagger UI — try every endpoint interactively).

## Example flow

```bash
curl -X POST "http://localhost:8000/ingest?file_id=doc1" -F "file=@working-sheet.xlsx"

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "prompt": "how many rows and columns in the working sheet",
  "file_id": "doc1"
}'

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "prompt": "give me user_id = U0396",
  "file_id": "doc1"
}'

curl http://localhost:8000/files
curl http://localhost:8000/files/doc1/tables
```

## Why this structure (vs. the original notebook)

- **One responsibility per file.** `docling_ingest.py` doesn't know about
  exporters; `exporters.py` doesn't know about OCR. Bugs are findable by
  filename, not by scrolling a 1,400-line script.
- **`state.py` is the only place holding mutable globals.** Every other
  module reads/writes through it via functions, never touches the dict
  directly — this is what makes swapping in Redis later a one-file change.
- **`data_query.py` replaces regex-guessed row/column lookups with real
  DuckDB SQL** generated against the actual schema and executed against the
  actual data — this is what fixes "give me user_id = X" actually finding
  the row (or truthfully returning zero rows), instead of vector-search
  guessing from a 5-row embedding summary.
- **Every exporter writes to `OUTPUT_DIR`**, and `/files/download/{filename}`
  serves it — this is the missing piece the notebook never had (files were
  written to the working directory with no way to retrieve them via API).

## Known limitations to design around before production

- `state.py` is single-process, in-memory — restart loses all ingested
  file state. Fine for a demo/single dev server; not fine for multiple
  uvicorn workers or a redeploy. Swap for Redis (session/active-file) +
  Postgres or S3 (cached tables/metadata) before scaling.
- No auth on any endpoint — add an API key / OAuth dependency before
  exposing this beyond localhost.
- `EMBED_DEVICE` defaults to `cpu` here (the notebook had `mps` hardcoded
  for Apple Silicon) — set it explicitly for your hardware.
