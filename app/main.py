"""FastAPI app entry point. Run with: uvicorn app.main:app --reload"""
import os
# Confirmed production failure: HuggingFace's tokenizers library forks a
# background worker process on first use when parallelism isn't explicitly
# disabled. Under uvicorn --reload that fork has been observed inheriting
# the server's own listening socket file descriptor — so after the real
# ASGI worker exits (a reload, a crash), that leftover forked process keeps
# `lsof -i :8000` showing something LISTENing, giving no indication the app
# itself is gone. New connections then hang forever with zero server-side
# logging, because nothing is actually there to accept() and route them.
# Must be set before torch/transformers/tokenizers are imported anywhere,
# so this line has to stay the first thing this entry module does.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import ingest, chat, files
from api.routes import jobs as jobs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: restore_state() re-ingests all files on boot — disabled for now
    # because large PDF files (49+ tables) take hours. Enable only after
    # implementing lazy restore (only on first access) or background ingestion.
    # from app.startup import restore_state
    # restore_state()
    yield


app = FastAPI(
    title="Agentic Document RAG API",
    description="Upload documents/spreadsheets, then query them via chat — "
                "overview, table listing, exact data queries (DuckDB), and "
                "exports to csv/excel/docx/pptx/chart.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(jobs_router.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
