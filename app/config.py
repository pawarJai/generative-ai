"""
All external-service / model configuration lives here — one place to change
the LLM, embedding model, or vector store. Never import langchain clients
directly in feature modules; import them from here.
"""
import functools
import os
import threading

# Confirmed production failure: file ingestion "stuck pending" for 30+
# minutes, reproduced in complete isolation (single process, nothing else
# running) — a stack sample during the hang showed PyTorch's `torch._dynamo`
# (torch.compile) mid-JIT-compiling Docling's internal rt_detr_v2 table/
# layout-detection model, having just hit a "Graph break" inside a
# Tensor.item() call and fallen into recompilation. torch.compile's first-run
# compilation cost is well known to run into many minutes on CPU, especially
# after a graph break forces a retry — and it buys nothing here, since this
# app processes each document once rather than running the same shape
# repeatedly in a tight loop, so compilation cost is never amortized. Must be
# set before torch/transformers/docling are imported anywhere, so this has to
# stay the first thing this module does (same reasoning as
# TOKENIZERS_PARALLELISM in app/main.py — but this module is the actual
# common import root for both the live server AND any standalone script, so
# it belongs here rather than only in the FastAPI entry point).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

load_dotenv()

CACHE_DIR = os.getenv("DOCLING_CACHE_DIR", "./docling_cache")
LOG_PATH = os.getenv("LOG_PATH", "./interaction_log.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
    encode_kwargs={"normalize_embeddings": True},
)

_raw_vector_db = Chroma(
    persist_directory=os.getenv("CHROMA_DIR", "./chroma_bge"),
    collection_name="docling_rag_store",
    embedding_function=embeddings,
)


class _SerializedVectorStore:
    """Funnels every ChromaDB call through one process-wide lock.

    DEFENSIVE ONLY — this was NOT the cause of the SIGSEGV crash that
    prompted it, and the honest record matters here. A macOS crash report
    showed nine threads blocked on one `std::__1::mutex::lock()` inside
    chromadb_rust_bindings.abi3.so and a tenth crashing on a null
    dereference at the next instruction, which looked like a thread-safety
    failure. It was not: the same crash reproduces with a SINGLE thread
    doing one add_documents(), and does NOT reproduce against a freshly
    created store. The real cause was on-disk corruption of the persisted
    store (see the CHROMA_DIR note below); those nine threads were
    Chroma's own internal tokio/sqlx workers, a symptom rather than the
    trigger.

    Kept because it is cheap and this app does legitimately touch the
    store from several threads at once — ingestion runs on a background
    worker (api/routes/ingest.py) calling embed_text ->
    get/delete/add_documents while /chat requests concurrently call
    similarity_search — and serializing that costs nothing measurable.
    Wrapping here covers all seven call sites across app/embedding.py,
    app/query/semantic.py, app/query/dispatch_legacy.py and
    app/graph/tools.py without each having to remember the lock.

    On corruption: Chroma's HNSW index is not crash-safe. SIGKILLing the
    server mid-write (or a `uvicorn --reload` restart landing during an
    embedding step) can leave the store in a state where the next write
    segfaults the whole process — which then looks like "upload stuck on
    pending forever", because the background ingestion task dies with no
    traceback. If that happens again: stop the server, move the
    CHROMA_DIR aside, and let it rebuild; re-upload the documents.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_lock", threading.RLock())

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_inner"), name)
        if not callable(attr):
            return attr
        lock = object.__getattribute__(self, "_lock")

        @functools.wraps(attr)
        def _locked(*args, **kwargs):
            with lock:
                return attr(*args, **kwargs)

        return _locked


vector_db = _SerializedVectorStore(_raw_vector_db)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    temperature=0.0,
    max_tokens=4096,
    # Confirmed production failure: with no timeout set, a stalled call to
    # OpenRouter blocked a /chat request for 15+ minutes with near-zero CPU
    # use — the process wasn't stuck computing, it was waiting on a network
    # read that had no deadline. Every LLM call in this app goes through
    # this one client, so bounding it here bounds all of them at once,
    # including the code-exec sandbox's retry loop, which calls this
    # multiple times per request and would otherwise multiply an unbounded
    # hang instead of just having one.
    timeout=90,
    max_retries=1,
)

# Confirmed production failure: file ingestion "stuck pending" for 8+
# minutes on a 560KB, single-digit-page PDF — reproduced in isolation, with
# a stack sample during the hang showing genuine (not stuck) but very heavy
# PyTorch CPU inference. Root cause: DocumentConverter() with no pipeline
# options runs Docling's OWN defaults — do_ocr=True (runs full OCR analysis
# on EVERY page, unconditionally) and TableFormerMode.ACCURATE (the
# slowest table-structure model) — neither of which this app needs:
#   - do_ocr=False: this codebase already has its own selective OCR
#     fallback (app/ingestion/ocr.py's page_text_is_thin -> ocr_page),
#     added specifically so only pages with genuinely thin extracted text
#     pay the OCR cost. Docling's own do_ocr=True ran OCR analysis on
#     every page regardless, unconditionally, doubling the work on thin
#     pages and paying for it on every other page for nothing.
#   - TableFormerMode.FAST: table-structure recognition's own documented
#     speed/accuracy tradeoff. Accuracy loss here is absorbed by this
#     project's existing downstream repair logic in app/tables/helpers.py
#     (_recover_header, _restore_dropped_labels, _columns_are_positional),
#     which was already built to tolerate imperfect raw extraction —
#     ACCURATE mode's extra precision was never actually being relied on.
_pdf_pipeline_options = PdfPipelineOptions()
_pdf_pipeline_options.do_ocr = False
_pdf_pipeline_options.table_structure_options.mode = TableFormerMode.FAST

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_pipeline_options),
    }
)
