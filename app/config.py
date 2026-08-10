"""
All external-service / model configuration lives here — one place to change
the LLM, embedding model, or vector store. Never import langchain clients
directly in feature modules; import them from here.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter

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

vector_db = Chroma(
    persist_directory=os.getenv("CHROMA_DIR", "./chroma_bge"),
    collection_name="docling_rag_store",
    embedding_function=embeddings,
)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    temperature=0.0,
    max_tokens=4096,
)

_converter = DocumentConverter()
