"""
Pydantic models. QueryPlan is the internal routing object (unchanged logic
from the notebook, now with 'data_query' added — see app/query/data_query.py).
The rest are the FastAPI request/response schemas for the HTTP layer.
"""
from typing import Dict, Optional, List, Literal
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    intent: Literal[
        "qa", "page_lookup", "list_columns", "export", "complex",
        "overview", "table_of_contents", "generate", "row_sample", "data_query",
        "chat_history", "ai_meta", "general", "list_files",
        "generate_quotation", "contract_analysis", "list_sheets"
    ] = Field(
        description="qa=general document Q&A via vector search; page_lookup=specific "
                    "page; list_columns=table headers/shape; export=pull EXISTING "
                    "tables FROM the uploaded document and save them; overview=summarize "
                    "the uploaded document; table_of_contents=list chapters/sections; "
                    "generate=CREATE brand-new content from a spec the user gave, never "
                    "sourced from the uploaded document; complex=multi-step requests; "
                    "row_sample=legacy deterministic row preview; data_query=run real "
                    "pandas code against the loaded tables for any factual/tabular "
                    "question (counts, filters, lookups, sheet names, column values) — "
                    "may span multiple uploaded files at once; "
                    "chat_history=show recent conversation history; "
                    "ai_meta=question about the assistant's own behaviour or errors; "
                    "general=question with no connection to any uploaded file (general "
                    "knowledge, math, weather, casual conversation) — never touches "
                    "vector search or file state; "
                    "list_files=meta question about which files have been uploaded; "
                    "list_sheets=the sheets/tabs of a spreadsheet, read from the "
                    "workbook itself rather than counted from extracted tables; "
                    "generate_quotation=build a pre-filled quotation Excel from an RFQ; "
                    "contract_analysis=analyze past contracts/bids for win-loss patterns."
    )
    page_number: Optional[int] = None
    sink: Optional[Literal["csv", "excel", "docx", "pptx", "chart"]] = Field(
        None, description="If set, ALSO save the result to this file format.")
    filename: Optional[str] = None
    rename: Optional[Dict[str, str]] = None
    columns: Optional[List[str]] = None
    file_scope: Optional[List[str]] = None
    n_rows: Optional[int] = None
    sheet_name: Optional[str] = None
    no_context: bool = Field(
        False, description="Suppress the document context band (letterhead, "
                           "section heading, provenance) that is otherwise "
                           "written above the table. Set when the user asks "
                           "for the bare table only.")


class IngestResponse(BaseModel):
    file_id: str
    original_filename: str
    kind: str
    tables_found: int
    ocr_pages: int = 0
    latency_sec: float


class FileSummary(BaseModel):
    file_id: str
    original_filename: str
    kind: str
    summary: Optional[str] = None
    # "loaded"  — parsed objects are in memory, usable right now
    # "on_disk" — registered and the upload still exists; restored on first use
    # "missing" — registered but the source upload is gone from disk
    status: str = "loaded"


class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    file_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    sink: Optional[str] = None
    filename: Optional[str] = None
    file_id: Optional[str] = None
    table: Optional[dict] = None
    sql: Optional[str] = None
    available_files: List[dict] = Field(default_factory=list,
        description="Every file currently ingested: [{file_id, name, kind}]")
    target_files: List[str] = Field(default_factory=list,
        description="Which file_id(s) this specific answer actually used.")
