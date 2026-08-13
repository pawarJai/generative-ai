"""Tools available to the LangGraph agent.
Each tool is a real function with a real docstring — the LLM reads
the docstring to decide when to use it. No regex routing.
"""
from typing import Optional
from langchain_core.tools import tool
from app import state as app_state
from app.tables.helpers import get_all_real_tables, get_page_markdown
from app.config import vector_db


def resolve_target_file(file_id: Optional[str], question: str = "") -> Optional[str]:
    """The one document a tool should read, in descending order of evidence.

    Confirmed production failure (22:44:10): the model called export_data
    without file_id for the prompt "in data file 2 page number 6 and 7 …",
    and the tool fell back to `FILE_ORDER[0]` — the OLDEST file loaded into
    the process, unrelated to the chat, the selection, or the file the user
    named. It exported data-file-1's page 6 (2 rows) and reported that
    data-file-2's page 7 "contained no extractable tables". Both halves of
    that sentence were about the wrong document, and the export still
    reported success, so nothing looked wrong.

    Order: the explicit argument, then the document named in the question,
    then the active file, and only then a lone loaded file. With several
    candidates and no signal, returns None — a tool saying "which file?"
    beats a tool silently answering about a document nobody mentioned.
    """
    if file_id:
        return file_id

    if question:
        from app.graph.agent import _find_named_files
        named = _find_named_files(question)
        if len(named) == 1:
            return named[0][1]

    active = app_state.get_active_file_id()
    if active:
        return active

    if len(app_state.FILE_ORDER) == 1:
        return app_state.FILE_ORDER[0]
    return None


@tool
def search_documents(query: str, file_id: str = None) -> str:
    """Search uploaded documents semantically for a TOPIC or CONCEPT — terms,
    conditions, company info, product details, specifications, anything
    written in a PDF or document where you don't know which page it's on.
    Also use when the user asks about text content in spreadsheets.

    Do NOT use this for a specific page number (use get_page_content) or for
    "table of contents" / "what sections exist" / "indexing" questions (use
    get_table_of_contents) — this tool returns the top few similar-looking
    chunks, not the exact page or a real structural list, and will guess
    wrong on those question types.

    Args:
        query: The search query in natural language
        file_id: Optional specific file to search. If None, searches all files.
    """
    filter_dict = {"file_id": file_id} if file_id else None
    try:
        results = vector_db.similarity_search(query, k=8, filter=filter_dict)
        if not results:
            return "No relevant content found in uploaded documents."
        chunks = "\n\n---\n\n".join(
            f"[From: {r.metadata.get('file_id', 'unknown')}]\n{r.page_content}"
            for r in results
        )
        return f"Found {len(results)} relevant sections:\n\n{chunks}"
    except Exception as e:
        return f"Search failed: {e}"


@tool
def get_page_content(page_number: int, file_id: str = None) -> str:
    """Get the EXACT content of a specific page number from an uploaded
    document. Use this whenever the user names a page number — "what's on
    page 6", "what does page 12 say" — instead of search_documents, which
    only returns similar-looking chunks and is not guaranteed to be that
    exact page. Only works for PDF/DOCX/PPTX/image files (not spreadsheets).

    Args:
        page_number: The page number to retrieve
        file_id: Optional specific file. If None, uses the active file.
    """
    target = file_id or app_state.get_active_file_id()
    if not target:
        return "No file selected."
    if app_state.FILE_KIND.get(target) != "docling":
        return (f"'{target}' is not a paginated document (it's a "
                f"spreadsheet or unrecognized file) — page lookup doesn't apply.")
    md = get_page_markdown(target, page_number)
    if md.strip():
        return md

    # A dead end used to end the turn here, and the model would improvise a
    # cause ("empty, corrupted, or not a valid spreadsheet"). Report the real
    # reason instead, and search the file's indexed text — Chroma persists
    # across restarts, so that content is still available.
    from app.tables.helpers import get_page_count
    from app.query.semantic import semantic_fallback

    n_pages = get_page_count(target)
    if n_pages is not None and page_number > n_pages:
        # Not a missing-content problem at all — don't muddy it with
        # semantically "close" text from an unrelated page.
        return (f"This document has {n_pages} pages, so page {page_number} "
                f"does not exist. The file is fine — tell the user the real "
                f"page range instead of suggesting a re-upload.")

    hit = semantic_fallback(target, f"page {page_number}")
    if hit:
        return (f"Page {page_number} has no separately extractable text (it may "
                f"be a scan or an image-only page). Closest matching content "
                f"from this file's indexed text — this is a SEMANTIC match, not "
                f"a guaranteed page-{page_number} match, so say so when "
                f"answering:\n\n{hit}")
    return (f"Page {page_number} has no extractable text, and no similar content "
            f"was found in this file's indexed text. The file itself is present "
            f"and was ingested successfully — do NOT tell the user it is empty, "
            f"corrupted, or needs re-uploading.")


@tool
def get_table_of_contents(file_id: str = None) -> str:
    """Get the real, extracted table of contents / heading list for an
    uploaded document — every section heading found during ingestion, with
    its page number. Use this for "table of contents", "what sections/
    chapters exist", "what page is section X on", "give me the indexing" —
    anything about the document's STRUCTURE. Do NOT use search_documents for
    these; it only returns a handful of similar text chunks, not a real
    structural list, and will produce an inconsistent or invented answer.

    Args:
        file_id: Optional specific file. If None, uses the active file.
    """
    target = file_id or app_state.get_active_file_id()
    if not target:
        return "No file selected."
    toc = app_state.FILE_META.get(target, {}).get("toc", [])
    if not toc:
        return f"No table of contents / headings were detected in '{target}'."
    lines = [f"- {t['text']} (p.{t['page']})" for t in toc]
    return f"Table of contents for '{target}' ({len(toc)} headings):\n" + "\n".join(lines)


@tool
def query_table_data(question: str, file_id: str = None) -> str:
    """Query tabular data (Excel sheets, CSV files) using natural language.
    Use this for: row counts, column values, filtering, aggregations,
    specific data lookups, exporting data. Any question about numbers,
    rows, or structured data in spreadsheets.

    Args:
        question: Natural language question about the data
        file_id: Optional specific file to query. If None, queries all files.
    """
    from app.query.code_exec import run_code_on_files
    # Resolved rather than defaulted to FILE_ORDER — see resolve_target_file.
    # Querying "all loaded files" sounds harmless here, but the answer then
    # depends on which files happen to be in memory, so the same question
    # gives different answers before and after a restart.
    target = resolve_target_file(file_id, question)
    target_files = [target] if target else list(app_state.FILE_ORDER)
    if not target_files:
        return "No files uploaded yet."
    result = run_code_on_files(target_files, question)
    if result.get("table"):
        import pandas as pd
        df = pd.DataFrame(result["table"]["rows"],
                          columns=result["table"]["columns"])
        return f"{result['text']}\n\n{df.to_markdown(index=False)}"
    return result["text"]


@tool
def list_uploaded_files() -> str:
    """List all files the user has uploaded in this session.
    Use when the user asks: 'what files do I have', 'how many files',
    'what did I upload', 'show my documents'.
    """
    import os
    from app.persistence import get_all_files

    # Scoped to THIS chat. Documents belong to the conversation they were
    # uploaded into, so "what did I upload" must mean this chat's documents —
    # answering from the whole registry reports other conversations' files.
    #
    # Read from the durable registry, not in-memory state: state is emptied by
    # every server restart, which made this tool answer "you have uploaded
    # 1 file" to a user with a hundred — and made the assistant recommend
    # re-uploading files that were still perfectly available.
    session = app_state.get_active_session_id()
    records = [r for r in get_all_files(session) if os.path.exists(r["path"])]
    if not records:
        if not app_state.FILE_ORDER:
            return ("No documents have been uploaded in this chat yet. "
                    "Documents belong to the chat they were uploaded into.")
        records = [{"file_id": fid,
                    "original_filename": app_state.FILE_ORIGINAL_NAME.get(fid, fid),
                    "kind": app_state.FILE_KIND.get(fid, "unknown"),
                    "path": ""} for fid in app_state.FILE_ORDER]

    lines = []
    for i, r in enumerate(records, 1):
        fid = r["file_id"]
        if fid in app_state.FILE_KIND:
            detail = f"{len(get_all_real_tables(fid))} table(s)"
        else:
            # Counting tables would force a full re-parse of every file just
            # to answer "what do I have" — report availability instead.
            detail = "available, loads on first use"
        lines.append(f"{i}. {r['original_filename']} — {r['kind']}, {detail}, id={fid}")
    return (f"{len(records)} document(s) uploaded in this chat:\n"
            + "\n".join(lines))


@tool
def export_data(question: str, format: str = "excel",
                filename: str = None, file_id: str = None) -> str:
    """Export data from uploaded files to a file.
    IMPORTANT: format must be one of: csv, excel, docx, pptx
    NEVER use 'pdf' as format — use 'excel' for spreadsheets.
    Use when the user says: 'export', 'save to', 'download', 'create a file'.

    Args:
        question: What data to export (e.g. 'the Group column from Working Sheet')
        format: Output format — 'csv', 'excel', 'docx', 'pptx'
        filename: Optional output filename
        file_id: Optional specific file to export from
    """
    from app.query.code_exec import run_code_on_files
    from app.export.exporters import EXPORTERS, verify_export
    from app.models import QueryPlan
    import pandas as pd
    import os

    VALID_FORMATS = {"csv", "excel", "docx", "pptx"}
    if format not in VALID_FORMATS:
        format = "excel"  # safe default — never produce a .pdf
    if filename and filename.lower().endswith(".pdf"):
        filename = filename[:-4] + ".xlsx"

    from app.graph.agent import (_extract_requested_pages,
                                  _deterministic_page_export,
                                  _resolve_source_specs,
                                  _deterministic_multi_export)

    default_name = filename or f"export.{'xlsx' if format == 'excel' else format}"

    # Multi-document FIRST. "Combine the tables from all my documents" names
    # no single file on purpose, so resolving a single target before this
    # made the ask-which-file guard below swallow the request and reply
    # "which file did you mean?" — the one question a combine-everything
    # request has already answered.
    specs = _resolve_source_specs(question, file_id)
    if len(specs) > 1:
        return _deterministic_multi_export(specs, question, default_name, format)

    # The single most damaging bug found in this project: with file_id
    # omitted this used to become FILE_ORDER[0] and export a document the
    # user never mentioned, while reporting success. See resolve_target_file.
    target = resolve_target_file(file_id, question)
    if not target and not app_state.FILE_ORDER:
        return "No files uploaded yet."
    if not target:
        names = ", ".join(f"{app_state.FILE_ORIGINAL_NAME.get(f, f)} (id={f})"
                          for f in app_state.FILE_ORDER)
        return ("Several documents are loaded and the request does not say "
                f"which one to export from: {names}. Ask the user which file "
                "they mean, or pass file_id.")
    target_files = [target]

    # "export page N and M" goes through exact page→table matching on the
    # real Docling objects instead of the code-exec LLM sandbox. The sandbox
    # was confirmed in production to either substitute a wrong-numbered
    # table or refuse outright when one of the requested pages holds no
    # table — the latter losing the pages that DO have data. The same
    # routing already guarded the recovery backstop; without it here, a
    # direct tool call from the agent still hit the old behaviour.
    # Read the page numbers out of the user's own words first. The model's
    # paraphrase drops them, or narrows them: asked to export "page number 6
    # to 10" it called this tool with a question naming no page at all, so
    # the request fell through to the sandbox, which answered that the
    # document "does not contain a table with the exact columns" — about a
    # table that is plainly there on pages 7 to 10.
    intent_text = app_state.get_current_user_prompt() or question
    pages = _extract_requested_pages(intent_text) or _extract_requested_pages(question)
    if pages and app_state.FILE_KIND.get(target_files[0]) == "docling":
        from app.graph.agent import _WHOLE_TABLE_RE, _wants_bare_table
        # Judge "give me the whole table" and "without the header" from the
        # user's own words too; the model's paraphrase routinely drops those
        # qualifiers, and "without header data" then produced a file with the
        # header band on it twice running.
        return _deterministic_page_export(
            target_files[0], pages, default_name, format,
            whole_table=bool(_WHOLE_TABLE_RE.search(intent_text)),
            no_context=_wants_bare_table(intent_text))

    result = run_code_on_files(target_files, question)
    if not result.get("table"):
        return f"Could not extract data: {result['text']}"

    df = pd.DataFrame(result["table"]["rows"], columns=result["table"]["columns"])
    out_filename = filename or f"export.{format}"
    plan = QueryPlan(intent="export", sink=format, filename=out_filename)
    export_fn = EXPORTERS.get(format, EXPORTERS["csv"])
    msg = export_fn(target_files[0], plan, tables=[df])

    # CRITICAL: verify the file actually landed on disk
    from app.config import OUTPUT_DIR
    path = os.path.join(OUTPUT_DIR, out_filename)
    ok, verify_msg = verify_export(path, len(df), format)
    if not ok:
        return f"Export attempted but verification failed: {verify_msg}"
    return f"{verify_msg}\n\nData preview:\n{df.head(3).to_markdown(index=False)}"


@tool
def modify_export(filename: str, change: str = "", file_id: str = None) -> str:
    """Change a file that has ALREADY been exported, in place.

    Use this — never export_data — when the user refers to a file that exists
    and wants it altered: 'add the header details to that excel', 'rename the
    first two columns', 'remove the SL no column', 'take the header off'.

    Args:
        filename: The existing output file, e.g. 'f1-123.xlsx'
        change: What to change, in the user's own words
        file_id: Optional source document, if the header must be re-read
    """
    from app.export.modify import modify

    # The user's literal words first. The phrasing that failed in production
    # — "we need to add this Full header section data also in this excel top
    # section" — is exactly the kind a paraphrase flattens into something with
    # no operation left in it.
    spoken = app_state.get_current_user_prompt() or ""
    instruction = spoken if _names_an_operation(spoken) else (change or spoken)
    if not filename:
        recent = sorted(app_state.WRITTEN_EXPORTS)
        if not recent:
            return "No file has been exported in this session yet."
        filename = recent[-1]
    return modify(filename, instruction, file_id)


def _names_an_operation(text: str) -> bool:
    from app.export.modify import parse_instruction
    ops = parse_instruction(text)
    return bool(ops["rename"] or ops["positional_rename"] or ops["drop"]
                or ops["context"] is not None)


@tool
def get_file_overview(file_id: str = None) -> str:
    """Get a summary/overview of an uploaded file — what it contains,
    its structure, key information. Use when the user asks:
    'what is this file about', 'summarize', 'overview', 'describe'.

    Args:
        file_id: Optional specific file. If None, summarizes the active file.
    """
    target = file_id or app_state.get_active_file_id()
    if not target:
        return "No file selected."
    meta = app_state.FILE_META.get(target, {})
    summary = meta.get("summary", "No summary available.")
    name = app_state.FILE_ORIGINAL_NAME.get(target, target)
    tables = get_all_real_tables(target)
    sheet_info = "\n".join(
        f"  - {t.attrs.get('page')}: {t.shape[0]} rows × {t.shape[1]} cols"
        for t in tables[:10]
    )
    return (f"File: {name}\n\nSummary: {summary}"
            + (f"\n\nSheets/Tables:\n{sheet_info}" if sheet_info else ""))


@tool
def generate_quotation(rfq_description: str, file_ids: list = None) -> str:
    """Generate a quotation document from an RFQ (Request for Quotation).

    Use when user says: 'generate quotation', 'create quote',
    'make quotation from RFQ', 'fill quotation template'.

    Extracts product requirements from RFQ documents, matches to
    available pricing data, calculates totals per shipset, and
    generates a pre-filled Excel for Product Manager review.

    Args:
        rfq_description: Description of what to quote (or "from uploaded RFQ")
        file_ids: Files to use (None = all uploaded files)
    """
    from app.query.code_exec import run_code_on_files
    import pandas as pd
    import os
    from app.config import OUTPUT_DIR
    from app.export.exporters import export_excel, verify_export
    from app.models import QueryPlan

    targets = file_ids or list(app_state.FILE_ORDER)
    if not targets:
        return "No files uploaded. Upload RFQ and product database first."

    # Step 1: Extract line items from RFQ
    items_result = run_code_on_files(
        targets,
        "Extract all product items with Tag No, description, quantity per "
        "shipset, and number of shipsets. Return as a DataFrame with columns: "
        "Tag_No, Description, Qty_per_Shipset, Shipsets, Total_Qty"
    )

    if not items_result.get("table"):
        return (f"Could not extract line items from uploaded files. "
                f"Make sure you've uploaded the RFQ document.\n"
                f"Details: {items_result['text']}")

    df = pd.DataFrame(items_result["table"]["rows"],
                      columns=items_result["table"]["columns"])

    # Step 2: Add pricing columns for Product Manager to fill
    for col in ["Unit_Rate_INR", "Total_Value_per_Shipset",
                 "Total_Value_All_Shipsets", "GST_Percent", "Grand_Total"]:
        if col not in df.columns:
            df[col] = ""

    # Step 3: Export for PM review, verified against disk like every other export
    filename = "quotation_draft.xlsx"
    plan = QueryPlan(intent="export", sink="excel", filename=filename)
    export_excel(targets[0], plan, tables=[df])

    path = os.path.join(OUTPUT_DIR, filename)
    ok, verify_msg = verify_export(path, len(df), "excel")
    if not ok:
        return f"Quotation generation failed: {verify_msg}"

    return (f"Quotation draft created: {filename} ({verify_msg})\n"
            f"Contains {len(df)} line items ready for Product Manager review.\n"
            f"Open the file to fill in Unit Rate (INR) and pricing details.")


@tool
def analyze_past_contracts(question: str, file_ids: list = None) -> str:
    """Analyze past contracts, bids, and quotations for patterns.

    Use when user asks: 'why did we lose', 'what went wrong',
    'analyze past bids', 'contract history', 'what should we improve',
    'client patterns', 'rejection reasons'.

    Searches through past contract documents and finds patterns in
    wins, losses, rejection reasons, pricing, and client preferences.

    Args:
        question: What to analyze about past contracts
        file_ids: Contract files to analyze (None = all uploaded)
    """
    from app.query.code_exec import run_code_on_files

    targets = file_ids or list(app_state.FILE_ORDER)
    if not targets:
        return "No files uploaded. Upload past contract documents first."

    # Semantic search for contract patterns
    semantic_result = search_documents.invoke({
        "query": question + " contract rejection win loss bid",
        "file_id": None
    })

    # Also check tabular data for pricing patterns
    table_result = run_code_on_files(
        targets,
        f"Find data relevant to: {question}. "
        "Look for columns with price, status, outcome, rejection, win/loss."
    )

    response = f"Contract Analysis:\n\n{semantic_result}"
    if table_result.get("table"):
        import pandas as pd
        df = pd.DataFrame(table_result["table"]["rows"],
                          columns=table_result["table"]["columns"])
        response += f"\n\nData found:\n{df.to_markdown(index=False)}"
    return response
