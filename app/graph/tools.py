"""Tools available to the LangGraph agent.
Each tool is a real function with a real docstring — the LLM reads
the docstring to decide when to use it. No regex routing.
"""
import os
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
                                  _deterministic_multi_export,
                                  _side_by_side_plan,
                                  _extract_requested_sheet,
                                  _deterministic_sheet_export)

    default_name = filename or f"export.{'xlsx' if format == 'excel' else format}"

    # A side-by-side request must never be answered by stacking, whichever
    # route reached this tool. The model choosing export_data is only one of
    # them: the unverified-claim backstop re-invokes export_data directly, so
    # a rule aimed at the model would not have covered the recovery path that
    # actually wrote the 202-row vertical file in session 477b296a.
    spoken = app_state.get_current_user_prompt() or question
    sbs = _side_by_side_plan(spoken, file_id)
    if sbs:
        return merge_sources_side_by_side(sbs, default_name)

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

    # A named SHEET of a spreadsheet, matched against the file's own real
    # sheet names first — before the sandbox ever sees a question, and
    # before _clean_export_question can mangle one. Confirmed production
    # failure: "export sheet name = cover sheet into sheet-01.xlsx" had its
    # sheet name deleted by the directive-stripping regex below, because it
    # sat between the words "export" and "file".
    sheet_name = (_extract_requested_sheet(intent_text, target_files[0])
                 or _extract_requested_sheet(question, target_files[0]))
    if sheet_name:
        return _deterministic_sheet_export(
            target_files[0], sheet_name, default_name, format)

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


def _merge_suffix(display_name: str) -> str:
    """A short tag that says which document a column came from.

    Derived from the file's own name — never from a fixed list of expected
    documents. A name ending in a number becomes f<number> ('data-file-2' ->
    'f2', 'tender 14.pdf' -> 'f14'), which is how users refer to these files
    in the chat; anything else falls back to a slug of the name.
    """
    import re as _re
    stem = _re.sub(r"[^a-z0-9]+", "", os.path.splitext(str(display_name))[0].lower())
    trailing = _re.search(r"(\d+)$", stem)
    return f"f{trailing.group(1)}" if trailing else (stem[:6] or "src")


def _side_frame(file_id: str, pages: Optional[list]):
    """One side of a horizontal merge: (dataframe, error, pages_used).

    Aligned, never pd.concat'd. Concatenating fragments whose columns differ
    unions them positionally, which is the defect that produced the file this
    feature exists to replace — outputs/test_f1-2f.xlsx, where data-file-2's
    34 rows sat under col_0..col_3 with every one of data-file-1's named
    columns blank beside them.
    """
    from app.graph.agent import _restore_file_if_needed
    from app.tables.helpers import get_all_real_tables, assemble_pages
    from app.tables.assembly import assemble

    err = _restore_file_if_needed(file_id)
    if err:
        return None, err, []
    if pages:
        df, _report, used = assemble_pages(file_id, pages)
        if df is None:
            return None, f"none of page(s) {pages} hold an extractable table", []
        return df, None, used
    tables = get_all_real_tables(file_id)
    if not tables:
        return None, "no extractable tables", []
    # With no pages named, "this document" means its main table — not every
    # table it contains. Assembling all 49 tables of data-file-1 gave 393
    # rows across columns that mostly do not co-occur, which is the sparse
    # union _group_by_schema was written to prevent. The largest group of
    # tables sharing a column signature IS the document's main table, split
    # across page breaks.
    from app.graph.agent import _group_by_schema
    groups, _skipped = _group_by_schema(tables)
    picked = groups[0][0] if groups else tables
    df, _report = assemble(picked)
    if df is None:
        return None, "no extractable tables", []
    used = sorted({t.attrs.get("page") for t in picked
                   if isinstance(t.attrs.get("page"), int)})
    return df, None, used


def _source_display_name(file_id: str) -> str:
    """What to call a source document. In-memory names are lost on a reload;
    the registry keeps them. Without the fallback a merge reported "5 columns
    from sbs-f2" — a raw file_id where the user expects a filename."""
    from app.graph.agent import _display_name
    from app.persistence import get_file
    stored = (app_state.FILE_ORIGINAL_NAME.get(file_id)
              or (get_file(file_id) or {}).get("original_filename") or file_id)
    return _display_name(file_id, stored)


def _distinct_suffixes(names: list) -> list:
    """One short, unique tag per source document, in order.

    Two documents whose names reduce to the same tag would produce duplicate
    column names and an unreadable file, so collisions get a positional
    letter — which also scales past two sources.
    """
    suffixes = [_merge_suffix(n) for n in names]
    seen = {}
    for i, sfx in enumerate(suffixes):
        seen.setdefault(sfx, []).append(i)
    for sfx, positions in seen.items():
        if len(positions) > 1:
            for rank, i in enumerate(positions):
                suffixes[i] = f"{sfx}-{chr(ord('a') + rank)}"
    return suffixes


def merge_sources_side_by_side(specs: list, output_filename: str) -> str:
    """Put N source documents beside each other in one sheet.

    `specs` is [{"file_id": ..., "pages": [...]}, ...] in left-to-right order.
    Two sources is the common case; three and four work the same way, because
    the columns of each document are simply appended to the right of the last.
    Row 1 of every source lands on spreadsheet row 1.
    """
    from app.export.exporters import verify_export
    from app.config import OUTPUT_DIR
    import pandas as pd

    specs = [s for s in specs if s.get("file_id")]
    # Same document twice contributes nothing a single copy would not — and
    # the same document arrives under two different file_ids routinely,
    # because one upload of "data-file-1.pdf" is registered per ingestion and
    # the model names a different one than the prompt resolver does. Keyed on
    # file_id alone, a three-document request came out with five blocks:
    # f1-a and f1-b were one document, pasted beside itself.
    by_document, order = {}, []
    for spec in specs:
        key = _source_display_name(spec["file_id"])
        if key not in by_document:
            by_document[key] = dict(spec)
            order.append(key)
        elif not by_document[key].get("pages") and spec.get("pages"):
            # Whichever copy carries page numbers is the informative one.
            by_document[key] = dict(spec)
    specs = [by_document[k] for k in order]

    if len(specs) < 2:
        loaded = ", ".join(
            f"{_source_display_name(f)} (id={f})"
            for f in app_state.FILE_ORDER) or "none"
        return ("A side-by-side merge needs at least TWO different documents, "
                "and the request does not say which ones. Ask the user to name "
                f"them. Loaded in this session: {loaded}.")

    names = [_source_display_name(s["file_id"]) for s in specs]
    frames, used_pages = [], []
    for spec, name in zip(specs, names):
        df, err, used = _side_frame(spec["file_id"], spec.get("pages") or None)
        if err:
            return f"Nothing to merge: '{name}' — {err}. No file was created."
        frames.append(df)
        used_pages.append(used)

    suffixes = _distinct_suffixes(names)

    def _prepare(df, sfx):
        # Provenance and the context letterhead belong to a single-document
        # export; side by side they would collide, so they are dropped and
        # said out loud below rather than half-written.
        out = df[[c for c in df.columns if not str(c).startswith("_source")]].copy()
        out.columns = [f"{c}_{sfx}" for c in out.columns]
        return out.reset_index(drop=True)

    prepared = [_prepare(df, sfx) for df, sfx in zip(frames, suffixes)]

    # HORIZONTAL merge = axis=1 (side by side, not stacked).
    merged = pd.concat(prepared, axis=1)

    if not str(output_filename).lower().endswith(".xlsx"):
        output_filename = f"{os.path.splitext(str(output_filename))[0] or 'merged'}.xlsx"
    output_filename = os.path.basename(output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    merged.to_excel(out_path, index=False)

    ok, verify_msg = verify_export(out_path, len(merged), "excel", skiprows=0)
    if not ok:
        return f"Horizontal merge attempted but verification failed: {verify_msg}"

    # So a later "add the header details" or "combine these two columns" can
    # modify this file instead of rebuilding it. The leftmost document is its
    # provenance.
    app_state.WRITTEN_EXPORTS[output_filename] = {
        "file_id": specs[0]["file_id"], "context": None,
        "pages": used_pages[0] or None, "sheets": 1,
    }

    # Sources of unequal length pad the short ones with blank cells. That is
    # not a flaw to hide behind a row count — it is the single thing the user
    # must know before reading the file across.
    lengths = [len(p) for p in prepared]
    pad = ""
    if len(set(lengths)) > 1:
        short = [f"'{n}' ({l} rows)" for n, l in zip(names, lengths)
                 if l < max(lengths)]
        pad = (f" Sources are of unequal length: the merged file has "
               f"{max(lengths)} rows, and {', '.join(short)} run out before "
               f"that, so their last cells are blank. Rows are matched by "
               f"position, not by any key column.")

    sides = []
    for i, (name, frame, pages) in enumerate(zip(names, prepared, used_pages)):
        where = f" page(s) {', '.join(map(str, pages))}" if pages else ""
        position = ("Left" if i == 0 else
                    "Right" if i == len(prepared) - 1 else f"Middle {i}")
        sides.append(f"{position} ({name}{where}): {len(frame.columns)} "
                     f"columns — {list(frame.columns)[:4]}")
    return (
        f"{verify_msg}\n"
        f"Shape: {merged.shape[0]} rows × {merged.shape[1]} columns, "
        f"from {len(specs)} documents placed side by side.\n"
        + "\n".join(sides) + "\n"
        + f"No header band was written; a merged file has {len(specs)} source "
        f"documents and one letterhead would misattribute the rest.{pad}"
    )


@tool
def merge_files_side_by_side(
    description: str = "",
    file1_id: str = None,
    file1_pages: list = None,
    file2_id: str = None,
    file2_pages: list = None,
    output_filename: str = "merged.xlsx",
    extra_sources: list = None,
) -> str:
    """Merge data from TWO OR MORE files SIDE BY SIDE — file 1's columns on
    the left, file 2's to their right, and so on. Like a SQL JOIN but without
    a key column. Row 1 of each file sits on the same spreadsheet row.

    Use when the user says: 'side by side', 'horizontal merge', 'left side
    right side', 'SQL join style', 'put both tables next to each other',
    'merge/concat these files horizontally'.

    DO NOT use this for stacking rows on top of each other — that is
    export_data.

    Args:
        description: What data to take from each file
        file1_id: First file's file_id (leftmost)
        file1_pages: Page numbers to use from file 1 (e.g. [8,9,10])
        file2_id: Second file's file_id
        file2_pages: Page numbers to use from file 2 (e.g. [6,7,8,9,10])
        output_filename: Output file name (must end in .xlsx)
        extra_sources: For a THIRD, FOURTH or later document, a list like
            [{"file_id": "...", "pages": [3,4]}] appended to the right
    """
    from app.graph.agent import _resolve_source_specs

    # Which documents, and which of their pages, comes from the USER's own
    # words — the same resolver export_data uses. The model's paraphrase drops
    # page numbers, and picking FILE_ORDER[0]/[-1] instead is the confirmed
    # worst bug in this project: it reads whichever documents the process
    # happens to have loaded, not the ones this chat is about.
    intent_text = app_state.get_current_user_prompt() or description or ""
    resolved = _resolve_source_specs(intent_text, None)

    specs = []
    if file1_id:
        specs.append({"file_id": file1_id, "pages": file1_pages})
    if file2_id:
        specs.append({"file_id": file2_id, "pages": file2_pages})
    for extra in (extra_sources or []):
        if isinstance(extra, dict) and extra.get("file_id"):
            specs.append({"file_id": extra["file_id"], "pages": extra.get("pages")})
        elif isinstance(extra, str):
            specs.append({"file_id": extra, "pages": None})

    # The model names two files even when the user named four, so whatever it
    # left out is taken from the user's own sentence rather than dropped.
    known = {s["file_id"] for s in specs}
    for spec in resolved:
        if spec["file_id"] not in known:
            specs.append({"file_id": spec["file_id"], "pages": spec["pages"]})
    # A file the model named but gave no pages for still has pages in the
    # user's sentence.
    by_fid = {s["file_id"]: s for s in resolved}
    for spec in specs:
        if not spec.get("pages") and spec["file_id"] in by_fid:
            spec["pages"] = by_fid[spec["file_id"]]["pages"]

    return merge_sources_side_by_side(specs, output_filename)


def _match_columns(requested: list, available: list):
    """Requested column names mapped onto the file's real ones, matched
    loosely on case/spacing/punctuation. Returns (resolved, missing) — the
    model paraphrases 'SL no_f1' as 'SL_no_f1' constantly, and failing that
    on an exact string comparison sends the user back to retype it."""
    import re as _re

    def key(name):
        return _re.sub(r"[^a-z0-9]+", "", str(name).lower())

    index = {}
    for col in available:
        index.setdefault(key(col), col)
    resolved, missing = [], []
    for want in requested:
        hit = index.get(key(want))
        (resolved if hit is not None else missing).append(hit if hit is not None else want)
    return resolved, missing


@tool
def combine_columns(
    source_filename: str,
    columns_to_combine: list,
    new_column_name: str,
    separator: str = " ",
    output_filename: str = None,
) -> str:
    """Combine two or more columns of an ALREADY EXPORTED file into one new
    column.

    Examples:
    - first_name + last_name -> full_name
    - city + state + country -> address_combined
    - code + description -> item_label

    Use when the user says: 'combine columns', 'merge columns', 'join
    columns', 'concatenate columns', 'create a new column from'.

    Args:
        source_filename: The .xlsx file to read (from the outputs folder)
        columns_to_combine: List of column names to combine
        new_column_name: Name for the new combined column
        separator: Character to put between values (default is a space)
        output_filename: Save as this name (default: source name + _combined)
    """
    from app.export.exporters import band_offset, verify_export
    from app.config import OUTPUT_DIR
    import pandas as pd

    source_filename = os.path.basename(str(source_filename or ""))
    source_path = os.path.join(OUTPUT_DIR, source_filename)
    if not source_filename or not os.path.exists(source_path):
        available = sorted(f for f in os.listdir(OUTPUT_DIR)
                           if f.endswith(".xlsx")) if os.path.isdir(OUTPUT_DIR) else []
        return (f"File '{source_filename}' is not in the outputs folder, so "
                f"nothing was changed.\nAvailable files: {available[-10:]}")

    if not columns_to_combine or len(columns_to_combine) < 2:
        return ("Combining needs at least two column names — say which "
                "columns to join.")

    # Exports from this app can carry a context band (letterhead rows) above
    # the real header. Reading the file without that offset would take the
    # band as the column names and report every real column as missing.
    try:
        sheet_names = pd.ExcelFile(source_path).sheet_names
    except Exception as e:
        return f"Could not read {source_filename}: {e}"

    sheets, offsets = {}, {}
    for name in sheet_names:
        off = band_offset(source_path, name)
        offsets[name] = off
        sheets[name] = pd.read_excel(source_path, sheet_name=name, skiprows=off)

    # The sheet that actually has the columns, not simply the first one — a
    # multi-document export writes one sheet per source file, and rewriting
    # only sheet 1 would silently delete the others.
    target_sheet, resolved, missing = None, [], []
    for name, df in sheets.items():
        hit, miss = _match_columns(columns_to_combine, list(df.columns))
        if not miss:
            target_sheet, resolved, missing = name, hit, []
            break
        if target_sheet is None:
            target_sheet, resolved, missing = name, hit, miss
    if missing:
        return (f"These columns are not in {source_filename}: {missing}\n"
                f"Available columns"
                f"{f' on sheet {target_sheet!r}' if len(sheets) > 1 else ''}: "
                f"{list(sheets[target_sheet].columns)}\nNothing was changed.")

    df = sheets[target_sheet]
    blank = {"nan", "none", "nat", "<na>", ""}

    def _join(row):
        # Values are stringified one at a time rather than with astype(str):
        # under pandas 3 that leaves a missing cell as a float NaN instead of
        # the string "nan", and joining it raised AttributeError mid-file.
        parts = []
        for value in row:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text.lower() not in blank:
                parts.append(text)
        return separator.join(parts)

    df[new_column_name] = df[resolved].apply(_join, axis=1)
    sheets[target_sheet] = df

    out_filename = os.path.basename(
        str(output_filename) if output_filename
        else source_filename.replace(".xlsx", "_combined.xlsx"))
    if not out_filename.lower().endswith(".xlsx"):
        out_filename += ".xlsx"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    with pd.ExcelWriter(out_path) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    total = sum(len(f) for f in sheets.values())
    ok, verify_msg = verify_export(out_path, total, "excel", skiprows=0)
    if not ok:
        return f"Column combine attempted but verification failed: {verify_msg}"

    app_state.WRITTEN_EXPORTS[out_filename] = dict(
        app_state.WRITTEN_EXPORTS.get(source_filename, {}),
        sheets=len(sheets))

    band_note = (f" The {offsets[target_sheet]} header-band row(s) above the "
                 f"table were not carried into the new file."
                 if offsets.get(target_sheet) else "")
    sample = df[[*resolved, new_column_name]].head(3).to_string(index=False)
    return (
        f"{verify_msg}\n"
        f"Combined {resolved} -> '{new_column_name}'"
        f"{f' on sheet {target_sheet!r}' if len(sheets) > 1 else ''}.{band_note}\n"
        f"Sample:\n{sample}"
    )


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
