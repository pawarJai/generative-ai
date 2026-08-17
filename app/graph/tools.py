"""Tools available to the LangGraph agent.
Each tool is a real function with a real docstring — the LLM reads
the docstring to decide when to use it. No regex routing.
"""
import contextvars
import os
import re
import threading
import uuid
from typing import Optional

from langchain_core.tools import tool

from app import state as app_state
from app.config import vector_db
from app.tables.helpers import get_all_real_tables, get_page_markdown

# --- repeated-identical-call guard -----------------------------------------
#
# Confirmed production failure: in one turn the model called
# get_page_range(13, 30) THIRTY-FIVE times in a row with identical arguments,
# each returning the identical result; in another it alternated
# search_by_keyword('A.1') and get_page_range(13, 30) about twenty times. The
# turn burned its whole step budget and ended in "Sorry, need more steps to
# process this request" without ever answering. Raising recursion_limit does
# not help — a loop that repeats the same call will consume any budget.
#
# A tool that has already returned the same answer twice will not return a
# different one the third time, so the third identical call is answered with
# an explicit instruction to stop and use what it already has. Read-only
# lookup tools only: never applied to anything that writes a file, where a
# repeat may be a legitimate retry after a fix.
# Counts live in a module-level map keyed by turn, NOT in a threading.local().
#
# They used to be thread-local, and the guard silently never fired: log id
# 9fac5f2d shows get_page_range(13, 30) and search_documents called ELEVEN
# times each with identical arguments, every one executed, the turn dying on
# recursion_limit after 76 seconds. reset_repeat_guard() runs on the
# agent-invoke worker thread, but LangGraph does not guarantee it executes a
# sync tool on that same thread — verified here: a sync tool invoked from an
# async caller ran on a different thread than the one that called reset. Each
# thread therefore saw its own empty counter, every call looked like the
# first, and nothing was ever blocked.
#
# The turn token comes from a contextvar when it propagates and falls back to
# the most recently started turn when it does not, so the guard works
# regardless of which thread the tool lands on.
_repeat_lock = threading.Lock()
_repeat_calls: dict = {}                     # turn token -> {signature: count}
_repeat_turn: contextvars.ContextVar = contextvars.ContextVar(
    "repeat_turn", default=None)
_repeat_latest: Optional[str] = None
_REPEAT_TURNS_KEPT = 16                      # bounds memory across sessions


def reset_repeat_guard(token: str = None) -> str:
    """Start a fresh identical-call history for one agent turn.

    Called once at the start of each turn. Pass the session id so concurrent
    sessions keep separate counts — otherwise two users asking the same
    question could block each other."""
    global _repeat_latest
    token = token or uuid.uuid4().hex
    _repeat_turn.set(token)
    with _repeat_lock:
        _repeat_calls[token] = {}
        _repeat_latest = token
        while len(_repeat_calls) > _REPEAT_TURNS_KEPT:
            _repeat_calls.pop(next(iter(_repeat_calls)))
    return token


def _output_file_hint(file_id) -> Optional[str]:
    """A correction when an EXPORTED file's name was passed where an uploaded
    document's file_id belongs.

    Confirmed production failure: list_sheets(file_id='spec_filled.xlsx') and
    get_file_overview(file_id='spec_filled.xlsx') both silently fell through
    to the session's active file and answered about a completely different
    document — "'data-file-1_8702_data-file-1.pdf' is not a spreadsheet" —
    which reads as a contradiction to anyone who just asked about an .xlsx.
    Files written into outputs/ are results, not uploaded documents, and are
    inspected with the file tools rather than the document tools."""
    name = str(file_id or "")
    if not name.lower().endswith((".xlsx", ".xls", ".csv")):
        return None
    from app.config import OUTPUT_DIR
    if not os.path.exists(os.path.join(OUTPUT_DIR, os.path.basename(name))):
        return None
    return (f"'{name}' is a file this app EXPORTED, not an uploaded document, "
            f"so it has no file_id and document tools cannot read it. To SEE "
            f"what is inside it (sheets, columns, row count, sample rows) "
            f"call inspect_output_file('{os.path.basename(name)}'). To change "
            f"it use the tools that take a `filename`: rename_column, "
            f"add_column, remove_column, filter_rows, handle_duplicates, "
            f"style_excel, modify_export. To attach another exported file's "
            f"columns to it by a shared key, use lookup_and_add_columns. "
            f"Do not pass it as file_id again.")


def _guard_repeat(tool_name: str, **key) -> Optional[str]:
    """A message to return INSTEAD of running the tool, once the same call has
    already been made twice in this turn; None while the call is still new."""
    signature = (tool_name, tuple(sorted((str(k), str(v)) for k, v in key.items())))
    token = _repeat_turn.get() or _repeat_latest
    with _repeat_lock:
        calls = _repeat_calls.setdefault(token, {})
        calls[signature] = calls.get(signature, 0) + 1
        count = calls[signature]
    if count < 3:
        return None
    return (f"STOP — you have already called {tool_name} with these exact "
            f"arguments {count} times in this turn, and it returned "
            f"the same thing every time. It will not return anything different. "
            f"Do NOT call it again with these arguments. Use the results you "
            f"already have to answer the user, or try a genuinely different "
            f"tool or different arguments. If the information truly is not in "
            f"what you have retrieved, say so plainly.")


def _build_template_export(columns: list, filename: str, out_format: str = "excel",
                           constants: Optional[dict] = None,
                           file_id: Optional[str] = None) -> str:
    """Build a file from an EXPLICIT column list — with or without a source
    document, unlike export_data (which always pulls FROM a document and
    refuses outright with nothing uploaded, or when none of the requested
    columns exist in what IS uploaded).

    Confirmed the actual gap in production: "create excel file based on
    this columns = Sr No, Group, ... export in w1-02.xlsx" was routed to
    export_data every time (rule: filename + export verb -> export_data),
    which either said "No files uploaded" with nothing active, or "none of
    these columns exist" with a document active whose real schema doesn't
    contain most of them. Both are TRUE, individually-correct, unhelpful
    answers to a request that was never really "pull this data from my
    document" in the first place — it was "build me a file with these
    headers, filled in wherever you actually can be, blank elsewhere."

    Every cell is filled from exactly one of three sources, and every
    caller learns which: a real, fuzzy-matched column in a resolved source
    document; an explicit constant the caller supplied; or blank. Nothing
    is ever invented to fill a gap.
    """
    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.export import spec as export_spec
    from app.export.exporters import EXPORTERS
    from app.export.spec import _LEADING_LIST_MARKER_RE
    from app.export.spec import _clean as _clean_spec
    from app.models import QueryPlan

    constants = {str(k): v for k, v in (constants or {}).items()}
    # The model routinely passes the user's own "1.Sr No" straight through
    # as the column name — confirmed live: a real request produced headers
    # literally reading "1.Sr No", "2.Group", etc. Stripped here rather
    # than trusted to already be clean, the same reasoning as every other
    # deterministic guard in this file. Punctuation after the digit is
    # required (see _LEADING_LIST_MARKER_RE) so a real column name that
    # happens to start with a digit, like "6 Ship Set", is left alone.
    columns = [_clean_spec(_LEADING_LIST_MARKER_RE.sub("", str(c)))
              for c in (columns or [])]
    columns = [c for c in columns if c]
    if not columns:
        return "No column names were given — say which columns the file should have."

    filename = filename or "template.xlsx"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "xlsx"
    out_format = {"xlsx": "excel", "xls": "excel"}.get(ext, out_format)

    # An optional source, resolved the same way export_data resolves one for
    # an omitted file_id: explicit argument, then a file named in the user's
    # own words, then the session's active file, then a lone loaded file.
    #
    # Confirmed production failure (session 7e43a023, 6 files uploaded): the
    # model called this tool twice for the identical request, passing
    # file_id the first time (resolved to data-file-2, 17 rows filled from
    # a matched table) and omitting it the second time. The old check here
    # — target = FILE_ORDER[0] only when EXACTLY ONE file is loaded — has no
    # fallback with several files uploaded, so the second call silently got
    # target=None and returned a fully blank template. Same request, same
    # session, two contradictory results depending on an argument the model
    # isn't guaranteed to repeat identically across its own retries.
    target = None
    if file_id and _is_known_file(file_id):
        target = file_id
    elif not file_id:
        target = resolve_target_file(None, app_state.get_current_user_prompt() or "")

    source_cols, source_df, row_count = {}, None, 0
    if target:
        from app.tables.helpers import get_all_real_tables as _get_all_real_tables
        candidates = [t for t in _get_all_real_tables(target) if len(t)]
        best, best_score = None, 0
        for t in candidates:
            score = sum(
                1 for c in columns
                if len(export_spec._norm_key(c)) >= 3 and any(
                    export_spec._fuzzy_contains(export_spec._norm_key(c),
                                                export_spec._norm_key(rc))
                    for rc in t.columns))
            if score > best_score:
                best, best_score = t, score
        if best is not None and best_score:
            source_df = best
            row_count = len(best)
            for c in columns:
                if len(export_spec._norm_key(c)) < 3:
                    continue
                for real_col in best.columns:
                    if export_spec._fuzzy_contains(export_spec._norm_key(c),
                                                   export_spec._norm_key(real_col)):
                        source_cols[c] = real_col
                        break

    if not row_count:
        row_count = 1 if constants else 0

    data = {}
    filled_source, filled_constant, blank = [], [], []
    for c in columns:
        const_key = next((k for k in constants
                          if export_spec._norm_key(k) == export_spec._norm_key(c)), None)
        if c in source_cols:
            vals = list(source_df[source_cols[c]])[:row_count]
            vals += [""] * (row_count - len(vals))
            if const_key:
                vals = [constants[const_key]] * row_count
                filled_constant.append(c)
            else:
                filled_source.append(c)
            data[c] = vals
        elif const_key is not None:
            data[c] = [constants[const_key]] * row_count
            filled_constant.append(c)
        else:
            data[c] = [""] * row_count
            blank.append(c)

    df = pd.DataFrame(data, columns=columns)
    plan = QueryPlan(intent="export", sink=out_format, filename=filename, no_context=True)
    export_fn = EXPORTERS.get(out_format, EXPORTERS["excel"])
    export_fn(target or "template", plan, tables=[df])

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"FAILED — {filename} was not written."
    try:
        written_cols = list(pd.read_excel(path).columns) if out_format == "excel" \
            else list(pd.read_csv(path).columns)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the turn
        return f"FAILED — {filename} could not be read back to verify: {e}"
    if written_cols != columns:
        return (f"FAILED — {filename} was written but its columns "
                f"({written_cols}) do not match what was requested "
                f"({columns}).")

    parts = [f"✓ Created {filename} — {len(columns)} columns, {row_count} row(s)."]
    if filled_source:
        parts.append(f"Filled from '{app_state.FILE_ORIGINAL_NAME.get(target, target)}': "
                     f"{', '.join(filled_source)}.")
    if filled_constant:
        parts.append(f"Set to a fixed value for every row: {', '.join(filled_constant)}.")
    if blank:
        parts.append(f"Left blank — not found in any uploaded document and no "
                     f"value was given for them: {', '.join(blank)}.")
    return " ".join(parts)


@tool
def create_excel_template(columns: list, filename: str, constants: dict = None,
                          file_id: str = None) -> str:
    """Build an Excel file from an EXPLICIT list of column headers — works
    WITHOUT any document uploaded, and does not require every column to
    exist in one if a document IS uploaded.

    Use this instead of export_data when the user hands you a column list
    to build a file AROUND ('create excel file based on this columns = ...',
    'build a template with these headers', 'make a file with columns X, Y,
    Z') rather than asking you to pull specific data you already know is in
    an uploaded document. If the user's message has no clear reference to
    an uploaded document at all, this is almost always the right tool —
    export_data will refuse with "no files uploaded" for the exact same
    request, which is correct but not what the user wants.

    Each column is filled from exactly one source, and the reply says
    which: a real, matching column in an uploaded document (if one is
    active or file_id is given); a fixed value from `constants`, applied to
    every row (e.g. the user said "set Yard No. to BY531-536 for all
    rows" -> constants={"Yard No.": "BY531-536"}); or left blank. Never
    invents a value for a column that has neither.

    Args:
        columns: Exact column headers, in the order the file should have them.
        filename: Output filename.
        constants: {column_name: value} to broadcast into every row of that
                  column, for values the user stated directly rather than
                  asked to be pulled from a document (e.g. a fixed yard
                  number, a fixed project code).
        file_id: Optional — a specific uploaded document to pull real data
                 from for any column that matches it. Omit if none was
                 named; a file mentioned in the user's own words, then the
                 session's active file, then a lone loaded file is used
                 automatically, in that order. Pass it explicitly and
                 consistently across retries of the same request — omitting
                 it on one call and supplying it on another can resolve to a
                 different document if the active file has since changed.
    """
    return _build_template_export(columns, filename, "excel", constants, file_id)


def _is_known_file(file_id: str) -> bool:
    """Whether a file_id refers to a document that actually exists — either
    loaded in memory now, or recorded in the durable upload registry (it can
    be lazily restored from there; see agent._restore_file_if_needed)."""
    if file_id in app_state.FILE_KIND or file_id in app_state.FILE_ORDER:
        return True
    try:
        from app.persistence import get_file
        return get_file(file_id) is not None
    except Exception:  # noqa: BLE001 -- registry unavailable must not break a query
        return False


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

    The explicit argument only counts if the file_id is REAL. Confirmed
    production failure (session 477b296a): asked to export a sheet, the model
    once answered with a made-up file_id, "data-file-5_9912" — a plausible
    shape, never uploaded, no registry record. That answer was checkpointed
    into the conversation, so from then on the model read its own invented id
    back out of the history and passed it to tools, which trusted it, found
    no tables under it, and reported the user's perfectly good workbook as
    "corrupted... please re-upload" — in four separate turns, over three
    hours, while the same questions answered correctly in a fresh session.
    An id nothing knows about is not evidence; fall through to the evidence
    that is real.
    """
    if file_id and _is_known_file(file_id):
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


def _best_spec_occurrence(full_text: str, spec_code: str) -> int:
    """Index of the occurrence of ``spec_code`` most likely to be that
    spec's OWN section, rather than a table-of-contents entry or a passing
    reference inside a data table.

    Confirmed production failure: "A.1" appears 9 times in the real
    document — occurrence #1 is the TOC line ("A.1. SCREW DOWN ... ..... 8"),
    #2-#5 are cells inside a quantity table, and only #6 is the actual
    "## A.1. SCREW DOWN NON-RETURN GLOBE VALVE" section heading with the
    specification text under it. Taking the FIRST match returned the TOC
    line every time, so the model saw dot-leader noise, concluded the spec
    "is not present / the document is corrupted", and told the user to
    contact the shipyard — about content that is genuinely in the file.
    Rank the candidates instead of taking whichever comes first.
    """
    import re as _re
    best_idx, best_score = -1, -1
    for m in _re.finditer(_re.escape(spec_code), full_text):
        i = m.start()
        line_start = full_text.rfind("\n", 0, i) + 1
        nl = full_text.find("\n", i)
        line = full_text[line_start:nl if nl != -1 else len(full_text)]
        stripped = line.lstrip()
        if "...." in line:            # dot-leader TOC entry — never the content
            score = 0
        elif stripped.startswith("#"):  # markdown heading — the real section
            score = 3
        elif stripped.startswith("|"):  # a cell in a table — a reference, not the spec
            score = 1
        else:
            score = 2
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _smart_file_select(question: str) -> Optional[str]:
    """Which uploaded file a question with no named file_id most likely
    means, based on what KIND of question it is — not just "whichever
    file was uploaded most recently".

    Confirmed production failure: upload a PDF then an Excel file, and
    every spec/technical question with no file_id silently searched the
    Excel (the most recently active file) instead of the PDF that
    actually has the specs. Preferring the file whose kind matches the
    question's own vocabulary fixes the common case without needing the
    model to always name a file explicitly.
    """
    if len(app_state.FILE_ORDER) == 1:
        return app_state.FILE_ORDER[0]

    q_lower = question.lower()
    spec_keywords = ("spec", "technical", "specification", "valve",
                     "material", "dimension", "standard", "pressure",
                     "rating", "drawing", "page", "annexure")
    prefers_pdf = any(k in q_lower for k in spec_keywords)

    data_keywords = ("price", "rate", "quantity", "total", "column",
                     "sheet", "row", "excel", "csv", "calculate")
    prefers_tabular = any(k in q_lower for k in data_keywords)

    for fid in reversed(app_state.FILE_ORDER):
        kind = app_state.FILE_KIND.get(fid, "")
        if prefers_pdf and kind == "docling":
            return fid
        if prefers_tabular and kind == "tabular":
            return fid

    return app_state.FILE_ORDER[-1] if app_state.FILE_ORDER else None


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
    import re

    repeated = _guard_repeat("search_documents", query=query, file_id=file_id)
    if repeated:
        return repeated

    # Short alphanumeric spec codes ("A.1", "A.14", "B.3") are exactly
    # where vector similarity fails — a 2-4 character token carries almost
    # no semantic signal for an embedding model to match on. Confirmed
    # production failure: "spec no number A.1" returned chunks that never
    # mention A.1 at all, even though the text is genuinely in the
    # document. A literal substring search against Docling's own already-
    # extracted document text finds it with certainty, no embedding
    # involved, before vector search even runs. Tries every uploaded file
    # when file_id isn't given (a cheap substring check, not a guess) —
    # deliberately NOT narrowed by _smart_file_select here, since we want
    # certainty across all documents, not a heuristic guess at one.
    short_spec = re.search(r'\b([A-Z]\.\d+)\b', query, re.IGNORECASE)
    if short_spec:
        spec_code = short_spec.group(1).upper()
        search_fids = [file_id] if file_id else list(app_state.FILE_ORDER)
        for search_fid in search_fids:
            doc = app_state.DOCLING_DOCS.get(search_fid)
            if doc is None:
                continue
            full_text = doc.export_to_markdown()
            idx = _best_spec_occurrence(full_text, spec_code)
            if idx >= 0:
                start = max(0, idx - 200)
                end = min(len(full_text), idx + 3000)
                src = app_state.FILE_ORIGINAL_NAME.get(search_fid, search_fid)
                return (f"Found '{spec_code}' in '{src}':\n\n"
                       f"{full_text[start:end]}")
        # Not found verbatim in any Docling document — fall through to
        # vector search rather than claiming it doesn't exist.

    fid = file_id or _smart_file_select(query)
    filter_dict = {"file_id": fid} if fid else None
    try:
        results = vector_db.similarity_search(query, k=8, filter=filter_dict)
        if not results:
            return "No relevant content found in uploaded documents."

        query_terms = [t for t in query.lower().split() if len(t) > 3]

        # A table-of-contents page is short and heading-dense, so it embeds
        # "close" to almost any question and can fill the entire top-k with
        # nothing but dot-leader lines ("Section 3 .......... 12") — every
        # result technically similar, none of them an actual answer. Confirmed
        # shape of the same failure class as the exact-spec-code miss above:
        # the real content IS in the document, embedding similarity just
        # didn't surface it. Fall back to a literal substring search against
        # the document's own already-extracted text instead of returning TOC
        # noise as if it were the answer. Restores a file lazily first, same
        # reasoning as get_page_content — DOCLING_DOCS can be empty for a
        # file that's genuinely fine, just not currently loaded.
        toc_only = all(
            "......" in r.page_content or len(r.page_content.strip()) < 100
            for r in results)
        if toc_only:
            from app.graph.agent import _restore_file_if_needed
            search_fids = [fid] if fid else list(app_state.FILE_ORDER)
            fallback_terms = query_terms or [query]
            for search_fid in search_fids:
                _restore_file_if_needed(search_fid)
                doc = app_state.DOCLING_DOCS.get(search_fid)
                if doc is None:
                    continue
                full = doc.export_to_markdown()
                for term in fallback_terms:
                    if term.lower() in full.lower():
                        idx = full.lower().find(term.lower())
                        src = app_state.FILE_ORIGINAL_NAME.get(search_fid, search_fid)
                        return (f"The indexed search only found table-of-"
                               f"contents entries; found the actual text in "
                               f"'{src}' instead:\n\n"
                               f"{full[max(0, idx - 100):idx + 2000]}")
            # No literal match either — fall through and report the TOC
            # results honestly rather than inventing content that isn't there.

        # Confirmed production failure: "spec no number A.1" returned chunks
        # that never actually mention A.1 at all — pure embedding-similarity
        # search, with no guarantee the top-k results contain the literal
        # terms asked for. A second call with different wording ("Spec No.
        # A.1") happened to embed closer to the right chunk and succeeded,
        # meaning the right chunk WAS in the store — vector similarity alone
        # just didn't rank it first. Re-sort the SAME k results by how many
        # of the query's real words they actually contain, so an exact-term
        # match outranks a merely similar-sounding one; only reorders what
        # was already retrieved, so this can't introduce a chunk the vector
        # search didn't already consider relevant enough to return.
        if query_terms:
            results_text = " ".join(r.page_content.lower() for r in results)
            key_terms_found = sum(1 for t in query_terms if t in results_text)
            if key_terms_found < 2:
                results = sorted(
                    results,
                    key=lambda r: sum(
                        1 for t in query_terms if t in r.page_content.lower()),
                    reverse=True)

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
    repeated = _guard_repeat("get_page_content", page=page_number,
                             file_id=file_id)
    if repeated:
        return repeated

    target = file_id or app_state.get_active_file_id()
    if not target:
        return "No file selected."

    # Confirmed root cause: app/state.py is pure in-memory, and DOCLING_DOCS/
    # FILE_KIND for an already-ingested file can be lost (server restart,
    # uvicorn --reload) independent of whether the file itself is fine.
    # FILE_KIND.get(target) then comes back as None — not "tabular", just
    # missing — and the old check here couldn't tell that apart from
    # "genuinely not a PDF", so a real 49-table PDF was reported as "it's a
    # spreadsheet". run_agent already does this same lazy-restore for the
    # turn's resolved active file; doing it here too covers a file_id a
    # tool was called with directly, which may not be that same file.
    from app.graph.agent import _restore_file_if_needed
    restore_error = _restore_file_if_needed(target)
    if restore_error:
        return restore_error

    # Belt-and-suspenders: trust DOCLING_DOCS itself, not only the
    # FILE_KIND label, in case the two are ever out of sync right after a
    # restore.
    if target not in app_state.DOCLING_DOCS and app_state.FILE_KIND.get(target) != "docling":
        return (f"'{target}' is not a paginated document (it's a "
                f"spreadsheet or unrecognized file) — page lookup doesn't apply.")
    md = get_page_markdown(target, page_number)
    if md.strip():
        return md

    # A dead end used to end the turn here, and the model would improvise a
    # cause ("empty, corrupted, or not a valid spreadsheet"). Report the real
    # reason instead, and search the file's indexed text — Chroma persists
    # across restarts, so that content is still available.
    from app.query.semantic import semantic_fallback
    from app.tables.helpers import get_page_count

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
def get_page_range(start_page: int, end_page: int, file_id: str = None) -> str:
    """Get content from a range of pages in one call, instead of calling
    get_page_content once per page. Use when the user asks for several
    consecutive pages: 'show me pages 13 to 16', 'read pages 31-35'.

    Args:
        start_page: First page number
        end_page: Last page number (inclusive)
        file_id: Optional specific file. If None, uses the active file.
    """
    repeated = _guard_repeat("get_page_range", start=start_page,
                             end=end_page, file_id=file_id)
    if repeated:
        return repeated

    target = file_id or app_state.get_active_file_id()
    if not target:
        return "No file selected."

    # Same lazy-restore reasoning as get_page_content — this state is
    # process-local and can be lost independent of the file being fine.
    from app.graph.agent import _restore_file_if_needed
    restore_error = _restore_file_if_needed(target)
    if restore_error:
        return restore_error

    if target not in app_state.DOCLING_DOCS and app_state.FILE_KIND.get(target) != "docling":
        return (f"'{target}' is not a paginated document (it's a "
                f"spreadsheet or unrecognized file) — page lookup doesn't apply.")

    if end_page < start_page:
        start_page, end_page = end_page, start_page

    parts = []
    missing = []
    for page_no in range(start_page, end_page + 1):
        md = get_page_markdown(target, page_no)
        if md.strip():
            parts.append(f"=== Page {page_no} ===\n{md}")
        else:
            missing.append(page_no)

    if not parts:
        return (f"No extractable text on pages {start_page}-{end_page} of "
                f"'{target}'. The file itself is present and was ingested "
                f"successfully — do NOT say it is empty or corrupted.")

    result = "\n\n".join(parts)
    if missing:
        result += (f"\n\n(Pages {', '.join(map(str, missing))} had no "
                   f"extractable text and are not included above.)")
    return result


@tool
def search_by_keyword(keyword: str, file_id: str = None,
                      context_chars: int = 300, max_results: int = 5) -> str:
    """Find every occurrence of an EXACT word, phrase, or code across
    uploaded documents AND tables — like Ctrl+F, not semantic similarity.

    Use when the user wants a literal match: 'where does PTFE appear',
    'find tag number 7210-V123', 'search for BY531-536', 'find material
    code PP1131'. Complements search_documents (meaning-based) and the
    exact-spec-code shortcut already built into it (only for codes shaped
    like "A.1") — this is the general-purpose version, for any term, in
    both document text and tabular data.

    Args:
        keyword: Exact text to find (case-insensitive)
        file_id: Optional specific file. If None, searches all uploaded files.
        context_chars: Characters of context around each document-text match
        max_results: Maximum results to return across all files
    """
    from app.graph.agent import _restore_file_if_needed
    from app.tables.helpers import get_all_real_tables

    repeated = _guard_repeat("search_by_keyword", keyword=keyword,
                             file_id=file_id)
    if repeated:
        return repeated

    fids = [file_id] if file_id else list(app_state.FILE_ORDER)
    if not fids:
        return "No documents uploaded."

    kw_lower = keyword.lower()
    results = []

    for fid in fids:
        if len(results) >= max_results:
            break
        _restore_file_if_needed(fid)
        fname = app_state.FILE_ORIGINAL_NAME.get(fid, fid)

        # Docling document text — a plain substring search, not a regex
        # scan: no backtracking risk regardless of document size.
        doc = app_state.DOCLING_DOCS.get(fid)
        if doc is not None:
            try:
                full_text = doc.export_to_markdown()
            except Exception:  # noqa: BLE001 -- one bad file must not stop the search
                full_text = ""
            # Rank occurrences the same way the spec-code shortcut does,
            # instead of returning them in document order. Confirmed
            # production failure: for "A.1" the first five occurrences are
            # all table-of-contents dot-leader lines, so every result was
            # TOC noise and the model concluded the real spec "is only
            # listed on page 8" — while the actual section sat further
            # down the document, unreturned. Heading matches first, TOC
            # dot-leader lines last.
            full_lower = full_text.lower()
            hits, search_from = [], 0
            while True:
                idx = full_lower.find(kw_lower, search_from)
                if idx == -1:
                    break
                hits.append(idx)
                search_from = idx + len(keyword)

            def _rank(i: int, full_text: str = full_text) -> int:
                ls = full_text.rfind("\n", 0, i) + 1
                nl = full_text.find("\n", i)
                line = full_text[ls:nl if nl != -1 else len(full_text)]
                s = line.lstrip()
                if "...." in line:
                    return 0            # TOC dot-leader entry
                if s.startswith("#"):
                    return 3            # real section heading
                if s.startswith("|"):
                    return 1            # a cell in a data table
                return 2

            for idx in sorted(hits, key=_rank, reverse=True):
                if len(results) >= max_results:
                    break
                start = max(0, idx - context_chars)
                end = min(len(full_text), idx + len(keyword) + context_chars)
                results.append(f"[{fname}]\n{full_text[start:end]}")

        # Tabular data — exact substring match per column, real column
        # names only, no schema assumptions about any one document.
        for df in get_all_real_tables(fid):
            if len(results) >= max_results:
                break
            for col in df.columns:
                try:
                    mask = df[col].astype(str).str.contains(
                        keyword, case=False, na=False, regex=False)
                except Exception:  # noqa: BLE001 -- skip an unusable column
                    continue
                if mask.any():
                    page = df.attrs.get("page", "?")
                    results.append(
                        f"[{fname}, page/sheet {page}, column '{col}']\n"
                        f"{df[mask].head(3).to_string(index=False)}")
                    break

    if not results:
        return f"'{keyword}' not found in any uploaded document or table."
    return (f"Found '{keyword}' in {len(results)} location(s):\n\n"
           + "\n\n---\n\n".join(results[:max_results]))


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
    target = resolve_target_file(file_id)
    if not target:
        return "No file selected."
    # A spreadsheet's structure is its sheet list, and the stored "toc" for a
    # tabular file is one entry per extracted TABLE — several per sheet.
    # Answering a structure question from it reported 17 headings for a
    # 10-sheet workbook, which the model then presented as "17 sheets".
    if app_state.FILE_KIND.get(target) == "tabular":
        return _sheet_listing(target)
    toc = app_state.FILE_META.get(target, {}).get("toc", [])
    if not toc:
        return f"No table of contents / headings were detected in '{target}'."
    lines = [f"- {t['text']} (p.{t['page']})" for t in toc]
    return f"Table of contents for '{target}' ({len(toc)} headings):\n" + "\n".join(lines)


@tool
def list_sheets(file_id: str = None) -> str:
    """List the sheets (tabs) in an uploaded spreadsheet, with each one's
    row and column count. Use this for ANY question about a spreadsheet's
    sheets: "how many sheets are there", "list the sheet names", "what tabs
    does this file have", "which sheet has X rows".

    This reads the workbook's real sheet list recorded at upload time. Do NOT
    use query_table_data for these questions — that runs generated code over
    the EXTRACTED TABLES, of which there are several per sheet, so it counts
    sheets wrong and reports internal table labels ("Cover Sheet_Raw",
    "Terms (block 2)") as if they were sheet names.

    Args:
        file_id: Optional specific file. If None, uses the active file.
    """
    hint = _output_file_hint(file_id)
    if hint:
        return hint
    target = resolve_target_file(file_id)
    if not target:
        return "No file selected."
    return _sheet_listing(target)


def _sheet_listing(target: str) -> str:
    """The sheet inventory itself, callable from either tool without going
    back through the tool wrapper."""
    from app.graph.agent import _sheet_base_name, _tabular_sheet_names

    name = app_state.FILE_ORIGINAL_NAME.get(target, target)
    if app_state.FILE_KIND.get(target) != "tabular":
        return (f"'{name}' is not a spreadsheet — it has pages, not sheets. "
                f"Use get_table_of_contents for its structure.")

    sheets = _tabular_sheet_names(target)
    if not sheets:
        return f"No sheets were recorded for '{name}'."

    # Group the extracted tables under the sheet each came from. The "_Raw"
    # table is the whole sheet as-is, so it — not a sub-block — gives the
    # sheet's true size.
    by_sheet = {}
    for t in get_all_real_tables(target):
        by_sheet.setdefault(_sheet_base_name(t.attrs.get("page")), []).append(t)

    rows = []
    for i, sheet in enumerate(sheets, 1):
        tables = by_sheet.get(sheet, [])
        raw = [t for t in tables
               if str(t.attrs.get("page", "")).endswith("_Raw")]
        best = raw[0] if raw else (max(tables, key=len) if tables else None)
        if best is None:
            rows.append(f"| {i} | {sheet} | — | — | empty (no table extracted) |")
        else:
            rows.append(f"| {i} | {sheet} | {len(best)} | {best.shape[1]} | "
                        f"{len(tables)} table(s) extracted |")

    return (f"**{name}** has **{len(sheets)} sheet(s)**:\n\n"
            f"| # | Sheet name | Rows | Columns | Notes |\n"
            f"|---|---|---|---|---|\n" + "\n".join(rows))


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
    # A question ABOUT an exported workbook ("what are the column names in
    # spec_details.xlsx?") is not a question about the uploaded PDF, but with
    # no file_id this fell through to the source document and answered from
    # its tables. Confirmed production failure: asked for spec_details.xlsx's
    # columns it returned ['7', '8', '9', '10', '13', ...] — Docling page
    # numbers — and the model, believing the file it had just written was
    # garbage, spent the rest of the turn re-reading pages one at a time
    # until the step budget ran out (log ids b2610545, 74f778b2).
    import re as _re
    # No spaces in the character class: with one, this matched "names in
    # spec_details.xlsx" out of "what are the column names in
    # spec_details.xlsx?", which resolves to no file at all and let the
    # question fall through to the source document exactly as before.
    named = _re.findall(r"[\w][\w\-.]*\.(?:xlsx|xlsm|csv)\b", question or "")
    for candidate in named:
        hint = _output_file_hint(candidate.strip())
        if hint:
            return hint

    # An exact material/spec/tag code (e.g. "PP1131NNE3EFFMWX0040",
    # "N45310510605002") is a literal substring lookup, not a question the
    # LLM sandbox needs to reason about — and the sandbox's own row/column
    # guessing is exactly what turns "find code X" into several minutes of
    # retries against columns that were never going to match. Matched
    # against EVERY real column of EVERY candidate table, not a fixed
    # schema, so this applies the same way to any document. Requires at
    # least one digit: a plain English word also goes fully uppercase in
    # `question.upper()`, and "EXPORT"/"COLUMN" etc. would otherwise match
    # the same shape as a real code with no digits in it at all.
    exact_codes = [
        c for c in _re.findall(
            r'\b([A-Z]{2}[A-Z0-9]{4,}(?:[A-Z0-9_\-\.]*[A-Z0-9])?)\b',
            question.upper())
        if any(ch.isdigit() for ch in c)
    ]
    if exact_codes:
        from app.tables.helpers import get_all_real_tables
        targets = [file_id] if file_id else list(app_state.FILE_ORDER)
        code_results = []
        for fid in targets:
            for df in get_all_real_tables(fid):
                for col in df.columns:
                    for code in exact_codes:
                        mask = df[col].astype(str).str.contains(
                            _re.escape(code), case=False, na=False)
                        if mask.any():
                            src = app_state.FILE_ORIGINAL_NAME.get(fid, fid)
                            code_results.append(
                                f"Found '{code}' in '{col}' ({src}):\n"
                                f"{df[mask].to_string(index=False)}")
        if code_results:
            return "\n\n".join(code_results)

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
    import os

    import pandas as pd

    from app.export.exporters import EXPORTERS, verify_export
    from app.models import QueryPlan
    from app.query.code_exec import run_code_on_files

    VALID_FORMATS = {"csv", "excel", "docx", "pptx"}
    if format not in VALID_FORMATS:
        format = "excel"  # safe default — never produce a .pdf
    if filename and filename.lower().endswith(".pdf"):
        filename = filename[:-4] + ".xlsx"

    from app.graph.agent import (
        _deterministic_multi_export,
        _deterministic_page_export,
        _deterministic_sheet_export,
        _extract_requested_pages,
        _extract_requested_sheet,
        _resolve_source_specs,
        _side_by_side_plan,
    )

    default_name = filename or f"export.{'xlsx' if format == 'excel' else format}"
    intent_text = app_state.get_current_user_prompt() or question

    # A deliberately broad request ("export all spec data") on a large
    # document is exactly the shape that used to eat the whole
    # recursion_limit chaining many spec-by-spec tool calls in one turn.
    # Checked against intent_text (the user's own words), not the raw
    # `question` argument — the model's paraphrase routinely drops
    # qualifiers like "all", the same reasoning every other intent check
    # in this file already follows. resolve_target_file is used here (not
    # a bare file_id-or-active-file fallback) so this reads the same
    # document the export itself would resolve to, in a multi-file session.
    _broad_keywords = ("all spec", "all specification", "all data",
                       "complete spec", "every spec")
    if any(k in intent_text.lower() for k in _broad_keywords):
        broad_target = resolve_target_file(file_id, intent_text)
        if broad_target:
            from app.tables.helpers import get_all_real_tables
            tables = [t for t in get_all_real_tables(broad_target) if len(t)]
            total_rows = sum(len(t) for t in tables)
            if total_rows > 100:
                return (
                    f"This file has {total_rows} rows across {len(tables)} "
                    f"tables. Exporting all of it at once risks running out "
                    f"of steps before finishing. Recommended: export one "
                    f"spec section at a time, e.g. 'export spec A.1 data' "
                    f"then 'export spec A.2 data', or specify page numbers: "
                    f"'export pages 13-15 to spec-data.xlsx'.")

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
        # "No files uploaded" is correct but not the whole answer when the
        # request itself hands over an explicit column list — that shape of
        # request is create_excel_template's job, not export_data's, and
        # the model routinely calls this tool for it anyway (both are
        # "create a file" on the surface). Confirmed production failure:
        # "create excel file based on this columns = Sr No, Group, ... "
        # with nothing uploaded got "No files have been uploaded, so I
        # cannot generate..." three turns running, when a blank template
        # with exactly those headers is a completely reasonable answer.
        from app.export import spec as export_spec
        tokens = export_spec.extract_requested_column_tokens(intent_text)
        if len(tokens) >= 3:
            return _build_template_export(tokens, default_name, format)
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
            target_files[0], sheet_name, default_name, format, prompt=intent_text)

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

    # Deterministic column-selection / row-filter / top-N / computed-column
    # handling, read from the user's own words against the file's REAL
    # columns — not from run_code_on_files below, an LLM sandbox.
    #
    # Confirmed production failure (session 477b296a, data-file-3): "export
    # two column data ... = Item Title, Item Quantity" wrote all 7 columns
    # to the output file, because this tool built the exported DataFrame
    # from the sandbox's result and handed it straight to the exporter with
    # no column filter ever applied — prep_tables's deterministic filter
    # (app.export.schema_map) was reachable from other callers but not this
    # one. Scoring each candidate table against the prompt, rather than
    # assuming target_files[0]'s first table, is needed because a tabular
    # file can hold several extracted tables and only one actually has the
    # named columns.
    from app.config import OUTPUT_DIR
    from app.export import spec as export_spec
    from app.tables.helpers import get_tables_for_scope

    candidate_tables = [t for t in get_tables_for_scope(target_files) if len(t)]
    best_table, best_score = None, 0
    for t in candidate_tables:
        cols = list(t.columns)
        score = (len(export_spec.extract_requested_columns(intent_text, cols))
                 + (2 if export_spec.extract_row_filter(intent_text, cols) else 0)
                 + (2 if export_spec.extract_computed_column(intent_text, cols) else 0)
                 + (1 if export_spec.extract_row_limit(intent_text) else 0))
        if score > best_score:
            best_table, best_score = t, score

    if best_table is not None and best_score > 0:
        result_df, changes = export_spec.parse_and_apply(best_table, intent_text)
        applied = "; ".join(changes) if changes else "no changes"
        if result_df.empty:
            return (f"The requested filter matched no rows in '{target}' — "
                    f"no file was created. Applied: {applied}.")
        plan = QueryPlan(intent="export", sink=format, filename=default_name)
        export_fn = EXPORTERS.get(format, EXPORTERS["csv"])
        msg = export_fn(target_files[0], plan, tables=[result_df])
        path = os.path.join(OUTPUT_DIR, default_name)
        ok, verify_msg = verify_export(path, len(result_df), format)
        if not ok:
            return f"Export attempted but verification failed: {verify_msg}\nApplied: {applied}"
        return (f"{verify_msg}\nApplied: {applied}\n\n"
                f"Data preview:\n{result_df.head(3).to_markdown(index=False)}")

    # best_score == 0 means NONE of the deterministic checks above found
    # anything — no real column matched, no filter, no limit, no computed
    # column. That is exactly the shape of an explicit-column-list request
    # naming a schema this document doesn't have (a quotation template
    # built from a spec, not data pulled from the file) — not a case the
    # LLM sandbox below is any better positioned to answer, and confirmed
    # in production to take several minutes doing it (repeated retries
    # against columns that were never going to match) before saying so.
    # Skipping straight to the honest, fast answer here also means the
    # user never has to depend on the model choosing create_excel_template
    # correctly on its own.
    if best_score == 0:
        from app.export import spec as export_spec
        tokens = export_spec.extract_requested_column_tokens(intent_text)
        if len(tokens) >= 3:
            return _build_template_export(tokens, default_name, format,
                                          file_id=target_files[0])

    result = run_code_on_files(target_files, question)
    if not result.get("table"):
        return f"Could not extract data: {result['text']}"

    df = pd.DataFrame(result["table"]["rows"], columns=result["table"]["columns"])
    plan = QueryPlan(intent="export", sink=format, filename=default_name)
    export_fn = EXPORTERS.get(format, EXPORTERS["csv"])
    msg = export_fn(target_files[0], plan, tables=[df])

    # CRITICAL: verify the file actually landed on disk
    path = os.path.join(OUTPUT_DIR, default_name)
    ok, verify_msg = verify_export(path, len(df), format)
    if not ok:
        return f"Export attempted but verification failed: {verify_msg}"

    # This is the one export path whose DataFrame comes from the LLM
    # sandbox's own guess at rows/columns rather than a real extracted
    # table — the only place a header row can come back as pandas's own
    # "Unnamed: N" placeholder instead of a real column name, a sign the
    # source table's header row was not read correctly. Column names are
    # never assumed here (no fixed list of expected headers for any one
    # document) — only pandas's own generic-placeholder pattern is
    # checked, so this applies the same way to any document.
    generic_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if generic_cols and len(generic_cols) >= max(1, len(df.columns) // 2):
        return (f"{verify_msg}\n\n"
                f"WARNING: {len(generic_cols)} of {len(df.columns)} column "
                f"headers came back unnamed ({generic_cols[:3]}) — the "
                f"source table's header row may not have been read "
                f"correctly. Actual columns: {list(df.columns)}")

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
    from app.tables.assembly import assemble
    from app.tables.helpers import assemble_pages, get_all_real_tables

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
    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.export.exporters import verify_export

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
    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.export.exporters import band_offset, verify_export

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
    hint = _output_file_hint(file_id)
    if hint:
        return hint
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
    # Label:value pairs from the document's own letterhead/header block —
    # Yard No., Spec No., Project — extracted at ingestion time (see
    # app/ingestion/docling_ingest.py) precisely because they sit OUTSIDE
    # any table and get_all_real_tables never sees them. Surfaced here so
    # a request naming one of these (e.g. "add a Yard No. column") has a
    # real value to read instead of the model guessing or asking the user
    # to re-type something already in the document.
    key_values = meta.get("key_values") or {}
    kv_info = "\n".join(f"  - {k}: {v}" for k, v in key_values.items())
    return (f"File: {name}\n\nSummary: {summary}"
            + (f"\n\nSheets/Tables:\n{sheet_info}" if sheet_info else "")
            + (f"\n\nDocument metadata (from its own header/letterhead):\n{kv_info}"
               if kv_info else ""))


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
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.export.exporters import export_excel, verify_export
    from app.models import QueryPlan
    from app.query.code_exec import run_code_on_files

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


# --- structured, single-purpose file-editing tools --------------------------
#
# modify_export already does renames/drops/header-band changes from a
# free-text `change` argument, resolved (deliberately) against the user's own
# literal words rather than the model's paraphrase — see its own docstring.
# That resolution has a real failure mode for a MULTI-STEP turn: a request
# combining "export this AND rename X to Y AND add a column Z" makes two
# modify_export-shaped calls in one turn, and both re-read the SAME full raw
# prompt. Confirmed production failure (session 556c0de1, 2026-08-15): the
# first call (inside export_data) applied the rename correctly; the second,
# separate modify_export call — meant only to add a new "Yard No." column —
# still had "rename Description to Group" in the raw prompt it re-parsed, so
# it re-attempted that already-completed rename (found no column named
# "Description" anymore, since it was already "Group"), reported "left
# unchanged", and never attempted the add-column instruction at all — while
# the model's own final answer then fabricated "a new Yard No. column has
# been added". These tools take structured arguments instead of a re-parsed
# sentence, so a second call in the same turn cannot be confused by the
# first call's already-applied clause.

@tool
def rename_column(filename: str, old_name: str, new_name: str) -> str:
    """Rename a column in an already-exported Excel or CSV file.
    Use when user says: rename column X to Y, change column name,
    call column X as Y — on a file that has ALREADY been exported.

    Prefer this over modify_export when the rename is the ONLY thing being
    asked for by this tool call, or when it is one of several distinct
    changes (export, rename, add) named in the same message — each gets
    its own tool call with its own clean arguments, rather than one
    modify_export call re-parsing the whole original sentence more than
    once.

    Args:
        filename: File in outputs/ folder to modify
        old_name: Current column name (exact match or closest match)
        new_name: New column name
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        available = [f for f in os.listdir(OUTPUT_DIR)
                     if f.endswith(('.xlsx', '.csv'))]
        return f"File '{filename}' not found. Available: {available}"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)
    cols = [str(c) for c in df.columns]

    # Already done? Confirmed production data loss: the export step had
    # ALREADY applied 'Description' -> 'Group', so by the time this tool ran
    # there was no 'Description' column left. The fuzzy match below (at the
    # old cutoff of 0.4) then matched 'Pressure Rating' and renamed THAT to
    # 'Group' — destroying a real data column and leaving two columns called
    # 'Group', which pandas silently mangled into 'Group.1'. Detecting the
    # already-applied rename first makes the repeat a no-op instead.
    if old_name not in cols and new_name in cols:
        return (f"'{filename}' already has a column named '{new_name}' and no "
                f"column named '{old_name}' — the rename looks like it was "
                f"already applied, so nothing was changed. Columns: {cols}")

    # Renaming onto a name that already exists would create two identical
    # headers; pandas then de-duplicates them on the next read and one
    # column's data becomes unreachable.
    if old_name in cols and new_name in cols and old_name != new_name:
        return (f"'{filename}' already has a column named '{new_name}', so "
                f"renaming '{old_name}' to it would create two columns with "
                f"the same name and make one of them unreadable. Nothing was "
                f"changed. Columns: {cols}")

    if old_name not in cols:
        import difflib
        # cutoff raised from 0.4 to 0.8: at 0.4, 'Description' matched
        # 'Pressure Rating' (see above). A rename is destructive and renames
        # exactly one column, so it must only fire on a near-certain match
        # (a typo or a case/spacing difference), never on a vague
        # resemblance. A wrong guess here silently corrupts the file.
        close = difflib.get_close_matches(old_name, cols, n=1, cutoff=0.8)
        if close:
            old_name = close[0]
        else:
            return (f"Column '{old_name}' not found in {filename}, so nothing "
                    f"was renamed — no other column was guessed at.\n"
                    f"Available columns: {cols}")
    df = df.rename(columns={old_name: new_name})
    if ext == 'xlsx':
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return (f"Renamed '{old_name}' -> '{new_name}' in {filename}.\n"
            f"All columns now: {list(df.columns)}")


@tool
def add_column(filename: str, column_name: str,
               value: str, position: int = -1) -> str:
    """Add a new column with a fixed value to every row in an exported file.
    Use when user says: add column X with value Y, add yard number,
    add a column called X and fill it with Y — on a file that has ALREADY
    been exported.

    Prefer this over modify_export when adding the column is the ONLY thing
    being asked for by this tool call — see rename_column's docstring for
    why a second, separate call with its own clean arguments matters when a
    single message asks for both a rename and a new column.

    If the user asks for a value that is "already in the document" (like a
    project code or ID number) rather than typing the literal value
    themselves, call get_file_overview FIRST — the document's own
    header/letterhead key-value pairs are extracted at ingestion time and
    listed there under "Document metadata". Use that value; do not guess
    one or ask the user to re-type something already in their document.

    Args:
        filename: File in outputs/ folder to modify
        column_name: Name of the new column
        value: Value to fill in every row (same value for all rows)
        position: Column position (0=first, -1=last). Default last.
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)
    if column_name in df.columns:
        return (f"Column '{column_name}' already exists in {filename}.\n"
                f"Use rename_column if you want to rename it, or choose a "
                f"different name.")
    # Confirmed production failure: this returned "Added column 'Yard No.' =
    # 'BY531-536' to all 0 rows" on an empty file, and the model reported that
    # to the user as a success — "All 368 rows now include these columns."
    # Adding a column to a table with no rows changes nothing a user can see,
    # so say that plainly instead of returning a success-shaped sentence.
    if len(df) == 0:
        return (f"FAILED — '{filename}' has no data rows, so adding column "
                f"'{column_name}' would produce a header with nothing under "
                f"it. Nothing was changed. Check that the file actually "
                f"contains the exported data before adding columns to it.")
    if position == -1 or position >= len(df.columns):
        df[column_name] = value
    else:
        df.insert(int(position), column_name, value)
    if ext == 'xlsx':
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return (f"Added column '{column_name}' = '{value}' to all {len(df)} rows.\n"
            f"All columns now: {list(df.columns)}")


@tool
def add_data_row(filename: str, row_data: list | dict) -> str:
    """Add one or more rows of data to an already-exported Excel or CSV file.
    Use when the user asks to add specific extracted values as a new row in
    the file, rather than filling a template with all tables from a document.
    Also use this to legitimately add duplicated data (e.g. multiple sizes for the same Spec No).

    Args:
        filename: File in outputs/ folder to modify
        row_data: A dictionary (for one row) or a list of dictionaries (for multiple rows) 
                  where keys are column names and values are the data to insert. 
                  Unmatched columns will be left blank.
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/ directory."
        
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)
    
    # Normalize to list of dicts
    if isinstance(row_data, dict):
        row_data = [row_data]
        
    new_rows_df = pd.DataFrame(row_data)
    
    if not df.empty or len(df.columns) > 0:
        # Ignore empty or all-NA columns in the new row that might cause a warning, just concat
        df = pd.concat([df, new_rows_df], ignore_index=True)
    else:
        df = new_rows_df
        
    if ext == 'xlsx':
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
        
    return f"Successfully added {len(row_data)} row(s) to {filename} with data: {row_data}"


@tool
def remove_column(filename: str, column_name: str) -> str:
    """Remove/drop a column from an already-exported Excel or CSV file.
    Use when user says: drop column X, remove yard number, delete the Project
    column — on a file that has ALREADY been exported.

    Prefer this over modify_export when removing the column is the ONLY thing
    being asked for by this tool call.

    Args:
        filename: File in outputs/ folder to modify
        column_name: Current column name to remove (exact match or closest match)
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        available = [f for f in os.listdir(OUTPUT_DIR)
                     if f.endswith(('.xlsx', '.csv'))]
        return f"File '{filename}' not found. Available: {available}"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)
    
    # Fuzzy match: find closest column name if exact not found
    if column_name not in df.columns:
        import difflib
        close = difflib.get_close_matches(column_name, [str(c) for c in df.columns],
                                          n=1, cutoff=0.4)
        if close:
            column_name = close[0]
        else:
            return (f"Column '{column_name}' not found.\n"
                    f"Available columns: {list(df.columns)}")
                    
    df = df.drop(columns=[column_name])
    
    if ext == 'xlsx':
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return (f"Removed column '{column_name}' from {filename}.\n"
            f"All columns now: {list(df.columns)}")


@tool
def filter_rows(filename: str, conditions: str,
                output_filename: str = None) -> str:
    """Filter rows in an already-exported file. Like a SQL WHERE clause.
    Use when user says: filter where X=Y, age >= 18, city = Pune,
    top 10 rows, bottom 5 rows, where column equals value,
    remove duplicates, keep only rows where X.

    Args:
        filename: Source file in outputs/ folder
        conditions: Natural language: 'city = Pune and age >= 18'
                    or 'top 10' or 'bottom 5' or 'remove duplicates'
                    or 'unique rows only'
        output_filename: Save result as this name (default: adds _filtered)
    """
    import os
    import re as _re

    import pandas as pd

    from app.config import OUTPUT_DIR, llm

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)
    original_count = len(df)
    cond_lower = conditions.lower().strip()

    # Handle top/bottom N and duplicate-removal without an LLM round trip —
    # these are unambiguous from the words alone.
    top_m = _re.search(r'top\s+(\d+)', cond_lower)
    bot_m = _re.search(r'bottom\s+(\d+)', cond_lower)
    dup_m = _re.search(r'(remove\s+dup|unique|no\s+dup)', cond_lower)

    if top_m:
        result = df.head(int(top_m.group(1)))
        applied = f"top {top_m.group(1)} rows"
    elif bot_m:
        result = df.tail(int(bot_m.group(1)))
        applied = f"bottom {bot_m.group(1)} rows"
    elif dup_m:
        result = df.drop_duplicates()
        applied = "removed duplicates"
    else:
        # A plain "<column> <op> <value>" comparison is handled directly with
        # pandas boolean indexing, never through df.query(). Confirmed
        # production failure: "Spec No. = 'A.1'" was sent to df.query() four
        # times in a row and failed every time with "unterminated string
        # literal" — df.query() parses its argument as Python, so a column
        # name containing a space or a dot ("Spec No.") is not a valid
        # identifier there and must be wrapped in backticks. The model has no
        # way to know that, so it kept re-sending the same broken string.
        # Comparing the column directly sidesteps the query grammar entirely,
        # and works for any column name however it is punctuated.
        result = None
        applied = None
        simple = _re.match(
            r'^\s*(?:where\s+)?(.+?)\s*(>=|<=|!=|==|=|>|<)\s*(.+?)\s*$',
            conditions, _re.IGNORECASE)
        if simple:
            raw_col, op, raw_val = (simple.group(1).strip(),
                                    simple.group(2),
                                    simple.group(3).strip().strip('"').strip("'"))
            import difflib
            cols = [str(c) for c in df.columns]
            col = next((c for c in cols if c.lower() == raw_col.lower()), None)
            if col is None:
                near = difflib.get_close_matches(raw_col, cols, n=1, cutoff=0.8)
                col = near[0] if near else None
            if col is not None:
                series = df[col]
                if op in ('=', '=='):
                    mask = series.astype(str).str.strip().str.lower() == raw_val.lower()
                elif op == '!=':
                    mask = series.astype(str).str.strip().str.lower() != raw_val.lower()
                else:
                    numeric = pd.to_numeric(series, errors='coerce')
                    try:
                        target = float(raw_val)
                    except ValueError:
                        return (f"'{raw_val}' is not a number, so '{op}' cannot "
                                f"be applied to column '{col}'.")
                    mask = {'>': numeric > target, '<': numeric < target,
                            '>=': numeric >= target, '<=': numeric <= target}[op]
                result = df[mask.fillna(False)]
                applied = f"{col} {op} {raw_val}"

        if result is None:
            # Genuinely complex condition — fall back to the LLM, but tell it
            # about the backtick rule so the generated query is valid for
            # these column names.
            code = llm.invoke(
                f"Convert this filter condition to a pandas df.query() string.\n"
                f"Output ONLY the query string. No explanation. No code. No "
                f"quotes around it.\n"
                f"IMPORTANT: wrap every column name in backticks, e.g. "
                f"`Spec No.` == 'A.1' — column names here contain spaces and "
                f"dots and are invalid Python identifiers without them.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Sample data: {df.head(2).to_dict(orient='records')}\n"
                f"Condition: {conditions}"
            ).content.strip().strip('"').strip("'").strip('`')
            try:
                result = df.query(code)
                applied = f"query: {code}"
            except Exception as e:  # noqa: BLE001 -- report, don't crash the turn
                return (f"Could not apply filter. Error: {e}\n"
                        f"Condition was: {conditions}\n"
                        f"Available columns: {list(df.columns)}\n"
                        f"Do NOT retry this same wording — it will fail again. "
                        f"Use a simple 'ColumnName = value' form instead, e.g. "
                        f"'Spec No. = A.1'.")

    out_name = output_filename or filename.replace(f'.{ext}', f'_filtered.{ext}')
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if ext == 'xlsx':
        result.to_excel(out_path, index=False)
    else:
        result.to_csv(out_path, index=False)
    return (f"Filter applied ({applied}).\n"
            f"Result: {len(result)} of {original_count} rows kept.\n"
            f"Saved to {out_name}")


@tool
def fill_excel_template(
    template_filename: str,
    data_file_ids: list = None,
    output_filename: str = None,
    column_mapping: dict = None
) -> str:
    """Fill an Excel template with data from uploaded files.
    The template has pre-defined columns and formatting — this tool
    fills in the rows from the data while preserving the template design.

    Use when user says: fill this template, populate the template,
    add data to my template file, use this Excel format.

    Args:
        template_filename: Template xlsx file (in uploads/ or outputs/)
        data_file_ids: Which uploaded files to pull data from
        output_filename: Save filled template as this name
        column_mapping: Map template column -> source column name
                        e.g. {"Item Description": "Description",
                               "Qty": "Total Quantity"}
    """
    import os

    import openpyxl
    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.tables.helpers import get_all_real_tables

    # Find template file
    template_path = None
    for search_dir in (OUTPUT_DIR, "./uploads"):
        p = os.path.join(search_dir, template_filename)
        if os.path.exists(p):
            template_path = p
            break
    if not template_path:
        return f"Template '{template_filename}' not found in outputs/ or uploads/"

    # Get source data
    fids = data_file_ids or list(app_state.FILE_ORDER)
    all_tables = []
    for fid in fids:
        all_tables.extend(get_all_real_tables(fid))
    if not all_tables:
        return "No data found in the specified files."

    source_df = pd.concat(all_tables, ignore_index=True)

    # Read template to understand its column structure
    wb = openpyxl.load_workbook(template_path)
    ws = wb.active

    # Find header row in template (first row with multiple non-empty cells)
    header_row_idx = None
    template_headers = []
    for row_idx, row in enumerate(ws.iter_rows(), 1):
        vals = [str(c.value).strip() for c in row if c.value and str(c.value).strip()]
        if len(vals) >= 2:
            header_row_idx = row_idx
            template_headers = [str(c.value).strip() if c.value else "" for c in row]
            break

    if not header_row_idx:
        return "Could not find header row in template."

    # Map template columns to source columns
    import difflib
    mapping = column_mapping or {}
    auto_mapping = {}
    source_cols = [str(c) for c in source_df.columns]
    for th in template_headers:
        if not th:
            continue
        if th in mapping:
            auto_mapping[th] = mapping[th]
        else:
            close = difflib.get_close_matches(th, source_cols, n=1, cutoff=0.4)
            if close:
                auto_mapping[th] = close[0]

    # Write data rows starting after template header
    data_start_row = header_row_idx + 1
    for df_idx, (_, row) in enumerate(source_df.iterrows()):
        excel_row = data_start_row + df_idx
        for col_idx, th in enumerate(template_headers, 1):
            if th and th in auto_mapping:
                src_col = auto_mapping[th]
                if src_col in source_df.columns:
                    val = row[src_col]
                    ws.cell(row=excel_row, column=col_idx,
                            value=None if pd.isna(val) else val)

    out_name = output_filename or template_filename.replace(".xlsx", "_filled.xlsx")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    wb.save(out_path)

    mapped = {k: v for k, v in auto_mapping.items() if v}
    unmapped = [th for th in template_headers if th and th not in auto_mapping]
    return (f"Template filled: {out_name}\n"
            f"Rows added: {len(source_df)}\n"
            f"Columns mapped: {mapped}\n"
            f"Columns not matched (left blank): {unmapped}")


@tool
def handle_duplicates(
    filename: str,
    mode: str = "show",
    subset_columns: list = None,
    keep: str = "first",
    output_filename: str = None
) -> str:
    """Find, show, or remove duplicate rows in an already-exported file.
    Use when user says: show duplicates, remove duplicates,
    keep unique rows, find duplicate entries, allow duplicates,
    deduplicate this file.

    Args:
        filename: File in outputs/ folder
        mode: 'show' = just show how many duplicates exist
              'remove' = remove duplicate rows and save
              'allow' = confirm duplicates are kept (no change)
        subset_columns: Check duplicates only in these columns (None = all)
        keep: Which duplicate to keep: 'first', 'last', or 'none'
        output_filename: Save result as this name (default: overwrites)
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)

    dup_mask = df.duplicated(subset=subset_columns, keep=(keep if keep != "none" else False))
    dup_count = int(dup_mask.sum())

    if mode == "show":
        if dup_count == 0:
            return f"No duplicate rows found in {filename} ({len(df)} total rows)."
        dup_rows = df[dup_mask]
        return (f"Found {dup_count} duplicate rows in {filename}.\n"
                f"Total rows: {len(df)}\n"
                f"Unique rows: {len(df) - dup_count}\n"
                f"Sample duplicates:\n{dup_rows.head(3).to_string(index=False)}\n\n"
                f"Say 'remove duplicates from {filename}' to clean them.")

    if mode == "allow":
        return (f"Duplicates allowed in {filename}. "
                f"File has {dup_count} duplicate rows out of {len(df)} total. "
                f"No changes made.")

    if mode == "remove":
        cleaned = df.drop_duplicates(subset=subset_columns,
                                     keep=(keep if keep != "none" else False))
        removed = len(df) - len(cleaned)
        out_name = output_filename or filename
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if ext == "xlsx":
            cleaned.to_excel(out_path, index=False)
        else:
            cleaned.to_csv(out_path, index=False)
        return (f"Removed {removed} duplicate rows.\n"
                f"Before: {len(df)} rows. After: {len(cleaned)} rows.\n"
                f"Saved to {out_name}")

    return f"Unknown mode '{mode}'. Use: show, remove, or allow."


@tool
def style_excel(
    filename: str,
    header_bg_color: str = None,
    header_font_color: str = None,
    header_bold: bool = True,
    row_alt_color: str = None,
    font_name: str = None,
    font_size: int = None,
    output_filename: str = None
) -> str:
    """Apply styling to an already-exported Excel file: colors, fonts,
    background. Use when user says: change color, make header blue, bold
    headers, alternating row colors, change font, set background color,
    make it look professional, style this file.

    Args:
        filename: Excel file in outputs/ folder to style
        header_bg_color: Header background color as hex e.g. '1F4E79' (dark blue)
                         Common: '1F4E79'=dark blue, '2E75B6'=blue,
                         '70AD47'=green, 'FF0000'=red, 'FFC000'=orange,
                         '000000'=black
        header_font_color: Header text color as hex e.g. 'FFFFFF' for white
        header_bold: Make header text bold (default True)
        row_alt_color: Alternating row color as hex e.g. 'D9E1F2' (light blue)
        font_name: Font for all cells e.g. 'Calibri', 'Arial', 'Times New Roman'
        font_size: Font size for data rows e.g. 11
        output_filename: Save styled file as this name (default: overwrites)
    """
    import os

    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"

    wb = openpyxl.load_workbook(path)

    for ws in wb.worksheets:
        max_row = ws.max_row
        max_col = ws.max_column

        for col_idx in range(1, max_col + 1):
            max_len = 0
            for row_idx in range(1, min(max_row + 1, 50)):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 50)

        for row_idx in range(1, max_row + 1):
            for col_idx in range(1, max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)

                font_kwargs = {}
                if font_name:
                    font_kwargs["name"] = font_name
                if row_idx == 1 and header_bold:
                    font_kwargs["bold"] = True
                if row_idx == 1 and header_font_color:
                    font_kwargs["color"] = header_font_color.lstrip("#")
                elif font_size:
                    font_kwargs["size"] = font_size
                if font_kwargs:
                    cell.font = Font(**font_kwargs)

                if row_idx == 1 and header_bg_color:
                    cell.fill = PatternFill(fill_type="solid",
                                            fgColor=header_bg_color.lstrip("#"))
                elif row_alt_color and row_idx > 1 and row_idx % 2 == 0:
                    cell.fill = PatternFill(fill_type="solid",
                                            fgColor=row_alt_color.lstrip("#"))

                cell.alignment = Alignment(wrap_text=False, vertical="center")

        ws.freeze_panes = "A2"

    out_name = output_filename or filename
    out_path = os.path.join(OUTPUT_DIR, out_name)
    wb.save(out_path)

    applied = []
    if header_bg_color:
        applied.append(f"header background: #{header_bg_color}")
    if header_font_color:
        applied.append(f"header text: #{header_font_color}")
    if header_bold:
        applied.append("header bold")
    if row_alt_color:
        applied.append(f"alternating rows: #{row_alt_color}")
    if font_name:
        applied.append(f"font: {font_name}")
    if font_size:
        applied.append(f"size: {font_size}")
    applied.append("auto-width columns")
    applied.append("frozen header row")

    return f"Styled {out_name}.\nApplied: {', '.join(applied)}"


@tool
def merge_output_with_data(
    output_filename: str,
    source_file_id: str = None,
    source_pages: list = None,
    join_on_column: str = None,
    output_filename_new: str = None
) -> str:
    """Merge an already-exported file (in outputs/) with data from an
    uploaded source document, side by side.
    Use when: 'add data to w2-99.xlsx from the PDF', 'merge my exported
    file with data-file-1', 'combine the output with source columns'.
    Args:
        output_filename: Existing file in outputs/ e.g. 'w2-99.xlsx'
        source_file_id: Uploaded source file_id
        source_pages: Pages from source e.g. [31,32,33]
        join_on_column: Join key column e.g. 'Spec No.' (None=positional)
        output_filename_new: Save result as this name
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.tables.helpers import get_all_real_tables

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    if not os.path.exists(out_path):
        return f"File '{output_filename}' not found in outputs/"
    ext = output_filename.lower().split('.')[-1]
    df_out = pd.read_excel(out_path) if ext == 'xlsx' else pd.read_csv(out_path)
    fid = source_file_id or app_state.get_active_file_id()
    if not fid:
        return "No source file specified."
    tables = get_all_real_tables(fid)
    if source_pages:
        pt = [t for t in tables
              if str(t.attrs.get('page')) in [str(p) for p in source_pages]]
        tables = pt if pt else tables
    if not tables:
        return f"No tables found for pages {source_pages}"
    df_src = pd.concat(tables, ignore_index=True)
    df_src = df_src[[c for c in df_src.columns if not str(c).startswith('_')]]
    df_src.columns = [f"{c}_src" if c in df_out.columns else c
                      for c in df_src.columns]
    if join_on_column:
        js = (f"{join_on_column}_src" if f"{join_on_column}_src" in df_src.columns
              else join_on_column)
        try:
            merged = df_out.merge(df_src, left_on=join_on_column, right_on=js,
                                  how='left')
        except Exception as e:  # noqa: BLE001 -- report, don't crash the turn
            # This tool joins an export against RAW TABLES INSIDE THE PDF. A
            # join key that lives in another exported workbook can never be
            # found here, and the bare KeyError ("Join on 'Spec No.' failed:
            # 'Spec No.'") gave the model nothing to act on — it retried this
            # same call until the step budget ran out, twice (log ids
            # b2610545, 74f778b2). Name the tool that can actually do it.
            return (f"Join on '{join_on_column}' failed: {e}\n"
                    f"This tool can only join '{output_filename}' against "
                    f"tables inside the SOURCE DOCUMENT, and no table there "
                    f"has a '{join_on_column}' column.\n"
                    f"If the data you want to attach is in another EXPORTED "
                    f"file (something already in outputs/), use "
                    f"lookup_and_add_columns instead — it joins two exported "
                    f"files on a shared key and finds the key columns itself. "
                    f"Do not retry this call.")
    else:
        merged = pd.concat([df_out.reset_index(drop=True),
                            df_src.reset_index(drop=True)], axis=1)
    out_new = output_filename_new or output_filename.replace(f'.{ext}', f'_merged.{ext}')
    merged.to_excel(os.path.join(OUTPUT_DIR, out_new), index=False)
    return (f"Merged {output_filename} with source.\n"
            f"Shape: {merged.shape[0]} rows x {merged.shape[1]} cols\n"
            f"Saved to: {out_new}")


@tool
def add_excel_dropdown(
    filename: str,
    column_name: str,
    options: list,
    output_filename: str = None
) -> str:
    """Add a dropdown/data validation list to a column in an Excel file.
    Users can only pick from the given options in that column.
    Use when: add dropdown to column X, create dropdown with values A B C,
    add data validation, limit column to specific values,
    make Rating column a dropdown with PN10 PN16 PN25.
    Args:
        filename: Excel file in outputs/
        column_name: Column to add dropdown to (exact or fuzzy match)
        options: List of allowed values e.g. ['PN10', 'PN16', 'PN25']
        output_filename: Save as this name (default: overwrites)
    """
    import difflib
    import os

    import openpyxl
    import pandas as pd
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"

    wb = openpyxl.load_workbook(path)
    ws = wb.active

    # Find header row and column index
    header_row = None
    col_idx = None
    for row in ws.iter_rows(min_row=1, max_row=5):
        for cell in row:
            if cell.value and difflib.get_close_matches(
                    str(column_name).lower(),
                    [str(cell.value).lower()], n=1, cutoff=0.5):
                header_row = cell.row
                col_idx = cell.column
                break
        if col_idx:
            break

    if not col_idx:
        df = pd.read_excel(path)
        return (f"Column '{column_name}' not found.\n"
                f"Columns: {list(df.columns)}")

    col_letter = get_column_letter(col_idx)
    data_start = header_row + 1
    data_end = ws.max_row + 100  # extra rows for future data

    dv = DataValidation(
        type="list",
        formula1=f'"{",".join(str(o) for o in options)}"',
        allow_blank=True,
        showDropDown=False  # False = show the dropdown arrow (openpyxl's
                            # flag is inverted relative to its name)
    )
    dv.sqref = f"{col_letter}{data_start}:{col_letter}{data_end}"
    dv.error = f"Please select from: {', '.join(str(o) for o in options)}"
    dv.errorTitle = "Invalid value"
    dv.prompt = f"Select from: {', '.join(str(o) for o in options)}"
    dv.promptTitle = column_name
    ws.add_data_validation(dv)

    out_name = output_filename or filename
    wb.save(os.path.join(OUTPUT_DIR, out_name))
    return (f"Dropdown added to column '{column_name}' in {out_name}.\n"
            f"Options: {options}\n"
            f"Applied to rows {data_start} to {data_end}.")


@tool
def summarize_document(
    file_id: str = None,
    summary_type: str = "brief",
    focus_on: str = None
) -> str:
    """Generate a smart summary of an uploaded document.
    Not just the cached overview — this reads the actual content and
    generates a targeted summary based on what you need.
    Use when: summarize this document, give me key points, what are the
    main requirements, executive summary, what does this tender ask for,
    what are the technical specifications, key clauses.
    Args:
        file_id: File to summarize (None=active file)
        summary_type: 'brief'=2-3 sentences, 'detailed'=full analysis,
                      'bullets'=key points as list, 'executive'=1 paragraph
        focus_on: What to focus on e.g. 'delivery terms', 'pricing',
                  'technical specs', 'payment', 'penalties', 'quantities'
    """
    from app.config import llm
    from app.tables.helpers import get_all_real_tables

    fid = file_id or app_state.get_active_file_id()
    if not fid:
        return "No file selected."

    doc_text = ""
    if app_state.FILE_KIND.get(fid) == 'docling':
        doc = app_state.DOCLING_DOCS.get(fid)
        if doc:
            doc_text = doc.export_to_markdown()[:8000]
    else:
        tables = get_all_real_tables(fid)
        if tables:
            doc_text = "\n\n".join(t.to_string(index=False) for t in tables[:5])[:8000]

    if not doc_text:
        cached = app_state.FILE_META.get(fid, {}).get('summary', '')
        if cached:
            return cached
        return "No content available for this file."

    name = app_state.FILE_ORIGINAL_NAME.get(fid, fid)
    focus_clause = f" Focus specifically on: {focus_on}." if focus_on else ""
    format_map = {
        'brief': "Write a 2-3 sentence summary.",
        'detailed': "Write a detailed analysis covering all major sections.",
        'bullets': "Write as a bullet-point list of key points (max 10 bullets).",
        'executive': "Write a single executive summary paragraph for a manager."
    }
    format_inst = format_map.get(summary_type, format_map['brief'])

    response = llm.invoke(
        f"Document: {name}\n\nContent:\n{doc_text}\n\n"
        f"{format_inst}{focus_clause}\n"
        f"Base the summary ONLY on the content shown. Do not invent details."
    ).content
    return f"Summary of {name}:\n\n{response}"


@tool
def compare_documents(
    file_id_1: str = None,
    file_id_2: str = None,
    compare_on: str = None,
    output_filename: str = None
) -> str:
    """Compare two uploaded documents and highlight differences.
    Use when: compare file 1 and file 2, what is different between these,
    which file has better terms, compare quantities, compare pricing,
    find differences in specifications.
    Args:
        file_id_1: First file to compare
        file_id_2: Second file to compare
        compare_on: What to compare e.g. 'quantities', 'pricing',
                    'delivery terms', 'specifications', 'columns'
        output_filename: Save comparison as Excel (optional)
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR, llm
    from app.tables.helpers import get_all_real_tables

    fids = list(app_state.FILE_ORDER)
    fid1 = file_id_1 or (fids[-2] if len(fids) >= 2 else None)
    fid2 = file_id_2 or (fids[-1] if len(fids) >= 1 else None)

    if not fid1 or not fid2:
        return "Need two files to compare."

    name1 = app_state.FILE_ORIGINAL_NAME.get(fid1, fid1)
    name2 = app_state.FILE_ORIGINAL_NAME.get(fid2, fid2)

    def _get_text(fid):
        if app_state.FILE_KIND.get(fid) == 'docling':
            doc = app_state.DOCLING_DOCS.get(fid)
            return doc.export_to_markdown()[:4000] if doc else ""
        tables = get_all_real_tables(fid)
        return ("\n".join(t.to_string(index=False) for t in tables[:3])[:4000]
                if tables else "")

    text1 = _get_text(fid1)
    text2 = _get_text(fid2)

    focus = f" Focus on comparing: {compare_on}." if compare_on else ""
    comparison = llm.invoke(
        f"Compare these two documents and list the KEY DIFFERENCES clearly.\n"
        f"{focus}\n\n"
        f"=== {name1} ===\n{text1}\n\n"
        f"=== {name2} ===\n{text2}\n\n"
        f"Format as a table with columns: Aspect | {name1} | {name2} | Difference"
    ).content

    if output_filename:
        try:
            lines = [ln for ln in comparison.split('\n') if '|' in ln]
            rows = [[c.strip() for c in ln.split('|') if c.strip()] for ln in lines]
            if rows:
                df = (pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1
                      else pd.DataFrame(rows))
                df.to_excel(os.path.join(OUTPUT_DIR, output_filename), index=False)
                comparison += f"\n\nComparison saved to {output_filename}"
        except Exception:  # noqa: BLE001 -- the prose comparison still stands
            pass

    return f"Comparison: {name1} vs {name2}\n\n{comparison}"


@tool
def calculate_totals(
    filename: str,
    quantity_column: str,
    rate_column: str,
    total_column_name: str = "Total Value",
    tax_percent: float = None,
    tax_column_name: str = "GST Amount",
    grand_total_column: str = "Grand Total",
    output_filename: str = None
) -> str:
    """Calculate totals, tax, and grand total for a pricing/quotation file.
    Multiplies quantity x rate to get total, optionally adds tax.
    Use when: calculate total, add total value column, compute quantity x rate,
    calculate GST, add tax calculation, compute grand total,
    multiply quantity by unit rate.
    Args:
        filename: Excel/CSV file in outputs/
        quantity_column: Column with quantities
        rate_column: Column with unit rates/prices
        total_column_name: Name for the total column (default 'Total Value')
        tax_percent: Tax percentage e.g. 18.0 for 18% GST (None=no tax)
        tax_column_name: Name for tax column (default 'GST Amount')
        grand_total_column: Name for grand total column
        output_filename: Save as this name (default: overwrites)
    """
    import difflib
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)

    def _find_col(name):
        if name in df.columns:
            return name
        close = difflib.get_close_matches(name, [str(c) for c in df.columns],
                                          n=1, cutoff=0.4)
        return close[0] if close else None

    qty_col = _find_col(quantity_column)
    rate_col = _find_col(rate_column)

    if not qty_col:
        return f"Quantity column '{quantity_column}' not found.\nColumns: {list(df.columns)}"
    if not rate_col:
        return f"Rate column '{rate_column}' not found.\nColumns: {list(df.columns)}"

    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce').fillna(0)
    df[total_column_name] = df[qty_col] * df[rate_col]

    summary = [f"Total column '{total_column_name}' = {qty_col} x {rate_col}"]
    summary.append(f"Sum of {total_column_name}: {df[total_column_name].sum():,.2f}")

    if tax_percent is not None:
        df[tax_column_name] = df[total_column_name] * (tax_percent / 100)
        df[grand_total_column] = df[total_column_name] + df[tax_column_name]
        summary.append(f"GST @ {tax_percent}%: {df[tax_column_name].sum():,.2f}")
        summary.append(f"Grand Total: {df[grand_total_column].sum():,.2f}")

    out_name = output_filename or filename
    if ext == 'xlsx':
        df.to_excel(os.path.join(OUTPUT_DIR, out_name), index=False)
    else:
        df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
    return "\n".join(summary) + f"\nSaved to {out_name}"


@tool
def split_excel_by_column(
    filename: str,
    split_column: str,
    output_prefix: str = None
) -> str:
    """Split an Excel file into multiple files, one per unique value in a column.
    Like splitting a product list by category, or a valve list by group.
    Use when: split by group, separate by category, one file per product type,
    split the data by valve type, create separate files for each department.
    Args:
        filename: Excel/CSV file in outputs/
        split_column: Column to split on e.g. 'Group', 'Category', 'Department'
        output_prefix: Prefix for output files e.g. 'valves' -> 'valves_Globe.xlsx'
    """
    import difflib
    import os
    import re as _re

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)

    if split_column not in df.columns:
        close = difflib.get_close_matches(split_column, [str(c) for c in df.columns],
                                          n=1, cutoff=0.4)
        if close:
            split_column = close[0]
        else:
            return f"Column '{split_column}' not found.\nColumns: {list(df.columns)}"

    prefix = output_prefix or filename.replace(f'.{ext}', '')
    created = []
    for value in df[split_column].dropna().unique():
        subset = df[df[split_column] == value]
        safe_val = _re.sub(r'[^\w\-]', '_', str(value))[:30]
        out_name = f"{prefix}_{safe_val}.xlsx"
        subset.to_excel(os.path.join(OUTPUT_DIR, out_name), index=False)
        created.append(f"{out_name} ({len(subset)} rows)")

    return (f"Split {filename} by '{split_column}' into {len(created)} files:\n"
            + "\n".join(created))


@tool
def pivot_table(
    filename: str,
    rows: str,
    values: str,
    columns: str = None,
    aggfunc: str = "sum",
    output_filename: str = None
) -> str:
    """Create a pivot table from an Excel/CSV file. Like Excel's PivotTable.
    Use when: create pivot table, summarize by group, total by category,
    group by product and sum quantity, pivot the data, aggregate by column.
    Args:
        filename: Source file in outputs/
        rows: Column to use as rows e.g. 'Group' or 'Category'
        values: Column to aggregate e.g. 'Quantity' or 'Total Value'
        columns: Column to spread as headers (optional)
        aggfunc: 'sum', 'count', 'mean', 'max', 'min'
        output_filename: Save pivot as this name
    """
    import difflib
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)

    def _fc(name):
        if name and name not in df.columns:
            close = difflib.get_close_matches(name, [str(c) for c in df.columns],
                                              n=1, cutoff=0.4)
            return close[0] if close else name
        return name

    rows = _fc(rows)
    values = _fc(values)
    columns = _fc(columns) if columns else None

    try:
        pivot = pd.pivot_table(
            df, values=values, index=rows,
            columns=columns, aggfunc=aggfunc,
            fill_value=0, margins=True, margins_name='Total'
        ).reset_index()
    except Exception as e:  # noqa: BLE001 -- report, don't crash the turn
        return f"Pivot failed: {e}\nColumns: {list(df.columns)}"

    out_name = output_filename or filename.replace(f'.{ext}', '_pivot.xlsx')
    pivot.to_excel(os.path.join(OUTPUT_DIR, out_name), index=False)
    return (f"Pivot table created: {out_name}\n"
            f"Rows: '{rows}', Values: '{values}' ({aggfunc})"
            + (f", Columns: '{columns}'" if columns else "") +
            f"\nShape: {pivot.shape[0]} rows x {pivot.shape[1]} cols")


@tool
def extract_key_values(
    file_id: str = None,
    keys_to_find: list = None,
    pages: list = None,
    output_filename: str = None
) -> str:
    """Extract specific key-value pairs from a document.
    Finds structured data like: Yard No., Project Name, Spec No.,
    EMD amount, delivery period, payment terms — even from unstructured text.
    Use when: find the yard number, what is the project name, extract EMD,
    get the tender reference number, find all key details, extract header info,
    what is the spec number for X.
    Args:
        file_id: File to extract from (None=active file)
        keys_to_find: Specific keys e.g. ['Yard No.', 'Project', 'EMD', 'Spec No.']
                      None = auto-detect all key-value pairs
        pages: Specific pages to search (None=all)
        output_filename: Save results as Excel (optional)
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR, llm
    from app.tables.helpers import get_all_real_tables

    fid = file_id or app_state.get_active_file_id()
    if not fid:
        return "No file selected."

    content = ""
    doc = app_state.DOCLING_DOCS.get(fid)
    if doc:
        if pages:
            for p in pages:
                try:
                    content += doc.export_to_markdown(page_no=p)[:2000] + "\n\n"
                except Exception:  # noqa: BLE001 -- skip an unreadable page
                    pass
        else:
            content = doc.export_to_markdown()[:6000]

    if not content:
        tables = get_all_real_tables(fid)
        content = "\n".join(t.to_string(index=False) for t in tables[:5])[:6000]

    name = app_state.FILE_ORIGINAL_NAME.get(fid, fid)
    keys_clause = (f"Extract ONLY these specific keys: {keys_to_find}"
                   if keys_to_find else
                   "Extract ALL key-value pairs you can find (labels and their values)")

    response = llm.invoke(
        f"Document: {name}\n\nContent:\n{content}\n\n"
        f"{keys_clause}.\n"
        f"Format as: Key: Value (one per line).\n"
        f"Only extract values that are explicitly stated. Do not guess."
    ).content

    if output_filename:
        try:
            pairs = []
            for line in response.split('\n'):
                if ':' in line and not line.startswith('#'):
                    k, v = line.split(':', 1)
                    pairs.append({'Key': k.strip(), 'Value': v.strip()})
            if pairs:
                df = pd.DataFrame(pairs)
                df.to_excel(os.path.join(OUTPUT_DIR, output_filename), index=False)
                response += f"\n\nSaved to {output_filename}"
        except Exception:  # noqa: BLE001 -- the prose answer still stands
            pass

    return f"Key values from {name}:\n\n{response}"


@tool
def validate_data(
    filename: str,
    rules: list = None
) -> str:
    """Validate data in an output file against rules.
    Finds missing values, invalid formats, out-of-range numbers, empty cells.
    Use when: check for errors, validate this data, find missing values,
    check data quality, are there any blank cells, find invalid entries,
    check if all required columns are filled.
    Args:
        filename: File in outputs/ to validate
        rules: Optional list of rules e.g.:
               ['no_empty:Quantity', 'positive:Unit Rate', 'max:100:Discount']
               None = auto-detect common issues (nulls, negatives, duplicates)
    """
    import os

    import pandas as pd

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"
    ext = filename.lower().split('.')[-1]
    df = pd.read_excel(path) if ext == 'xlsx' else pd.read_csv(path)

    issues = []

    if rules:
        for rule in rules:
            parts = rule.split(':')
            rule_type = parts[0].lower()
            col = parts[-1] if len(parts) > 1 else None
            if col and col not in df.columns:
                issues.append(f"Rule references unknown column: '{col}'")
                continue
            if rule_type == 'no_empty' and col:
                empty = df[col].isna().sum() + (df[col].astype(str) == '').sum()
                if empty > 0:
                    issues.append(f"'{col}': {empty} empty/null values")
            elif rule_type == 'positive' and col:
                numeric = pd.to_numeric(df[col], errors='coerce')
                neg = (numeric <= 0).sum()
                if neg > 0:
                    issues.append(f"'{col}': {neg} zero or negative values")
            elif rule_type == 'max' and len(parts) >= 3:
                limit = float(parts[1])
                numeric = pd.to_numeric(df[col], errors='coerce')
                over = (numeric > limit).sum()
                if over > 0:
                    issues.append(f"'{col}': {over} values exceed {limit}")
    else:
        null_summary = df.isnull().sum()
        for col, count in null_summary.items():
            if count > 0:
                pct = count / len(df) * 100
                issues.append(f"'{col}': {count} missing values ({pct:.0f}%)")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"Duplicate rows: {dup_count}")
        for col in df.select_dtypes(include='number').columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"'{col}': {neg} negative values")

    if not issues:
        return (f"Validation passed for {filename}.\n"
                f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
                f"No issues found.")
    return (f"Validation issues in {filename} "
            f"({df.shape[0]} rows x {df.shape[1]} cols):\n\n"
            + "\n".join(f"- {i}" for i in issues) +
            f"\n\nTotal issues: {len(issues)}")


@tool
def create_summary_sheet(
    filename: str,
    group_by_column: str = None,
    sum_columns: list = None,
    count_column: str = None,
    output_filename: str = None
) -> str:
    """Add a Summary sheet to an Excel file with totals and counts.
    Like Excel's summary tab — shows grand totals, subtotals by group.
    Use when: add summary tab, create summary sheet, add totals sheet,
    show totals by group, summarize the data, add a dashboard sheet.
    Args:
        filename: Excel file in outputs/
        group_by_column: Column to group by e.g. 'Group' or 'Category'
        sum_columns: Columns to sum e.g. ['Quantity', 'Total Value']
        count_column: Column to count records for (None=auto, unused —
                     Records is always counted from the first column)
        output_filename: Save as this name (default: overwrites)
    """
    import difflib
    import os

    import openpyxl
    import pandas as pd
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    from app.config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found in outputs/"

    df = pd.read_excel(path)

    def _fc(name):
        if name and name not in df.columns:
            close = difflib.get_close_matches(name, [str(c) for c in df.columns],
                                              n=1, cutoff=0.4)
            return close[0] if close else None
        return name

    grp = _fc(group_by_column)
    sum_cols = [c for c in (sum_columns or []) if _fc(c)]
    if not sum_cols:
        sum_cols = list(df.select_dtypes(include='number').columns)[:5]

    if grp:
        summary_df = df.groupby(grp).agg(
            **{c: pd.NamedAgg(column=c, aggfunc='sum') for c in sum_cols},
            Records=pd.NamedAgg(column=df.columns[0], aggfunc='count')
        ).reset_index()
    else:
        summary_data = {'Metric': ['Total Records'], 'Value': [len(df)]}
        for c in sum_cols:
            summary_data['Metric'].append(f'Total {c}')
            summary_data['Value'].append(df[c].sum())
        summary_df = pd.DataFrame(summary_data)

    wb = openpyxl.load_workbook(path)
    if 'Summary' in wb.sheetnames:
        del wb['Summary']
    ws = wb.create_sheet('Summary', 0)  # Insert as first sheet

    ws.append(list(summary_df.columns))
    for cell in ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill(fill_type='solid', fgColor='1F4E79')

    for _, row in summary_df.iterrows():
        ws.append(list(row))

    for col_idx in range(1, ws.max_column + 1):
        max_len = max((len(str(ws.cell(r, col_idx).value or ''))
                       for r in range(1, ws.max_row + 1)), default=8)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 40)

    out_name = output_filename or filename
    wb.save(os.path.join(OUTPUT_DIR, out_name))
    return (f"Summary sheet added to {out_name}.\n"
            f"Groups: {len(summary_df)} rows\n"
            f"Columns: {list(summary_df.columns)}")


@tool
def web_search_procurement(query: str) -> str:
    """Search for procurement-related information online.
    Searches for: GeM tender details, government specifications,
    material standards (IS/BS/EN/ASTM), valve specifications,
    current market prices, competitor info, regulatory requirements.
    Use when: search for specification of X, find IS standard for Y,
    what is EN 558, look up ASTM A216, find current price of Z,
    search GeM for this product, find government spec.
    Args:
        query: Search query e.g. 'EN 558 face to face dimensions globe valve'
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return f"Search results for: {query}\n\n{result}"
    except ImportError:
        pass

    try:
        import json
        import urllib.parse
        import urllib.request
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.urlopen(url, timeout=10)
        data = json.loads(req.read().decode())
        abstract = data.get('AbstractText', '')
        related = [r.get('Text', '') for r in data.get('RelatedTopics', [])[:3]
                  if r.get('Text')]
        if abstract:
            return (f"Search: {query}\n\nResult: {abstract}\n\n"
                    f"Related: {chr(10).join(related)}")
        return f"No direct results found for: {query}\nTry a more specific search."
    except Exception as e:  # noqa: BLE001 -- report, don't crash the turn
        return (f"Web search unavailable: {e}\n"
                f"For procurement standards, try: www.bis.gov.in (IS standards), "
                f"www.en-standard.eu (EN standards), gem.gov.in (GeM tenders)")

@tool
def smart_extract_to_template(
    template_filename: str,
    specs_to_find: list = None,
    pages: list = None,
    file_id: str = None,
    output_filename: str = None
) -> str:
    """Intelligently extract complex/vertical tables into an Excel template.
    Use when simple column matching fails because the data uses key-value pairs 
    or nested tables (like spec property tables + size tables).
    
    Args:
        template_filename: Template xlsx file (in uploads/ or outputs/)
        specs_to_find: List of spec numbers to extract e.g. ['A.1', 'A.2', 'A.3']
        pages: Optional specific pages to search to avoid context limits
        file_id: File to extract from (None=active file)
        output_filename: Save filled template as this name
    """
    import os
    import json
    import openpyxl
    import pandas as pd
    from app.config import OUTPUT_DIR, llm
    from app import state as app_state

    fid = file_id or app_state.get_active_file_id()
    if not fid: return "No file selected."
    doc = app_state.DOCLING_DOCS.get(fid)
    if not doc: return "File not parsed with Docling."

    # Find template file
    template_path = None
    for search_dir in (OUTPUT_DIR, "./uploads"):
        p = os.path.join(search_dir, template_filename)
        if os.path.exists(p):
            template_path = p
            break
    if not template_path:
        return f"Template '{template_filename}' not found in outputs/ or uploads/"

    # Get template columns
    wb = openpyxl.load_workbook(template_path)
    ws = wb.active
    header_row_idx = None
    template_headers = []
    for row_idx, row in enumerate(ws.iter_rows(), 1):
        vals = [str(c.value).strip() for c in row if c.value and str(c.value).strip()]
        if len(vals) >= 2:
            header_row_idx = row_idx
            template_headers = [str(c.value).strip() if c.value else "" for c in row]
            break
            
    if not template_headers:
        return "Could not find headers in template."

    # Get Markdown content
    content_chunks = []
    if pages:
        for p in pages:
            try:
                content_chunks.append(doc.export_to_markdown(page_no=p))
            except Exception: pass
    else:
        # Default chunking: maybe group by 3 pages
        current_chunk = ""
        for p in range(1, len(getattr(doc, 'pages', [])) + 1):
            try:
                text = doc.export_to_markdown(page_no=p)
                current_chunk += text + "\n\n"
                if len(current_chunk) > 12000:
                    content_chunks.append(current_chunk)
                    current_chunk = ""
            except Exception: pass
        if current_chunk:
            content_chunks.append(current_chunk)
            
    if not content_chunks:
        return "Could not extract text from document."

    schema_instruction = json.dumps(template_headers)
    specs_instruction = f" Specifically look for these specs: {specs_to_find}." if specs_to_find else " Extract all specifications you find."

    extracted_data = []
    
    for i, chunk in enumerate(content_chunks):
        if not chunk.strip(): continue
        
        prompt = (
            f"You are a data extraction AI. Extract the requested specifications from the following document text.\n"
            f"The text often contains vertical tables for properties and a separate table for sizes. "
            f"You MUST expand the sizes so each size has its own row along with all the properties for that specification.\n\n"
            f"Output a valid JSON array of objects. Each object must have these exact keys:\n{schema_instruction}\n\n"
            f"Instructions:\n{specs_instruction}\n"
            f"- If a column does not apply or isn't found, use an empty string \"\".\n"
            f"- Output ONLY the raw JSON array. No markdown blocks, no explanation.\n\n"
            f"Text Chunk ({i+1}/{len(content_chunks)}):\n{chunk}"
        )
        
        try:
            response = llm.invoke(prompt).content.strip()
            # Clean up markdown code blocks if the LLM still wraps it
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            chunk_data = json.loads(response.strip())
            if isinstance(chunk_data, list):
                extracted_data.extend(chunk_data)
        except Exception as e:
            print(f"Error parsing chunk {i}: {e}")
            continue

    if not extracted_data:
        return "LLM could not extract any matching data or failed to return valid JSON."

    source_df = pd.DataFrame(extracted_data)
    
    # Write to excel starting after header
    data_start_row = header_row_idx + 1
    for df_idx, row in source_df.iterrows():
        excel_row = data_start_row + df_idx
        for col_idx, th in enumerate(template_headers, 1):
            if th and th in source_df.columns:
                val = row.get(th)
                ws.cell(row=excel_row, column=col_idx, value=None if pd.isna(val) else val)

    out_name = output_filename or template_filename.replace(".xlsx", "_smart_filled.xlsx")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    wb.save(out_path)

    return f"Successfully extracted {len(extracted_data)} rows into {out_name}."



# --- repeated-section export ------------------------------------------------
#
# Matches an enumerated section heading: "## A.1. SCREW DOWN NON-RETURN GLOBE
# VALVE", "## 9.2 Testing". The code is captured separately from the title so
# sections can be grouped into a series (all "A.n") and sorted numerically —
# "A.10" must sort after "A.9", which plain string sorting gets wrong.
_SECTION_HEADING_RE = re.compile(
    r'^#{1,6}\s*([A-Za-z]{1,3}\.\d+|\d+\.\d+)\.?[\s ]+(\S.*?)\s*$', re.M)


def _section_sort_key(code: str):
    """'A.10' after 'A.9', not before it."""
    prefix, _, num = code.rpartition(".")
    try:
        return (prefix, int(num))
    except ValueError:
        return (prefix, 0)


def _find_sections(target: str, n_pages: int):
    """Every enumerated heading in the document, as
    [{code, title, page}, ...] in page order."""
    found = []
    for page_no in range(1, n_pages + 1):
        md = get_page_markdown(target, page_no)
        if not md:
            continue
        for m in _SECTION_HEADING_RE.finditer(md):
            found.append({"code": m.group(1).rstrip("."),
                          "title": m.group(2).strip(),
                          "page": page_no})
    return found


def _band_labels(grids) -> set:
    """Row-group labels ("Material", "Dimensions") that Docling sometimes
    fuses onto the front of a real attribute name, e.g. "Material Bonnet".

    A cell qualifies only when it sits in column 0 of a >=3-column PROPERTY
    table with both a label and a value beside it — and does so in at least
    TWO different tables. That second condition is what makes this safe:
    a genuine spanning band recurs on section after section, whereas the
    one-off damage Docling does to a badly merged table does not.

    Requiring it caught three real false positives on this document.
    Without the two-table rule, "Rating"/"PN10" were learned as bands from
    the 2-row SIZE matrix; "Hydraulic" was learned from a single mangled row
    on page 28 (['Hydraulic', 'Test Seat', ...]) and then silently rewrote
    every section's "Hydraulic Test" attribute to "Test"; and "End
    connection" was learned the same way and ate its own column. Derived
    from the document, never hardcoded.
    """
    from collections import defaultdict
    seen: dict = defaultdict(set)
    for i, grid in enumerate(grids):
        if len(grid) < 3:          # 2-row matrix tables have no bands at all
            continue
        for row in grid:
            if len(row) >= 3 and row[0] and row[1] and row[-1]:
                seen[row[0]].add(i)
    return {label for label, tables in seen.items() if len(tables) >= 2}


def _merge_band_prefixed_keys(records: list, bands: set) -> list:
    """Fold "Material Bonnet" into "Bonnet" — but only where the evidence
    says that is a merge rather than a rename.

    Stripping a band prefix unconditionally is wrong: "Design Pressure"
    appears in 9 sections and bare "Pressure" in 1, so stripping "Design"
    would rename the well-populated column to the name of a scrap one. The
    prefix is only removed when the remainder is ALREADY a more common key
    across the series, which is exactly the case where removing it unifies
    two spellings of one attribute into a single column instead of
    inventing a new one.
    """
    from collections import Counter
    freq = Counter(k for rec in records for k, v in rec.items() if v)

    renames = {}
    for key in freq:
        for band in bands:
            if key != band and key.startswith(band + " "):
                rest = key[len(band):].strip()
                if rest and freq.get(rest, 0) > freq[key]:
                    renames[key] = rest
                break
    if not renames:
        return records

    merged_records = []
    for rec in records:
        merged: dict = {}
        for key, value in rec.items():
            key = renames.get(key, key)
            if merged.get(key):
                if value and value not in merged[key]:
                    merged[key] = f"{merged[key]} {value}".strip()
            else:
                merged[key] = value
        merged_records.append(merged)
    return merged_records


def _read_property_grid(grid, into: dict) -> None:
    """Vertical key/value spec table -> {attribute: value}.

    Docling does NOT place the attribute in a consistent column: the same
    logical table yields ['', 'Body', 'NAB'], ['Material', 'Disc', 'NAB'],
    ['Design Pressure', '', 'PN10'] and ['Body', 'GM to BS 1400 LG 4C'] on
    different pages of one document. So the column index is ignored entirely
    and each row is read by position instead:

        value = last non-empty cell
        key   = nearest non-empty cell before it

    A row with a value but no key is a continuation (a wrapped cell, e.g.
    Hydraulic Test splitting into "Body - 1.5 x ..." / "Seat - 1.1 x ...")
    and is appended to the previous attribute. A repeated attribute appends
    rather than overwrites, for the same reason.
    """
    last_key = None
    for row in grid:
        cells = [c.strip() for c in row]
        idx = [i for i, c in enumerate(cells) if c]
        if not idx:
            continue
        value = cells[idx[-1]]
        key = cells[idx[-2]] if len(idx) >= 2 else None

        if key is None:
            # Continuation of the attribute above — never a new column.
            if last_key and value:
                into[last_key] = f"{into[last_key]} {value}".strip()
            continue

        if not key:
            continue
        if key in into and into[key]:
            if value and value not in into[key]:
                into[key] = f"{into[key]} {value}".strip()
        else:
            into[key] = value
        last_key = key


def _read_matrix_grid(grid, into: dict) -> None:
    """2-row matrix table (header row over a single value row) -> fields.

    This is the "SIZE: -" table, where the header repeats across the merged
    span ("Rating | Size of the valves (NB) | Size of the valves (NB) | ...")
    over one row of values ("PN10 | 40 | 50 | 65"). Values under a repeated
    header are collected into one comma-joined field, so a spec with four
    sizes stays one row instead of exploding into four.
    """
    header, values = grid[0], grid[1]
    collected: dict = {}
    for i, name in enumerate(header):
        name = name.strip()
        if not name or i >= len(values):
            continue
        val = values[i].strip()
        if not val:
            continue
        collected.setdefault(name, [])
        if val not in collected[name]:
            collected[name].append(val)
    for name, vals in collected.items():
        joined = ", ".join(vals)
        into[name] = f"{into[name]}, {joined}" if into.get(name) else joined


@tool
def export_document_sections(output_filename: str = "sections.xlsx",
                             section_prefix: str = None,
                             start_page: int = None,
                             end_page: int = None,
                             file_id: str = None) -> str:
    """Export EVERY repeated numbered section of a document to one Excel row
    per section — the right tool when a document repeats the same layout for
    many items, one item per page, and the user wants all of them at once.

    Use for requests like "export all spec numbers A.1 to A.18", "get the
    details for every spec into Excel", "put all the valve specifications in
    one sheet". Each section's key/value table becomes columns, so section
    A.1 and A.18 line up in the same spreadsheet.

    Prefer this over calling get_page_range page by page and building the
    table yourself: it reads all sections in ONE call, takes every value
    straight from the document's own tables, and cannot invent a value.
    Column names come from the document, not from a fixed list.

    Args:
        output_filename: Excel file to write, e.g. 'spec.xlsx'
        section_prefix: Only sections whose code starts with this, e.g. 'A'
                        for A.1-A.18. None = the largest series in the file.
        start_page: Optional first page to look at
        end_page: Optional last page to look at
        file_id: Optional specific file. If None, uses the active file.
    """
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    from app.config import OUTPUT_DIR
    from app.tables.helpers import get_page_count, get_table_grids_on_page

    target = resolve_target_file(file_id, "")
    if not target:
        return "No file selected."

    from app.graph.agent import _restore_file_if_needed
    restore_error = _restore_file_if_needed(target)
    if restore_error:
        return restore_error

    if target not in app_state.DOCLING_DOCS:
        return (f"'{target}' is not a paginated document — this tool reads "
                f"sections out of a PDF/DOCX/PPTX.")

    n_pages = get_page_count(target) or 0
    sections = _find_sections(target, n_pages)
    if not sections:
        return (f"No numbered section headings (like 'A.1' or '9.2') were "
                f"found in '{target}', so there are no repeated sections to "
                f"export. Use export_data or get_page_range instead.")

    # Group by series prefix and take the requested one — or, when the user
    # did not say, the series with the most sections, which is the repeated
    # per-item block they almost certainly mean.
    by_prefix: dict = {}
    for sec in sections:
        by_prefix.setdefault(sec["code"].rpartition(".")[0], []).append(sec)

    if section_prefix:
        wanted = section_prefix.strip().rstrip(".")
        chosen = by_prefix.get(wanted)
        if not chosen:
            return (f"No sections starting with '{wanted}' in '{target}'. "
                    f"Series present: "
                    f"{', '.join(f'{k}.n ({len(v)})' for k, v in by_prefix.items())}")
    else:
        chosen = max(by_prefix.values(), key=len)

    if start_page is not None:
        chosen = [s for s in chosen if s["page"] >= start_page]
    if end_page is not None:
        chosen = [s for s in chosen if s["page"] <= end_page]
    if not chosen:
        return "No sections fall inside the page range given."

    chosen.sort(key=lambda s: (s["page"], _section_sort_key(s["code"])))

    # A section owns the pages from its own heading up to (not including) the
    # next section's page. The last one owns only its own page — extending it
    # to the end of the document would sweep in unrelated annexures.
    spans = []
    for i, sec in enumerate(chosen):
        first = sec["page"]
        last = chosen[i + 1]["page"] - 1 if i + 1 < len(chosen) else first
        spans.append((sec, first, max(first, last)))

    # Bands are learned across the WHOLE series before any row is built —
    # a leaked prefix on page 13 is only recognisable because some other
    # page uses that same word as a standalone spanning label.
    page_grids: dict = {}
    for _sec, first, last in spans:
        for page_no in range(first, last + 1):
            if page_no not in page_grids:
                page_grids[page_no] = get_table_grids_on_page(target, page_no)
    bands = _band_labels([g for grids in page_grids.values() for g in grids])

    rows = []
    empty_sections = []
    for sec, first, last in spans:
        # Title comes from the HEADING, never from the table's own header
        # cell: in this document A.18's table still says "SELF CLOSING VALVE"
        # while its heading correctly reads "FOOT VALVE WITH STRAINER".
        record: dict = {"Section": sec["code"], "Title": sec["title"],
                        "Page": first}
        props: dict = {}
        for page_no in range(first, last + 1):
            for grid in page_grids.get(page_no, []):
                if len(grid) >= 3 and len(grid[0]) >= 2:
                    _read_property_grid(grid, props)
                elif len(grid) == 2:
                    _read_matrix_grid(grid, props)
        if not props:
            empty_sections.append(sec["code"])
        record.update(props)
        rows.append(record)

    # Band folding runs across the finished series, not per row — whether
    # "Material Bonnet" should become "Bonnet" is only answerable once every
    # section's keys have been counted.
    rows = _merge_band_prefixed_keys(rows, bands)

    # Columns ordered by how many sections actually carry them, most-used
    # first (ties broken by first appearance, so equally-common attributes
    # still read in document order).
    #
    # Not cosmetic. Docling mangles the occasional source table badly enough
    # that a value gets read as an attribute name — page 27 of this document
    # exports as a 4-column grid with a spurious middle column, yielding
    # one-off "columns" like "4C". Those are real extracted text and are NOT
    # dropped, because a genuinely unique attribute looks identical to them
    # (a gate valve really does have a "Wedge" that no other section has).
    # Sorting by support puts every attribute the series shares up front and
    # lets the long tail of one-offs trail at the right, where it is obvious
    # what it is, instead of interleaving noise with the real schema.
    from collections import Counter
    support = Counter(k for rec in rows for k, v in rec.items() if v)
    first_seen: dict = {}
    for record in rows:
        for key in record:
            first_seen.setdefault(key, len(first_seen))

    pinned = ["Section", "Title", "Page"]
    columns = [c for c in pinned if c in first_seen]
    columns += sorted((k for k in first_seen if k not in pinned),
                      key=lambda k: (-support.get(k, 0), first_seen[k]))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sections"
    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="1F4E79")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for record in rows:
        ws.append([record.get(c, "") for c in columns])

    ws.freeze_panes = "A2"
    for col_idx in range(1, ws.max_column + 1):
        width = max((len(str(ws.cell(r, col_idx).value or ""))
                     for r in range(1, ws.max_row + 1)), default=8)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(width + 3, 45)

    if not output_filename.lower().endswith((".xlsx", ".xlsm")):
        output_filename += ".xlsx"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    wb.save(out_path)

    # Read the workbook back off disk before claiming anything about it, and
    # emit the "✓ Verified:" token the rest of the pipeline keys on.
    #
    # Not decoration. run_agent's anti-hallucination guard treats a model
    # message that names a file as FABRICATED unless an export tool returned
    # a verified result this turn. This tool wrote a correct 18-row spec.xlsx
    # and, because it said only "Created", the guard discarded the true answer,
    # replaced it with "**No file was created.**", and ran a recovery export
    # that pointed the download at a stale empty spec-data.xlsx from the day
    # before. The extraction was right and the user still received an empty
    # file (log ids 15619f2a, b0a2d5cc).
    from app.export.exporters import verify_export
    ok, verdict = verify_export(out_path, len(rows), "excel", skiprows=0)
    if not ok:
        return (f"FAILED to write {output_filename} — {verdict}. "
                f"No file was produced; do not tell the user one was.")

    codes = ", ".join(s["code"] for s, _f, _l in spans)
    msg = (f"{verdict}\n{output_filename} holds {len(rows)} sections "
           f"({codes}) x {len(columns)} columns, from pages "
           f"{spans[0][1]}-{spans[-1][2]} of "
           f"{app_state.FILE_ORIGINAL_NAME.get(target, target)}.\n"
           f"Download this exact filename: {output_filename}\n"
           f"Columns: {columns}")
    if empty_sections:
        msg += (f"\nNo table data found for: {', '.join(empty_sections)} — "
                f"these rows have the heading only.")
    return msg


def _resolve_workbook(filename: str):
    """Locate a spreadsheet by name in outputs/ then uploads/."""
    from app.config import OUTPUT_DIR
    for folder in (OUTPUT_DIR, "./uploads"):
        path = os.path.join(folder, os.path.basename(str(filename)))
        if os.path.exists(path):
            return path
    return None


def _norm_key(value) -> str:
    """Normalise a join key so 'A.1', ' a.1 ' and 'A.1.' all match.

    Only ever used for MATCHING — the values written to the sheet stay
    exactly as the document spelled them.
    """
    import re as _re
    text = _re.sub(r"\s+", " ", str(value).strip())
    return text.rstrip(".").casefold()


def _pick_join_columns(df_target, df_lookup):
    """Guess which column on each side is the shared key, by overlap.

    The two sides genuinely disagree on naming in real files — the valve list
    calls it "Spec No." while the extracted spec sheet calls it "Section" —
    and the model cannot be relied on to spot that. Whichever pair of columns
    actually shares the most values IS the key, which is evidence rather than
    a guess about names.
    """
    best = (0.0, None, None)
    for lookup_col in df_lookup.columns:
        lookup_vals = {_norm_key(v) for v in df_lookup[lookup_col].dropna()}
        lookup_vals.discard("")
        if not lookup_vals:
            continue
        for target_col in df_target.columns:
            target_vals = [_norm_key(v) for v in df_target[target_col].dropna()]
            target_vals = [v for v in target_vals if v]
            if not target_vals:
                continue
            coverage = sum(v in lookup_vals for v in target_vals) / len(target_vals)
            if coverage > best[0]:
                best = (coverage, target_col, lookup_col)
    return best


@tool
def lookup_and_add_columns(target_filename: str,
                           lookup_filename: str,
                           target_key_column: str = None,
                           lookup_key_column: str = None,
                           columns_to_add: list = None,
                           output_filename: str = None) -> str:
    """Attach detail columns to an existing spreadsheet by matching a shared
    key — a VLOOKUP between two files in outputs/.

    Use when the user wants rows ENRICHED rather than stacked: "map the spec
    number data onto my valve list", "add the A.1-A.18 details to each row by
    Spec No.", "add first name and address to each user id". Every row in
    target_filename keeps its place and gains the matching row's columns from
    lookup_filename; many target rows may share one lookup row.

    This is NOT merge_output_with_data, which joins an export against raw
    tables inside a PDF and cannot see a second exported file.

    Args:
        target_filename: File in outputs/ whose rows are being enriched,
                         e.g. 'w2-99.xlsx'
        lookup_filename: File in outputs/ holding the detail rows,
                         e.g. 'spec_details.xlsx'
        target_key_column: Key column in the target, e.g. 'Spec No.'
                           (None = detected from shared values)
        lookup_key_column: Key column in the lookup, e.g. 'Section'
                           (None = detected from shared values)
        columns_to_add: Which lookup columns to bring across (None = all)
        output_filename: Save as this name (default: overwrites the target)
    """
    import difflib

    import pandas as pd

    from app.config import OUTPUT_DIR
    from app.export.exporters import verify_export

    target_path = _resolve_workbook(target_filename)
    if not target_path:
        return (f"'{target_filename}' not found in outputs/. Export it first, "
                f"or check the name with list_uploaded_files.")
    lookup_path = _resolve_workbook(lookup_filename)
    if not lookup_path:
        return (f"'{lookup_filename}' not found in outputs/. If the detail "
                f"data has not been extracted yet, run "
                f"export_document_sections first, then call this again.")
    if os.path.abspath(target_path) == os.path.abspath(lookup_path):
        return ("target_filename and lookup_filename are the same file — "
                "a lookup needs a separate file to pull the details from.")

    def _read(path):
        return (pd.read_csv(path) if path.lower().endswith(".csv")
                else pd.read_excel(path))

    df_target = _read(target_path)
    df_lookup = _read(lookup_path)
    if df_target.empty:
        return f"'{target_filename}' has no data rows — nothing to enrich."
    if df_lookup.empty:
        return f"'{lookup_filename}' has no data rows — nothing to look up."

    def _match_col(name, columns):
        if name is None:
            return None
        names = [str(c) for c in columns]
        if str(name) in names:
            return str(name)
        close = difflib.get_close_matches(str(name), names, n=1, cutoff=0.8)
        return close[0] if close else None

    tkey = _match_col(target_key_column, df_target.columns)
    lkey = _match_col(lookup_key_column, df_lookup.columns)

    # Fill in whichever side was not given (or was named wrongly) from the
    # actual shared values rather than failing outright.
    if not tkey or not lkey:
        coverage, guess_t, guess_l = _pick_join_columns(df_target, df_lookup)
        if coverage < 0.5 or not guess_t:
            return (f"Could not find a shared key between '{target_filename}' "
                    f"and '{lookup_filename}' — no pair of columns has "
                    f"matching values (best overlap {coverage:.0%}).\n"
                    f"{target_filename} columns: {list(df_target.columns)}\n"
                    f"{lookup_filename} columns: {list(df_lookup.columns)}\n"
                    f"Name the key columns explicitly and try again.")
        tkey = tkey or guess_t
        lkey = lkey or guess_l

    df_target["__key"] = df_target[tkey].map(_norm_key)
    df_lookup = df_lookup.copy()
    df_lookup["__key"] = df_lookup[lkey].map(_norm_key)

    # One row per key. Duplicates would multiply the target's rows — a silent
    # row explosion is far worse than taking the first and saying so.
    dupes = df_lookup["__key"].duplicated().sum()
    df_lookup = df_lookup.drop_duplicates(subset="__key", keep="first")

    wanted = [c for c in df_lookup.columns if c not in (lkey, "__key")]
    if columns_to_add:
        chosen = []
        for name in columns_to_add:
            hit = _match_col(name, wanted)
            if hit:
                chosen.append(hit)
        if not chosen:
            return (f"None of {columns_to_add} are columns in "
                    f"'{lookup_filename}'. Available: {wanted}")
        wanted = chosen

    # Never silently overwrite a column the target already has.
    renames = {c: f"{c} ({os.path.splitext(os.path.basename(lookup_path))[0]})"
               for c in wanted if c in df_target.columns}
    slim = df_lookup[["__key"] + wanted].rename(columns=renames)

    merged = df_target.merge(slim, on="__key", how="left")
    added = [renames.get(c, c) for c in wanted]

    # Count matches by KEY membership, never by whether an added column came
    # back non-null: a key can match perfectly onto a lookup row that happens
    # to have nothing in the first requested column. Measuring nullity instead
    # reported "62 of 64" for a join where all 64 keys matched and only A.17
    # and A.18 lacked a 'Body' value in the source document.
    lookup_keys = set(df_lookup["__key"])
    matched_mask = df_target["__key"].isin(lookup_keys)
    matched = int(matched_mask.sum())
    unmatched_keys = sorted({str(k) for k in
                             df_target.loc[~matched_mask, tkey].dropna()})

    merged = merged.drop(columns=["__key"])
    out_name = output_filename or os.path.basename(target_path)
    if not out_name.lower().endswith((".xlsx", ".xlsm", ".csv")):
        out_name += ".xlsx"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if out_name.lower().endswith(".csv"):
        merged.to_csv(out_path, index=False)
        fmt = "csv"
    else:
        merged.to_excel(out_path, index=False)
        fmt = "excel"

    ok, verdict = verify_export(out_path, len(merged), fmt, skiprows=0)
    if not ok:
        return f"FAILED to write {out_name} — {verdict}."

    msg = (f"{verdict}\n"
           f"Matched {tkey} -> {lkey}: {matched} of {len(merged)} rows found a "
           f"match and gained {len(added)} columns.\n"
           f"Download this exact filename: {out_name}\n"
           f"Columns added: {added}")
    if unmatched_keys:
        msg += (f"\nNo match in '{lookup_filename}' for these {tkey} values, "
                f"whose new columns are blank: {unmatched_keys}")
    if dupes:
        msg += (f"\nNote: '{lookup_filename}' had {dupes} duplicate {lkey} "
                f"value(s); the first row of each was used.")
    return msg


@tool
def inspect_output_file(filename: str, max_rows: int = 5) -> str:
    """Show what is actually inside a file this app EXPORTED — its sheets,
    row count, column names and the first few rows.

    Use whenever a question is about an .xlsx/.csv in outputs/ rather than an
    uploaded document: "what columns does spec.xlsx have", "is the data
    filled in", "check the file you just made", "what's the key column".
    Also use it BEFORE lookup_and_add_columns if unsure which columns two
    files share.

    query_table_data cannot answer this — it reads the uploaded source
    document, not the workbooks written into outputs/.

    Args:
        filename: Exported file name, e.g. 'w2-99.xlsx'
        max_rows: How many leading rows to show (default 5)
    """
    import pandas as pd

    path = _resolve_workbook(filename)
    if not path:
        return (f"'{filename}' is not in outputs/. Use list_uploaded_files to "
                f"see uploaded documents, or export something first.")

    try:
        if path.lower().endswith(".csv"):
            sheets = {"(csv)": pd.read_csv(path)}
        else:
            names = pd.ExcelFile(path).sheet_names
            sheets = {n: pd.read_excel(path, sheet_name=n) for n in names}
    except Exception as e:  # noqa: BLE001 -- report, don't kill the turn
        return f"Could not read '{filename}': {e}"

    parts = [f"{os.path.basename(path)} — {len(sheets)} sheet(s)"]
    for name, df in sheets.items():
        parts.append(f"\n=== {name} === {len(df)} rows x {len(df.columns)} columns")
        parts.append(f"Columns: {[str(c) for c in df.columns]}")
        if df.empty:
            parts.append("NO DATA ROWS — this sheet has headers only.")
            continue
        filled = {str(c): int(df[c].notna().sum()) for c in df.columns}
        blank = [c for c, n in filled.items() if n == 0]
        if blank:
            parts.append(f"Completely empty columns: {blank}")
        parts.append(df.head(max(1, max_rows)).to_string(index=False,
                                                         max_colwidth=30))
    return "\n".join(parts)
