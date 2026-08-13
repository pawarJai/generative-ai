"""LangGraph agent — replaces planner.py + dispatch.py entirely.

The agent uses tool-calling instead of intent classification:
- No regex routing
- No keyword matching
- The LLM reads tool docstrings and decides which to call
- Chat history is maintained automatically by LangGraph
- State persists across turns via SQLite checkpointer
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage
from app.config import llm
from app.graph.tools import (
    search_documents, query_table_data, list_uploaded_files,
    export_data, get_file_overview, get_page_content, get_table_of_contents,
    generate_quotation, analyze_past_contracts, modify_export
)
# Real-world tools for general-knowledge questions that need LIVE data —
# already implemented and tested in the legacy agent (app/agent/tools.py's
# agent_executor). Reused directly rather than duplicated: a LangChain @tool
# object works with any tool-calling agent, this one included. Without
# these, "answer general questions directly" only works for facts already
# in the LLM's training data — anything needing live data (weather, news)
# had no way to be answered and the model could only honestly decline.
from app.agent.tools import web_search, get_weather, get_news, calculate
from typing import Dict, List, Optional
import os
import re
import sqlite3

_EXPORT_TOOL_NAMES = {"export_data", "generate_quotation", "modify_export"}
_FABRICATED_FILENAME_RE = re.compile(r"\.(xlsx|csv|docx|pptx|pdf)\b", re.IGNORECASE)
_SUCCESS_PHRASE_RE = re.compile(
    r"\b(file\s+is\s+ready|ready\s+for\s+download|download\s+ready|"
    r"creat(?:e|ed)\s+(?:the\s+)?file\s+manually|"
    r"has\s+been\s+(successfully\s+)?(created|saved|exported)|"
    r"successfully\s+(created|saved|exported)|"
    r"(created|saved|exported)\s+successfully)\b",
    re.IGNORECASE,
)


def _catch_unverified_export_claim(response_text: str, turn_messages: list,
                                    prompt: str, file_id: Optional[str],
                                    fallback_filename: Optional[str] = None) -> tuple:
    """Deterministic backstop for a confirmed failure mode: after repeated
    export_data errors, the model announced "let me bypass the tool and
    create the file manually", fabricated a plausible-looking table in the
    chat response, and claimed the file was ready for download — while the
    real file on disk was untouched since the last (failed/wrong) tool
    call. A system-prompt rule asks the model not to do this, but it
    already ignored that once under real pressure, so this only trusts a
    "file created" claim when a matching tool call in THIS turn actually
    returned a verified success.

    The original regex required the success phrase and the filename to sit
    in the SAME sentence (no period/newline between them). Confirmed via a
    real production message this missed the actual failing case: "File is
    ready." as its own sentence, followed by "You can now download
    `f1-0121.xlsx`" in the NEXT sentence — same bug class already fixed
    once in _catch_skipped_export_request's regex, just never applied here.
    Now checks for a success phrase and a filename ANYWHERE in the
    response, no proximity requirement — a false positive here just adds
    an unnecessary correction note, but a false negative lets a fabricated
    file claim reach the user, so err toward catching more.

    Previously this only apologized and asked the user to specify columns
    and resend. Now, same escalation as _catch_skipped_export_request: try
    the real deterministic export in this same turn before falling back to
    an apology, so a fabricated claim gets replaced by an actual file
    wherever possible instead of another round-trip.

    A claim about a file is not on its own a request for one. "still you not
    able to givw currect data to me like file i did not get in last chat remind
    last 2 chats" carried a filename and a success phrase forward from an
    earlier answer, so this fired, discarded the whole reply and ran an export
    nobody had asked for — the user's actual question went unanswered and the
    response was a sandbox error about pandas DataFrames. A recovery export now
    runs only when THIS message asks for a file; otherwise the claim is
    corrected and the answer is left standing.

    Returns (final_response_text, forced_tool_name_or_None, recovery_attempted).
    """
    if not (_SUCCESS_PHRASE_RE.search(response_text) and _FABRICATED_FILENAME_RE.search(response_text)):
        return response_text, None, False

    last_export_result = None
    for msg in turn_messages:
        if getattr(msg, "name", None) in _EXPORT_TOOL_NAMES:
            last_export_result = str(msg.content)

    if last_export_result and "Verified:" in last_export_result:
        return response_text, None, False  # a real tool success this turn — trust it

    response_text = _strip_invented_links(response_text)

    if not _asks_for_a_file(prompt, fallback_filename):
        # The claim is about an earlier turn. Correct it where it will be read
        # — a note under 60 lines of confident markdown was scrolled past for
        # three turns while the user hunted a file that did not exist.
        return (
            "**No file was created this turn.** Any filename below is from an "
            "earlier message; check the file list before trusting it.\n\n---\n"
            + response_text
        ), None, False

    detail = f"\n\nThe export attempt this turn returned:\n{last_export_result}" \
        if last_export_result else ""

    if not file_id:
        return (
            "I claimed a file was created; nothing had written one." + detail +
            "\n\nNo file is currently active to export from — select a file "
            "and resend the request."
        ), None, True

    export_result = _attempt_recovery_export(prompt, file_id, fallback_filename)
    if "Verified:" in export_result:
        return (
            f"{export_result}\n\n---\nThe file above was written just now: my "
            f"first answer announced it without anything having created it, so "
            f"I ran the export for real rather than leaving you a claim." + detail
        ), "export_data", True

    return (
        "**No file was created.** I claimed one was; nothing had written it, "
        f"and running the export directly did not succeed either:\n\n{export_result}"
        + detail
    ), None, True


_FILENAME_RE = re.compile(r"\.(xlsx|csv|docx|pptx|pdf)\b", re.IGNORECASE)
_EXPORT_VERB_RE = re.compile(r"\b(create|make|generate|export|save|download)\b", re.IGNORECASE)
_CREATE_VERB_RE = re.compile(r"\b(create|make|generate|export|save|write)\b", re.IGNORECASE)
_FULL_FILENAME_RE = re.compile(r"\b[\w][\w\-]*\.(xlsx|csv|docx|pptx|pdf)\b", re.IGNORECASE)


def _asks_for_a_file(prompt: str, fallback_filename: Optional[str] = None) -> bool:
    """Whether THIS message asks for a file to be created.

    Real prompts ramble between the export verb and the filename, often past
    100 characters, so proximity is not required — but a name has to be
    available. `fallback_filename` is the one the user gave earlier in the
    same chat: "export into f1-14.xlsx" and then "we need to export all this
    table data" is one request for one file, and reading the second message as
    asking for nothing left the model to fill the silence — it announced 9
    rows, 6,145 bytes and a download link for a file that was never written.

    One definition, shared by both export backstops, so neither can recover an
    export the other has decided was never requested.
    """
    prompt = prompt or ""
    if _FILENAME_RE.search(prompt) and _EXPORT_VERB_RE.search(prompt):
        return True
    # Leaning on a name from an earlier message needs a verb that actually
    # asks for one to be made. "download" is an export verb when it sits
    # beside a filename, but on its own it is usually a complaint — "i not
    # able to download last given file" must not silently re-run an export.
    return bool(fallback_filename and _CREATE_VERB_RE.search(prompt))
# Page numbers are read around the word "page", in either direction and for a
# list of any length. A pattern that hardcoded "page N (sep M)?" read "page
# number 8 … check for this data is present on 9 ,10 page" as page 8 alone and
# exported 27 rows of a 64-row table, and read "page 3, 5 and 9" as pages 3
# and 5. Nothing here is specific to a phrasing: an anchor, optional filler
# words, and a separated list of numbers.
_PAGE_ANCHOR_RE = re.compile(r"\bpages?\b", re.IGNORECASE)
_PAGE_FILLER = (r"(?:number|numbers|no\.?|num|like|such\s+as|from|"
                r"starting(?:\s+at)?|is|are|:|=)")
_PAGE_SEP = (r"(?:,|&|\band\b|\bor\b|\bto\b|-|–|\bthrough\b|\buntil\b|"
             r"\bupto\b|\bup\s+to\b|/)")
# What may sit between "page", its filler word and the number. Whitespace was
# the only thing allowed, so "page number-8" and "page-9" — both of which real
# users type — matched nothing at all. In a single-document export that lost
# the page; in a two-document one it was worse than lost, because the OTHER
# document's page number was then the nearest one and got attributed here.
_PAGE_CONNECT = r"[\s\-–—:=]"
_PAGES_AFTER_RE = re.compile(
    rf"(?:{_PAGE_CONNECT}*{_PAGE_FILLER})*{_PAGE_CONNECT}*(\d+)"
    rf"((?:\s*{_PAGE_SEP}\s*\d+)*)", re.IGNORECASE)
_PAGES_BEFORE_RE = re.compile(
    rf"(\d+)((?:\s*{_PAGE_SEP}\s*\d+)*)\s*$", re.IGNORECASE)
# A range can straddle the anchor word: "this table is spread 6 to page number
# 10" puts the start before "page" and the end after it. Read forwards that is
# page 10 alone, and a five-page export came back as one page. The list before
# the anchor ends on a separator here, not on a digit, so _PAGES_BEFORE_RE
# cannot see it.
_PAGES_LEADING_RANGE_RE = re.compile(rf"(\d+)\s*({_PAGE_SEP})\s*$", re.IGNORECASE)
_PAGE_PAIR_RE = re.compile(rf"\s*({_PAGE_SEP})\s*(\d+)", re.IGNORECASE)
# 'to', '-', 'through' mean every page between the two; ',' and 'and' mean
# just the two named. "page number 6 to 10" was being read as the pair (6, 10),
# so an export of a five-page table silently returned one page of it.
_PAGE_RANGE_WORDS = {"to", "-", "–", "through", "until", "upto", "up to"}
# A range wider than this is a misparse (a year, an amount), not a page span.
_MAX_PAGE_SPAN = 100
# A number glued to a filename or an id ("data-file -1", "f1-133") is not a
# page number, however close it sits to the word "page".
_GLUED_TO_TOKEN = re.compile(r"[\w.\-/]")
_EXPORT_DIRECTIVE_RE = re.compile(
    r"\b(create|make|generate|export|save|download)\b[^.,]{0,40}?\b(excel|csv|xlsx|docx|pptx|file)\b|"
    r"\bfile\s*name\s*(give|is|:)?\b|\bsave\s+as\b|\bname\s+it\b",
    re.IGNORECASE,
)
_EXT_TO_FORMAT = {"xlsx": "excel", "csv": "csv", "docx": "docx", "pptx": "pptx", "pdf": "excel"}


def _extract_requested_filename(prompt: str) -> Optional[str]:
    m = _FULL_FILENAME_RE.search(prompt)
    if not m:
        return None
    name = m.group(0)
    if name.lower().endswith(".pdf"):
        name = name[:-4] + ".xlsx"
    return name


_SAVED_TO_RE = re.compile(
    r"saved to\s+([\w\-.]+\.(?:xlsx|csv|docx|pptx))", re.IGNORECASE)


def _resolve_export_filename(turn_messages: list, response_text: str) -> Optional[str]:
    """The name of the file this turn actually produced, for the UI's
    download link.

    Confirmed root cause of the "file not found in outputs/" 404: the UI
    fell back to scraping the first filename out of the response prose,
    which is frequently the SOURCE document (data-file-2.pdf, living in
    uploads/) rather than the export. So the name is taken from the export
    tool's own verified result, and returned only if the file is really on
    disk in OUTPUT_DIR — an unresolvable name yields None and no link,
    which is honest, instead of a link that 404s."""
    from app.config import OUTPUT_DIR

    candidates = []
    for msg in turn_messages:
        if getattr(msg, "name", None) in _EXPORT_TOOL_NAMES:
            candidates += _SAVED_TO_RE.findall(str(msg.content))
    candidates += _SAVED_TO_RE.findall(response_text or "")
    # Last resort: any export-shaped filename in the prose, still subject
    # to the exists-on-disk check below.
    candidates += [m.group(0) for m in
                   re.finditer(r"\b[\w][\w\-.]*\.(?:xlsx|csv|docx|pptx)\b",
                               response_text or "", re.IGNORECASE)]

    for name in candidates:
        if os.path.isfile(os.path.join(OUTPUT_DIR, name)):
            return name
    return None


_SOURCE_SENTENCE_RE = re.compile(r"Source: '([^']+)'[^.]*\.")


def _ensure_source_stated(response_text: str, turn_messages: list) -> str:
    """Guarantee the answer names the document an export actually read.

    The export tool now reports its source, but the model summarises tool
    output and dropped it ("data from page 7 of the document"). The whole
    point of Step 2 is that a wrong-document export cannot look like a right
    one, and that only holds if the document reaches the user — so this is
    deterministic rather than another system-prompt request.
    """
    if not response_text:
        return response_text
    for msg in turn_messages:
        if getattr(msg, "name", None) not in _EXPORT_TOOL_NAMES:
            continue
        m = _SOURCE_SENTENCE_RE.search(str(msg.content))
        if not m:
            continue
        if m.group(1) in response_text:
            return response_text  # the model kept it — nothing to add
        return f"{response_text}\n\n*{m.group(0).strip()}*"
    return response_text


def _page_mentions(prompt: str) -> List[tuple]:
    """Every "page …" in a prompt, as (position, pages_named_there).

    Each mention is read on both sides of the word and for a list of any
    length, because that is how people write it:

        "page number 6 to 10"                    -> 6, 7, 8, 9, 10
        "page 3, 5 and 9"                        -> 3, 5, 9
        "check ... is present on 9 ,10 page"     -> 9, 10
        "this table is spread 6 to page 10"      -> 6, 7, 8, 9, 10

    The last shape cost a real export 37 rows: the user listed the pages the
    table continued onto, the numbers came *before* the word, and the pattern
    only ever looked after it. Positions are kept because a multi-document
    export has to attach each page number to the file it sits nearest.
    """
    mentions: List[tuple] = []

    def read(first: str, tail: str) -> List[int]:
        found: List[int] = []

        def add(n: int):
            if n not in found and 0 < n < 10_000:
                found.append(n)

        prev = int(first)
        add(prev)
        for pair in _PAGE_PAIR_RE.finditer(tail or ""):
            sep = re.sub(r"\s+", " ", pair.group(1).strip().lower())
            n = int(pair.group(2))
            if sep in _PAGE_RANGE_WORDS and prev < n <= prev + _MAX_PAGE_SPAN:
                for k in range(prev + 1, n + 1):
                    add(k)
            else:
                add(n)
            prev = n
        return found

    def not_glued(text: str, start: int) -> bool:
        """A number glued to a filename or an id ("data-file -1", "f1-133") is
        not a page number, however close it sits to the word."""
        return not (start and _GLUED_TO_TOKEN.match(text[start - 1]))

    for anchor in _PAGE_ANCHOR_RE.finditer(prompt or ""):
        after = _PAGES_AFTER_RE.match(prompt, anchor.end())
        if after:
            pages = read(after.group(1), after.group(2))
            lead = _PAGES_LEADING_RANGE_RE.search(prompt[:anchor.start()])
            if lead and not_glued(prompt, lead.start()):
                sep = re.sub(r"\s+", " ", lead.group(2).strip().lower())
                start = int(lead.group(1))
                if sep in _PAGE_RANGE_WORDS and 0 < start < pages[0] \
                   and pages[0] - start <= _MAX_PAGE_SPAN:
                    pages = list(range(start, pages[0])) + pages
            mentions.append((anchor.start(), pages))
            continue
        before = _PAGES_BEFORE_RE.search(prompt[:anchor.start()])
        if before and not_glued(prompt, before.start()):
            mentions.append((before.start(), read(before.group(1), before.group(2))))
    return mentions


def _extract_requested_pages(prompt: str) -> list:
    """Every page number a request names, in the order they appear."""
    pages = []
    for _pos, found in _page_mentions(prompt):
        for n in found:
            if n not in pages:
                pages.append(n)
    return pages


def _clean_export_question(prompt: str, filename: Optional[str]) -> str:
    """The raw user prompt often contains export-instruction phrasing
    ("create excel file... file name give f1-03.xlsx") alongside the
    actual data question. Confirmed via direct testing: passing that raw
    phrasing straight into the code-exec sandbox's LLM makes it think it's
    being asked to WRITE a file itself ("Writing files is not allowed in
    this sandbox"), and it refuses instead of extracting data. Strip the
    filename and the create/save-a-file phrasing, leaving just the data
    question."""
    q = prompt
    if filename:
        q = q.replace(filename, "")
    q = _EXPORT_DIRECTIVE_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip(" .,-")
    return q or prompt


_WHOLE_TABLE_RE = re.compile(
    r"\bwhole (?:table|data|sheet)\b|\bentire (?:table|data)\b"
    r"|\ball (?:the )?(?:rows?|data|records?)\b|\bfull table\b"
    r"|\bwhere (?:this |the )?table (?:is )?end|\btable (?:is )?end(?:s|ing)?\b"
    r"|\bcontinue[sd]? (?:on|to|across)\b|\btill the end\b|\buntil the end\b"
    # How the same request is actually phrased in production: "this table data
    # is not fit only one page so check also pages like 9, 10 so on".
    r"|\b(?:other|another|next|upcom\w*|following|remaining|subsequent)\s+pages?\b"
    r"|\b(?:multiple|several|more than one|other) pages?\b|\bso on\b"
    r"|\bnot fit\b|\bspread(?:s|ing)? (?:across|over|on|to)\b"
    # "page number 8 table start all table data", "page number-8 started
    # data": naming where a table STARTS is a request for the whole table,
    # not for that one page. Read as a single page, a 64-row schedule came
    # back as its first 23 rows.
    r"|\b(?:table|data)\s+start(?:s|ed|ing)?\b"
    r"|\bstart(?:s|ed|ing)?\s+(?:from\s+|at\s+)?(?:all\s+)?(?:the\s+)?(?:table|data)\b",
    re.IGNORECASE)


def _wants_bare_table(prompt: str) -> bool:
    """True when the user asked for the table WITHOUT the document header band.

    "export data in f1-023.xlsx file without header data" was parsed correctly
    and then ignored: only modify_export ever asked the question, so a fresh
    export always carried the band and the user had to say it twice. The same
    parser answers for both paths — there is one definition of what "without
    the header" means, not two.

    A message that says nothing about the band inherits what the chat last
    said about it. "if i need header than i will tell but right now i don't
    need" is a standing instruction: the band came back on the very next
    export, which is the third time the same complaint was filed.
    """
    from app.export.modify import parse_instruction
    stated = parse_instruction(prompt or "").get("context")
    if stated is None:
        from app import state as app_state
        remembered = app_state.get_band_preference(app_state.get_active_session_id())
        return remembered is False
    return stated is False


def _deterministic_page_export(file_id: str, pages: list, out_filename: str,
                               out_format: str, whole_table: bool = False,
                               no_context: bool = False) -> str:
    """Bypasses the code-exec LLM sandbox entirely for the single
    highest-frequency failure pattern in this whole project: "export page
    N and M". The sandbox's LLM has twice — in real production turns, not
    just testing — either refused when a requested page had no table, or
    silently substituted a DIFFERENT, wrong-numbered table and fabricated
    a code comment claiming it "represents" the requested page (verified:
    it picked tables '11'/'12' for a request about pages 6/7, with a
    comment asserting they were pages 6 and 7). Reading real Docling table
    objects directly via get_all_real_tables() — already the ground-truth
    source used elsewhere in this codebase (app/tables/helpers.py) — means
    page-to-table mapping is exact, not an LLM guess."""
    from app.tables.helpers import get_all_real_tables
    from app.export.exporters import EXPORTERS, verify_export
    from app.models import QueryPlan
    from app import state as app_state
    import os
    from app.config import OUTPUT_DIR

    # Name the document in every outcome. A confirmed production export read
    # the wrong file entirely and reported "page 7 contained no extractable
    # tables" — true of the file it read, false of the file the user asked
    # about, and indistinguishable from a correct answer because no document
    # was ever named.
    src = _display_name(file_id, app_state.FILE_ORIGINAL_NAME.get(file_id) or file_id)

    # "the whole table", "where does it end", "all pages" — follow the table
    # across every page it continues onto instead of stopping at the page the
    # user happened to name. A 64-row schedule was being exported as 17 rows.
    #
    # Every named page is expanded, not just the first: expanding only from
    # min(pages) would silently drop a second, unrelated page the user also
    # asked for. The expanded list then goes through the same assembly and
    # verification as any other export.
    span_note = ""
    if whole_table and pages:
        from app.tables.helpers import span_pages
        expanded = list(pages)
        for page in pages:
            for continued in span_pages(file_id, page):
                if continued not in expanded:
                    expanded.append(continued)
        if len(expanded) > len(pages):
            added = sorted(p for p in expanded if p not in pages)
            span_note = (f" The table the user named does not fit on one page: "
                         f"page(s) {', '.join(map(str, added))} continue it, and "
                         f"are included.")
            pages = sorted(expanded)

    # Aligned, not concatenated. pd.concat over fragments with different
    # column counts and no headers is what produced outputs/f1-66.xlsx, where
    # page 8 onward had the item text under 'Evaluation Schedules' and the
    # quantities under 'Consignee Address' — a file that opened cleanly, was
    # reported as "65 rows, 5 columns, export complete", and was wrong.
    from app.tables.helpers import assemble_pages
    df, report, found = assemble_pages(file_id, pages)

    all_tables = get_all_real_tables(file_id)
    missing = [p for p in pages if p not in found]
    if df is None:
        return (f"None of the requested page(s) {pages} have an extractable table "
                f"in '{src}' (file_id {file_id}) — no file was created. Checked "
                f"against {len(all_tables)} real tables found across that whole "
                f"document. If the user meant a different document, say so rather "
                f"than reporting this one as empty.")

    plan = QueryPlan(intent="export", sink=out_format, filename=out_filename,
                     no_context=no_context)
    export_fn = EXPORTERS.get(out_format, EXPORTERS["excel"])
    export_fn(file_id, plan, tables=[df])

    path = os.path.join(OUTPUT_DIR, out_filename)
    ok, verify_msg = verify_export(path, len(df), out_format)
    source = (f" Source: '{src}' (file_id {file_id}), page(s) "
              f"{', '.join(map(str, found))}.")
    caveat = span_note + (
        f" Note: page(s) {missing} have no extractable table in '{src}', "
        f"so they are not included." if missing else "")

    # Stopping in the middle of a table is the "you again give me only 27
    # records" failure: page 8 holds 27 rows of a 64-row schedule and nothing
    # in the answer said the other 37 existed. Say it, with the pages and the
    # count, so a short export can never look like a complete one.
    if found:
        from app.tables.helpers import span_pages
        beyond = [p for p in span_pages(file_id, min(found)) if p not in found]
        if beyond:
            extra = sum(t.shape[0] for t in all_tables
                        if t.attrs.get("page") in beyond)
            caveat += (f" Note: this table CONTINUES onto page(s) "
                       f"{', '.join(map(str, beyond))} — {extra} further rows "
                       f"that are not in this file. Tell the user the export "
                       f"holds {len(df)} of {len(df) + extra} rows and offer to "
                       f"export the whole table.")
    # Anything the aligner could not place is stated, not hidden behind a
    # success message — an unmatched column means the file is not simply "the
    # table" the user asked for.
    if report is not None and not report.clean:
        caveat += f" Note: {report.describe()}."
    if not ok:
        return f"Export attempted but verification failed: {verify_msg}{source}{caveat}"
    return f"{verify_msg}{source}{caveat}"


_ALL_FILES_RE = re.compile(
    r"\b(both|all|each|every)\b[^.]{0,20}?"
    r"\b(files?|documents?|docs?|uploads?|pdfs?|sheets?|spreadsheets?)\b",
    re.IGNORECASE)


def _normalize_with_offsets(text: str) -> tuple:
    """Lowercased alphanumeric-only text, plus a map back to the original
    offsets. File handles have to be matched on the squashed form ("data
    file 2" == "data-file-2" == "datafile2"), but assigning page numbers to
    the right file needs real positions in the user's actual sentence."""
    norm, idx_map = [], []
    for i, ch in enumerate(text.lower()):
        if ch.isalnum():
            norm.append(ch)
            idx_map.append(i)
    return "".join(norm), idx_map


def _find_file_mentions(prompt: str) -> List[tuple]:
    """EVERY mention of a registered file, as (position_in_prompt, file_id).

    A document named twice is named twice. Confirmed production failure,
    shipped as outputs/f1-2f.xlsx: "export data data-file-1 and data-file-2
    tables like data-file-1 page number 8 ..." names data-file-1 in a preamble
    AND again where the page number is, but only the first occurrence was
    recorded — so data-file-1's only anchor was the preamble, page 8 landed
    nearer data-file-2 and was attributed to it, and data-file-1 fell through
    to "all tables" and contributed two table groups nobody asked for.
    """
    from app.persistence import get_all_files

    norm, idx_map = _normalize_with_offsets(prompt)
    if not norm:
        return []
    records = [r for r in get_all_files() if os.path.exists(r["path"])]

    candidates = []
    for record in reversed(records):  # newest upload of a name wins
        for key in _name_keys(record["original_filename"]):
            for m in re.finditer(re.escape(key), norm):
                candidates.append((len(key), m.start(), key, record["file_id"]))

    # Longest handle first, then claim its span: "data-file-22" must not also
    # register as a mention of "data-file-2", and two handles for the same
    # upload must not both count at one spot.
    claimed, hits = [], []
    for _, pos, key, fid in sorted(candidates, key=lambda c: (-c[0], c[1])):
        end = pos + len(key)
        if any(pos < c_end and c_pos < end for c_pos, c_end in claimed):
            continue
        claimed.append((pos, end))
        hits.append((idx_map[pos], fid))
    return sorted(hits)


def _find_named_files(prompt: str) -> List[tuple]:
    """The distinct files a prompt names, as (first_position, file_id), in the
    order they first appear.

    One entry per document, because callers use the length of this to decide
    "one file or several". Attributing a page number to a file needs every
    mention instead — that is _find_file_mentions.
    """
    seen, hits = set(), []
    for pos, fid in _find_file_mentions(prompt):
        if fid not in seen:
            seen.add(fid)
            hits.append((pos, fid))
    return hits


def _nearest_index(positions: List[int], target: int) -> int:
    return min(range(len(positions)), key=lambda i: abs(positions[i] - target))


def _resolve_source_specs(prompt: str, default_file_id: Optional[str]) -> List[dict]:
    """The documents an export should draw from, as
    [{file_id, pos, pages}] in the order they were mentioned.

    Returns fewer than two entries when this is an ordinary single-document
    export — callers then keep using the existing single-file path, so the
    multi-document code can never change the behaviour of the exports that
    already work.
    """
    from app import state as app_state
    from app.persistence import get_all_files

    # Resolve scope from what the USER wrote. Tools are handed the model's
    # paraphrase, and one such rewrite ("all tables from all uploaded files")
    # turned a request about a single tender into a 13-sheet, 907-row export
    # spanning every document in the chat.
    user_prompt = app_state.get_current_user_prompt()
    if user_prompt:
        prompt = user_prompt

    named = _find_named_files(prompt)
    if len(named) == 1:
        # One document named explicitly means one document, whatever else the
        # sentence says. "combine all the table columns from data-file-2"
        # must not be read as "combine all my files".
        return []
    if not named:
        # "combine all my documents" names nothing explicitly. It means the
        # documents of THIS chat — the registry is scoped by session, so this
        # no longer depends on which files a restart happened to leave loaded,
        # and it can never reach into another conversation's uploads.
        if not _ALL_FILES_RE.search(prompt):
            return []
        session = app_state.get_active_session_id()
        fids = [r["file_id"] for r in get_all_files(session)] if session else []
        if len(fids) < 2:
            fids = list(app_state.FILE_ORDER)
        if len(fids) < 2:
            return []
        named = [(0, fid) for fid in fids]

    positions = [p for p, _ in named]
    specs = [{"file_id": fid, "pos": p, "pages": []} for p, fid in named]
    by_fid = {s["file_id"]: s for s in specs}

    # A page number belongs to whichever file MENTION it sits closest to,
    # which reads correctly in both orders users actually write:
    # "data-file-2 page 7" and "page 7 of data-file-2". Nearest mention, not
    # nearest document: a document listed once in a preamble and again beside
    # its page number has two anchors, and only the second one means anything.
    mentions = [m for m in _find_file_mentions(prompt) if m[1] in by_fid] or named
    mention_pos = [p for p, _ in mentions]

    for position, found in _page_mentions(prompt):
        owner = by_fid[mentions[_nearest_index(mention_pos, position)][1]]
        for n in found:
            if n not in owner["pages"]:
                owner["pages"].append(n)
    return specs


def _label_of(df) -> str:
    return str(df.attrs.get("page", ""))


_PROVENANCE_COLS = {"_source_file", "_source_page"}


def _is_headerless(df) -> bool:
    """True when ingestion recovered no real column names for this table —
    every column is a col_N / positional placeholder. Such a table carries no
    evidence about what its columns MEAN, so it can be stacked next to another
    document's table but never merged into its named columns."""
    from app.tables.assembly import is_generic
    names = [c for c in df.columns if str(c) not in _PROVENANCE_COLS]
    return bool(names) and all(is_generic(c) for c in names)


def _page_tag(pages: List[int]) -> str:
    """Compact sheet-name tag for a page list: p8, p8-10, p8,12."""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p{pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"p{pages[0]}-{pages[-1]}"
    return "p" + ",".join(map(str, pages))


def _select_tables_for_spec(spec: dict, prompt: str, positions: List[int],
                            tables: list, whole_table: bool = False,
                            pages_named_elsewhere: bool = False) -> tuple:
    """The tables one source document contributes, and how they were chosen.

    Three ways, in descending order of how explicit the user was: the page
    numbers they gave, a sheet/block name they typed ("the Working Sheet"),
    or everything the document has.

    Returns (tables, detail, tag, missing_pages, resolved_pages).
    """
    if spec["pages"]:
        pages = list(spec["pages"])
        # The same span-following the single-document export already does.
        # Without it "data-file-1 page number 8, the table starts there" gave
        # page 8's fragment alone — 23 rows of a 64-row schedule.
        if whole_table:
            from app.tables.helpers import span_pages
            for page in list(pages):
                for continued in span_pages(spec["file_id"], page):
                    if continued not in pages:
                        pages.append(continued)
        pages.sort()
        picked = [t for t in tables if _label_of(t).isdigit()
                  and int(_label_of(t)) in pages]
        got = sorted({int(_label_of(t)) for t in picked})
        return (picked, f"page {', '.join(map(str, got))}" if got else "",
                _page_tag(got),
                [p for p in spec["pages"] if p not in got], got)

    # Sheet/block labels are real strings on tabular tables ("Working Sheet"),
    # so a named sheet can be matched against the document instead of guessed.
    norm_prompt, idx_map = _normalize_with_offsets(prompt)
    by_label = []
    for t in tables:
        label = _label_of(t)
        if not label or label.isdigit():
            continue
        key = re.sub(r"[^a-z0-9]", "", label.lower())
        if len(key) < 4:
            continue
        pos = norm_prompt.find(key)
        # ...but only if the mention is closer to THIS file than to the other
        # source, so two documents with a "Sheet1" don't both claim it.
        if pos >= 0 and positions[_nearest_index(positions, idx_map[pos])] == spec["pos"]:
            by_label.append(t)
    if by_label:
        labels = sorted({_label_of(t) for t in by_label})
        return by_label, "sheet " + ", ".join(labels), ", ".join(labels), [], []

    # The user named page numbers and none of them resolved to this document.
    # Falling through to "everything" here is how f1-2f.xlsx acquired two
    # table groups nobody asked for (130 rows across pages 21-25 and 31-35,
    # from a request that named page 8). Contributing nothing, and saying so,
    # is the honest answer to "I could not tell which pages you meant".
    if pages_named_elsewhere:
        return [], "", "", [], []

    # Everything. Tabular ingestion keeps a "<sheet>_Raw" copy of each sheet
    # alongside the cleaned one; including both would silently duplicate
    # every row in the output.
    cleaned = [t for t in tables if not _label_of(t).endswith("_Raw")]
    return (cleaned or tables), "all tables", "all", [], []


MAX_GROUPS_PER_DOC = 3


def _group_by_schema(tables: list, max_groups: int = MAX_GROUPS_PER_DOC) -> tuple:
    """Split one document's tables into logical tables, largest first.

    Concatenating every table in a document produces a union of every column
    that appears anywhere — measured on the real corpus: 49 tables in
    data-file-1 became 406 rows x 76 columns, and data-file-5 became 593 x
    151, almost all of it empty. Tables that share a column signature are the
    same logical table split across pages (exactly the page 6/7 product
    schedule), so grouping on that signature restores real tables instead of
    a sparse union.

    Returns (groups, skipped) where each group is (tables, label).
    """
    buckets: Dict[tuple, list] = {}
    for t in tables:
        buckets.setdefault(tuple(str(c) for c in t.columns), []).append(t)

    ordered = sorted(buckets.values(), key=lambda g: sum(len(t) for t in g), reverse=True)
    groups = []
    for group in ordered[:max_groups]:
        labels = [_label_of(t) for t in group if _label_of(t)]
        nums = sorted(int(l) for l in labels if l.isdigit())
        if nums:
            label = f"p{nums[0]}" if len(nums) == 1 else f"p{nums[0]}-{nums[-1]}"
        else:
            label = labels[0] if labels else "table"
        groups.append((group, label))

    skipped = ordered[max_groups:]
    return groups, (len(skipped), sum(len(t) for g in skipped for t in g))


_SHEET_SAFE_RE = re.compile(r"[\[\]:*?/\\]")


def _display_name(file_id: str, stored_name: str) -> str:
    """What to call a document in front of the user.

    Uploads are saved as "{file_id}_{original}", and that mangled string is
    what the registry keeps — so a user who uploaded data-file-2.pdf would
    otherwise see their source labelled
    "data-file-2_0773_data-file-2.pdf" in every sheet name and provenance
    cell. Stripping the known file_id prefix is exact, not a guess at which
    part of the name is real."""
    name = stored_name or file_id
    prefix = f"{file_id}_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _source_sheet_name(display_name: str, tag: str) -> str:
    """Excel caps sheet names at 31 characters. Truncating the whole string
    silently ate the useful half — "data-file-5 sheet Working Sheet" became
    "data-file-5 sheet W" — so the document name gives way to the part that
    says WHICH slice of it this is."""
    stem = _SHEET_SAFE_RE.sub("-", os.path.splitext(display_name)[0]).strip()
    tag = _SHEET_SAFE_RE.sub("-", tag or "").strip()[:20]
    if not tag:
        return stem[:31] or "Sheet"
    return f"{stem[: 31 - len(tag) - 1]} {tag}".strip()


def _deterministic_multi_export(specs: List[dict], prompt: str,
                                out_filename: str, out_format: str) -> str:
    """Build one file out of several uploaded documents, with every row
    traceable back to the document and page it came from.

    Same reasoning as _deterministic_page_export, which this extends: the
    page/sheet -> table mapping is read off the real Docling/tabular objects
    rather than asked of the code-exec sandbox, because the sandbox was
    repeatedly caught substituting a differently-numbered table and
    asserting in a comment that it was the requested one. With two source
    documents in play that failure would be invisible — the wrong rows still
    look plausible next to the right ones.
    """
    from app.tables.helpers import get_all_real_tables, _restore_dropped_labels
    from app.tables.assembly import assemble
    from app.export.exporters import EXPORTERS, verify_export
    from app.models import QueryPlan
    from app.persistence import get_file
    from app import state as app_state
    from app.config import OUTPUT_DIR
    import pandas as pd

    positions = [s["pos"] for s in specs]
    sheets, notes, described = [], [], []
    # Judged from the user's own words, like every other scope decision here.
    whole_table = bool(_WHOLE_TABLE_RE.search(prompt))
    pages_named = bool(_page_mentions(prompt))

    # Two uploads can carry the same filename (a v1 and v2 of the same
    # tender). Labelling both rows "tender.pdf" would make the provenance
    # columns useless for the one case they exist to resolve.
    display = {s["file_id"]: _display_name(
        s["file_id"], app_state.FILE_ORIGINAL_NAME.get(s["file_id"])
        or (get_file(s["file_id"]) or {}).get("original_filename") or s["file_id"])
        for s in specs}
    ambiguous = {n for n in display.values()
                 if list(display.values()).count(n) > 1}

    for spec in specs:
        fid = spec["file_id"]
        name = display[fid]
        label = f"{name} [{fid}]" if name in ambiguous else name
        err = _restore_file_if_needed(fid)
        if err:
            notes.append(err)
            continue
        tables = get_all_real_tables(fid)
        if not tables:
            notes.append(f"'{name}' has no extractable tables, so it "
                         f"contributes nothing to this file.")
            continue

        picked, detail, tag, missing, resolved = _select_tables_for_spec(
            spec, prompt, positions, tables, whole_table=whole_table,
            pages_named_elsewhere=pages_named and not spec["pages"])
        if missing:
            notes.append(f"page(s) {missing} of '{name}' have no extractable "
                         f"table and are not included.")
        if not picked:
            if pages_named and not spec["pages"]:
                notes.append(
                    f"the request names page numbers but none of them could be "
                    f"tied to '{name}', so nothing was taken from it — say "
                    f"which of its pages you want (e.g. \"{name} page 8\").")
            continue

        # When the user named pages or a sheet, they picked the table — keep
        # it whole. When they said "everything", split the document into its
        # logical tables by column signature instead of unioning all of them
        # into one mostly-empty frame.
        if resolved:
            # Same label recovery the single-document page export gets:
            # Docling drops the row-spanning leftmost column on some pages of
            # a multi-page table, and padding it with blanks loses real data.
            groups = [(_restore_dropped_labels(fid, picked), tag)]
        elif tag == "all":
            groups, (n_skipped, rows_skipped) = _group_by_schema(picked)
            if n_skipped:
                notes.append(f"'{name}' also has {n_skipped} smaller table "
                             f"group(s) ({rows_skipped} rows) not included — "
                             f"ask for them by page if you need them.")
        else:
            groups = [(picked, tag)]

        for group, group_tag in groups:
            frames = []
            for t in group:
                d = t.copy()
                d.attrs.update(t.attrs)
                # assemble() reads provenance off attrs, so the columns get
                # added AFTER alignment — adding them first would make them
                # part of what is being aligned.
                d.attrs["source_file"] = label
                frames.append(d)
            # Aligned, not concatenated. pd.concat over fragments whose
            # columns differ unions them positionally, which is exactly the
            # failure this project already fixed inside a single document —
            # across documents it is less visible, not less wrong.
            merged, report = assemble(frames, provenance=True)
            if merged is None:
                continue
            for item in report.unmatched:
                notes.append(f"column {item['column']!r} from {item['part']} "
                             f"did not match that table's own columns and was "
                             f"kept in its own column rather than folded in.")
            # One sheet per logical table, built here rather than by grouping
            # on sheet name afterwards — two documents sharing a name would
            # have been silently merged into a single sheet by that grouping.
            merged.attrs["sheet_name"] = _source_sheet_name(name, group_tag)
            # The document itself, separate from the sheet label — a deck's
            # title slide lists source documents, not sheet names.
            merged.attrs["source_file"] = label
            sheets.append(merged)
        described.append(
            f"'{label}' ({detail}"
            + (f", {len(groups)} tables" if tag == "all" else "") + ")")

    if not sheets:
        return ("No data could be taken from the requested documents, so no "
                "file was created. " + " ".join(notes))

    # Excel only. The other exporters flatten their table list into a single
    # output — export_csv concatenates it — so adding a Combined frame that
    # is itself the concatenation of the others would duplicate every row.
    #
    # The gate used to be exact column-name equality, which no pair of real
    # documents ever satisfies: one side of the confirmed failure had proper
    # headers and the other had col_0..col_3 because header recovery lost
    # them, so "their columns differ" was reported on every single
    # multi-document export and a merged sheet was never once produced. The
    # alignment already used within a document answers the same question
    # properly — by column content, not by spelling — and names whatever it
    # could not match instead of dropping it.
    combined_sheet, unmatched_in_combined, stacked = False, [], []
    if out_format == "excel" and len(sheets) > 1:
        # Signature alignment needs name evidence on at least one side. Where
        # header recovery failed and a table is nothing but col_0..col_3, the
        # matcher is comparing text-shaped values to text-shaped values and
        # scores 0.909 for putting a contact name under 'Spec No.' and a
        # postal address under 'Material code' — measured, on exactly these
        # two documents. So a headerless table is STACKED beside a headed one,
        # never mapped onto it: every value keeps its own column, the rows
        # still land in one sheet, and no column claims to be a column it is
        # not. Two headed tables, or two headerless ones, still align.
        headed = [s for s in sheets if not _is_headerless(s)]
        stacked = [s for s in sheets if _is_headerless(s)] if headed else []
        if stacked:
            combined = pd.concat(sheets, ignore_index=True, sort=False)
            combined_report = None
        else:
            combined, combined_report = assemble(sheets)
        if combined is not None:
            combined.attrs["sheet_name"] = "Combined"
            if combined_report is not None:
                unmatched_in_combined = [i["column"] for i in combined_report.unmatched]
            sheets.insert(0, combined)
            combined_sheet = True

    plan = QueryPlan(intent="export", sink=out_format, filename=out_filename,
                     no_context=_wants_bare_table(prompt))
    EXPORTERS.get(out_format, EXPORTERS["excel"])(specs[0]["file_id"], plan, tables=sheets)

    path = os.path.join(OUTPUT_DIR, out_filename)
    total = sum(len(s) for s in sheets)
    ok, verify_msg = verify_export(path, total, out_format)
    summary = (f" Sources: {', '.join(described)}. Every row carries its "
               f"_source_file and _source_page.")
    if out_format == "excel" and combined_sheet and stacked:
        names = ", ".join(sorted({str(s.attrs.get("sheet_name")) for s in stacked}))
        summary += (
            f" There is one sheet per source table, plus a Combined sheet with "
            f"all rows in it. In Combined, {names} keeps its own col_N columns "
            f"beside the other document's instead of being merged into them: "
            f"ingestion recovered no header row for that table, so there is no "
            f"evidence its columns mean the same things, and mapping them would "
            f"have filed values under headings they do not belong to.")
    elif out_format == "excel" and combined_sheet:
        summary += (" There is one sheet per source table, plus a Combined "
                    "sheet holding all of them merged on aligned columns.")
        if unmatched_in_combined:
            summary += (f" In the Combined sheet, {len(unmatched_in_combined)} "
                        f"column(s) had no counterpart in the other document "
                        f"({', '.join(repr(c) for c in unmatched_in_combined[:4])}) "
                        f"and are kept as their own columns rather than merged "
                        f"into one that only looks right.")
    elif out_format == "excel":
        summary += " There is one sheet per source table."
    if notes:
        summary += " Note: " + " ".join(notes)
    if not ok:
        return f"Export attempted but verification failed: {verify_msg}{summary}"
    return f"{verify_msg}{summary}"


def _attempt_recovery_export(prompt: str, file_id: str,
                             fallback_filename: Optional[str] = None) -> str:
    """Shared deterministic recovery logic used by both export backstops:
    given the raw user prompt and the active file_id, actually perform the
    export rather than just telling the user what should happen. Prefers
    the exact page-number-based extraction (bypassing the code-exec LLM
    sandbox entirely — see _deterministic_page_export) whenever the prompt
    names page numbers, since that's the failure mode confirmed repeatedly
    in real production logs; falls back to the normal LLM-driven
    export_data tool for everything else."""
    filename = (_extract_requested_filename(prompt) or fallback_filename
                or "export.xlsx")
    ext = filename.rsplit(".", 1)[-1].lower()
    fmt = _EXT_TO_FORMAT.get(ext, "excel")

    specs = _resolve_source_specs(prompt, file_id)
    if len(specs) > 1:
        return _deterministic_multi_export(specs, prompt, filename, fmt)

    pages = _extract_requested_pages(prompt)
    if pages:
        return _deterministic_page_export(
            file_id, pages, filename, fmt,
            whole_table=bool(_WHOLE_TABLE_RE.search(prompt)),
            no_context=_wants_bare_table(prompt))

    from app.graph.tools import export_data
    cleaned_question = _clean_export_question(prompt, filename)
    try:
        return export_data.invoke({
            "question": cleaned_question, "format": fmt,
            "filename": filename, "file_id": file_id,
        })
    except Exception as e:
        return f"Export failed: {e}"


def _catch_skipped_export_request(prompt: str, response_text: str, turn_messages: list,
                                   file_id: Optional[str],
                                   fallback_filename: Optional[str] = None) -> tuple:
    """Mirror image of _catch_unverified_export_claim: confirmed failure
    mode where the user gave an explicit, unambiguous export request with
    a named output filename ("create excel file... file name give
    f1-03.xlsx"), and the model answered a page-lookup-shaped part of the
    same message, then ended with "let me know if you'd like..." instead
    of actually calling export_data — no file, of any kind, got created.

    This used to just append a note asking the user to re-send the
    request. Confirmed in real production logs TWICE (not once) that
    strengthening the system-prompt rule alone does not reliably stop the
    model from doing this again — the model violated an already-explicit
    version of Rule 6 a second time on the identical prompt shape. A
    system-prompt instruction is a request; it is not deterministic. So
    this now actually RUNS the export in this same turn instead of asking
    the user to trigger it again — the same escalation already proven
    necessary for _catch_unverified_export_claim's failure mode.

    Whether the message asks for a file at all is `_asks_for_a_file` — the
    same test the other backstop uses, so the two cannot disagree about it.

    This runs only when `_catch_unverified_export_claim` has not already
    recovered. Its recovery happens outside the graph and so leaves no tool
    message behind, which the `turn_messages` scan below reads as "no export
    was attempted" — two successful exports of the same request ran back to
    back, and the user was shown the identical result twice under two
    different apologies (confirmed, log ids 5529d728 and c94da052).

    Returns (final_response_text, forced_tool_used_name_or_None).
    """
    if not _asks_for_a_file(prompt, fallback_filename):
        return response_text, None

    for msg in turn_messages:
        if getattr(msg, "name", None) in _EXPORT_TOOL_NAMES:
            return response_text, None  # export was actually attempted — nothing to catch

    if not file_id:
        return (
            response_text +
            "\n\n---\n**Note:** you asked me to create a file, but I didn't call "
            "the export tool this turn, and no file is currently active to export "
            "from. Select a file and resend the request."
        ), None

    export_result = _attempt_recovery_export(prompt, file_id, fallback_filename)

    if "Verified:" in export_result:
        return (
            f"{export_result}\n\n---\nThe answer above offered the file instead "
            f"of creating it, so I ran the export rather than making you ask "
            f"twice."
        ), "export_data"

    return (
        "**No file was created.** I answered without exporting, then ran the "
        f"export directly — that came back honest, not successful:\n\n"
        f"{export_result}\n\n---\n" + _strip_invented_links(response_text)
    ), None


def run_export_backstops(prompt: str, response_text: str, turn_messages: list,
                         file_id: Optional[str],
                         fallback_filename: Optional[str] = None) -> tuple:
    """Both export backstops, in order, with at most one recovery between them.

    They catch opposite failures — a file claimed but never written, and a file
    requested but never offered — and each recovers by exporting outside the
    graph, which leaves no tool message behind. The second then read that
    absence as "no export was attempted" and exported the same request again:
    two writes, and the identical verified result shown to the user twice under
    two different apologies (log ids 5529d728, c94da052).

    Returns (final_response_text, forced_tool_used_name_or_None).
    """
    response_text, forced_tool, recovered = _catch_unverified_export_claim(
        response_text, turn_messages, prompt, file_id, fallback_filename)
    if recovered:
        return response_text, forced_tool

    response_text, skipped_tool = _catch_skipped_export_request(
        prompt, response_text, turn_messages, file_id, fallback_filename)
    return response_text, skipped_tool or forced_tool


def _restore_file_if_needed(file_id: str) -> Optional[str]:
    """Confirmed root cause of a whole class of failures in this session:
    app/state.py is pure in-memory, and `uvicorn --reload` restarts the
    process on every source file change — wiping every ingested file's
    DOCLING_DOCS/TABULAR_TABLES/FILE_META. A file_id the user (or a prior
    conversation turn) still references then looks like it "contains no
    detectable tables... appears empty or corrupted", when the file itself
    was never actually broken — it was just never reloaded after a restart.

    app/startup.py already has restore_state() for this, but it's disabled
    because eagerly re-ingesting every registered file at boot blocks
    startup for minutes with several large PDFs. This is the lazy version:
    restore only the one file actually being asked about, only when it's
    actually missing, right before it's needed.

    This used to fail silently (return None either way) whenever restoration
    wasn't possible, which let downstream tools report a generic "file
    appears empty or corrupted" for a file_id that was actually fine — the
    real problem (registry has no record, or the registered upload is gone
    from disk, e.g. uploads/ got cleared externally) was invisible. Now it
    returns None on success and a human-readable reason on failure, so the
    caller can surface the true cause instead of guessing.
    """
    from app import state as app_state
    if file_id in app_state.FILE_KIND:
        return None  # already loaded, nothing to do

    from app.persistence import get_file
    record = get_file(file_id)
    if not record:
        return (f"'{file_id}' is not a recognized file_id — it has no upload "
                f"record at all. Check the file_id or upload the file first.")

    import os
    if not os.path.exists(record["path"]):
        return (f"'{file_id}' (\"{record['original_filename']}\") was uploaded "
                f"previously, but its source file is no longer on disk at "
                f"{record['path']!r}. It cannot be recovered — please "
                f"re-upload the file.")

    try:
        from app.ingestion.universal import universal_ingest
        universal_ingest(record["path"], file_id)
        return None
    except Exception as e:
        print(f"[_restore_file_if_needed] failed to restore {file_id}: {e}")
        return (f"'{file_id}' (\"{record['original_filename']}\") failed to "
                f"restore from its saved upload: {e}")


_FALSE_MISSING_RE = re.compile(
    r"\b(re-?upload|upload (?:it|the file) again|"
    r"(?:appears?|seems?) to be (?:empty|corrupted|damaged)|"
    r"is (?:empty|corrupted|damaged)|"
    r"no detectable (?:tables?|data)|"
    r"not a valid (?:spreadsheet|excel|file|document)|"
    r"contains no (?:tables?|data|content))\b",
    re.IGNORECASE,
)


def _catch_false_missing_file_claim(response_text: str, prompt: str,
                                     file_id: Optional[str]) -> str:
    """Deterministic backstop against telling a user their file is broken
    when it demonstrably is not.

    Confirmed production failure (20:18:34): the model replied "the file
    data-file-4 contains no detectable tables or data — it appears to be
    empty, corrupted, or not a valid spreadsheet… re-upload the file", for a
    file that was registered, present on disk, and holds 4 extractable
    tables plus indexed chunks in Chroma. Step 4 fixed the routing that
    caused it; this catches any remaining path that reaches the same wrong
    conclusion, because a user acting on that advice destroys their own
    working state.

    Only fires when the claim is provably false — registry record present,
    source file on disk. Otherwise the response is left alone, since the
    file really may be gone."""
    if not file_id or not response_text or not _FALSE_MISSING_RE.search(response_text):
        return response_text
    # "how many files do I have" is answered by _catch_wrong_file_inventory.
    # This check fired on the incidental phrase "re-upload the missing file"
    # and pasted a wall of unrelated document text under an inventory answer.
    if _INVENTORY_ASK_RE.search(prompt or ""):
        return response_text

    from app.persistence import get_file
    from app.query.semantic import semantic_fallback, has_indexed_content
    from app.tables.helpers import get_all_real_tables

    record = get_file(file_id)
    if not record or not os.path.exists(record["path"]):
        return response_text  # the claim may well be true — don't override

    # Only count tables when the file is actually loaded in this process —
    # get_all_real_tables() returns [] for a not-yet-restored file, and
    # reporting that as "0 tables" would be its own false statement.
    from app import state as app_state
    n_tables = len(get_all_real_tables(file_id)) if file_id in app_state.FILE_KIND else None
    indexed = has_indexed_content(file_id)
    if not n_tables and not indexed:
        return response_text  # nothing to contradict it with

    evidence = []
    if n_tables:
        evidence.append(f"{n_tables} extractable table(s)")
    if indexed:
        evidence.append("its text is indexed for search")

    facts = (f"\n\n---\n\n**Correction (automatic check against the file "
             f"registry and vector store):** "
             f"`{record['original_filename']}` is present on disk and was "
             f"ingested successfully — {', and '.join(evidence)}. "
             f"It is not empty or corrupted, and re-uploading it is not "
             f"necessary.")

    hit = semantic_fallback(file_id, prompt)
    if hit:
        facts += (f"\n\nClosest matching content found in this file by "
                  f"semantic search:\n\n{hit}")
    return response_text + facts


# A quoted name before the word "column" is tried first: in
# 'the "Unit Rate (INR)" column values' the unquoted branch would otherwise
# latch onto "values" from the trailing "column values".
_COLUMN_ASK_RE = re.compile(
    r"[\"'`]([A-Za-z][\w ()/&.%-]{2,40})[\"'`]\s+column"
    r"|column\s+(?:name[ds]?\s+)?[\"'`]?([A-Za-z][\w ()/&.%-]{2,40}?)[\"'`]?\s*"
    r"(?:$|[,.?!]|\s+(?:in|of|from|data|values?|list|column))",
    re.IGNORECASE)

# Words that are part of the request, never the name of a column.
_NOT_A_COLUMN = {"values", "value", "data", "name", "names", "header",
                 "headers", "unique", "all", "the", "this", "that", "list"}


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _catch_invented_column(response_text: str, prompt: str,
                           file_id: Optional[str]) -> str:
    """Stop the model presenting values for a column that does not exist.

    Confirmed production failure (08:07:47): asked for "unique data of column
    EvaluationSchedules", the model answered "Schedule 1 … Schedule 8" and
    called them "the only unique values found". None of those eight strings
    occur anywhere in the document. The real column holds Foot Valve, Drain
    Valve, … and Docling dropped it during extraction, so the agent had
    nothing — and filled the gap with invention.

    A fabricated column is the most damaging failure this system can produce:
    a quotation built on it looks perfectly ordinary. So the requested column
    is checked against the schema actually extracted from the file, and the
    three real situations are reported honestly — including the one that
    applies here, where the column name survives in a header row whose data
    rows were never extracted.
    """
    if not file_id or not response_text or not prompt:
        return response_text
    m = _COLUMN_ASK_RE.search(prompt)
    if not m:
        return response_text
    asked = (m.group(1) or m.group(2) or "").strip()
    if not asked or asked.lower() in _NOT_A_COLUMN:
        return response_text

    from app import state as app_state
    if file_id not in app_state.FILE_KIND:
        return response_text  # not loaded here; no schema to check against

    from app.tables.helpers import get_all_real_tables
    target = _norm_col(asked)
    all_cols = set()
    for df in get_all_real_tables(file_id):
        all_cols.update(str(c) for c in df.columns)
    if target in {_norm_col(c) for c in all_cols}:
        return response_text  # the column is real — nothing to correct

    # Listing every column across a 68-table document is noise. Drop the
    # col_N placeholders (they are "we found no name", not a name a user
    # could ask for) and keep the list short enough to actually read.
    named = sorted(c for c in all_cols if not re.fullmatch(r"col_\d+", c.strip()))
    shown = ", ".join(f"`{c}`" for c in named[:20])
    if len(named) > 20:
        shown += f", … ({len(named) - 20} more)"
    real_cols = shown or "none with recognisable names"

    # A name can survive in a header-only table (header and body split across
    # a page break, body missing the column). That is a different, and much
    # more useful, answer than "no such column".
    header_only = _header_only_column_names(file_id)
    if target in {_norm_col(c) for c in header_only}:
        return response_text + (
            f"\n\n---\n\n**Correction (checked against the data actually "
            f"extracted from this file):** a column named `{asked}` appears in "
            f"a table header in this document, but **no data rows were "
            f"extracted for it** — the table's body was read without that "
            f"column. **Any values listed above for it are not from your "
            f"document — ignore them.** The columns that do carry data are: "
            f"{real_cols}.")

    return response_text + (
        f"\n\n---\n\n**Correction (checked against the data actually extracted "
        f"from this file):** there is no column named `{asked}` in this "
        f"document. **Any values listed above for it are not from your "
        f"document — ignore them.** The real columns are: {real_cols}.")


def _header_only_column_names(file_id: str) -> set:
    """Column names that exist in the document as a header row whose table
    carries no extracted data rows — invisible to get_all_real_tables, which
    filters them out, but the only record that such a column ever existed."""
    from app import state as app_state
    from app.tables.helpers import _is_header_only_table

    doc = app_state.DOCLING_DOCS.get(file_id)
    if doc is None:
        return set()
    names = set()
    for t in doc.tables:
        try:
            raw = t.export_to_dataframe(doc)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            continue  # a malformed table must not break the honesty check
        if _is_header_only_table(raw):
            names.update(str(c).strip() for c in raw.columns)
    return names


_INVENTORY_ASK_RE = re.compile(
    r"\bhow many (?:files?|documents?|uploads?)\b"
    r"|\bwhat (?:files?|documents?)\b"
    r"|\bwhich (?:files?|documents?)\b"
    r"|\b(?:files?|documents?) (?:do i|have i|did i)\b"
    r"|\b(?:what|show)\b[^.?!]{0,25}\b(?:did i upload|my documents?|my files?)\b"
    r"|\blist (?:my |the |all )?(?:files?|documents?)\b",
    re.IGNORECASE)


def _catch_wrong_file_inventory(response_text: str, prompt: str) -> str:
    """Answer "how many files did I upload" from the registry, always.

    Confirmed production failure (08:45:02): five documents had just been
    uploaded into the chat, and the answer was "You have uploaded one file",
    naming only the active one — then insisted "I did not give wrong data".
    The logged intent was `general`: the model never called
    list_uploaded_files at all. It read the active-file note out of its
    context and reported that as the whole inventory.

    A question with an exact, cheap, authoritative answer should never depend
    on the model choosing to look it up, so the real list is appended from
    the session registry regardless of what the model said.
    """
    if not response_text or not prompt or not _INVENTORY_ASK_RE.search(prompt):
        return response_text

    import os
    from app import state as app_state
    from app.persistence import get_all_files

    session = app_state.get_active_session_id()
    if not session:
        return response_text
    records = [r for r in get_all_files(session) if os.path.exists(r["path"])]
    if not records:
        return response_text  # nothing authoritative to correct it with

    listed = "\n".join(
        f"{i}. `{_display_name(r['file_id'], r['original_filename'])}` "
        f"({r['kind']}, id={r['file_id']})"
        for i, r in enumerate(records, 1))
    return response_text + (
        f"\n\n---\n\n**From the file registry for this chat — "
        f"{len(records)} document(s):**\n{listed}")


def _name_keys(original_filename: str) -> set:
    """Short handles a user would actually type for an uploaded file.

    Uploads are stored as "{file_id}_{original}", e.g.
    "data-file-4_3738_data-file-4.pdf", so the useful handle has to be dug
    out of the middle. Digit-only segments are the generated collision
    suffix, never something the user types."""
    stem = os.path.splitext(original_filename)[0]
    keys = {stem}
    keys.update(seg for seg in stem.split("_") if len(seg) >= 4 and not seg.isdigit())
    return {re.sub(r"[^a-z0-9]", "", k.lower()) for k in keys if k}


def _resolve_named_file(prompt: str, current_file_id: Optional[str]) -> Optional[str]:
    """The file the user NAMED in their message, when it isn't the one the
    UI has selected.

    Confirmed production failure (20:18:34): the prompt said "on data-file-4
    page number 24" while the UI sent file_id=data-file-5_1455, a
    spreadsheet. The page lookup ran against the wrong file, found nothing,
    and the model told the user their document was "empty, corrupted, or not
    a valid spreadsheet" — while data-file-4_3738 was registered, present on
    disk, and holds 4 extractable tables.

    resolve_file_scope() in app/tables/helpers.py cannot cover this: it
    requires the FULL stored filename to appear in the prompt and reads
    in-memory state, which is empty after a restart. This matches short
    handles ("data-file-4", "data file 4") against the durable registry.

    Returns None when the active file already matches what the user named,
    so a correct UI selection is never churned for an equivalent re-upload.
    """
    from app.persistence import get_all_files

    normalized = re.sub(r"[^a-z0-9]", "", prompt.lower())
    records = [r for r in get_all_files() if os.path.exists(r["path"])]

    current = next((r for r in records if r["file_id"] == current_file_id), None)
    if current and any(k in normalized for k in _name_keys(current["original_filename"])):
        return None  # the selected file is the one the user is talking about

    # Registry is ordered oldest-first; the newest upload of a named document
    # is the one the user means.
    for record in reversed(records):
        if any(k in normalized for k in _name_keys(record["original_filename"])):
            return record["file_id"]
    return None


# A turn that could not read the document it was pointed at says so in one of
# these ways. Such a turn must not become what the NEXT message's "this table"
# means: one failed export against the wrong spreadsheet overwrote the focus
# with that spreadsheet, and every message after it inherited the mistake.
_TURN_FAILED_RE = re.compile(
    r"no extractable table|could not extract|no file was created|"
    r"have no extractable|not a recognized file_id|appears (?:to be )?empty|"
    r"is empty or corrupted|export failed", re.IGNORECASE)

# The app serves downloads from its own /files/download/ route. An absolute
# http(s) link to a file is therefore always invented — the model produced
# "[Click here to download f1-14.xlsx](https://example.com/f1-14.xlsx)" for a
# file that had never been written, and the user chased a 404 for three turns.
_INVENTED_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*https?://[^)]*\)")


def _strip_invented_links(text: str) -> str:
    return _INVENTED_LINK_RE.sub(r"\1", text or "")


def _turn_defines_focus(named: bool, refocused: bool, existing_focus: Optional[str],
                        response_text: str) -> bool:
    """Whether this turn's document becomes what the next "this table" means.

    A turn that could not read the document it was pointed at does not get to
    define it. A failed export against the stale spreadsheet recorded that
    spreadsheet as the conversation's subject, and the very next message —
    which did name pages only the real document has — found the focus already
    agreeing with the wrong selection, so nothing was left to correct it
    (log ids 34abf85d → 3fd484e8, and four fabricated answers after that).

    A document the user named outright always counts, failure or not: they
    said which one they meant.
    """
    if named or refocused or not existing_focus:
        return True
    return not _TURN_FAILED_RE.search(response_text or "")


def _pages_answerable(file_id: Optional[str], pages: List[int]) -> int:
    """How many of `pages` this document actually has a table on.

    Evidence, not wording — the same question `_deterministic_page_export`
    asks, borrowed here so the document can be chosen before the export runs
    instead of after it has already failed.
    """
    if not file_id:
        return 0
    if _restore_file_if_needed(file_id):
        return 0
    try:
        from app.tables.helpers import get_all_real_tables
        available = {t.attrs.get("page") for t in get_all_real_tables(file_id)}
    except Exception:
        return 0
    return len(available & set(pages))


def _refocus_on_conversation(prompt: str, selected: Optional[str],
                             focus: Optional[str],
                             selection_changed: bool = False) -> Optional[str]:
    """The document this message is about, when it names none itself.

    Confirmed production failure (09:00–09:10): data-file-5.xlsx was uploaded
    and auto-selected by the UI, the conversation then moved to data-file-2.pdf
    by name, and the next two messages — "export THAT page number started
    table", "THIS table data available on page number 6 to 10" — carried the
    stale chip. Both ran against the spreadsheet and reported that the pages
    the user was looking at did not exist.

    A chip the user has just clicked is a deliberate act and always wins. A
    chip that has been sitting there unchanged, while the conversation moved
    to another document, is not evidence of anything — comparing this turn's
    selection with the previous turn's is what tells the two apart.

    Beyond that the switch is decided on evidence: when the message names
    pages, the document that actually has them keeps the turn, so a live
    selection is never churned for a document that cannot answer either.

    Returns the file_id to use instead, or None to keep the selection.
    """
    if not focus or focus == selected:
        return None
    if selection_changed:
        return None            # the user picked this file for this message
    pages = _extract_requested_pages(prompt)
    if not pages:
        # "we need all this table data" — no page to test, but the selection
        # is stale and the conversation is unambiguous. Read as belonging to
        # the selected file, this exported nothing and the model invented a
        # file rather than come back empty (log ids 34abf85d, 3fd484e8).
        return None if _restore_file_if_needed(focus) else focus
    if _pages_answerable(selected, pages):
        return None
    return focus if _pages_answerable(focus, pages) else None


TOOLS = [
    search_documents,
    query_table_data,
    list_uploaded_files,
    export_data,
    modify_export,
    get_file_overview,
    get_page_content,
    get_table_of_contents,
    generate_quotation,
    analyze_past_contracts,
    web_search,
    get_weather,
    get_news,
    calculate,
]

SYSTEM_PROMPT = """You are ONEX AI — a private procurement intelligence
assistant for industrial and government quotation management.

You help with:
- Quotation generation from RFQ documents
- Contract and bid analysis
- Product pricing and discount calculations
- Document search and data extraction
- Exporting data to Excel, CSV, Word, PDF

You have these tools available:
- search_documents: for a TOPIC or CONCEPT question where you don't know
  which page it's on — terms, specs, descriptions written in documents
- get_page_content: when the user names a specific page number — this
  returns the EXACT page, unlike search_documents which only guesses
- get_table_of_contents: for "table of contents", "what sections/chapters
  exist", "what page is X on", "indexing" — returns the real extracted
  heading list, not a guess from search results
- query_table_data: for questions about numbers, rows, columns, data
  in spreadsheets — filtering, counting, aggregating
- list_uploaded_files: when user asks what files they have uploaded
- export_data: when user wants to save/download/export data to a file
- modify_export: when user wants to CHANGE a file that already exists —
  add or remove the document header block above the table, rename columns,
  drop columns. Never re-export from scratch to satisfy one of these.
- get_file_overview: when user wants a summary of a file
- generate_quotation: when asked to create a quotation from RFQ
- analyze_past_contracts: for bid history, win/loss patterns, improvements
- get_weather: current weather for a location — you have no live weather
  data of your own, ALWAYS use this tool for weather questions, never
  guess or say you don't have access to a weather service
- web_search: current events, news, or any fact that may have changed
  since your training data — use this rather than answering from memory
  when the question needs up-to-date information
- get_news: latest headlines about a topic or location
- calculate: for arithmetic you want verified rather than computed mentally

Rules:
1. ALWAYS use a tool to answer questions about uploaded files.
2. For general knowledge questions with a STATIC answer (math, geography,
   definitions), answer directly WITHOUT using any tool. For questions
   that need LIVE or current data (weather, news, today's anything),
   ALWAYS use the matching tool (get_weather, web_search, get_news) —
   never say you don't have access to that information, you do, use it.
3. For tabular data questions (Excel/CSV), prefer query_table_data
   over search_documents.
4. A page number in the question ALWAYS means get_page_content, never
   search_documents — search_documents cannot guarantee it returns that
   exact page's content.
5. A question about structure — table of contents, sections, chapters,
   "what page is X on", "indexing" — ALWAYS means get_table_of_contents,
   never search_documents, even if a previous answer already tried to
   guess this from search results.
6. If the user's message contains BOTH a filename (like f1-03.xlsx) AND
   an export verb (create, make, export, save, generate), call export_data
   IMMEDIATELY — do NOT answer the page content first. The message might
   ALSO mention page numbers (e.g. "create excel from page 6 and 7") but
   the presence of a filename + export verb means the user wants a FILE,
   not a page content answer. Pass the FULL user message as the `question`
   parameter to export_data so it can extract what data to export.
   An offer to create the file later ("let me know if you'd like...") is
   NOT acceptable when the user already told you to create it.
7. If the user asks to add header details, company/project/spec information,
   or a title block to a file that already exists, call modify_export with
   that filename. That information is recovered from the source document —
   do NOT ask the user to supply it, and do NOT answer that the request is
   ambiguous. If no filename is given, the most recent export is meant.
8. NEVER fabricate data — if a tool returns empty, say so honestly.
9. Be concise. Don't repeat tool results verbatim — summarize them.
10. If a tool returns an error, tell the user clearly what failed and why.
11. NEVER claim a file was "created", "saved", or is "ready for download"
    unless export_data or generate_quotation actually returned a verified
    success message THIS turn. If repeated attempts keep failing or
    returning wrong data, say exactly that — which columns/rows are wrong
    and why — do NOT invent a "manually created" file or make up data
    that looks plausible. A fabricated success is worse than an honest
    failure.
12. NEVER write a download URL, and never state a file's size or row count
    from memory. The user downloads files through this app's own file list;
    a link you compose goes nowhere. Confirmed: a made-up
    "https://example.com/f1-14.xlsx" for a file that was never written sent
    the user chasing a 404 through three more messages. Report only what the
    tool returned THIS turn, in one or two sentences.
13. If a request repeats one you already answered, do NOT restate the earlier
    answer more confidently. Repetition means the last answer was wrong —
    call the tool again and report what it actually returns.
"""


# Injected scaffolding (active-file notes, priority export instructions) is
# not part of the user-visible conversation.
_VISIBLE_ROLES = {"human": "user", "ai": "assistant"}


def thread_messages(session_id: str) -> List[dict]:
    """The user-visible turns of one conversation, oldest first.

    Every turn is checkpointed, but nothing in the agent could read a
    checkpoint back — asked "remind last 2 chats" the model answered that it
    has no memory of previous conversations, and once the export backstop got
    hold of that turn the reply became a sandbox error about DataFrames. The
    history endpoints in api/routes/chat.py already knew how to replay a
    thread; this is that reader, moved somewhere both can use it.
    """
    try:
        snapshot = agent.get_state({"configurable": {"thread_id": session_id}})
    except Exception as e:
        raise RuntimeError(f"Could not read session: {e}") from e

    out = []
    for msg in snapshot.values.get("messages", []):
        role = _VISIBLE_ROLES.get(getattr(msg, "type", None))
        content = getattr(msg, "content", "")
        # An assistant message with empty content is a tool-call step, not
        # something the user ever saw.
        if role and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


def recent_exchanges(session_id: str, count: int) -> List[dict]:
    """The last `count` user messages of a thread, each with the answer it
    got. "chats" in "remind last 2 chats" means exchanges, not messages."""
    msgs = thread_messages(session_id)
    pairs = []
    for i, msg in enumerate(msgs):
        if msg["role"] != "user":
            continue
        answer = next((m["content"] for m in msgs[i + 1:] if m["role"] == "assistant"), "")
        pairs.append({"asked": msg["content"], "answered": answer})
    return pairs[-count:] if count > 0 else pairs


# "remind last 2 chats", "what did I ask before" — how many exchanges back.
_RECALL_COUNT_RE = re.compile(
    r"\b(?:last|previous|past|recent)\s+(\d+|one|two|three|four|five)\b", re.IGNORECASE)
_RECALL_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_DEFAULT_RECALL = 3


def _recall_count(prompt: str) -> int:
    m = _RECALL_COUNT_RE.search(prompt or "")
    if not m:
        return _DEFAULT_RECALL
    token = m.group(1).lower()
    n = _RECALL_WORDS.get(token, int(token) if token.isdigit() else _DEFAULT_RECALL)
    return max(1, min(n, 20))


def _is_recall_request(prompt: str) -> bool:
    """Whether the user is asking about this conversation rather than about a
    document. The cue pattern is the one `app/query/planner.py` has always
    used for its `chat_history` intent — unreachable since the LangGraph
    migration, but correct, and there is no reason for a second copy."""
    from app.query.planner import CONVERSATION_RECALL_RE
    return bool(CONVERSATION_RECALL_RE.search((prompt or "").lower()))


# SQLite checkpointer — state survives server restarts
# This replaces the fragile in-memory state.py chat history
CHECKPOINT_DB_PATH = "./agent_checkpoints.db"
_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)

# The model's ceiling is 262,144 tokens. The first version of this budgeted
# 800k chars on a 4-chars-per-token assumption and STILL overflowed at 258,049
# tokens — this content is tables, numbers and Devanagari, which tokenize
# closer to 2.5 chars/token. Measured against that ratio, 240k chars is about
# 96k tokens, leaving a wide margin for the system prompt, the tool schemas
# and the reply. A message cap backs it up, because a pathological single
# message should not be able to consume the whole budget.
_MAX_HISTORY_CHARS = 240_000
_MAX_HISTORY_MESSAGES = 40
_MAX_TOOL_CHARS = 6_000


def _trim_history(state: dict) -> dict:
    """Bound what is sent to the model, without touching what is stored.

    Confirmed production failure: one chat reached 184 turns / 114 MB of
    checkpoints, and every request then died with "number of input tokens
    (262325) has exceeded max_seq_len (262144)" — the whole conversation was
    replayed each turn, including entire exported tables. The chat became
    permanently unusable, which is the worst possible failure for a feature
    whose selling point is that your history is never lost.

    So the transcript stays complete in the checkpointer (history replay and
    session resume still show everything) and only the model's *input* is
    trimmed: the newest messages that fit, plus any system scaffolding, which
    carries the active file and export instructions for this turn.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"llm_input_messages": messages}

    # Only THIS turn's scaffolding. Every turn appends its own active-file and
    # export notes to the thread, so after 184 turns the history held hundreds
    # of stale "the active file is …" messages — all contradicting each other
    # and all being resent. Keeping the last few keeps the current turn's
    # instructions while discarding the accumulated noise.
    system = [m for m in messages if getattr(m, "type", None) == "system"][-3:]
    rest = [m for m in messages if getattr(m, "type", None) != "system"]

    kept, total = [], sum(len(str(getattr(m, "content", ""))) for m in system)
    for msg in reversed(rest):
        # A single huge tool result (a 900-row table) can blow the budget on
        # its own; truncate it rather than dropping the turn it belongs to.
        content = str(getattr(msg, "content", ""))
        if len(content) > _MAX_TOOL_CHARS:
            msg = msg.model_copy(update={
                "content": content[:_MAX_TOOL_CHARS]
                + f"\n… [truncated {len(content) - _MAX_TOOL_CHARS} characters]"})
            content = str(msg.content)
        if kept and (total + len(content) > _MAX_HISTORY_CHARS
                     or len(kept) >= _MAX_HISTORY_MESSAGES):
            break
        kept.append(msg)
        total += len(content)

    trimmed = list(reversed(kept))
    # A tool message whose originating AI tool-call was trimmed away is
    # rejected by the API, so drop any leading orphaned tool results.
    while trimmed and getattr(trimmed[0], "type", None) == "tool":
        trimmed.pop(0)
    return {"llm_input_messages": system + trimmed}


agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    pre_model_hook=_trim_history,
)


def run_agent(prompt: str, session_id: str = "default",
               file_id: str = None) -> dict:
    """Single entry point replacing chat() in dispatch.py.

    Returns same contract as old chat():
    {"response": str, "intent": str, "file_id": str, "table": dict|None}
    """
    from app import state as app_state
    import time

    # "general_chat" is a frontend placeholder for "no document selected",
    # not a file. Passing it through made the Step 4 guard reject every
    # message in a chat with no documents — "'general_chat' is not a
    # recognized file_id… upload the file first" — for users who only
    # wanted to talk.
    if file_id in ("general_chat", "", "null", "undefined"):
        file_id = None

    # Tools resolve "all my documents" against the chat's own documents, and
    # they only receive the arguments the model passes them — so the session
    # is published here, the same way the active file already is.
    app_state.set_active_session_id(session_id)
    app_state.set_current_user_prompt(prompt)

    # A file named explicitly in the message wins over the UI's selection —
    # asking about data-file-4 while data-file-5 happens to be selected must
    # answer from data-file-4, not report data-file-4 as empty.
    # Did the user just click a different chip, or has this one been sitting
    # there while the conversation moved on? Recorded before anything can
    # redirect it, so what is compared is what the UI actually sent.
    selection_changed = file_id != app_state.get_last_selection(session_id)
    app_state.set_last_selection(session_id, file_id)

    redirected_from = None
    refocused = False
    named_file_id = _resolve_named_file(prompt, file_id)
    if named_file_id and named_file_id != file_id:
        redirected_from, file_id = file_id, named_file_id
    elif not named_file_id:
        # No document named here — "this table", "that page". The UI keeps the
        # newest upload selected long after the conversation has moved on, so
        # fall back to the document this chat is actually about.
        focused = _refocus_on_conversation(
            prompt, file_id, app_state.get_session_focus(session_id),
            selection_changed)
        if focused and focused != file_id:
            redirected_from, file_id = file_id, focused
            refocused = True

    # Set active file context so tools know what file to default to.
    # ACTIVE_FILE_ID is module-level and outlives the request, so a chat with
    # no document would otherwise inherit whichever file the *previous*
    # request happened to leave there — one conversation silently answering
    # from another's document.
    if not file_id:
        app_state.set_active_file_id(None)
    if file_id:
        app_state.set_active_file_id(file_id)
        restore_error = _restore_file_if_needed(file_id)
        if restore_error:
            # Hard guard: don't let the agent run against a file_id that
            # can't actually be loaded — every tool would just report an
            # empty/missing file and the model would guess at why, giving
            # the user a misleading "corrupted" style answer instead of the
            # real reason (unregistered id, or the source upload is gone
            # from disk and needs re-uploading).
            return {
                "response": restore_error,
                "intent": "error",
                "file_id": file_id,
                "table": None,
                "tool_used": None,
                "available_files": [
                    {"file_id": fid,
                     "name": app_state.FILE_ORIGINAL_NAME.get(fid, fid),
                     "kind": app_state.FILE_KIND.get(fid, "unknown")}
                    for fid in app_state.FILE_ORDER
                ],
            }

    config = {"configurable": {"thread_id": session_id}}

    # How many messages exist in this thread BEFORE this turn, so we can
    # isolate what THIS turn actually did — the checkpointer returns the
    # full accumulated history, and scanning all of it for "the tool used"
    # locks onto turn 1's tool forever (confirmed: turn 2 of a real session
    # correctly called search_documents but was reported as the overview
    # tool from turn 1, because the old code scanned from the start).
    try:
        prior_state = agent.get_state(config)
        n_prior_messages = len(prior_state.values.get("messages", []))
    except Exception:
        n_prior_messages = 0

    messages = []
    if file_id:
        # The LLM otherwise has no way to know a file is active — it only
        # sees the raw prompt text. Without this, a question like "what's
        # on page 6" with file_id passed by the caller still gets answered
        # with "please provide a file", because nothing in the LLM's input
        # mentions one exists.
        fname = app_state.FILE_ORIGINAL_NAME.get(file_id, file_id)
        note = (f"The active file for this message is file_id={file_id!r} "
                f"(\"{fname}\"). Use this file_id when calling tools.")
        if redirected_from and refocused:
            note += (f" The UI still had {redirected_from!r} selected — the file "
                     f"most recently uploaded — but this message refers back to "
                     f"the document the conversation is about, and that one has "
                     f"the pages asked for while the selected file does not. It "
                     f"has already been resolved and loaded for you. Answer from "
                     f"it, and say which document you used.")
        elif redirected_from:
            note += (f" The UI had {redirected_from!r} selected, but the user's "
                     f"message names this file instead, so it has already been "
                     f"resolved and loaded for you — do not tell the user the "
                     f"file is missing, empty, or needs re-uploading.")
        messages.append({"role": "system", "content": note})

    # The active-file note above is the only thing the model knows about
    # files, so asked "how many did I upload" it answers "one" — naming the
    # active file. Give it the chat's real inventory instead of letting it
    # infer one from context.
    if _INVENTORY_ASK_RE.search(prompt):
        from app.persistence import get_all_files
        owned = [r for r in get_all_files(session_id) if os.path.exists(r["path"])]
        if owned:
            listing = "; ".join(
                f"{_display_name(r['file_id'], r['original_filename'])} "
                f"(id={r['file_id']})" for r in owned)
            messages.append({"role": "system", "content": (
                f"This chat has {len(owned)} uploaded document(s): {listing}. "
                f"Use exactly this list and this count when answering — do not "
                f"infer the number from the active file.")})
    # A question about the conversation is answered from the conversation.
    # The thread is checkpointed and the model can see the recent part of it,
    # but asked to recall it the model has answered that it cannot — so the
    # turns are stated outright rather than left to be noticed.
    if _is_recall_request(prompt):
        try:
            # This turn is not in the checkpoint yet, so everything the
            # reader returns is genuinely earlier than the current message.
            past = recent_exchanges(session_id, _recall_count(prompt))
        except RuntimeError:
            past = []
        if past:
            lines = "\n".join(
                f"{i}. The user asked: {p['asked'][:400]!r}\n"
                f"   You answered: {p['answered'][:600]!r}"
                for i, p in enumerate(past, 1))
            messages.append({"role": "system", "content": (
                f"This message asks about this conversation, not about a "
                f"document. Here are the last {len(past)} exchange(s) in this "
                f"chat, oldest first:\n{lines}\n"
                f"Answer from these. Do not search a document, do not export "
                f"anything, and never tell the user you cannot remember "
                f"previous messages — they are right here.")})
        else:
            messages.append({"role": "system", "content": (
                "This message asks about earlier messages, but this "
                "conversation has no earlier turns. Say exactly that.")})

    # An unambiguous export request (an output filename + an export verb)
    # was still being answered with a long description of the page followed
    # by an offer to export. The backstop below recovers the file, but the
    # user still waits through the essay, so the intent is stated up front
    # as its own instruction rather than relying on a numbered rule buried
    # in the system prompt.
    requested_name = _extract_requested_filename(prompt)
    # The name carries to the next message: "export into f1-14.xlsx" followed
    # by "we need all this table data" is one request for one file.
    app_state.set_last_requested_export(session_id, requested_name)
    # Likewise for the header band: said once, it holds until said otherwise.
    from app.export.modify import parse_instruction
    app_state.set_band_preference(session_id,
                                  parse_instruction(prompt).get("context"))
    fallback_filename = requested_name or app_state.get_last_requested_export(session_id)
    if requested_name and _EXPORT_DIRECTIVE_RE.search(prompt):
        ext = requested_name.rsplit(".", 1)[-1].lower()
        # A request that draws on two documents is where the model is most
        # tempted to work file-by-file — read one, describe it, ask whether
        # to continue. export_data resolves and combines both sources itself,
        # so it needs exactly one call with the whole message.
        multi = _resolve_source_specs(prompt, file_id)
        messages.append({
            "role": "system",
            "content": (
                f"PRIORITY: this message is a file-creation request for "
                f"{requested_name!r}. Call export_data FIRST, with "
                f"filename={requested_name!r} and "
                f"format={_EXT_TO_FORMAT.get(ext, 'excel')!r}, passing the "
                f"user's full message as `question`. Do not call "
                f"get_page_content first, do not describe the page contents, "
                f"and do not offer to create the file — create it, then "
                f"reply in at most two sentences reporting the result."
                + (f" This request draws on {len(multi)} uploaded documents; "
                   f"export_data resolves and combines all of them from the "
                   f"message itself, so make ONE call with the full message "
                   f"and do not export each document separately."
                   if len(multi) > 1 else "")
            )
        })

    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    try:
        result = agent.invoke({"messages": messages}, config=config)
        response_text = result["messages"][-1].content
        turn_messages = result["messages"][n_prior_messages:]

        # Determine which tool THIS turn used — only scan messages added
        # after n_prior_messages, not the whole persisted thread history.
        tool_used = None
        for msg in turn_messages:
            if hasattr(msg, "name") and msg.name in [t.name for t in TOOLS]:
                tool_used = msg.name
                break

        # Two confirmed failure modes, mirror images of each other: a file
        # claimed but never written ("let me bypass the tool and create the
        # file manually", with an invented table), and a file demanded but
        # never created, the answer ending in an offer to export instead.
        # Rules 6 and 11 of the system prompt ask the model not to do either;
        # it has done both under real pressure, so these are deterministic
        # backstops rather than requests. See run_export_backstops for why
        # they must not both fire.
        active_file_id = file_id or app_state.get_active_file_id()
        response_text, forced_tool = run_export_backstops(
            prompt, response_text, turn_messages, active_file_id,
            fallback_filename)
        if forced_tool:
            tool_used = forced_tool

        # Never let a "your file is empty / corrupted / re-upload it" answer
        # reach the user for a file that is provably fine.
        response_text = _catch_false_missing_file_claim(
            response_text, prompt, active_file_id)
        response_text = _ensure_source_stated(response_text, turn_messages)
        response_text = _catch_invented_column(
            response_text, prompt, active_file_id)
        response_text = _catch_wrong_file_inventory(response_text, prompt)
        # Downloads are served from this app's own /files/download/ route, so
        # an absolute http(s) link to a file is always invented.
        response_text = _strip_invented_links(response_text)

        intent = _tool_to_intent(tool_used)
        export_filename = _resolve_export_filename(turn_messages, response_text)

        # What the NEXT message's "this table" will mean. Recorded per session,
        # so two chats open side by side never inherit each other's document.
        #
        if _turn_defines_focus(bool(named_file_id), refocused,
                               app_state.get_session_focus(session_id),
                               response_text):
            app_state.set_session_focus(session_id, file_id)

        # Log the interaction
        from app.logging_utils import log_interaction
        from app.models import QueryPlan
        # The log was recording "filename": null on every successful export
        # because only the intent was passed here — which made the primary
        # debugging surface for this project useless for the one field that
        # matters most when an export goes wrong.
        # sink is a Literal of format names ("excel"), not file extensions
        # ("xlsx") — mapped through the same table the export path uses.
        plan = QueryPlan(
            intent=intent,
            sink=(_EXT_TO_FORMAT.get(export_filename.rsplit(".", 1)[-1].lower())
                  if export_filename else None),
            filename=export_filename,
        )
        log_interaction(session_id, file_id, prompt, plan,
                        response_text, success=True,
                        latency=time.time() - t0)
        return {
            "response": response_text,
            "intent": intent,
            "file_id": file_id or app_state.get_active_file_id(),
            "table": None,  # table data is now embedded in response_text as markdown
            "tool_used": tool_used,
            # The exact file the UI should offer for download. Left None
            # unless it really exists in OUTPUT_DIR, so the download button
            # can never point at something that 404s.
            "filename": export_filename,
            "available_files": [
                {"file_id": fid,
                 "name": app_state.FILE_ORIGINAL_NAME.get(fid, fid),
                 "kind": app_state.FILE_KIND.get(fid, "unknown")}
                for fid in app_state.FILE_ORDER
            ],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "response": f"Something went wrong: {e}",
            "intent": "error",
            "file_id": file_id,
            "table": None,
            "available_files": [],
        }


def _tool_to_intent(tool_name: str) -> str:
    mapping = {
        "search_documents": "qa",
        "query_table_data": "data_query",
        "list_uploaded_files": "list_files",
        "export_data": "export",
        "modify_export": "export",
        "get_file_overview": "overview",
        "get_page_content": "page_lookup",
        "get_table_of_contents": "table_of_contents",
        "generate_quotation": "generate_quotation",
        "analyze_past_contracts": "contract_analysis",
        None: "general",
    }
    return mapping.get(tool_name, "general")
