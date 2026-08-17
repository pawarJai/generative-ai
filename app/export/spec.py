"""Deterministic parsing of export requests, matched against a table's REAL
columns — not free-form English column-list phrasing handed to an LLM.

Confirmed production failure (session 477b296a, data-file-3): "export two
column data ... = Item Title, Item Quantity" wrote all 7 columns to the
output file. export_data's fallback path built the DataFrame via
run_code_on_files (an LLM sandbox) and handed it straight to the exporter
with plan.columns never set — so the deterministic column filter that
already exists in app.export.schema_map.prep_tables was never reached for
that path. This module gives export_data (and app.export.modify, for
already-written files) a way to resolve column selection, row filtering,
row-count limits and simple computed columns directly from the user's own
words against the real column list, the same reasoning
_extract_requested_sheet in app.graph.agent already uses for sheet names:
match against what actually exists, don't parse English grammar.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "twenty": 20, "fifty": 50, "hundred": 100}


def _count(word: str) -> Optional[int]:
    word = word.strip().lower()
    if word.isdigit():
        return int(word)
    return _WORD_NUMBERS.get(word)


def _norm_key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def _fuzzy_contains(key: str, normalized: str, max_dist: int = 1) -> bool:
    """Whether ``key`` appears in ``normalized``, tolerating a single typo'd
    character — same reasoning as app.graph.agent's file-handle fuzzy
    matcher, generalized to column names. Confirmed real: a request typed
    'matrial code' for the real column 'Material code' and the export
    silently dropped that column instead of including it, because the old
    plain substring check never matches a typo. Skipped for short keys,
    where a one-character edit is indistinguishable from unrelated text."""
    if key in normalized:
        return True
    if len(key) < 5:
        return False
    for start in range(0, max(1, len(normalized) - len(key) + 1 + max_dist)):
        for wlen in (len(key) - 1, len(key), len(key) + 1):
            if wlen <= 0 or start + wlen > len(normalized):
                continue
            if _edit_distance(normalized[start:start + wlen], key) <= max_dist:
                return True
    return False


def _clean(s: str) -> str:
    return re.sub(r"^[\s\"'“”]+|[\s\"'“”]+$", "", s).strip()


# A rename clause naming a REAL column ("rename the column name = Description
# in to Group") must never reach extract_requested_columns below — that
# function matches any real column name found anywhere in the prompt, with
# no requirement that the mention actually be a selection request. Confirmed
# production failure (session 7e43a023, 2026-08-14): the rename clause's own
# old-name "Description" is a real column, so extract_requested_columns read
# it as "export only the Description column", silently dropping the other 7
# real columns from an 8-column export — the same bug _mask_span already
# guards against for filter clauses ("where Item Title = X"), just not yet
# for rename ones. Not required to cleanly capture old/new names here (that
# is app.export.modify's job) — only to cover the span so it gets masked.
_RENAME_CLAUSE_RE = re.compile(
    r"\brename\b\s+(?:the\s+)?(?:column\s+)?(?:name\s*[:=]\s*)?"
    r"[\"'“]?(.+?)[\"'”]?\s+\b(?:in\s*to|into|to|as)\b\s+[\"'“]?([^\"'”,.]+)",
    re.IGNORECASE)


# --- explicit column selection --------------------------------------------

def extract_requested_columns(prompt: str, available_columns: List[str]) -> List[str]:
    """Real column names the prompt names, in the TABLE's own order.

    Only returns a non-empty result when the match is a strict subset of the
    real columns — if every column name happens to appear (or none do) there
    is no actual selection being made, and treating either as "export only
    these" would silently narrow or no-op a request that named no columns at
    all.
    """
    if not prompt or not available_columns:
        return []
    normalized = re.sub(r"[^a-z0-9]", "", prompt.lower())
    hits = set()
    # Longest names first so a short column name can't shadow-match inside a
    # longer one that also appears (same reasoning as sheet-name matching).
    for col in sorted(available_columns, key=lambda c: len(str(c)), reverse=True):
        key = _norm_key(col)
        if key and len(key) >= 3 and _fuzzy_contains(key, normalized):
            hits.add(col)
    if not hits or len(hits) == len(available_columns):
        return []
    return [c for c in available_columns if c in hits]


# --- reporting: requested columns that aren't real data columns at all -----

_REQUESTED_LIST_START_RE = re.compile(
    # "columsn" is a confirmed real, repeated typo (transposed letters) for
    # "columns" — not a one-off, so it's matched literally rather than left
    # to rely on fuzzy tolerance, which is sized for column NAMES, not this
    # trigger phrase itself.
    r"colu(?:mn|msn)s?\s+(?:we\s+need(?:\s+to\s+export)?\s*)?(?:is|are)?\s*[:=]\s*",
    re.IGNORECASE)
# A numbered item's own '1.' must not be read as end-of-list — 'columns we
# need is = 1.Yard No.,2.matrial code' has its very first period one
# character after the '='. Stripped separately per item, below, instead of
# being part of the end-of-list test.
_LIST_ITEM_NUM_RE = re.compile(r"^\s*\d+\s*[.):]?\s*")
# Same idea, but with the punctuation REQUIRED rather than optional — for
# stripping a marker off a column name a caller (typically the model,
# copying the user's numbered list verbatim into a tool argument) may not
# have cleaned up itself. The optional punctuation in _LIST_ITEM_NUM_RE is
# fine during list PARSING, where surrounding context (a comma, another
# marker) already establishes that a leading digit is a marker — but as a
# standalone normalization step it would also strip the real leading digit
# off a genuine column name like "6 Ship Set", which is exactly one of the
# columns in this domain's own schema.
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*\d+\s*[.):]\s*")
# Splits right before each numbered marker, keeping the number attached to
# the item it introduces — for a list with NO commas at all between items
# ("1.Sr No 2.Group 3.Tag No. 4.Yard No. ..."), which the trailing-comma
# split below cannot separate at all, since there is no comma anywhere.
# The negative lookbehind is load-bearing, and needs to reject exactly the
# right things: without it, "10.TestingStd" itself contains a match at "0."
# (a digit immediately following another digit), splitting the marker "10."
# into "1" + "0.TestingStd" and silently truncating the list at the first
# two-digit item — and the filename "w1-02.xlsx" matches at "02." too. A
# real marker can be preceded by whitespace ("... Std 10.TestingStd") OR by
# a comma with no space ("Yard No.,2.matrial code") — both must be allowed
# — so the rejection is targeted at what actually means "still inside a
# token": a letter, digit, or hyphen immediately before the digit run.
_NUMBERED_ITEM_SPLIT_RE = re.compile(r"(?<![\w-])(?=\d+\s*[.):])")
# Trailing filler after the real list — "..., Item Quantity, do it" — reads
# exactly like a short list item (2 words) by word-count alone. None of
# these ever plausibly names a spreadsheet column on their own.
_LIST_FILLER_WORDS = {"do", "it", "please", "now", "thanks", "thank", "you",
                      "ok", "okay", "asap", "export", "save", "give",
                      "create", "generate", "make", "file", "excel", "csv"}


def extract_requested_column_tokens(prompt: str) -> List[str]:
    """The comma-separated list after 'columns we need is =' / 'columns = '
    / 'column we need to export=', stripped of leading numbering ('1.',
    '2)'). Used only to report which of the user's own named columns could
    not be matched to anything real in the source data — not for selection
    itself (extract_requested_columns stays existence-based), so a name
    that coincidentally isn't real doesn't silently vanish with no
    explanation.

    The list is read comma by comma rather than up to one end-of-list
    marker: a real column name can itself contain a period ('Yard No.',
    an abbreviation) or a comma-free multi-word phrase, so there is no
    single punctuation mark that reliably means "the list is over" — but a
    genuine list item is short (a few words, often numbered), and free
    prose describing the request is not. The first comma-delimited chunk
    that doesn't look like a list item ends the list.
    """
    if not prompt:
        return []
    m = _REQUESTED_LIST_START_RE.search(prompt)
    if not m:
        return []
    rest = prompt[m.end():]
    # A list numbered at least twice is delimited by its OWN numbering, not
    # commas, ONLY when there aren't already enough commas to be doing that
    # job themselves — "1.Sr No 2.Group 3.Tag No. ..." has one comma in the
    # WHOLE 31-item message (well after the list, before trailing prose),
    # so few commas relative to markers means numbering is the real
    # delimiter. "1.Yard No.,2.matrial code ,3 item" has a comma between
    # every pair of items — numbered-splitting would still work for items 1
    # and 2 there, but item 3 ("3 item", no period after the digit) has no
    # marker punctuation to split on and would be silently dropped; comma-
    # splitting (which also strips each chunk's own leading "N.") handles
    # it correctly, so it wins whenever there are enough commas to trust.
    markers = len(re.findall(r"\d+\s*[.):]", rest))
    commas = rest.count(",")
    if markers >= 2 and commas < markers - 1:
        raw_items = [c for c in _NUMBERED_ITEM_SPLIT_RE.split(rest) if c.strip()]
    else:
        raw_items = re.split(r"\s*,\s*", rest)

    tokens = []
    for raw in raw_items:
        numbered = bool(_LIST_ITEM_NUM_RE.match(raw))
        item = _clean(_LIST_ITEM_NUM_RE.sub("", raw))
        # A numbered item's own text can still run into trailing prose past
        # the list ("31.Total Value for All Shipset, export in w1-02.xlsx")
        # when it's the LAST item and nothing numbered follows to bound it —
        # a comma there separates the item from what comes after it.
        if "," in item:
            item = item.split(",", 1)[0].strip()
        words = item.split()
        if not words or not (numbered or len(words) <= 3):
            break
        if all(w.lower() in _LIST_FILLER_WORDS for w in words):
            break
        tokens.append(item)
    return tokens


def unmatched_requested_columns(prompt: str, selected_columns: List[str]) -> List[str]:
    """Requested column tokens with no counterpart among the columns that
    were actually selected — e.g. 'Yard No.' asked for on a table where it
    is letterhead text, not a data column, and so was never a candidate for
    extract_requested_columns to find in the first place."""
    tokens = extract_requested_column_tokens(prompt)
    if not tokens:
        return []
    selected_norm = " ".join(_norm_key(c) for c in selected_columns)
    return [t for t in tokens if not _fuzzy_contains(_norm_key(t), selected_norm)]


# --- row filter: "where <col> = <value>" ----------------------------------

_FILTER_RE = re.compile(
    r"\bwhere\b\s+(?:the\s+)?[\"'“]?(.+?)[\"'”]?\s*"
    r"(?:=|==|is exactly|is|equals?)\s*[\"'“]?([^\"'”,.\n]+)",
    re.IGNORECASE)


def extract_row_filter(prompt: str, available_columns: List[str]
                       ) -> Optional[Tuple[str, str]]:
    """(real column name, value) from 'where <col> = <value>'. A filter
    naming a column that doesn't match anything real is not honoured —
    silently filtering on a column that doesn't exist would just return an
    empty table with no explanation."""
    if not prompt or not available_columns:
        return None
    m = _FILTER_RE.search(prompt)
    if not m:
        return None
    col_text, value = _clean(m.group(1)), _clean(m.group(2))
    key = _norm_key(col_text)
    if not key:
        return None
    best = None
    for col in available_columns:
        ck = _norm_key(col)
        if ck == key:
            return col, value
        if (key in ck or ck in key) and best is None:
            best = col
    return (best, value) if best else None


def apply_row_filter(df: pd.DataFrame, filt: Tuple[str, str]) -> pd.DataFrame:
    col, value = filt
    mask = df[col].astype(str).str.strip().str.casefold() == value.strip().casefold()
    return df[mask]


# --- row limit: "top 10 rows" / "bottom 5 rows" ---------------------------

_TOP_RE = re.compile(r"\b(?:top|first)\s+(\d+|[a-z]+)\s+rows?\b", re.IGNORECASE)
_BOTTOM_RE = re.compile(r"\b(?:bottom|last)\s+(\d+|[a-z]+)\s+rows?\b", re.IGNORECASE)


def extract_row_limit(prompt: str) -> Optional[Tuple[str, int]]:
    """('head'|'tail', n) — 'top'/'first' N rows or 'bottom'/'last' N rows.
    Whichever phrase appears; a prompt naming both is not expected and only
    the first match wins."""
    if not prompt:
        return None
    m = _TOP_RE.search(prompt)
    if m:
        n = _count(m.group(1))
        if n:
            return "head", n
    m = _BOTTOM_RE.search(prompt)
    if m:
        n = _count(m.group(1))
        if n:
            return "tail", n
    return None


def apply_row_limit(df: pd.DataFrame, limit: Tuple[str, int]) -> pd.DataFrame:
    kind, n = limit
    return df.head(n) if kind == "head" else df.tail(n)


# --- computed column: "add column Total = Price * Quantity" --------------

_COMPUTE_RE = re.compile(
    r"\b(?:add|create|calculate|make)\b.{0,25}?\bcolumn\b\s+"
    r"[\"'“]?([\w \-]+?)[\"'”]?\s*(?:=|:|as)\s*"
    r"([\w .\-]+?)\s*([*/+\-])\s*([\w .\-]+?)"
    r"(?:$|[.,\n]|\bin\b|\band export\b|\bexport\b)",
    re.IGNORECASE)

_OPS = {"*": lambda a, b: a * b, "/": lambda a, b: a / b,
        "+": lambda a, b: a + b, "-": lambda a, b: a - b}


def _resolve_col(text: str, available_columns: List[str]) -> Optional[str]:
    key = _norm_key(text)
    if not key:
        return None
    for col in available_columns:
        if _norm_key(col) == key:
            return col
    return None


def extract_computed_column(prompt: str, available_columns: List[str]
                            ) -> Optional[Dict]:
    """{'name', 'left', 'op', 'right_col' XOR 'right_val'} from
    'add a column Total = Price * Quantity' or 'add column Total = Price *
    1.18'. Both operands must resolve — a real column on the left always,
    a real column or a plain number on the right — or nothing is returned;
    a formula this module can't ground stays with whatever else already
    handles the request rather than silently doing nothing."""
    if not prompt or not available_columns:
        return None
    m = _COMPUTE_RE.search(prompt)
    if not m:
        return None
    new_name = _clean(m.group(1))
    left_col = _resolve_col(_clean(m.group(2)), available_columns)
    if not new_name or left_col is None:
        return None
    op = m.group(3)
    right_text = _clean(m.group(4))
    right_col = _resolve_col(right_text, available_columns)
    right_val = None
    if right_col is None:
        try:
            right_val = float(right_text)
        except ValueError:
            return None
    return {"name": new_name, "left": left_col, "op": op,
            "right_col": right_col, "right_val": right_val}


def apply_computed_column(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    out = df.copy()
    left = pd.to_numeric(out[spec["left"]], errors="coerce")
    right = (pd.to_numeric(out[spec["right_col"]], errors="coerce")
             if spec["right_col"] else spec["right_val"])
    out[spec["name"]] = _OPS[spec["op"]](left, right)
    return out


# --- orchestration ----------------------------------------------------

def has_any_spec(prompt: str, available_columns: List[str]) -> bool:
    """True when the prompt names at least one operation this module can
    ground against real columns — the signal a caller uses to prefer this
    deterministic path over an LLM sandbox."""
    if not prompt or not available_columns:
        return False
    return bool(
        extract_computed_column(prompt, available_columns)
        or extract_row_filter(prompt, available_columns)
        or extract_row_limit(prompt)
        or extract_requested_columns(prompt, available_columns)
    )


def _mask_span(text: str, match: Optional[re.Match]) -> str:
    if not match:
        return text
    start, end = match.span()
    return text[:start] + (" " * (end - start)) + text[end:]


def parse_and_apply(df: pd.DataFrame, prompt: str) -> Tuple[pd.DataFrame, List[str]]:
    """Apply every deterministic operation this prompt names, in a fixed
    order (compute, then filter, then limit, then column selection — a
    computed column can be filtered/exported like any other; a limit applies
    to the filtered rows; column selection is the final narrowing).
    Returns (result, human-readable list of what changed).

    Each matched clause is masked out of the prompt before the LATER stages
    run: "where Item Title = jay-pawar" would otherwise leak the column name
    "Item Title" into the column-selection stage and turn a row filter into
    an unwanted column filter too.
    """
    changes: List[str] = []
    out = df
    masked = prompt or ""

    computed = extract_computed_column(prompt, list(out.columns))
    if computed:
        masked = _mask_span(masked, _COMPUTE_RE.search(prompt))
        out = apply_computed_column(out, computed)
        changes.append(f"added computed column '{computed['name']}'")

    filt = extract_row_filter(prompt, list(out.columns))
    if filt:
        masked = _mask_span(masked, _FILTER_RE.search(prompt))
        before = len(out)
        out = apply_row_filter(out, filt)
        changes.append(f"filtered where '{filt[0]}' = '{filt[1]}' "
                       f"({before} -> {len(out)} rows)")

    limit = extract_row_limit(prompt)
    if limit:
        kind, n = limit
        masked = _mask_span(masked, (_TOP_RE if kind == "head" else _BOTTOM_RE).search(prompt))
        out = apply_row_limit(out, limit)
        changes.append(f"kept {'top' if kind == 'head' else 'bottom'} {n} rows")

    masked = _mask_span(masked, _RENAME_CLAUSE_RE.search(prompt))

    cols = extract_requested_columns(masked, list(out.columns))
    if cols:
        out = out[cols]
        changes.append(f"kept only columns: {', '.join(str(c) for c in cols)}")

    missing = unmatched_requested_columns(prompt, list(out.columns))
    if missing:
        changes.append(
            f"could NOT find these requested column(s) anywhere in the "
            f"source data, so they are NOT in this file: "
            f"{', '.join(missing)}")

    return out, changes
