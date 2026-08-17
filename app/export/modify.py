"""Change a file that has already been exported.

Everything in this project until now rebuilt an export from scratch. So when a
user said "now add the header details to this excel", there was no code path
that could touch the file they were looking at — the request fell through to
the code-exec sandbox, which answered that it could not extract data. It failed
three turns running in the production log, including the turn where the user
had spelled out every single value by hand.

The operations here are the ones a person actually asks for after seeing a
result: put the document's identification back on top, rename the columns I
name, drop the ones I don't want. They are applied to the bytes on disk and the
file is re-verified afterwards, so a modification can never quietly shorten a
table.

Intent is read from the user's own words, not from a paraphrase. That rule is
load-bearing here: the phrasing that failed in production —"we need to add this
Full header section data also in this excel top section" — carries the request
in words a model routinely drops on the way to a tool call.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app import state
from app.config import OUTPUT_DIR
from app.export import spec as export_spec
from app.export.exporters import EXPORTERS, band_offset, verify_export
from app.models import QueryPlan

# "the first two columns", "first 2 columns"
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_POSITIONAL_RENAME_RE = re.compile(
    r"\b(?:rename|change|call|set|make)\b[^.]{0,40}?\bfirst\s+(?:(\w+)\s+)?columns?\b"
    r"[^.]{0,20}?\b(?:to|as|into)\b\s*(.+)", re.IGNORECASE)

_NAMED_RENAME_RE = re.compile(
    # "rename the column name = Description in to Group" is a real, repeated
    # user phrasing — "column NAME =" before the old name (not just
    # "column"), and "in to" as a typo'd "into". Without accounting for
    # both, the old capture group swallowed "name = Description in" whole,
    # which never matches any real column, so the rename silently failed —
    # and the model, composing the final answer, described the failure as
    # "the column 'Description' was not found" while the file it had just
    # written still literally had that header. Confirmed production
    # failure, session 7e43a023, 2026-08-14.
    r"\brename\b\s+(?:the\s+)?(?:column\s+)?(?:name\s*[:=]\s*)?"
    r"[\"'“]?(.+?)[\"'”]?\s+"
    r"\b(?:in\s*to|into|to|as)\b\s+[\"'“]?([^\"'”,.]+)", re.IGNORECASE)

_DROP_RE = re.compile(
    r"\b(?:remove|delete|drop|get rid of)\b\s+(?:the\s+)?[\"'“]?(.+?)[\"'”]?\s+columns?\b",
    re.IGNORECASE)

# Words a person uses for the block of identification above a table.
_CONTEXT_WORDS = re.compile(
    r"\b(header|letterhead|top section|title block|company name|company|"
    r"project|spec (?:no|number)|context|document detail|header detail|"
    r"header section|meta ?data)\b", re.IGNORECASE)
_ADD_WORDS = re.compile(
    r"\b(add|include|put|insert|append|need|want|keep|with|also)\b", re.IGNORECASE)
# Deliberately no bare "no": the column heading "Spec No." made every rename
# of it read as an instruction to strip the header band.
_REMOVE_WORDS = re.compile(
    r"\b(remove|delete|drop|without|strip|exclude|only the table|just the table|"
    r"bare|plain)\b", re.IGNORECASE)
# A clause opened by "if" describes a condition, not this instruction. "do not
# add header on table if i need header than i will tell but right now i don't
# need" says remove twice and add never — but "i need header" sits closer to a
# "header" than the negation does, so the band was written anyway and the user
# reported the same complaint for the third time. The clause runs to the next
# delimiter or to "but", whichever comes first.
_CONDITIONAL_RE = re.compile(r"\bif\b.*?(?=[.,;\n]|\bbut\b|$)",
                             re.IGNORECASE | re.DOTALL)
# "i don't need table header" read as an ADD, because the negation sits on the
# far side of the verb and only the verb was being measured. Negated wanting is
# rewritten to a plain remove verb, in place, so the distances stay honest.
_NEGATED_ADD_RE = re.compile(
    r"\b(?:do\s*n[o']?t|does\s*n[o']?t|did\s*n[o']?t|dont|don't|no|not|never)\s+"
    r"(?:need|want|require|include|add|put|keep)\b", re.IGNORECASE)

_SPLIT_TARGETS_RE = re.compile(r"\s*(?:,|\band\b|&|/)\s*", re.IGNORECASE)


def _count(word: str) -> Optional[int]:
    word = word.strip().lower()
    if word.isdigit():
        return int(word)
    return _WORD_NUMBERS.get(word)


def _clean_name(text: str) -> str:
    """Trim quotes and whitespace only.

    Not the trailing period: 'Spec No.' and 'Sr. No.' are real column headings
    in this domain, and a rule that strips a final full stop cannot tell them
    from a sentence. Matching is made tolerant instead — see `_norm`."""
    return re.sub(r"^[\s\"'“”]+|[\s\"'“”]+$", "", text).strip()


def _norm(name) -> str:
    """Comparison form of a column name: case, spacing and trailing
    punctuation ignored, so 'Spec No.' finds 'Spec No' and vice versa."""
    return re.sub(r"[\s.:]+$", "", re.sub(r"\s+", " ", str(name))).strip().casefold()


def parse_instruction(text: str) -> Dict:
    """What the user asked for, as an explicit set of operations.

    Deterministic on purpose. Every rule here fires on the user's literal
    words; nothing is inferred by a model that might paraphrase the request
    into something else.
    """
    ops: Dict = {"rename": {}, "positional_rename": [], "drop": [],
                 "context": None, "raw_text": text}
    if not text:
        return ops

    # Row-reducing operations (filter, top/bottom-N) and computed columns need
    # the file's real columns to resolve, which aren't available yet at parse
    # time — apply_ops does that once the frame is loaded. This flag only
    # answers "is there something here worth loading the file for", from the
    # instruction text alone, so modify() can still short-circuit unrecognized
    # instructions before touching disk.
    ops["may_reduce_rows"] = bool(
        export_spec._FILTER_RE.search(text)
        or export_spec._TOP_RE.search(text)
        or export_spec._BOTTOM_RE.search(text))
    ops["may_add_column"] = bool(export_spec._COMPUTE_RE.search(text))

    # Spans already claimed by a rename or drop are column names. They are
    # masked out before the header test, because a column called "Spec No."
    # or "Project" would otherwise make every rename of it read as a request
    # about the letterhead.
    claimed: List[Tuple[int, int]] = []

    match = _POSITIONAL_RENAME_RE.search(text)
    if match:
        # "the first column" names one; "the first two columns" names two.
        n = _count(match.group(1)) if match.group(1) else 1
        names = [_clean_name(p) for p in _SPLIT_TARGETS_RE.split(match.group(2))]
        names = [name for name in names if name]
        if n:
            ops["positional_rename"] = names[:n]
            claimed.append(match.span())

    if not ops["positional_rename"]:
        for match in _NAMED_RENAME_RE.finditer(text):
            old, new = _clean_name(match.group(1)), _clean_name(match.group(2))
            # "rename the first two columns to X and Y" is positional and was
            # already handled; don't also read it as a literal column called
            # "the first two columns".
            if old and new and "column" not in old.lower():
                ops["rename"][old] = new
                claimed.append(match.span())

    for match in _DROP_RE.finditer(text):
        name = _clean_name(match.group(1))
        if name and not _CONTEXT_WORDS.fullmatch(name):
            ops["drop"].extend(n for n in _SPLIT_TARGETS_RE.split(name) if n)
            claimed.append(match.span())

    masked = list(text)
    for start, end in claimed:
        masked[start:end] = " " * (end - start)
    masked = "".join(masked)
    # Same length in, same length out — every offset below still points at the
    # word it did before.
    masked = _CONDITIONAL_RE.sub(lambda m: " " * len(m.group(0)), masked)
    masked = _NEGATED_ADD_RE.sub(
        lambda m: "remove".ljust(len(m.group(0))), masked)

    if _CONTEXT_WORDS.search(masked):
        # "remove" wins over "add" only when it is the nearer verb, because
        # "add the header, drop the SL no column" says both.
        add_at = _nearest(masked, _ADD_WORDS, _CONTEXT_WORDS)
        remove_at = _nearest(masked, _REMOVE_WORDS, _CONTEXT_WORDS)
        if remove_at is not None and (add_at is None or remove_at < add_at):
            ops["context"] = False
        elif add_at is not None:
            ops["context"] = True
    return ops


def _nearest(text: str, verb: re.Pattern, target: re.Pattern) -> Optional[int]:
    """Distance from the closest verb match to the closest target match."""
    verbs = [m.start() for m in verb.finditer(text)]
    targets = [m.start() for m in target.finditer(text)]
    if not verbs or not targets:
        return None
    return min(abs(v - t) for v in verbs for t in targets)


def load_export(path: str) -> Tuple[List[pd.DataFrame], List[str]]:
    """Read an exported file back, skipping any context band already in it."""
    if path.lower().endswith(".csv"):
        return [pd.read_csv(path, dtype=str).fillna("")], ["Sheet1"]
    names = pd.ExcelFile(path).sheet_names
    frames = []
    for name in names:
        df = pd.read_excel(path, sheet_name=name, skiprows=band_offset(path, name),
                           dtype=str)
        frames.append(df.fillna(""))
    return frames, names


def apply_ops(df: pd.DataFrame, ops: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Apply renames and drops, reporting exactly what changed."""
    changes: List[str] = []
    out = df.copy()

    if ops.get("positional_rename"):
        new_names = list(out.columns)
        for i, name in enumerate(ops["positional_rename"][:len(new_names)]):
            if str(new_names[i]) != name:
                changes.append(f"column {i + 1} '{new_names[i]}' → '{name}'")
                new_names[i] = name
        out.columns = new_names

    if ops.get("rename"):
        lookup = {_norm(c): c for c in out.columns}
        mapping = {}
        for old, new in ops["rename"].items():
            actual = lookup.get(_norm(old))
            if actual is None:
                changes.append(f"no column named '{old}' — left unchanged")
                continue
            mapping[actual] = new
            changes.append(f"'{actual}' → '{new}'")
        out = out.rename(columns=mapping)

    for name in ops.get("drop", []):
        lookup = {_norm(c): c for c in out.columns}
        actual = lookup.get(_norm(name))
        if actual is None:
            changes.append(f"no column named '{name}' — nothing dropped")
        else:
            out = out.drop(columns=[actual])
            changes.append(f"dropped column '{actual}'")

    # Filter / top-N / computed-column, resolved against THIS frame's real
    # columns now that it's loaded — parse_instruction could only detect
    # that the instruction text mentions one, not resolve it.
    raw_text = ops.get("raw_text") or ""
    if ops.get("may_add_column"):
        computed = export_spec.extract_computed_column(raw_text, list(out.columns))
        if computed:
            out = export_spec.apply_computed_column(out, computed)
            changes.append(f"added computed column '{computed['name']}'")
    if ops.get("may_reduce_rows"):
        filt = export_spec.extract_row_filter(raw_text, list(out.columns))
        if filt:
            before = len(out)
            out = export_spec.apply_row_filter(out, filt)
            changes.append(f"filtered where '{filt[0]}' = '{filt[1]}' "
                           f"({before} -> {len(out)} rows)")
        limit = export_spec.extract_row_limit(raw_text)
        if limit:
            kind, n = limit
            out = export_spec.apply_row_limit(out, limit)
            changes.append(f"kept {'top' if kind == 'head' else 'bottom'} {n} rows")

    return out, changes


def modify(filename: str, instruction: str, file_id: Optional[str] = None) -> str:
    """Apply ``instruction`` to an already-exported file and rewrite it."""
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        known = sorted(state.WRITTEN_EXPORTS)[-5:]
        hint = f" Files this session wrote: {', '.join(known)}." if known else ""
        return (f"There is no file called {os.path.basename(filename)} to "
                f"modify.{hint}")

    ops = parse_instruction(instruction)
    if not any([ops["rename"], ops["positional_rename"], ops["drop"],
                ops["context"] is not None, ops.get("may_reduce_rows"),
                ops.get("may_add_column")]):
        return ("I can change an existing export in these ways: add or remove "
                "the document header band, rename columns (by name or by "
                "position, e.g. 'rename the first two columns to Category and "
                "Description'), drop columns, keep only rows where a column "
                "equals a value, keep only the top/bottom N rows, or add a "
                "computed column (e.g. 'add a column Total = Price * Qty'). "
                f"Say which one you want for {os.path.basename(path)}.")

    frames, sheet_names = load_export(path)
    if not frames:
        return f"{os.path.basename(path)} has no readable table in it."

    rows_before = sum(len(f) for f in frames)
    provenance = state.WRITTEN_EXPORTS.get(os.path.basename(path), {})
    all_changes: List[str] = []
    for i, frame in enumerate(frames):
        frames[i], changes = apply_ops(frame, ops)
        frames[i].attrs["sheet_name"] = sheet_names[i]
        all_changes.extend(changes)

    rows_after = sum(len(f) for f in frames)
    if ops.get("may_reduce_rows") and rows_after == 0:
        return (f"That filter matched no rows in {os.path.basename(path)} — "
                f"nothing was changed. Applied: "
                f"{'; '.join(all_changes) if all_changes else 'nothing'}.")
    # A filter or top/bottom-N limit intentionally shrinks the table; the
    # verification below must not treat that as a truncated/broken export.
    expected_rows = rows_after if ops.get("may_reduce_rows") else rows_before

    want_context = ops["context"]
    context = provenance.get("context")
    if want_context and context is None:
        # Nothing recorded — recover it from the source document if we still
        # know which one produced this file.
        source = file_id or provenance.get("file_id")
        if source:
            from app.tables.context import context_for, describe_source
            pages = provenance.get("pages")
            context = context_for(source, pages, source=describe_source(source, pages))
    if want_context:
        if not context:
            return (f"{os.path.basename(path)} was not produced from a document "
                    "whose header I can still read, so there is nothing "
                    "authentic to put above the table. Re-run the export and "
                    "the header will be included.")
        for frame in frames:
            frame.attrs["context"] = context
        all_changes.append("added the document header band")
    elif want_context is False:
        for frame in frames:
            frame.attrs.pop("context", None)
        all_changes.append("removed the document header band")
    else:
        for frame in frames:
            if context is not None:
                frame.attrs["context"] = context

    # Written beside the original and moved into place only once it verifies,
    # so a failed modification leaves the file the user already has intact
    # rather than replacing it with a broken one.
    fmt = "csv" if path.lower().endswith(".csv") else "excel"
    staged = f"~modify-{os.path.basename(path)}"
    staged_path = os.path.join(OUTPUT_DIR, staged)
    plan = QueryPlan(intent="export", sink=fmt, filename=staged,
                     no_context=(want_context is False))
    EXPORTERS[fmt](provenance.get("file_id") or file_id or "modified", plan,
                   tables=frames)

    ok, message = verify_export(staged_path, expected_rows, fmt)
    if not ok:
        if os.path.exists(staged_path):
            os.remove(staged_path)
        return (f"The change was not saved — {message}. "
                f"{os.path.basename(path)} is unchanged on disk.")

    os.replace(staged_path, path)
    state.WRITTEN_EXPORTS[os.path.basename(path)] = {
        **provenance,
        "context": context if want_context is not False else None,
    }
    state.WRITTEN_EXPORTS.pop(staged, None)
    detail = "; ".join(all_changes) if all_changes else "no change was needed"
    return (f"{message.replace(staged, os.path.basename(path))} "
            f"Changed: {detail}.")
