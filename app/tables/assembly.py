"""Assembling one rectangular table out of many extracted fragments.

Every "export pages 6 to 10", "combine these documents", "the whole table"
request ends up concatenating DataFrames that came from different pages,
sheets or files. Doing that with pd.concat is unsound, and produced a file
that passed every check we had while being wrong in the worst possible way.

Measured, from outputs/f1-66.xlsx (agent reported "65 rows, 5 columns,
export complete"):

    row 0   Evaluation Schedules | Item/Category | Consignee/Reporting
            Officer | Consignee Address | Quantity | Consignee / Reporting
            Officer      <- the same header twice, differing by two spaces
    row 18  '0' | '1' | '' | '3' | '' | '2'
                          <- Docling's positional column indices, as data
    row 19+ item text under 'Evaluation Schedules', officer under
            'Item/Category', quantity under 'Consignee Address'
                          <- every value under the wrong heading

Docling extracted those pages consistently; our assembly corrupted them.
The root causes, all confirmed against docling_cache:

  * Docling flags row 0 as column_header on pages 7/9/10 even though it is
    real data, and flags nothing on page 8. So page 8 arrives with integer
    column labels [0,1,2,3] and pages 7/9/10 arrive with a data row eaten
    as the header.
  * Only the page carrying the header (page 6 here) can name its columns;
    continuation pages arrive as col_0..col_N.
  * A continuation page whose merged left column was dropped is 4 wide
    against a 5-name header, so matching by position shifts every field.

So alignment cannot rely on column names — most fragments have none. It
relies on what the values look like, and it preserves left-to-right order,
because table columns cannot cross. Anything it cannot match confidently is
kept as its own column and reported, never merged into a neighbour: a
missing column is recoverable, a silently mismatched one is not.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math
import re

import pandas as pd

from app.tables.helpers import _columns_are_generic, _repair_split_words

# Columns are matched as a monotone alignment (no crossing), so a dropped
# column becomes a gap rather than a one-place shift of everything after it.
# A gap must cost less than a bad pairing, or the aligner "fixes" a missing
# column by sliding the rest along — precisely the f1-66.xlsx corruption.
_GAP_PENALTY = -0.30
_MIN_PAIR_SCORE = 0.05

_NUMERIC_RE = re.compile(r"^-?[\d\s,]*\d(\.\d+)?$")


def column_key(name) -> str:
    """Normalised identity of a column name.

    'Consignee/Reporting Officer' and 'Consignee / Reporting Officer' are one
    column that pandas unioned into two half-empty ones. PDF headers also
    arrive broken mid-word ('Q ua nti ty'), so the split-word repair already
    used for display runs here too.
    """
    text = _repair_split_words(str(name or ""))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_generic(name) -> bool:
    """True for our placeholder names and Docling's positional labels."""
    text = str(name).strip()
    return bool(re.fullmatch(r"col_\d+", text) or re.fullmatch(r"\d+", text))


@dataclass(frozen=True)
class Signature:
    """What a column's values look like, learned from the values themselves.

    Nothing here encodes this tender's vocabulary — no column names, no valve
    words. It is the only way to align fragments that have no headers at all.
    """
    numeric: float = 0.0
    blank: float = 0.0
    mean_len: float = 0.0
    distinct: float = 0.0
    alpha: float = 0.0

    def distance(self, other: "Signature") -> float:
        # Length is compared in log space: the gap between a 3-character
        # quantity and a 60-character address matters, the gap between 55 and
        # 60 characters does not.
        len_a = math.log1p(self.mean_len)
        len_b = math.log1p(other.mean_len)
        len_gap = abs(len_a - len_b) / math.log1p(200)
        return (0.35 * abs(self.numeric - other.numeric)
                + 0.30 * min(1.0, len_gap)
                + 0.20 * abs(self.distinct - other.distinct)
                + 0.15 * abs(self.alpha - other.alpha))


def signature(values: Sequence) -> Signature:
    """Fingerprint a column from a sample of its values."""
    texts = [("" if v is None else str(v)).strip() for v in list(values)[:200]]
    if not texts:
        return Signature()
    filled = [t for t in texts if t and t.lower() not in ("nan", "none")]
    if not filled:
        return Signature(blank=1.0)

    numeric = sum(1 for t in filled if _NUMERIC_RE.match(t)) / len(filled)
    alpha = sum(1 for t in filled if any(ch.isalpha() for ch in t)) / len(filled)
    return Signature(
        numeric=numeric,
        blank=1.0 - len(filled) / len(texts),
        mean_len=sum(len(t) for t in filled) / len(filled),
        distinct=len({t.lower() for t in filled}) / len(filled),
        alpha=alpha,
    )


def _name_similarity(a, b) -> Optional[float]:
    """Name agreement in 0..1, or None when at least one name is a
    placeholder and therefore carries no information."""
    if is_generic(a) or is_generic(b):
        return None
    key_a, key_b = column_key(a), column_key(b)
    if not key_a or not key_b:
        return None
    if key_a == key_b:
        return 1.0
    if key_a in key_b or key_b in key_a:
        return 0.8
    tokens_a = set(re.findall(r"[a-z0-9]+", _repair_split_words(str(a)).lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", _repair_split_words(str(b)).lower()))
    if not tokens_a or not tokens_b:
        return None
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def pair_score(name_a, sig_a: Signature, name_b, sig_b: Signature) -> float:
    """How strongly two columns look like the same column, in 0..1."""
    content = 1.0 - min(1.0, sig_a.distance(sig_b))
    name = _name_similarity(name_a, name_b)
    if name is None:
        return content
    return 0.4 * name + 0.6 * content


def align_columns(ref: List[Tuple[object, Signature]],
                  cand: List[Tuple[object, Signature]]) -> List[Optional[int]]:
    """Map each reference column to a candidate column index, or None.

    Needleman-Wunsch over the two column sequences. Order-preserving by
    construction, which matters: a table's columns keep their left-to-right
    order across a page break, so an alignment that lets them cross is
    always wrong. Gaps model the merged column Docling drops on some pages.
    """
    n, m = len(ref), len(cand)
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + _GAP_PENALTY
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + _GAP_PENALTY

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = scores[i - 1][j - 1] + pair_score(
                ref[i - 1][0], ref[i - 1][1], cand[j - 1][0], cand[j - 1][1])
            scores[i][j] = max(match,
                               scores[i - 1][j] + _GAP_PENALTY,
                               scores[i][j - 1] + _GAP_PENALTY)

    mapping: List[Optional[int]] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        match = scores[i - 1][j - 1] + pair_score(
            ref[i - 1][0], ref[i - 1][1], cand[j - 1][0], cand[j - 1][1])
        if abs(scores[i][j] - match) < 1e-9:
            score = pair_score(ref[i - 1][0], ref[i - 1][1],
                               cand[j - 1][0], cand[j - 1][1])
            # A pairing this weak is not evidence of anything; leaving the
            # reference column empty and reporting the candidate as unmatched
            # is honest, whereas forcing it fabricates a column of values.
            if score >= _MIN_PAIR_SCORE:
                mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif abs(scores[i][j] - (scores[i - 1][j] + _GAP_PENALTY)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return mapping


def drop_label_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove body rows that are not data.

    Two kinds, both observed in outputs/f1-66.xlsx: a row equal to the frame's
    own header, and a row of consecutive integers starting at 0 — Docling's
    positional column labels, which get_all_real_tables mistook for a
    misdetected header row and pushed into the body.
    """
    if df.empty:
        return df, 0

    header_keys = [column_key(c) for c in df.columns]
    keep, dropped = [], 0
    for idx, row in df.iterrows():
        cells = [("" if v is None else str(v)).strip() for v in row.tolist()]
        filled = [c for c in cells if c]
        is_positional = (
            len(filled) >= 2
            and all(c.isdigit() for c in filled)
            and [int(c) for c in filled] == sorted({int(c) for c in filled})
            and min(int(c) for c in filled) == 0
            and max(int(c) for c in filled) < len(cells)
        )
        is_header_echo = (
            len(filled) >= 2
            and [column_key(c) for c in cells] == header_keys
        )
        if is_positional or is_header_echo:
            dropped += 1
            continue
        keep.append(idx)
    if dropped:
        return df.loc[keep].reset_index(drop=True), dropped
    return df, 0


def repair_numeric_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Rejoin digits a narrow PDF column broke apart — '10 2' -> '102'.

    Page 10 of the tender reports quantities as '10 2', '13 8', '22 2'. Those
    are single numbers split by the column width, and they were exported
    verbatim, so a quantity of 138 read as '13 8'. Only columns whose values
    are overwhelmingly numeric are touched, and only cells that contain
    nothing but digits and spaces — decided from the data, never from a
    column name.
    """
    out = df
    for column in df.columns:
        sig = signature(df[column])
        if sig.numeric < 0.9:
            continue
        repaired = [
            re.sub(r"\s+", "", str(v)) if isinstance(v, str) and re.fullmatch(r"[\d\s]+", v)
            else v
            for v in df[column].tolist()
        ]
        if repaired != df[column].tolist():
            if out is df:
                out = df.copy()
            out[column] = repaired
    return out


def looks_like_label_column(values) -> bool:
    """A short, heavily repeated column — a merged group label rather than
    prose. Used to decide where split-word repair is safe to apply to values."""
    sig = signature(values)
    return sig.mean_len < 30 and sig.distinct < 0.6 and sig.alpha > 0.5


@dataclass
class AlignmentReport:
    """What assembly did and, more importantly, what it could not do."""
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    parts: List[Dict] = field(default_factory=list)
    unmatched: List[Dict] = field(default_factory=list)
    reference_page: Optional[object] = None

    @property
    def clean(self) -> bool:
        return not self.unmatched

    def describe(self) -> str:
        bits = [f"{self.rows_out} rows assembled from {len(self.parts)} fragment(s)"]
        if self.rows_dropped:
            bits.append(f"{self.rows_dropped} non-data row(s) removed")
        for item in self.unmatched:
            bits.append(
                f"column {item['column']!r} from {item['part']} did not match "
                f"the table's own columns and was kept separate")
        return "; ".join(bits)


def _pick_reference(frames: List[pd.DataFrame]) -> int:
    """The fragment whose columns everything else is aligned to: the one with
    the most real (non-placeholder) names, then the widest, then the longest.
    Named columns are what the user asked to keep, so they lead."""
    best, best_rank = 0, None
    for i, df in enumerate(frames):
        named = sum(0 if is_generic(c) else 1 for c in df.columns)
        rank = (named, df.shape[1], df.shape[0])
        if best_rank is None or rank > best_rank:
            best, best_rank = i, rank
    return best


def _part_label(df: pd.DataFrame, index: int) -> str:
    page = df.attrs.get("page")
    source = df.attrs.get("source_file") or df.attrs.get("sheet_name")
    if source and page is not None:
        return f"{source} page {page}"
    if source:
        return str(source)
    if page is not None:
        return f"page {page}"
    return f"fragment {index + 1}"


def assemble(frames: List[pd.DataFrame],
             provenance: bool = False) -> Tuple[Optional[pd.DataFrame], AlignmentReport]:
    """Combine table fragments into one aligned table.

    Returns (dataframe, report). The report names every column that could not
    be matched; callers should surface that rather than presenting the result
    as a clean export.
    """
    report = AlignmentReport()
    usable = [df for df in frames if df is not None and df.shape[1] > 0]
    if not usable:
        return None, report

    cleaned = []
    for df in usable:
        report.rows_in += len(df)
        trimmed, dropped = drop_label_rows(df)
        report.rows_dropped += dropped
        if len(trimmed):
            trimmed.attrs.update(df.attrs)
            cleaned.append(trimmed)
    if not cleaned:
        return None, report

    ref_idx = _pick_reference(cleaned)
    reference = cleaned[ref_idx]
    ref_cols = [(c, signature(reference[c])) for c in reference.columns]
    out_columns = [_repair_split_words(c) if not is_generic(c) else str(c)
                   for c in reference.columns]
    report.reference_page = reference.attrs.get("page")

    pieces = []
    for i, df in enumerate(cleaned):
        label = _part_label(df, i)
        if i == ref_idx:
            piece = df.copy()
            piece.columns = out_columns
        else:
            cand = [(c, signature(df[c])) for c in df.columns]
            mapping = align_columns(ref_cols, cand)
            used = {j for j in mapping if j is not None}
            data = {}
            for pos, j in enumerate(mapping):
                name = out_columns[pos]
                data[name] = (df.iloc[:, j].tolist() if j is not None
                              else [None] * len(df))
            piece = pd.DataFrame(data)
            for j, column in enumerate(df.columns):
                if j not in used:
                    # Kept, not merged, not dropped: an unmatched column is a
                    # question for the user, not something to silently fold
                    # into whichever column happened to be adjacent.
                    spare = str(column) if not is_generic(column) else f"unmatched_{j}"
                    piece[spare] = df.iloc[:, j].tolist()
                    report.unmatched.append({"part": label, "column": str(column)})

        piece.attrs.update(df.attrs)
        report.parts.append({"part": label, "rows": int(len(piece)),
                             "page": df.attrs.get("page")})
        if provenance:
            piece["_source_file"] = df.attrs.get("source_file") or ""
            piece["_source_page"] = df.attrs.get("page")
        pieces.append(piece)

    out = repair_numeric_cells(pd.concat(pieces, ignore_index=True, sort=False))
    out.attrs["page"] = reference.attrs.get("page")
    out.attrs["pages"] = [p["page"] for p in report.parts if p["page"] is not None]
    report.rows_out = int(len(out))
    return out, report
