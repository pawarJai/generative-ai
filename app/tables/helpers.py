"""
Table access layer — every other module reads tables through these functions,
never by touching app.state.DOCLING_DOCS / TABULAR_TABLES directly. Keeping
one access point means a future storage-backend swap (e.g. moving cached
tables to Postgres) only requires changing this file.
"""
import re
from typing import List, Dict, Optional
import pandas as pd
from app import state


def dedupe_columns(cols) -> List[str]:
    """Guarantees unique column names per table — required before any
    pd.concat, since duplicate/blank headers (common in real PDFs/xlsx)
    otherwise crash with InvalidIndexError."""
    seen: Dict[str, int] = {}
    out = []
    for i, c in enumerate(cols):
        name = str(c).strip()
        if name.lower().startswith("unnamed:"):
            name = ""
        name = name or f"col_{i}"
        
        base_name = name
        while name in seen:
            seen[base_name] += 1
            name = f"{base_name}_{seen[base_name]}"
        seen[name] = 0
        out.append(name)
    return out


def looks_like_real_table(df: pd.DataFrame) -> bool:
    """Content-based junk filter for Docling's occasional false-positive
    'table' detections on glossary/index pages (a column of sorted single
    words, no numbers)."""
    sample = df.astype(str).values.flatten()
    if len(sample) == 0:
        return False
    single_alpha_words = sum(1 for v in sample if v.strip().isalpha() and " " not in v.strip())
    has_digit = any(any(ch.isdigit() for ch in v) for v in sample)
    if single_alpha_words / len(sample) > 0.8 and not has_digit:
        return False
    return True


def get_page_markdown(file_id: str, page_no: int) -> str:
    if file_id not in state.DOCLING_DOCS:
        return ""
    doc = state.DOCLING_DOCS[file_id]
    return doc.export_to_markdown(page_no=page_no)


def get_page_count(file_id: str) -> Optional[int]:
    """Number of pages in a paginated document, or None if not applicable.
    Lets callers distinguish "this page has no text" from "this document
    doesn't have that many pages" — two very different answers."""
    doc = state.DOCLING_DOCS.get(file_id)
    if doc is None:
        return None
    try:
        return len(doc.pages)
    except Exception:
        return None


def get_tables_on_page(file_id: str, page_no: int) -> List[pd.DataFrame]:
    doc = state.DOCLING_DOCS[file_id]
    out = []
    for t in doc.tables:
        if t.prov and t.prov[0].page_no == page_no:
            out.append(t.export_to_dataframe())
    return out


def get_table_grids_on_page(file_id: str, page_no: int) -> List[List[List[str]]]:
    """Raw cell grids (list of rows of cell text) for every table on a page.

    Deliberately NOT export_to_dataframe(), which is lossy for the two table
    shapes this document family actually uses:

      * A vertical key/value spec table exports with integer column names and
        the real header stranded in row 0, so the caller cannot tell a header
        from data.
      * A 2-row matrix table (header row + single value row, e.g. "Rating |
        Size (NB) | Size (NB)" over "PN10 | 40 | 50") exports as an EMPTY
        DataFrame with the values fused into de-duplicated column names
        ("Size of the valves (NB).40"). The rows are simply gone.

    The grid keeps merged-cell blanks exactly where Docling put them, which is
    what makes the band/label/value reconstruction in
    export_document_sections possible at all.
    """
    doc = state.DOCLING_DOCS.get(file_id)
    if doc is None:
        return []
    out = []
    for t in doc.tables:
        if not (t.prov and t.prov[0].page_no == page_no):
            continue
        try:
            grid = [[(c.text or "").strip() for c in row] for row in t.data.grid]
        except Exception:
            continue
        if grid:
            out.append(grid)
    return out


def _columns_look_like_data(columns) -> bool:
    """Docling's export_to_dataframe() always treats row 0 of the source
    table as the header. For tables with no real header row, this silently
    turns a genuine data row into column names — the row itself vanishes
    from the data, and every downstream consumer (exports, code_exec) gets
    a corrupted schema and produces inconsistent results per call, since
    there's no real structure to reason about.

    A reliable, cheap signal: a genuine header is never a bare number
    (no one names a column "72") — a data row commonly is, e.g. a
    quantity or spec value in the last column."""
    numeric = re.compile(r"^-?\d+(\.\d+)?$")
    return any(numeric.match(str(c).strip()) for c in columns)


def _columns_are_positional(columns) -> bool:
    """True when the 'columns' are Docling's own positional labels — 0,1,2,…

    Docling emits these whenever a table has no column_header cells at all
    (page 8 of data-file-2, while pages 7/9/10 flag row 0). They are indices,
    not a misdetected header, and _columns_look_like_data() calls them
    numeric. Restoring them 'back' into the body wrote the literal row
    ['0','1','2','3'] into outputs/f1-66.xlsx and inflated every row count by
    one, so the export reported 65 rows for a 64-row table."""
    try:
        values = [int(str(c).strip()) for c in columns]
    except (TypeError, ValueError):
        return False
    return values == list(range(len(values)))


# Words that are legitimately short and lowercase inside a real header
# ("Cost of Goods", "Rate per Unit") — never merged into their neighbour by
# the split-word repair below.
_HEADER_STOPWORDS = {
    "of", "the", "and", "in", "for", "to", "per", "no", "by", "on", "at",
    "or", "a", "an", "as", "vs", "id", "qty", "up", "is", "if", "n/a",
}


def _repair_split_words(text: str) -> str:
    """PDF header cells often arrive with words broken mid-token by the
    column's narrow width — 'Q ua nti ty', 'Evaluat ion Schedu les'. Rejoin
    a fragment onto the previous token when it is short, all-lowercase and
    not a real short word, which leaves genuine headers like 'Cost of Goods'
    untouched."""
    tokens = str(text).split()
    if not tokens:
        return str(text).strip()
    out = [tokens[0]]
    for tok in tokens[1:]:
        if (len(tok) <= 3 and tok.isalpha() and tok.islower()
                and tok.lower() not in _HEADER_STOPWORDS):
            out[-1] += tok
        else:
            out.append(tok)
    return re.sub(r"\s+([/,)])", r"\1", " ".join(out)).strip()


def _is_header_only_table(df: pd.DataFrame) -> bool:
    """A table Docling emitted that carries the header row but no data —
    what happens when a table's header and body land on opposite sides of a
    page break. These are dropped by the min_rows filter, so their column
    names are the only surviving record of the real schema."""
    if len(df) > 0 or df.shape[1] < 2:
        return False
    vals = [str(c).strip() for c in df.columns]
    if any(not v or v.lower() in ("nan", "none") for v in vals):
        return False
    if len(set(vals)) < 2:
        return False
    numeric = re.compile(r"^-?\d+(\.\d+)?$")
    return not any(numeric.match(v) for v in vals)


def _columns_are_generic(cols) -> bool:
    return all(re.fullmatch(r"col_\d+", str(c).strip()) for c in cols)


def _is_generic(col) -> bool:
    return bool(re.fullmatch(r"col_\d+", str(col).strip()))


def _same_name(a, b) -> bool:
    return re.sub(r"\s+", " ", str(a)).strip().casefold() == \
        re.sub(r"\s+", " ", str(b)).strip().casefold()


def _continues_table(df: pd.DataFrame, columns: List) -> bool:
    """Whether ``df`` is the next page of a table whose columns are ``columns``.

    Originally this asked only "are this page's columns still col_0, col_1…",
    on the reasoning that a page starting a NEW table brings its own header.
    Header recovery then began naming continuation pages from the page before
    them — which is right, and which silently broke this test: pages 9 and 10
    of the tender's 64-row schedule now arrive carrying the real column names,
    so the span stopped at page 8 and the export was 27 rows. A page continues
    the table when it is the same width and every column is either still
    unnamed or is the same name the anchor uses in that position.
    """
    width = df.shape[1]
    if width not in (len(columns), len(columns) - 1):
        return False
    if _columns_are_generic(df.columns):
        return True            # unnamed continuation, possibly a column short
    if width != len(columns):
        return False
    return all(_is_generic(c) or _same_name(c, ref)
               for c, ref in zip(df.columns, columns))


def _restore_dropped_column(df: pd.DataFrame, name: str, file_id, page, table,
                            carry_in: str = ""):
    """Put back a leftmost column Docling lost, rebuilt from the PDF text.
    Returns None unless it produced exactly one value per body row."""
    if not file_id or table is None:
        return None
    try:
        from app.persistence import get_file
        from app.ingestion.column_recovery import recover_left_column
        record = get_file(file_id)
        if not record:
            return None
        values = recover_left_column(record["path"], page, table, carry_in)
    except Exception as e:                      # noqa: BLE001 - recovery is best-effort
        print(f"[_restore_dropped_column] {file_id} p{page}: {e}")
        return None

    if not values or len(values) != len(df):
        return None
    out = df.copy()
    out.insert(0, _repair_split_words(name), values)
    out.attrs["recovered_column"] = name
    return out


def _column_x_ranges(table) -> Dict[int, tuple]:
    """Left/right extent of every column of a Docling table."""
    spans: Dict[int, list] = {}
    for c in getattr(getattr(table, "data", None), "table_cells", []) or []:
        if c.bbox is None:
            continue
        i = c.start_col_offset_idx
        cur = spans.setdefault(i, [c.bbox.l, c.bbox.r])
        cur[0] = min(cur[0], c.bbox.l)
        cur[1] = max(cur[1], c.bbox.r)
    return {i: (v[0], v[1]) for i, v in spans.items()}


def _header_by_x_overlap(body_table, header_table, n_cols: int):
    """Column names taken from a header-only table by HORIZONTAL POSITION.

    The existing recovery matches a header to a body by counting columns, and
    silently gives up when the counts differ. They differ constantly: page 6
    of this tender carries the header as a 4-column table
    ('Evaluat ion Item/Category', 'Consignee /Reporting Officer',
    'Consignee Address', 'Q ua nti') while the body on page 7 has 5 columns,
    because Docling merged the two leftmost header cells. So the real names
    were sitting in the document and were thrown away, and every export came
    out as col_0..col_4 — which the model then "fixed" by inventing names.

    Position does not care about counts. Each header cell is assigned to the
    body column it overlaps most, which puts 'Schedu'/'les' (x 46-84) on the
    body's first column (x 41-78) and 'Evaluat ion Item/Category' (x 46-166)
    on its second (x 91-346), exactly as the page is laid out. Several cells
    landing on one column are the wrapped lines of one label and are rejoined
    in row order — a fragment starting lower-case continues the word above it
    ('Schedu' + 'les' -> 'Schedules'), anything else is a new word.
    """
    header_cells = [c for c in
                    (getattr(getattr(header_table, "data", None), "table_cells", []) or [])
                    if c.bbox is not None and (c.text or "").strip()]
    body_spans = _column_x_ranges(body_table)
    if not header_cells or len(body_spans) < n_cols:
        return None

    buckets: Dict[int, list] = {}
    for cell in header_cells:
        best_col, best_overlap = None, 0.0
        for i in range(n_cols):
            lo, hi = body_spans.get(i, (0, 0))
            overlap = min(cell.bbox.r, hi) - max(cell.bbox.l, lo)
            if overlap > best_overlap:
                best_col, best_overlap = i, overlap
        if best_col is not None:
            buckets.setdefault(best_col, []).append(cell)

    if len(buckets) < n_cols:
        return None

    names = []
    for i in range(n_cols):
        parts = sorted(buckets[i], key=lambda c: (c.start_row_offset_idx,
                                                  c.bbox.t))
        text = ""
        for part in parts:
            piece = (part.text or "").strip()
            if not piece:
                continue
            if not text:
                text = piece
            elif piece[:1].islower():
                text += piece          # wrapped word: Schedu + les
            else:
                text += " " + piece
        names.append(_repair_split_words(text) if text else f"col_{i}")
    return names


def _recover_header(df: pd.DataFrame, page, candidates: Dict[int, List[List[str]]],
                    file_id: str = None, table=None) -> pd.DataFrame:
    """Give a headerless table (col_0, col_1, …) the real column names taken
    from a header-only table on the same page or the page just before it.
    Nothing is invented here — the names come out of the document."""
    if page is None or not _columns_are_generic(df.columns):
        return df
    n = df.shape[1]

    # Position first: it recovers names the count-based match below throws
    # away whenever Docling merges or splits a header cell, which is the
    # normal case rather than the exception.
    if table is not None and file_id:
        doc = state.DOCLING_DOCS.get(file_id)
        for p in (page, page - 1):
            for other in (doc.tables if doc is not None else []):
                if not (other.prov and other.prov[0].page_no == p):
                    continue
                if other is table:
                    continue
                try:
                    if not _is_header_only_table(other.export_to_dataframe(doc)):
                        continue
                except Exception:
                    continue
                names = _header_by_x_overlap(table, other, n)
                if names:
                    df = df.copy()
                    df.columns = dedupe_columns(names)
                    df.attrs["header_recovered_from_page"] = p
                    return df

    for p in (page, page - 1):
        for header in candidates.get(p, []):
            if len(header) == n:
                chosen = header
            elif len(header) == n + 1:
                # One extra header name against an N-column body is PROOF a
                # column was dropped — a row-spanning/merged label column,
                # always leftmost in practice. Try to rebuild it from the
                # PDF's own text before falling back to renaming the columns
                # that survived; dropping it silently is what let the agent
                # answer "Evaluation Schedules" out of thin air.
                restored = _restore_dropped_column(df, header[0], file_id,
                                                   page, table)
                if restored is not None:
                    df = restored
                    n = df.shape[1]
                    chosen = header
                else:
                    chosen = header[1:]
            else:
                continue
            df = df.copy()
            df.columns = dedupe_columns([_repair_split_words(c) for c in chosen])
            df.attrs["header_recovered_from_page"] = p
            return df
    return df


def get_all_real_tables(file_id: str, min_cols: int = 3, min_rows: int = 1) -> List[pd.DataFrame]:
    """All real tables for a file — native Docling tables + OCR-derived +
    directly-loaded tabular tables, whichever apply."""
    if state.FILE_KIND.get(file_id) == "tabular":
        return state.TABULAR_TABLES.get(file_id, [])

    if file_id not in state.DOCLING_DOCS:
        return []

    doc = state.DOCLING_DOCS[file_id]

    # Header-only tables are collected before the main pass so a body table
    # can adopt the header that was split away from it by a page break.
    header_candidates: Dict[int, List[List[str]]] = {}
    for t in doc.tables:
        raw = t.export_to_dataframe(doc)
        if _is_header_only_table(raw):
            pg = t.prov[0].page_no if t.prov else None
            if pg is not None:
                header_candidates.setdefault(pg, []).append(
                    [str(c).strip() for c in raw.columns])

    dfs = []
    for t in doc.tables:
        df = t.export_to_dataframe(doc)
        if _columns_are_positional(df.columns):
            # Nothing to put back — these were never a header row.
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
        elif _columns_look_like_data(df.columns):
            # Put the misdetected "header" back as row 0, and use generic
            # names — matches the fallback naming already used elsewhere
            # (tabular_ingest.py) for headerless data.
            restored = pd.DataFrame([list(df.columns)], columns=range(len(df.columns)))
            df.columns = range(len(df.columns))
            df = pd.concat([restored, df], ignore_index=True)
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
        df.columns = dedupe_columns(df.columns)
        if df.shape[1] >= min_cols and df.shape[0] >= min_rows and looks_like_real_table(df):
            page = t.prov[0].page_no if t.prov else None
            df = _expand_merged_label_column(t, df)
            df = _recover_header(df, page, header_candidates, file_id, t)
            df.attrs["page"] = page
            dfs.append(df)

    docling_pages = {df.attrs.get("page") for df in dfs if df.attrs.get("page") is not None}

    for df in state.TABULAR_TABLES.get(file_id, []):
        if df.shape[1] >= min_cols and df.shape[0] >= min_rows:
            if df.attrs.get("page") in docling_pages and df.attrs.get("source") == "ocr":
                continue
            dfs.append(df)
    return dfs


def _expand_merged_label_column(table, df: pd.DataFrame) -> pd.DataFrame:
    """Repeat a merged group-label down every row it actually spans.

    A vertically-merged label cell ("Globe Valve" covering twelve size rows)
    is NOT reported by Docling as a span — every cell comes back with
    row_span=1. Worse, the label's wrapped lines are emitted as SEPARATE
    one-row cells at whatever grid rows their text happens to sit on, so
    "Globe Valve" arrived as 'Globe' on row 7 and 'Valve' on row 8 with the
    twelve rows it covers left blank. The user sees a bucket column that is
    empty almost everywhere and split across two rows.

    A plain forward-fill is the wrong repair and was rejected three times:
    it stamps the last seen label onto every following blank, which put
    "Globe Valve" on rows the document lists as Gate, Swing Check and Ball.

    What IS recoverable is geometry. A merged cell centres its text
    vertically, so for the run of rows [s..e] that the label really covers,
    (top[s] + bottom[e]) / 2 lands on the text's own centre. Each label is
    therefore given the LARGEST run of rows that
      * contains the rows its own text overlaps,
      * does not reach into a neighbouring label's text rows, and
      * still centres on that text within a fraction of a row height.
    Rows outside every label's run are left blank for the cross-page carry
    to resolve — they belong to a group whose label sits on another page.

    Returns df unchanged when the leftmost column is not a sparse label
    column (e.g. pages where Docling dropped it entirely, or where column 0
    is ordinary per-row data).
    """
    try:
        cells = [c for c in table.data.table_cells
                 if c.start_col_offset_idx == 0 and (c.text or "").strip()
                 and c.bbox is not None]
        n_rows = len(df)
        if not cells or n_rows < 3:
            return df
        # Sparse == a merged label column. A fully populated column 0 is
        # ordinary data and must not be touched.
        if len({c.start_row_offset_idx for c in cells}) > n_rows * 0.6:
            return df

        # Vertical extent of each grid row, measured from the OTHER columns
        # so a missing label cell cannot distort it.
        tops: Dict[int, float] = {}
        bots: Dict[int, float] = {}
        for c in table.data.table_cells:
            if c.start_col_offset_idx == 0 or c.bbox is None:
                continue
            r = c.start_row_offset_idx
            if r >= n_rows:
                continue
            tops[r] = min(tops.get(r, c.bbox.t), c.bbox.t)
            bots[r] = max(bots.get(r, c.bbox.b), c.bbox.b)
        rows = sorted(set(tops) & set(bots))
        if len(rows) < 3:
            return df
        heights = sorted(bots[r] - tops[r] for r in rows)
        row_height = heights[len(heights) // 2]
        if row_height <= 0:
            return df
        tolerance = 0.4 * row_height

        # Fragments are the wrapped lines of ONE label only when they are
        # also vertically CONTIGUOUS. Grid-row adjacency alone is not enough:
        # 'Foot Valve' (rows 0) and 'Drain Valves' (row 1) are two separate
        # one-row groups sitting on consecutive rows, and joining them
        # produced a bogus "Foot Valve Drain Valves" label. Consecutive lines
        # of one wrapped label nearly touch ('Globe' ends at 398, 'Valve'
        # starts at 399); separate labels do not (79 -> 106).
        cells.sort(key=lambda c: c.start_row_offset_idx)
        line_gap = 0.5 * row_height
        groups: List[list] = []
        for c in cells:
            if (groups
                    and c.start_row_offset_idx
                    - groups[-1][-1].start_row_offset_idx <= 1
                    and c.bbox.t - groups[-1][-1].bbox.b < line_gap):
                groups[-1].append(c)
            else:
                groups.append([c])

        labels = []
        for grp in groups:
            text = " ".join((c.text or "").strip() for c in grp).strip()
            top = min(c.bbox.t for c in grp)
            bot = max(c.bbox.b for c in grp)
            covered = [r for r in rows if bots[r] > top and tops[r] < bot]
            if not covered:
                covered = [min(rows, key=lambda r: abs((tops[r] + bots[r]) / 2
                                                       - (top + bot) / 2))]
            labels.append({"text": text, "centre": (top + bot) / 2,
                           "rows": covered})

        assigned = [""] * n_rows
        for i, lab in enumerate(labels):
            lo = (max(labels[i - 1]["rows"]) + 1) if i else rows[0]
            hi = (min(labels[i + 1]["rows"]) - 1) if i + 1 < len(labels) else rows[-1]
            must_lo, must_hi = min(lab["rows"]), max(lab["rows"])
            lo, hi = min(lo, must_lo), max(hi, must_hi)

            best = None
            for s in range(lo, must_lo + 1):
                for e in range(must_hi, hi + 1):
                    if s not in tops or e not in bots:
                        continue
                    centre = (tops[s] + bots[e]) / 2
                    if abs(centre - lab["centre"]) > tolerance:
                        continue
                    size = e - s + 1
                    if best is None or size > best[0]:
                        best = (size, s, e)
            if best is None:
                best = (len(lab["rows"]), must_lo, must_hi)
            for r in range(best[1], best[2] + 1):
                if 0 <= r < n_rows:
                    assigned[r] = lab["text"]

        if not any(assigned):
            return df
        out = df.copy()
        out.iloc[:, 0] = assigned
        out.attrs["merged_labels_expanded"] = True
        return out
    except Exception:
        # Geometry is a bonus, never a reason to lose the table.
        return df


def _prepend_column(df: pd.DataFrame, name, values,
                    reference_columns) -> pd.DataFrame:
    """Add a leading column, even when `name` is already taken.

    pd.DataFrame.insert() refuses a duplicate label outright — "cannot insert
    col_0, already exists". That is not an edge case: when Docling recovers no
    header at all, EVERY page is named col_0..col_N, so the reference page's
    first column name always collides with the short page's own first column.
    Confirmed crash exporting pages 6-10 of data-file-2, whose reference page
    has 5 generic columns and whose pages 8 and 9 have 4. The same code ran
    fine on data-file-1 only because its headers were real words ('Sl No'),
    which happened not to clash with 'col_0'.

    The column goes in under a private placeholder that cannot collide, then
    the frame takes the reference page's header when the widths now agree —
    these are pages of ONE table, so sharing its header is both correct and
    what lets assemble() line them up by name.
    """
    out = df.copy()
    placeholder = "__label__"
    while placeholder in out.columns:
        placeholder += "_"
    out.insert(0, placeholder, values)

    if len(out.columns) == len(reference_columns):
        out.columns = list(reference_columns)
    else:
        cols = list(out.columns)
        cols[0] = name
        out.columns = dedupe_columns(cols)
    return out


def _restore_dropped_labels(file_id: str, frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Rebuild the merged group-label column on every page that lost it.

    Docling drops the row-spanning leftmost column on some pages of a
    multi-page table and keeps it on others (pages 8 and 9 of the tender lose
    it, pages 7 and 10 keep it). Padding those pages with blanks loses real
    data; carrying the previous page's label down the whole page was worse —
    it stamped "Globe Valve" on 34 rows the PDF labels Gate, Swing Check and
    Ball. So each page is rebuilt from its own PDF text, and the carried label
    only fills the rows above that page's first real label, which is what a
    merged label spanning a page break actually means.
    """
    from app.tables.assembly import is_generic, looks_like_label_column

    if not frames:
        return frames
    ordered = sorted(frames, key=lambda d: (d.attrs.get("page") or 0))
    reference = max(ordered, key=lambda d: (
        sum(0 if is_generic(c) else 1 for c in d.columns), d.shape[1], d.shape[0]))
    width = reference.shape[1]
    first = str(reference.columns[0]) if width else ""

    # Pages whose label column Docling dropped: either already rebuilt one page
    # at a time (attrs set by _recover_header) or still one column short.
    pending = [df for df in ordered
               if df.attrs.get("recovered_column") or df.shape[1] == width - 1]
    solved = _solve_labels_across_pages(file_id, pending)

    out = []
    for df in ordered:
        page = df.attrs.get("page")
        values = solved.get(page) if solved else None
        if values is not None and len(values) == len(df):
            if df.shape[1] == width - 1:
                df = _prepend_column(df, first, values, reference.columns)
            else:
                df = df.copy()
                df.iloc[:, 0] = values
        elif df.shape[1] == width - 1 and width > 1:
            df = _prepend_column(df, first, [None] * len(df), reference.columns)

        if df.shape[1] == width and width:
            labels = df.iloc[:, 0]
            # 'Butterfl y Valves' — the same narrow-column word splitting that
            # mangles headers also mangles these labels, and it reached the
            # exported file verbatim.
            if looks_like_label_column(labels):
                df = df.copy()
                df.iloc[:, 0] = [_repair_split_words(v) if isinstance(v, str) and v.strip()
                                 else v for v in labels.tolist()]
        out.append(df)

    return _carry_labels_over_page_breaks(out, width)


def _carry_labels_over_page_breaks(frames: List[pd.DataFrame], width: int) -> List[pd.DataFrame]:
    """Fill only the blank labels at the TOP of a page, from the page before.

    A group that crosses a page break leaves the first rows of the next page
    with an empty label cell — page 10 of the tender opens with one more Ball
    Valve row before the Butterfly group starts. Those rows, and only those,
    belong to the previous page's last group.

    Deliberately not a plain forward-fill down the whole column: that is what
    stamped 'Globe Valve' onto 34 rows the tender lists as Gate, Swing Check
    and Ball. A blank appearing *after* a label on the same page is left
    blank, because there we have no evidence at all.
    """
    def blank(value) -> bool:
        return value is None or str(value).strip() in ("", "nan", "None")

    carry = ""
    out = []
    for df in frames:
        if width and df.shape[1] == width and len(df):
            values = df.iloc[:, 0].tolist()
            if carry and blank(values[0]):
                df = df.copy()
                for i, value in enumerate(values):
                    if not blank(value):
                        break
                    df.iloc[i, 0] = carry
            seen = [str(v).strip() for v in df.iloc[:, 0].tolist() if not blank(v)]
            if seen:
                carry = seen[-1]
        out.append(df)
    return _carry_labels_backwards(out, width, blank)


def _carry_labels_backwards(frames: List[pd.DataFrame], width: int,
                            blank) -> List[pd.DataFrame]:
    """Fill only the blank labels at the BOTTOM of a page, from the page after.

    The mirror of the forward carry above, and needed for the same reason.
    A merged label whose text is centred inside its cell puts that text on
    whichever page holds the middle of the group — so when a group straddles
    a page break its label can land on the SECOND page, leaving the tail of
    the first page unlabelled. Confirmed: the Gate Valve group starts on the
    last three rows of page 7 and its label sits on page 8, so those three
    rows came out blank while every other row had its bucket filled.

    Deliberately narrow, for the same reason the forward carry is: only a
    RUN OF BLANKS THAT REACHES THE BOTTOM of a page is filled, and only from
    the very first label on the following page. A blank above a filled cell
    is left alone — there we have no evidence.
    """
    for i in range(len(frames) - 1):
        df = frames[i]
        nxt = frames[i + 1]
        if not (width and df.shape[1] == width and len(df)):
            continue
        if not (nxt.shape[1] == width and len(nxt)):
            continue
        values = df.iloc[:, 0].tolist()
        if not blank(values[-1]):
            continue
        following = [str(v).strip() for v in nxt.iloc[:, 0].tolist()
                     if not blank(v)]
        if not following:
            continue
        label = following[0]
        df = df.copy()
        for j in range(len(values) - 1, -1, -1):
            if not blank(values[j]):
                break
            df.iloc[j, 0] = label
        frames[i] = df
    return frames


def _solve_labels_across_pages(file_id: str, frames: List[pd.DataFrame]) -> Dict:
    """Merged group labels for a whole multi-page table, solved together.

    A merged cell spanning a page break carries one label, centred over the
    entire run, so it is drawn on whichever page holds the middle of that run
    — page 7's Gate Valve label is printed on page 8. Reading each page on its
    own therefore mislabels the boundary rows every time.
    """
    if not frames:
        return {}
    try:
        from app.persistence import get_file
        from app.ingestion.column_recovery import recover_label_column
        record = get_file(file_id)
        if not record:
            return {}
        specs = []
        for df in frames:
            page = df.attrs.get("page")
            table = _docling_table_on(file_id, page, df.shape)
            if page is None or table is None:
                return {}
            specs.append((page, table))
        return recover_label_column(record["path"], specs) or {}
    except Exception as e:                      # noqa: BLE001 - recovery is best-effort
        print(f"[_solve_labels_across_pages] {file_id}: {e}")
        return {}


def assemble_pages(file_id: str, pages: List[int]):
    """One aligned table built from every real table on the given pages.

    Replaces the pd.concat that produced outputs/f1-66.xlsx, where fragments
    with different column counts and no headers were unioned positionally and
    every value from page 8 onward landed under the wrong heading. Returns
    (dataframe, report, pages_used) — the report names anything that could not
    be aligned, so the caller can tell the user instead of shipping a file
    that merely looks like a table.
    """
    from app.tables.assembly import assemble

    all_tables = get_all_real_tables(file_id)
    frames, used = [], []
    for page in pages:
        matches = [t for t in all_tables if t.attrs.get("page") == page]
        if not matches:
            continue
        used.append(page)
        frames.extend(matches)
    if not frames:
        return None, None, []

    df, report = assemble(_restore_dropped_labels(file_id, frames))
    # The rows alone are not the document. Carry the letterhead, the section
    # heading and the provenance line along with them so whatever writes the
    # file can say which tender these 64 valves belong to.
    from app.tables.context import attach
    attach(df, file_id, used)
    return df, report, used


def span_pages(file_id: str, start_page: int) -> List[int]:
    """Which pages the table beginning on ``start_page`` actually runs across.

    Page discovery only — no assembly, no column recovery — so a caller that
    just needs to warn "this table continues onto pages 9 and 10" can ask
    without paying for the join.
    """
    by_page: Dict[int, List[pd.DataFrame]] = {}
    for df in get_all_real_tables(file_id):
        page = df.attrs.get("page")
        if isinstance(page, int):
            by_page.setdefault(page, []).append(df)

    anchor = anchor_page = None
    for candidate in (start_page, start_page + 1):
        for df in by_page.get(candidate, []):
            if anchor is None or df.shape[0] > anchor.shape[0]:
                anchor, anchor_page = df, candidate
        if anchor is not None:
            break
    if anchor is None:
        return []

    columns = list(anchor.columns)
    pages, page = [anchor_page], anchor_page + 1
    while True:
        nxt = None
        for df in by_page.get(page, []):
            if _continues_table(df, columns):
                if nxt is None or df.shape[0] > nxt.shape[0]:
                    nxt = df
        if nxt is None:
            break
        pages.append(page)
        page += 1
    return pages


def get_table_span(file_id: str, start_page: int):
    """The whole logical table that begins on start_page, following it across
    every page it continues onto.

    A long table is emitted by Docling as one table per page: the tender's
    product schedule is pages 7, 8, 9 and 10 (17+17+17+13 rows). Only the
    header page carries column names, so the continuation pages come back as
    col_0…col_3 and look like unrelated tables. Exporting "page 6 and 7"
    therefore returned 17 rows of a 64-row table with no indication the rest
    existed.

    A page continues the previous one when it holds a table of the same width
    (or one narrower, before its dropped left column is restored) whose columns
    are still unnamed, or carry the same names as the anchor — see
    _continues_table. Returns (dataframe, pages_used).
    """
    pages = span_pages(file_id, start_page)
    if not pages:
        return None, []
    anchor_page = pages[0]

    # Discovering which pages the table runs across is separate from joining
    # them: the join itself goes through the aligner, which matches columns by
    # what the values look like rather than by position. Positional joining is
    # what put page 8's item text under 'Evaluation Schedules' and its
    # quantities under 'Consignee Address' in outputs/f1-66.xlsx.
    out, _report, _used = assemble_pages(file_id, pages)
    if out is None:
        return None, []
    out.attrs["page"] = anchor_page
    out.attrs["pages"] = pages
    return out, pages


def _docling_table_on(file_id: str, page_no: int, shape):
    """The raw Docling table object for a page, needed to rebuild a dropped
    column — get_all_real_tables returns DataFrames, which have lost it."""
    doc = state.DOCLING_DOCS.get(file_id)
    if doc is None:
        return None
    for t in doc.tables:
        if t.prov and t.prov[0].page_no == page_no:
            data = getattr(t, "data", None)
            if data is not None and getattr(data, "num_rows", None) == shape[0]:
                return t
    return None


def extraction_report(file_id: str) -> Dict:
    """Where extraction lost or guessed at structure, per table.

    Built because a dropped column was invisible: page 7 of a tender lost its
    merged "Evaluation Schedules" column, nothing reported it, and the agent
    invented values when asked for it. This makes that class of loss
    measurable instead of surfacing only when an answer looks wrong.

    - recovered_column: a dropped column was detected and rebuilt from the PDF
    - missing_column:   proven dropped (header has one name more than the
                        body) but NOT rebuilt — data is still absent
    - unnamed_columns:  columns still called col_N, i.e. no header was found
    """
    tables = get_all_real_tables(file_id)
    rows = []
    for df in tables:
        generic = [c for c in df.columns if re.fullmatch(r"col_\d+", str(c).strip())]
        rows.append({
            "page": df.attrs.get("page"),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "recovered_column": df.attrs.get("recovered_column"),
            "header_recovered_from_page": df.attrs.get("header_recovered_from_page"),
            "unnamed_columns": len(generic),
            "columns": [str(c) for c in df.columns],
        })
    return {
        "file_id": file_id,
        "tables": len(rows),
        "tables_with_recovered_column": sum(1 for r in rows if r["recovered_column"]),
        "tables_with_recovered_header": sum(
            1 for r in rows if r["header_recovered_from_page"]),
        "tables_fully_unnamed": sum(
            1 for r in rows if r["unnamed_columns"] == r["cols"]),
        "detail": rows,
    }


def get_tables_for_scope(file_ids: List[str]) -> List[pd.DataFrame]:
    """Tables from several files at once, each tagged with its source file_id."""
    all_tables = []
    for fid in file_ids:
        for df in get_all_real_tables(fid):
            d = df.copy()
            d.attrs["source_file"] = fid
            all_tables.append(d)
    return all_tables


def resolve_file_scope(prompt: str) -> List[str]:
    """Deterministically parses 'last file' / 'the excel file' / 'all files'
    into concrete file_ids — never left to LLM guessing."""
    p = prompt.lower()
    if not state.FILE_ORDER:
        return []
    if re.search(r"\b(all files?|every file|across (all|every) (file|document)s?|combine all)\b", p):
        return list(state.FILE_ORDER)
    if re.search(r"\b(last|latest|most recent|newest) (file|upload|document|pdf)\b", p):
        return [state.FILE_ORDER[-1]]
    if re.search(r"\b(first|earliest) (file|upload|document|pdf)\b", p):
        return [state.FILE_ORDER[0]]
    # Collect EVERY filename mentioned, not just the first match — a prompt like
    # "combine file-A and file-B" previously returned only file-A because this
    # used to `return` on the first hit.
    named = [fid for fid in state.FILE_ORDER
             if state.FILE_ORIGINAL_NAME.get(fid, "") and
             state.FILE_ORIGINAL_NAME.get(fid, "").lower() in p]
    if named:
        return named
    ext_words = {"excel": (".xlsx", ".xls"), "csv": (".csv",), "pdf": (".pdf",),
                 "powerpoint": (".pptx",), "image": (".png", ".jpg", ".jpeg")}
    for kw, exts in ext_words.items():
        if kw in p:
            matches = [fid for fid in state.FILE_ORDER
                       if state.FILE_ORIGINAL_NAME.get(fid, "").lower().endswith(exts)]
            if matches:
                return matches
    return []
