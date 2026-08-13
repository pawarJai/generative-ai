"""Recovering table columns that Docling's table model dropped.

Confirmed on data-file-2.pdf: page 6 carries a 5-column header
(Evaluation Schedules | Item/Category | Consignee/Reporting Officer |
Consignee Address | Quantity) and page 7 carries the 17 body rows — but with
only FOUR columns. The leftmost column, a merged group label reading
Foot Valve / Drain Valves / Globe Valve, is absent from the table model
entirely, and page 7 has exactly one text item ('7 / 55') because the rest of
the page was absorbed into the table region. So the data reached no export,
no query and no answer, and when a user asked for that column the LLM filled
the void with invention.

The text is still in the PDF: docling_parse (already a Docling dependency)
reports 130 textline cells on that page, the first being 'Foot' at x=46.5 —
left of the body's first column, which starts at x=91.8. This module rebuilds
the missing column from those cells by geometry.

Deliberately conservative: it only runs when a recovered header proves a
column is missing (N+1 header names against an N-column body), and it returns
None rather than guessing whenever the geometry is not unambiguous.
"""
from typing import Dict, List, Optional, Tuple

# Parsing a PDF page is far too slow to repeat per get_all_real_tables() call.
_CACHE: Dict[Tuple[str, int, int], Optional[List[str]]] = {}
_PARSERS: Dict[str, object] = {}


def _page_lines(pdf_path: str, page_no: int) -> Tuple[List[tuple], float]:
    """Textline cells of one page as (top, bottom, left, text), TOPLEFT
    coordinates to match Docling's table-cell boxes."""
    from docling_parse.pdf_parser import DoclingPdfParser

    doc = _PARSERS.get(pdf_path)
    if doc is None:
        doc = DoclingPdfParser().load(pdf_path)
        _PARSERS[pdf_path] = doc
    page = doc.get_page(page_no).export_to_dict()
    height = page["dimension"]["rect"]["r_y2"]

    lines = []
    for cell in page.get("textline_cells", []):
        rect = cell.get("rect") or {}
        text = (cell.get("text") or "").strip()
        if not text:
            continue
        left = min(rect["r_x0"], rect["r_x3"])
        right = max(rect["r_x1"], rect["r_x2"])
        top = height - max(rect["r_y2"], rect["r_y3"])
        bottom = height - min(rect["r_y0"], rect["r_y1"])
        lines.append((top, bottom, left, right, text))
    return lines, height


def _cluster(lines: List[tuple]) -> List[Tuple[float, float, str]]:
    """Group wrapped lines into one label each, as (top, bottom, text).

    A label like "Globe Valve" is two textlines; without clustering they land
    in two different table rows and the column comes out as 'Globe' then
    'Valve'. Lines closer than about one line-height belong together.
    """
    if not lines:
        return []
    lines = sorted(lines)
    blocks, current = [], [lines[0]]
    for line in lines[1:]:
        previous = current[-1]
        line_height = previous[1] - previous[0]
        if line[0] - previous[1] <= max(4.0, line_height * 0.8):
            current.append(line)
        else:
            blocks.append(current)
            current = [line]
    blocks.append(current)
    return [(b[0][0], b[-1][1], " ".join(x[4] for x in b)) for b in blocks]


def _geometry(pdf_path: str, page_no: int, table):
    """Row bands and left-gap text blocks for one page's table.

    Returns (rows, blocks) in TOPLEFT page coordinates, or None when the
    table has no usable gap to the left of its body — meaning there is
    nowhere a dropped column could have lived.
    """
    try:
        cells = [c for c in table.data.table_cells
                 if getattr(c, "start_col_offset_idx", None) == 0 and c.bbox]
        if not cells:
            return None
        body_left = min(c.bbox.l for c in cells)
        table_left = table.prov[0].bbox.l
    except (AttributeError, IndexError, TypeError, ValueError):
        return None

    if body_left - table_left < 10:
        return None

    bands: Dict[int, List[float]] = {}
    for c in cells:
        idx = c.start_row_offset_idx
        if idx in bands:
            bands[idx] = [min(bands[idx][0], c.bbox.t), max(bands[idx][1], c.bbox.b)]
        else:
            bands[idx] = [c.bbox.t, c.bbox.b]
    rows = [tuple(bands[i]) for i in sorted(bands)]
    if not rows:
        return None

    try:
        lines, _ = _page_lines(pdf_path, page_no)
    except Exception as e:                      # noqa: BLE001 - third-party parser
        print(f"[column_recovery] page parse failed {pdf_path} p{page_no}: {e}")
        return None

    in_gap = [ln for ln in lines
              if ln[3] <= body_left - 2 and ln[2] >= table_left - 2]
    return rows, _cluster(in_gap)


def _spans_from_centres(centres: List[float], top: float, bottom: float,
                        tolerance: float) -> Optional[List[Tuple[float, float]]]:
    """Row spans for merged labels whose text is vertically centred.

    A merged group-label cell is drawn once, centred over all the rows it
    covers, so its text sits in the MIDDLE of its group — not at the top.
    Assigning each label to the row it visually lands on, then carrying it
    downward, therefore starts every group about half a group too late. That
    is what put 'Globe Valve' on rows the tender lists as Gate Valve.

    Centring makes the spans solvable: if a label's centre is c and its group
    starts at s, the group ends at 2c - s, and the next starts there. Solve
    forward from the top of the body; if the last group does not land on the
    bottom of the body, the first group must have begun on the previous page,
    so solve backward from the bottom instead. Returns None when neither
    direction produces monotone spans covering the body — better no column
    than a confidently mislabelled one.
    """
    if not centres:
        return None

    def forward() -> Optional[List[Tuple[float, float]]]:
        spans, edge = [], top
        for c in centres:
            far = 2 * c - edge
            if far < edge - 1e-6:
                return None
            spans.append((edge, far))
            edge = far
        # The final group may run past the bottom when it continues onto a
        # page outside this run — that is expected, and still covers every
        # row here. Falling short of the bottom is not.
        return spans if edge >= bottom - tolerance else None

    def backward() -> Optional[List[Tuple[float, float]]]:
        spans, edge = [], bottom
        for c in reversed(centres):
            near = 2 * c - edge
            if near > edge + 1e-6:
                return None
            spans.append((near, edge))
            edge = near
        spans.reverse()
        return spans

    def partial() -> Optional[List[Tuple[float, float]]]:
        """Label what the labels actually cover and stop.

        A page whose last group continues onto the next page has no label for
        that group at all — page 7 shows three labels for four groups. Rather
        than stretching the last label down to the page edge, which is how the
        tender's Gate Valve rows came out as 'Globe Valve', those rows are
        left blank for the caller to fill from the wider context or report as
        unknown.
        """
        spans, edge = [], top
        for c in centres:
            far = 2 * c - edge
            if far < edge - 1e-6:
                return None
            spans.append((edge, far))
            edge = far
        return spans or None

    return forward() or backward() or partial()


def recover_label_column(pdf_path: str, pages: List[Tuple[int, object]],
                         carry_in: str = "") -> Optional[Dict[int, List[str]]]:
    """Rebuild a dropped merged-label column across every page of one table.

    Solving page by page cannot work, and the measurements say why. On page 7
    of the tender the labels sit at y=77.5, 120.3 and 398.5, while the true
    group centres are 79.1, 120.4 and 397.1 — the label is drawn once,
    vertically centred over the rows it covers. But page 7 has FOUR groups and
    only THREE labels: the Gate Valve group starts on page 7 and finishes on
    page 8, so its single label is centred over the whole run and lands on
    page 8. Read page 8 alone and that label looks like it belongs to page 8's
    second row.

    So the pages are laid end to end into one coordinate line, and the groups
    are solved along it: a group centred at c that starts at s ends at 2c - s,
    and the next starts there. On this document that reproduces the true group
    boundary at y=644.4 to within 0.0pt, and every group total then matches
    the tender's own RFQ-Authorities summary sheet exactly.

    Returns {page_no: [label per body row]}, or None if the geometry does not
    resolve — a wrong label is worse than a blank one, because it would be
    exported and quoted as though the document said it.
    """
    measured, offset = [], 0.0
    for page_no, table in pages:
        geo = _geometry(pdf_path, page_no, table)
        if geo is None:
            return None
        rows, blocks = geo
        top = min(r[0] for r in rows)
        bottom = max(r[1] for r in rows)
        measured.append({"page": page_no, "rows": rows, "blocks": blocks,
                         "top": top, "offset": offset})
        offset += bottom - top
    total = offset
    if not measured:
        return None

    row_spans, row_pages = [], []
    centres, labels = [], []
    for page in measured:
        for top, bottom in page["rows"]:
            row_spans.append((top - page["top"] + page["offset"],
                              bottom - page["top"] + page["offset"]))
            row_pages.append(page["page"])
        for btop, bbottom, text in page["blocks"]:
            centres.append((btop + bbottom) / 2 - page["top"] + page["offset"])
            labels.append(text)

    if not centres or not row_spans:
        return None
    order = sorted(range(len(centres)), key=lambda i: centres[i])
    centres = [centres[i] for i in order]
    labels = [labels[i] for i in order]

    height = total / max(1, len(row_spans))
    spans = _spans_from_centres(centres, 0.0, total, height)
    if spans is None:
        return None

    out: Dict[int, List[str]] = {page["page"]: [] for page in measured}
    for (top, bottom), page_no in zip(row_spans, row_pages):
        centre = (top + bottom) / 2
        label = carry_in
        for (s_top, s_bottom), text in zip(spans, labels):
            if s_top - 1e-6 <= centre <= s_bottom + 1e-6:
                label = text
                break
        out[page_no].append(label)
    return out


def recover_left_column(pdf_path: str, page_no: int, table,
                        carry_in: str = "") -> Optional[List[str]]:
    """Values for a leftmost column Docling dropped, one per body row.

    Single-page entry point, kept because get_all_real_tables() rebuilds each
    table independently. When the table runs over several pages the caller
    should use recover_label_column() instead: a group whose label is centred
    across a page break cannot be resolved from one page's geometry alone.

    Returns None when the column cannot be rebuilt unambiguously — a wrong
    column is far worse than a missing one, since it would be exported and
    quoted as though it came from the document.
    """
    try:
        rows = sum(1 for c in table.data.table_cells
                   if getattr(c, "start_col_offset_idx", None) == 0 and c.bbox)
    except (AttributeError, TypeError):
        return None

    # Parsing a PDF page is far too slow to repeat per get_all_real_tables().
    key = (pdf_path, page_no, rows, carry_in)
    if key in _CACHE:
        return _CACHE[key]

    recovered = recover_label_column(pdf_path, [(page_no, table)], carry_in)
    result = recovered.get(page_no) if recovered else None
    result = result if result and any(result) else None
    _CACHE[key] = result
    return result
