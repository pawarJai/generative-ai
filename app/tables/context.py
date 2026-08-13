"""Recover the meaning that surrounds a table.

A table exported as bare rows loses the thing that makes it a document: whose
letterhead it is under, which project and revision it belongs to, and which
numbered section of the spec it answers. In the tender this project was built
against, that band is the difference between "64 rows of valves" and "the item
schedule of Cochin Shipyard spec N45310510605002 REV.00 for project NGMV" — and
a quotation priced off the former cannot be traced back to the tender at all.

None of it is hardcoded. The rules here are positional and statistical:

* The band is the top-of-page text of the page being exported, kept verbatim,
  minus any line that appears on that page only. That single test is what keeps
  ``PROJECT -NGMV`` and drops ``Page 3 of 31`` without knowing what either
  means, and it works the same way on a document with a completely different
  layout. Taking the band from the exporting page rather than from a
  document-wide consensus matters on real uploads: this 83-page PDF is a bundle
  of several documents ("Page 3 of 31"), and its letterhead is extractable text
  on only 30 of those pages, so any global majority vote drops the company name
  from the very page that carries it.
* The section is the nearest heading above the table that is not part of that
  repeating band.
* Recurring lines at the *bottom* are the footer. ``UNCLASSIFIED`` on a defence
  tender is a classification marking; silently dropping it from an export is a
  real loss, not tidying.

Docling splits a single visual line into one text item on one page and three on
another (``Spec. No. N45310510605002 REV.00`` versus ``Spec. No.`` +
``N45310510605002`` + ``REV.00``), so fragments are joined by y-position before
anything else happens.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app import state

# A line whose top edge falls in the first 13% of the page height is a
# candidate for the letterhead band; the last 7% for the footer. Measured
# against the tender: the letterhead occupies 818->781pt of an 842pt page
# (top 7%), the section heading below it sits at 744pt (top 12%) and is
# correctly excluded from the band by the recurrence test rather than by
# this cut, which only limits how much of the page is examined.
_BAND_TOP_FRACTION = 0.13
_FOOTER_FRACTION = 0.07

# Two text items belong to the same visual line if their top edges are within
# this many points. Docling reports the three fragments of the spec number at
# 818.83 exactly, but rotated or slightly-baselined text drifts a point or two.
_LINE_TOLERANCE = 3.0

# Two fragments at the same height belong to the same line only if the gap
# between them is smaller than this. Words within a line sit 2-5pt apart; the
# cells of a letterhead grid are tens of points apart.
_GAP_FRACTION = 0.03
_MIN_GAP = 18.0

# A top-of-page line must appear on at least this many pages to count as
# letterhead rather than as content that happens to sit high on one page.
# Two is enough because the line is already taken from the exporting page —
# recurrence is only being used to reject what varies page to page.
_MIN_RECUR_PAGES = 2

# Documents this short cannot support a recurrence test at all, so every
# top-of-page line on the table's own page is kept. Carrying one line too many
# beats dropping the only identification a one-page document has.
_SHORT_DOC_PAGES = 2

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^\w]+")

_HEADING_LABELS = {"section_header", "title", "subtitle-level-1"}


@dataclass(frozen=True)
class _Line:
    """One visual line of text, with its distance from the top of the page."""
    top: float
    left: float
    right: float
    text: str
    label: str


@dataclass
class TableContext:
    """Everything around a table that gives its rows meaning."""
    document: List[str] = field(default_factory=list)
    section: Optional[str] = None
    footer: List[str] = field(default_factory=list)
    source: Optional[str] = None

    def lines(self) -> List[str]:
        """The band as it should be written above the table, in reading order.

        Every entry except ``source`` is verbatim document text. ``source`` is
        ours, and says so, so a reader can always tell which line the document
        asserted and which line we did.
        """
        out = list(self.document)
        if self.section:
            out.append(self.section)
        if self.source:
            out.append(self.source)
        out.extend(self.footer)
        return out

    def styled(self) -> List[Tuple[str, str]]:
        """The band as (text, role) pairs, for exporters that can format.

        Roles are ``title`` (the document's own first line), ``detail``,
        ``heading`` (the section) and ``note`` (our provenance line and any
        footer marking). An exporter that cannot format ignores the role.
        """
        out: List[Tuple[str, str]] = []
        for i, line in enumerate(self.document):
            out.append((line, "title" if i == 0 else "detail"))
        if self.section:
            out.append((self.section, "heading"))
        if self.source:
            out.append((self.source, "note"))
        out.extend((line, "note") for line in self.footer)
        return out

    def __bool__(self) -> bool:
        return bool(self.document or self.section or self.footer or self.source)


def attach(df, file_id: str, pages: Optional[List[int]] = None):
    """Record on a dataframe the context of the table it was built from.

    Carried in ``df.attrs`` so it survives being handed between the assembler
    and an exporter without changing any signature in between.
    """
    try:
        ctx = context_for(file_id, pages, source=describe_source(file_id, pages))
    except Exception:
        # Context is an enrichment. A document whose geometry we cannot read
        # must still export its rows.
        return df
    if ctx:
        df.attrs["context"] = ctx
    return df


def styled_lines(df) -> List[Tuple[str, str]]:
    """The band to write above ``df``, or empty if it has none."""
    ctx = df.attrs.get("context") if hasattr(df, "attrs") else None
    return ctx.styled() if isinstance(ctx, TableContext) else []


def _page_height(doc, page_no: int) -> float:
    page = getattr(doc, "pages", {}).get(page_no)
    size = getattr(page, "size", None)
    height = getattr(size, "height", None)
    return float(height) if height else 842.0


def _page_width(doc, page_no: int) -> float:
    page = getattr(doc, "pages", {}).get(page_no)
    size = getattr(page, "size", None)
    width = getattr(size, "width", None)
    return float(width) if width else 595.0


def _from_top(bbox, height: float) -> float:
    """Distance from the top of the page, whichever origin Docling used."""
    origin = str(getattr(bbox, "coord_origin", "BOTTOMLEFT"))
    if "TOPLEFT" in origin.upper():
        return float(bbox.t)
    return height - float(bbox.t)


def _text_items(doc) -> List[tuple]:
    """(page_no, top, left, text, label) for every text item that has a page."""
    out = []
    for item in getattr(doc, "texts", []) or []:
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        prov = getattr(item, "prov", None) or []
        if not prov:
            continue
        page_no = getattr(prov[0], "page_no", None)
        bbox = getattr(prov[0], "bbox", None)
        if page_no is None or bbox is None:
            continue
        height = _page_height(doc, page_no)
        label = str(getattr(item, "label", "") or "").split(".")[-1].lower()
        out.append((page_no, _from_top(bbox, height), float(bbox.l),
                    float(bbox.r), text, label))
    return out


def _merge_into_lines(items: List[tuple], page_width: float) -> List[_Line]:
    """Join text fragments that share a visual line, left to right.

    Fragments only merge when they are also horizontally adjacent. A
    letterhead is drawn as a grid of separate boxes at the same height —
    ``YARD No. BY531-536`` on the left and ``Page 3 of 31`` on the right — and
    joining those produced a band line that changed on every page, so the
    letterhead stopped recurring and the page number rode into the export.
    """
    if not items:
        return []
    gap_limit = max(_MIN_GAP, page_width * _GAP_FRACTION)
    rows: List[List[tuple]] = []
    for row in sorted(items, key=lambda r: (r[1], r[2])):
        if rows and abs(row[1] - rows[-1][0][1]) <= _LINE_TOLERANCE:
            rows[-1].append(row)
        else:
            rows.append([row])

    lines: List[_Line] = []
    for band in rows:
        parts = sorted(band, key=lambda r: r[2])
        run = [parts[0]]
        for part in parts[1:]:
            if part[2] - max(p[3] for p in run) > gap_limit:
                lines.append(_join(run))
                run = [part]
            else:
                run.append(part)
        lines.append(_join(run))
    return lines


def _join(group: List[tuple]) -> _Line:
    parts = sorted(group, key=lambda r: r[2])
    text = " ".join(p[4] for p in parts)
    labels = [p[5] for p in parts]
    label = next((l for l in labels if l in _HEADING_LABELS), labels[0])
    return _Line(top=min(p[1] for p in parts), left=min(p[2] for p in parts),
                 right=max(p[3] for p in parts), text=_tidy(text), label=label)


def _tidy(text: str) -> str:
    from app.tables.helpers import _repair_split_words
    return _WS_RE.sub(" ", _repair_split_words(text)).strip()


def _key(text: str) -> str:
    """Identity of a line: its whole text, casing and punctuation ignored.

    Deliberately the whole line and not a prefix. ``Page 3 of 31`` and
    ``Page 4 of 31`` must never share an identity — that is the entire
    mechanism by which the page number is kept out of a five-page export.
    """
    return " ".join(w for w in _NONWORD_RE.sub(" ", text.casefold()).split() if w)


@dataclass
class _Band:
    """Per-page lines plus, for each line, the set of pages it appears on."""
    lines_by_page: Dict[int, List[_Line]]
    header_pages: Dict[str, set]
    footer_pages: Dict[str, set]
    n_pages: int

    def repeats(self, key: str) -> bool:
        """Seen in the same region of more than one page — i.e. letterhead."""
        pages = (self.header_pages.get(key) or set()) | (self.footer_pages.get(key) or set())
        return len(pages) >= _MIN_RECUR_PAGES

    def recurs(self, key: str) -> bool:
        """Belongs in the band. Lenient for documents too short to test."""
        return True if self.n_pages <= _SHORT_DOC_PAGES else self.repeats(key)


_CACHE: Dict[tuple, _Band] = {}


def _band(file_id: str) -> Optional[_Band]:
    doc = state.DOCLING_DOCS.get(file_id)
    if doc is None:
        return None
    cache_key = (file_id, id(doc))
    hit = _CACHE.get(cache_key)
    if hit is not None:
        return hit

    items = _text_items(doc)
    if not items:
        return None

    # Fragments are merged per page, never across pages: two different pages
    # have text at the same height, and joining those would invent a line the
    # document never contained.
    by_page: Dict[int, List[_Line]] = {}
    per_page_items: Dict[int, List[tuple]] = {}
    for row in items:
        per_page_items.setdefault(row[0], []).append(row)
    for page_no, rows in per_page_items.items():
        by_page[page_no] = _merge_into_lines(rows, _page_width(doc, page_no))

    header_pages: Dict[str, set] = {}
    footer_pages: Dict[str, set] = {}
    for page_no, lines in by_page.items():
        height = _page_height(doc, page_no)
        for line in lines:
            key = _key(line.text)
            if not key:
                continue
            if line.top <= height * _BAND_TOP_FRACTION:
                header_pages.setdefault(key, set()).add(page_no)
            elif line.top >= height * (1 - _FOOTER_FRACTION):
                footer_pages.setdefault(key, set()).add(page_no)

    band = _Band(lines_by_page=by_page, header_pages=header_pages,
                 footer_pages=footer_pages, n_pages=len(by_page))
    _CACHE[cache_key] = band
    return band


def _band_lines(band: _Band, doc, page_no: int, region: str) -> List[str]:
    """Recurring lines from one region of one page, in reading order."""
    height = _page_height(doc, page_no)
    out = []
    for line in sorted(band.lines_by_page.get(page_no, []),
                       key=lambda l: (l.top, l.left)):
        in_region = (line.top <= height * _BAND_TOP_FRACTION if region == "header"
                     else line.top >= height * (1 - _FOOTER_FRACTION))
        if in_region and band.recurs(_key(line.text)):
            out.append(line.text)
    return out


def _table_top(doc, page_no: int) -> Optional[float]:
    """Distance from the page top to the highest table on that page."""
    tops = []
    for table in getattr(doc, "tables", []) or []:
        prov = getattr(table, "prov", None) or []
        if prov and getattr(prov[0], "page_no", None) == page_no:
            bbox = getattr(prov[0], "bbox", None)
            if bbox is not None:
                tops.append(_from_top(bbox, _page_height(doc, page_no)))
    return min(tops) if tops else None


def section_above(file_id: str, page_no: int, look_back: int = 5) -> Optional[str]:
    """The nearest heading above the page's table that is not letterhead.

    Walks back through earlier pages when the page holds no heading of its
    own, which is what a table continuing across a page break looks like.
    """
    band = _band(file_id)
    doc = state.DOCLING_DOCS.get(file_id)
    if band is None or doc is None:
        return None

    for offset in range(look_back + 1):
        page = page_no - offset
        if page < 1:
            break
        lines = band.lines_by_page.get(page)
        if not lines:
            continue
        height = _page_height(doc, page)
        limit = _table_top(doc, page) if offset == 0 else height
        limit = limit if limit is not None else height
        # Only "above the table" and "not letterhead" — no height cut. The
        # tender's section heading sits 11.6% down the page, inside the band
        # region, and a height cut excluded the one line the user actually
        # asked for. Recurrence is what separates letterhead from heading.
        candidates = [
            l for l in lines
            if l.top < limit
            and not band.repeats(_key(l.text))
            and l.label in _HEADING_LABELS
        ]
        if candidates:
            return max(candidates, key=lambda l: l.top).text
    return None


def context_for(file_id: str, pages: Optional[List[int]] = None,
                source: Optional[str] = None) -> TableContext:
    """The context band for a table drawn from ``pages`` of ``file_id``."""
    band = _band(file_id)
    doc = state.DOCLING_DOCS.get(file_id)
    if band is None or doc is None:
        return TableContext(source=source)

    # The band comes from a page that actually carries one. A requested page
    # can be blank or all-drawing, and an empty band there would silently drop
    # the letterhead from an export whose other pages have it.
    candidates = list(pages or sorted(band.lines_by_page)[:1])
    anchor = candidates[0]
    document: List[str] = []
    for page_no in candidates:
        document = _band_lines(band, doc, page_no, "header")
        if document:
            anchor = page_no
            break

    section = section_above(file_id, anchor)
    if section:
        # A heading high enough to fall inside the band region would otherwise
        # be printed twice.
        document = [l for l in document if _key(l) != _key(section)]

    return TableContext(
        document=document,
        section=section,
        footer=_band_lines(band, doc, anchor, "footer"),
        source=source,
    )


def describe_source(file_id: str, pages: Optional[List[int]] = None) -> str:
    """The provenance line: our own statement, never confused with document text."""
    name = state.FILE_ORIGINAL_NAME.get(file_id) or file_id
    name = re.sub(r"^[\w-]*?_\d{4}_", "", str(name))
    if pages:
        span = (f"page {pages[0]}" if len(pages) == 1
                else f"pages {pages[0]}-{pages[-1]}" if pages == list(range(pages[0], pages[-1] + 1))
                else "pages " + ", ".join(map(str, pages)))
        return f"Source: {name} — {span}"
    return f"Source: {name}"
