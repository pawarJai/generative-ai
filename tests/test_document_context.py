"""Tests for recovering the context that surrounds a table.

The failure these exist to prevent is in interaction_log.jsonl on 2026-08-12:
the user asked three times for the document's header details to be carried
into an exported sheet, supplied every value by hand on the second attempt,
and got "Could not extract data" each time — because nothing in the codebase
ever read the letterhead, and nothing could write above a header row.

The synthetic pages here reproduce the geometry of the real tender: a
letterhead drawn as separate boxes at the same height (so a naive line-merge
glues the page number onto the yard number), a section heading that sits high
enough to fall inside the letterhead region, and a page number that changes on
every page.
"""
import json
import os
from types import SimpleNamespace

import pytest

from app import state
from app.tables.context import _key, context_for, describe_source, section_above

PAGE_H, PAGE_W = 842.0, 595.0


def _text(text, top, left, right, label="text"):
    """A text item positioned like Docling's: BOTTOMLEFT origin, so the
    stored `t` counts up from the bottom of the page."""
    bbox = SimpleNamespace(l=left, r=right, t=PAGE_H - top, b=PAGE_H - top - 10,
                           coord_origin="CoordOrigin.BOTTOMLEFT")
    return SimpleNamespace(text=text, label=label,
                           prov=[SimpleNamespace(page_no=None, bbox=bbox)])


def _doc(pages):
    """pages: {page_no: [text items]} — items are re-stamped with their page."""
    texts = []
    for page_no, items in pages.items():
        for item in items:
            item.prov[0].page_no = page_no
            texts.append(item)
    size = SimpleNamespace(width=PAGE_W, height=PAGE_H)
    return SimpleNamespace(
        texts=texts,
        tables=[],
        pages={p: SimpleNamespace(size=size, page_no=p) for p in pages},
    )


def _letterhead(page_of):
    """The tender's header: four boxes, two of them sharing a line."""
    return [
        _text("ACME SHIPYARD LTD", 23, 119, 256, label="section_header"),
        _text("Spec. No. N453105 REV.00", 23, 416, 538),
        _text("PROJECT -NGMV", 29, 294, 373, label="section_header"),
        _text("YARD No. BY531-536", 47, 284, 380, label="section_header"),
        _text(f"Page {page_of} of 31", 47, 449, 509),
        _text("PURCHASE SPECIFICATION FOR VALVES", 61, 193, 425),
        _text("UNCLASSIFIED", 816, 286, 333, label="page_footer"),
    ]


def _register(doc, file_id="ctx_test"):
    state.DOCLING_DOCS[file_id] = doc
    return file_id


def test_letterhead_is_kept_and_the_page_number_is_not():
    """The whole point: 'PROJECT -NGMV' belongs on a five-page export and
    'Page 3 of 31' does not, and nothing in the code knows what either means."""
    fid = _register(_doc({p: _letterhead(p - 5) for p in range(6, 12)}))
    ctx = context_for(fid, [6, 7, 8])

    assert "PROJECT -NGMV" in ctx.document
    assert "ACME SHIPYARD LTD" in ctx.document
    assert "Spec. No. N453105 REV.00" in ctx.document
    assert not any("Page " in line for line in ctx.document)


def test_boxes_at_the_same_height_stay_separate():
    """'YARD No. BY531-536' and 'Page 3 of 31' are drawn on one line in two
    boxes. Glued together they form a line that differs on every page, so the
    yard number stops recurring and drops out of the band entirely."""
    fid = _register(_doc({p: _letterhead(p) for p in range(1, 6)}))
    ctx = context_for(fid, [3])

    assert "YARD No. BY531-536" in ctx.document


def test_fragments_of_one_line_are_joined():
    """The same spec number is one item on one page and three on another."""
    def split():
        return [_text("Spec. No.", 23, 416, 455), _text("N453105", 23, 458, 505),
                _text("REV.00", 23, 508, 538)]

    fid = _register(_doc({1: split(), 2: split(), 3: split()}))
    ctx = context_for(fid, [1])

    assert "Spec. No. N453105 REV.00" in ctx.document


def test_section_heading_inside_the_band_region_is_still_a_section():
    """The tender's section title sits 11.6% down the page — inside the
    letterhead region. A height cut excluded the one line the user asked for."""
    pages = {p: _letterhead(p) + [
        _text(f"{p}. SECTION {p} TITLE:", 98, 71, 278, label="section_header")]
        for p in range(1, 6)}
    fid = _register(_doc(pages))
    ctx = context_for(fid, [3])

    assert ctx.section == "3. SECTION 3 TITLE:"
    assert ctx.section not in ctx.document  # never printed twice


def test_section_carries_over_a_page_break():
    """A table continuing onto the next page keeps the heading it started
    under — the page it spills onto has no heading of its own."""
    pages = {1: _letterhead(1) + [_text("1. ITEM AND QUANTITY:", 98, 71, 278,
                                        label="section_header")],
             2: _letterhead(2), 3: _letterhead(3)}
    fid = _register(_doc(pages))

    assert section_above(fid, 2) == "1. ITEM AND QUANTITY:"


def test_a_document_with_no_letterhead_gets_no_band():
    """Not every document has one. Inventing a band is worse than omitting it."""
    pages = {p: [_text(f"body text on page {p}", 400, 71, 400)] for p in range(1, 6)}
    fid = _register(_doc(pages))
    ctx = context_for(fid, [2])

    assert ctx.document == []


def test_footer_marking_survives():
    """UNCLASSIFIED on a defence tender is a classification marking, not noise."""
    fid = _register(_doc({p: _letterhead(p) for p in range(1, 6)}))

    assert context_for(fid, [2]).footer == ["UNCLASSIFIED"]


def test_source_line_is_ours_and_says_so():
    state.FILE_ORIGINAL_NAME["src_test"] = "data-file-1_1580_data-file-1.pdf"

    assert describe_source("src_test", [8, 9, 10]) == "Source: data-file-1.pdf — pages 8-10"
    assert describe_source("src_test", [8, 12]) == "Source: data-file-1.pdf — pages 8, 12"


def test_key_ignores_case_and_punctuation_but_not_the_number():
    assert _key("PROJECT -NGMV") == _key("Project - NGMV")
    assert _key("Page 3 of 31") != _key("Page 4 of 31")


CACHE = "docling_cache/b309da66464232bf.json"


@pytest.mark.skipif(not os.path.exists(CACHE), reason="docling cache not present")
def test_real_tender_band_is_exactly_the_letterhead():
    """Ground truth: the six lines the user typed out by hand in the chat log,
    recovered from the document without any of them being in the code."""
    from docling_core.types.doc.document import DoclingDocument

    with open(CACHE) as fh:
        state.DOCLING_DOCS["ctx_real"] = DoclingDocument.model_validate(json.load(fh))

    ctx = context_for("ctx_real", [8, 9, 10])
    joined = " | ".join(ctx.document)

    assert "COCHIN SHIPYARD LTD" in joined
    assert "PROJECT -NGMV" in joined
    assert "YARD No. BY531-536" in joined
    assert "N45310510605002" in joined
    assert "PURCHASE TECHNICAL SPECIFICATION FOR VALVES (NON-CLASS) LOT-2" in joined
    assert ctx.section == "1. ITEM AND QUANTITY REQUIRED:"
    assert "of 31" not in joined       # the per-page number never travels
    assert ctx.footer == ["UNCLASSIFIED"]


@pytest.mark.skipif(not os.path.exists(CACHE), reason="docling cache not present")
def test_band_follows_the_page_and_is_not_one_fixed_answer():
    """Page 14 is a different section of the same document; page 1 is a
    different letterhead entirely. If either matched page 8's band, the band
    would be coming from somewhere other than the page."""
    from docling_core.types.doc.document import DoclingDocument

    with open(CACHE) as fh:
        state.DOCLING_DOCS["ctx_real2"] = DoclingDocument.model_validate(json.load(fh))

    assert context_for("ctx_real2", [14]).section.startswith("A.2.")
    assert context_for("ctx_real2", [20]).section.startswith("A.8.")
    assert context_for("ctx_real2", [1]).document != context_for("ctx_real2", [8]).document
