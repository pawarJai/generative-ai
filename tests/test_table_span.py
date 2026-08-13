"""Tests for following a table across the pages it continues onto.

From interaction_log.jsonl, 2026-08-12 14:44–14:51: three consecutive turns
asking for the schedule that starts on page 8 of data-file-1 — "make sure this
table data is not fit only one page so check also pages like 9, 10 so on and
give all row data there was 64 row this table you agin give me only 27 records"
— each returned 27 rows and reported success.

The span detector existed and was correct when it was written: a continuation
page carried no column names, so "columns are still col_0, col_1…" identified
one. Header recovery then began naming continuation pages from the page before
them, and that test silently stopped matching anything. Page 8 (27 rows) was
exported as the whole table while pages 9 (23) and 10 (14) sat right there.
"""
import json
import os

import pandas as pd
import pytest

from app.tables import helpers
from app.tables.helpers import _continues_table, span_pages

COLUMNS = ["SL no", "Item", "Spec No.", "Material code",
           "Quantity Per Vessel (nos)", "Total Quantity for 6 vessels (nos)"]


def _page(n, page, columns=COLUMNS):
    df = pd.DataFrame({c: [f"{c}-{page}-{i}" for i in range(n)] for c in columns})
    df.attrs["page"] = page
    return df


def _install(monkeypatch, frames):
    monkeypatch.setattr(helpers, "get_all_real_tables", lambda file_id, **kw: frames)


# --- the regression ----------------------------------------------------------

def test_span_follows_pages_whose_header_was_recovered(monkeypatch):
    """The exact shape of the production failure: all three pages carry the
    same real column names, because header recovery gave them the names."""
    _install(monkeypatch, [_page(27, 8), _page(23, 9), _page(14, 10)])

    assert span_pages("f", 8) == [8, 9, 10]


def test_span_still_follows_unnamed_continuation_pages(monkeypatch):
    """The original case must keep working: continuation pages with no header
    at all, one of them a column short because Docling dropped it."""
    generic = [f"col_{i}" for i in range(len(COLUMNS))]
    _install(monkeypatch, [_page(27, 8),
                           _page(23, 9, generic),
                           _page(14, 10, generic[:-1])])

    assert span_pages("f", 8) == [8, 9, 10]


def test_a_different_table_does_not_join_the_span(monkeypatch):
    """Same width, different headings — a new table, not a continuation."""
    other = ["Sr", "Description", "Standard", "Code", "Rate", "Amount"]
    _install(monkeypatch, [_page(27, 8), _page(9, 9, other)])

    assert span_pages("f", 8) == [8]


def test_a_narrower_named_table_does_not_join_the_span(monkeypatch):
    """Being one column short is only evidence of a dropped column when the
    page brought no names of its own."""
    _install(monkeypatch, [_page(27, 8), _page(9, 9, COLUMNS[:-1])])

    assert span_pages("f", 8) == [8]


def test_the_span_stops_at_the_first_gap(monkeypatch):
    _install(monkeypatch, [_page(27, 8), _page(23, 9), _page(11, 12)])

    assert span_pages("f", 8) == [8, 9]


def test_a_page_with_no_table_has_no_span(monkeypatch):
    _install(monkeypatch, [_page(27, 8)])

    assert span_pages("f", 40) == []


def test_continuation_test_tolerates_spacing_and_case():
    assert _continues_table(_page(3, 9, ["SL  NO", "item", "Spec No.", "Material code",
                                         "Quantity Per Vessel (nos)",
                                         "Total Quantity for 6 vessels (nos)"]),
                            COLUMNS)


def test_partly_recovered_headers_still_continue():
    """Header recovery sometimes names only some columns (page 3 of the tender
    came back as ['col_0', 'HANDWHEEL', 'col_2', …])."""
    mixed = ["col_0", "Item", "col_2", "col_3", "col_4", "col_5"]

    assert _continues_table(_page(3, 9, mixed), COLUMNS)


# --- the words that ask for the whole table ----------------------------------

def test_the_production_prompt_asks_for_the_whole_table():
    """Verbatim from interaction_log.jsonl, id=710bc034."""
    from app.graph.agent import _WHOLE_TABLE_RE, _extract_requested_pages

    prompt = ("on file data file -1 export page number 8 table started that "
              "table also data there in other page number export this all data "
              "in f1-133.xlsx make sure this table data not only there onpage 8 "
              "that data slo i thing available in upcomming page numbers so "
              "check and and than export it make sure i need header also on "
              "this file make sure this table data is not fit only one page so "
              "check also pages like 9, 10 so on and give all row data")

    # Two independent routes to the same 64 rows: the user both named the
    # continuation pages ("pages like 9, 10") and asked for the whole table.
    assert _extract_requested_pages(prompt) == [8, 9, 10]
    assert _WHOLE_TABLE_RE.search(prompt)


@pytest.mark.parametrize("phrase", [
    "export page 8, the table continues on the next pages",
    "page 8 data is not fit in one page, check other pages",
    "give me page 8 and the upcoming pages so on",
    "the table on page 8 is spread across multiple pages",
])
def test_continuation_phrasings_are_recognised(phrase):
    from app.graph.agent import _WHOLE_TABLE_RE

    assert _WHOLE_TABLE_RE.search(phrase)


@pytest.mark.parametrize("prompt,expected", [
    # The numbers come BEFORE the word — verbatim from log id=9fd081df, where
    # the user listed the continuation pages and got 27 of 64 rows anyway.
    ("check for this data is present on 9 ,10 page", [9, 10]),
    ("in data-file -1 page number 8 i need to export data in f1-023.xlsx file "
     "without header data also make sure this table data is not only one pages "
     "check for this data is present on 9 ,10 page", [8, 9, 10]),
    # A list of any length, not just a pair.
    ("export page 3, 5 and 9", [3, 5, 9]),
    ("give me pages like 9, 10 so on", [9, 10]),
    # Ranges, discrete lists and the guard against absurd spans are unchanged.
    ("page number 6 to 10 we have table", [6, 7, 8, 9, 10]),
    ("page 6 and 7 how many columns", [6, 7]),
    ("page 2 to 2000", [2, 2000]),
    # "3 of 31" is a page stamp; 31 is the document length, not a request.
    ("what is on page 3 of 31", [3]),
    # "to excel" is not a range, and a number glued to a filename or an id is
    # not a page number however close it sits to the word.
    ("export the table on page 8 to excel", [8]),
    ("in data-file -1 page number 8", [8]),
])
def test_page_numbers_are_read_around_the_word_not_after_it(prompt, expected):
    from app.graph.agent import _extract_requested_pages

    assert _extract_requested_pages(prompt) == expected


def test_page_numbers_stay_attached_to_the_file_they_were_named_with():
    """A multi-document export splits pages by which file mention is nearest."""
    from app.graph.agent import _page_mentions

    mentions = _page_mentions("data-file-2 page 7 and data-file-5 page 3")

    assert [pages for _pos, pages in mentions] == [[7], [3]]
    assert mentions[0][0] < mentions[1][0]


# --- "without the header" has to reach the exporter ---------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("export page 8 in f1.xlsx without header data", True),
    ("i don't need table header", True),
    ("i do not want the header section", True),
    ("make sure i need header also on this file", False),
    ("export page 8 into f1-133.xlsx", False),
    ("rename 'Spec No.' to Drawing Ref", False),
])
def test_a_bare_table_is_asked_for_in_the_users_own_words(prompt, expected):
    """Verbatim from log id=9fd081df: "export data in f1-023.xlsx file without
    header data" was parsed correctly and then ignored, because only
    modify_export ever asked the question. One parser now answers for both."""
    from app.graph.agent import _wants_bare_table

    assert _wants_bare_table(prompt) is expected


def test_a_plain_single_page_request_is_not_a_whole_table_request():
    """"export page 8" must still mean page 8 — the span is opt-in."""
    from app.graph.agent import _WHOLE_TABLE_RE

    assert not _WHOLE_TABLE_RE.search("export page 8 into f1-133.xlsx")


# --- against the real document -----------------------------------------------

CACHE = "docling_cache/b309da66464232bf.json"


@pytest.mark.skipif(not os.path.exists(CACHE), reason="docling cache not present")
def test_the_real_schedule_spans_pages_8_to_10():
    """Ground truth read off the running server's /files/{id}/tables:
    page 8 has 27 rows, page 9 has 23, page 10 has 14 — 64 in total, which is
    the number the user counted by hand."""
    from docling_core.types.doc.document import DoclingDocument

    from app import state
    from app.tables.helpers import get_table_span

    with open(CACHE) as fh:
        state.DOCLING_DOCS["span_real"] = DoclingDocument.model_validate(json.load(fh))
    state.FILE_KIND["span_real"] = "docling"

    assert span_pages("span_real", 8) == [8, 9, 10]

    df, pages = get_table_span("span_real", 8)
    assert pages == [8, 9, 10]
    assert len(df) == 64
    assert list(df.columns) == COLUMNS
    # The tender's own identification travels with it — an export of the span
    # is still traceable to the document it came from.
    assert df.attrs.get("context")
