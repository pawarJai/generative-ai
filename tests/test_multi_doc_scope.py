"""Regression tests for the multi-document export that shipped outputs/f1-2f.xlsx.

The prompt was:

    "i need to export data data-file-1 and data-file-2 tables like data-file-1
     page number 8 table start all table data and data-file-2 page number-8
     started data into excel file f1-2f.xlsx do it"

and the file it produced contained pages 21-25 and 31-35 of data-file-1 —
130 rows from a request that named page 8 twice and nothing else. Four
independent faults lined up to do that, and each gets its own test here so a
future change cannot quietly reintroduce one of them behind the others:

1. "page number-8" parsed as no page at all (hyphen, not space).
2. Only the FIRST mention of a filename was recorded, so the page number was
   attributed to whichever document the preamble happened to list nearer.
3. A document that ended up with no pages fell through to "export all of it".
4. "the table starts on page 8" was read as page 8 alone.

Scope resolution is pure text work, so the registry is stubbed rather than
requiring the real PDFs; the assembly-level tests are skipped without them.
"""
import os
import pytest

from app.graph import agent as A


PROMPT = ("i need to export data data-file-1 and data-file-2 tables like "
          "data-file-1 page number 8 table start all table data and "
          "data-file-2 page number-8 started data into excel file f1-2f.xlsx do it")


@pytest.fixture
def registry(monkeypatch):
    """Two registered documents, without touching the real registry or disk."""
    records = [
        {"file_id": "fid1", "original_filename": "data-file-1.pdf",
         "path": "/tmp/data-file-1.pdf", "kind": "docling"},
        {"file_id": "fid2", "original_filename": "data-file-2.pdf",
         "path": "/tmp/data-file-2.pdf", "kind": "docling"},
    ]
    monkeypatch.setattr("app.persistence.get_all_files",
                        lambda session_id=None: records)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(A.app_state if hasattr(A, "app_state") else A,
                        "__name__", A.__name__)  # no-op, keeps monkeypatch scope tidy
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", None)
    return records


@pytest.mark.parametrize("text,expected", [
    ("page number-8", [8]),          # the form that failed
    ("page number 8", [8]),
    ("page-9", [9]),
    ("page no-7", [7]),
    ("page:12", [12]),
    ("page 8 to 10", [8, 9, 10]),    # ranges still expand
    ("page 3, 5 and 9", [3, 5, 9]),  # lists still stay lists
    ("check for this data is present on 9 ,10 page", [9, 10]),  # numbers before
])
def test_page_numbers_are_read_however_they_are_punctuated(text, expected):
    assert [p for _, p in A._page_mentions(text)] == [expected]


def test_a_filename_is_not_read_as_a_page_number():
    """'data-file-2' and 'f1-133' must never contribute a page. The hyphen is
    now a valid page connector, so this guard carries more weight than before."""
    assert A._page_mentions("data-file-2 has no page word") == []
    assert A._page_mentions("f1-133 export") == []


def test_every_mention_of_a_document_is_recorded(registry):
    """data-file-1 is named twice: in the preamble and again beside its page
    number. Recording only the first cost the second one its meaning."""
    mentions = A._find_file_mentions(PROMPT)
    assert [f for _, f in mentions] == ["fid1", "fid2", "fid1", "fid2"]
    # ...while the deduplicated view callers use for "how many files" is still
    # one entry per document.
    assert [f for _, f in A._find_named_files(PROMPT)] == ["fid1", "fid2"]


def test_each_page_number_goes_to_the_document_it_was_written_next_to(registry):
    """The whole bug, in one assertion: page 8 for BOTH files, because each
    document has its own 'page number 8' sitting right after its name."""
    specs = {s["file_id"]: s["pages"] for s in A._resolve_source_specs(PROMPT, None)}
    assert specs == {"fid1": [8], "fid2": [8]}


def test_naming_where_a_table_starts_asks_for_the_whole_table():
    assert A._WHOLE_TABLE_RE.search(PROMPT)
    assert A._WHOLE_TABLE_RE.search("page 8 table start all table data")
    assert A._WHOLE_TABLE_RE.search("page number-8 started data")
    # ...but an ordinary single-page export is still a single page.
    assert not A._WHOLE_TABLE_RE.search("export page 8 into out.xlsx")


def test_unattributed_pages_never_become_the_whole_document():
    """When the user named pages and none resolved to this document, the
    document contributes nothing. Falling through to 'all tables' is what put
    130 unrequested rows into f1-2f.xlsx."""
    picked, detail, tag, missing, resolved = A._select_tables_for_spec(
        {"file_id": "fid1", "pos": 0, "pages": []}, "page 8", [0], [],
        pages_named_elsewhere=True)
    assert picked == [] and resolved == [] and tag == ""


def test_headerless_table_is_stacked_not_merged():
    """A table whose columns are all col_N carries no evidence of what its
    columns mean. Aligning it into a named table scored 0.909 for putting a
    contact name under 'Spec No.' — measured on the real documents."""
    import pandas as pd
    named = pd.DataFrame({"Spec No.": ["A.1"], "Material code": ["PP1131"]})
    unnamed = pd.DataFrame({"col_0": ["Dijeev D"], "col_1": ["Cochin Shipyard"]})
    assert not A._is_headerless(named)
    assert A._is_headerless(unnamed)
    # provenance columns must not make a headerless table look headed
    unnamed["_source_file"] = "data-file-2.pdf"
    assert A._is_headerless(unnamed)


REAL_PDF_1 = "uploads/data-file-1_1580_data-file-1.pdf"
REAL_PDF_2 = "uploads/data-file-2_0060_data-file-2.pdf"


@pytest.mark.skipif(not (os.path.exists(REAL_PDF_1) and os.path.exists(REAL_PDF_2)),
                    reason="real tender PDFs not available")
def test_end_to_end_export_takes_only_the_requested_pages(monkeypatch):
    """The failing request, run for real against the registered documents:
    every sheet must trace back to a page the user actually named (8, plus the
    pages that table continues onto)."""
    from app import state as app_state

    # set_current_user_prompt() writes a module-level global with no built-in
    # reset. Called directly (as this did before), it leaked PROMPT into
    # every test that ran afterward in the same session — confirmed live:
    # tests/test_sheet_export.py, added later, inherited this PROMPT instead
    # of its own and resolved to data-file-1/data-file-2 instead of the
    # tabular file it was actually testing. monkeypatch reverts automatically.
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", PROMPT)
    specs = A._resolve_source_specs(PROMPT, None)
    if len(specs) != 2:
        pytest.skip("data-file-1 / data-file-2 are not in the file registry")
    # Both documents, each carrying the page number written beside its name.
    assert [s["pages"] for s in specs] == [[8], [8]]

    result = A._deterministic_multi_export(specs, PROMPT, "test_f1-2f.xlsx", "excel")
    assert "Verified:" in result
    # Pages 21-25 and 31-35 were the rows nobody asked for.
    for unwanted in ("p21-25", "p31-35"):
        assert unwanted not in result, f"exported {unwanted}, which was never requested"
    assert "Combined" in result
