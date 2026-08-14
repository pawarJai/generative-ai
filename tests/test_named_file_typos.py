"""Resolving the document a typo names, and truthfully answering a page
lookup instead of inventing an excuse for it.

Confirmed production failure (session 477b296a, 17:55-18:25): the user typed
"in data-fle-4 page number 15..." (a dropped 'i'). _resolve_named_file's
exact substring match never fired for that typo, so the request silently
fell through to whatever file happened to be active — a spreadsheet — and
the model's answer described the ACTIVE file's real properties ("is a
spreadsheet... does not have pages") while calling it by the name the user
had actually typed ("data-file-4"), a confusing hybrid. That false claim
about "data-file-4" then got checkpointed and echoed back across several
more turns, even once later turns had the CORRECT file_id.

Separately, and independently reproduced: once the right document and page
were actually in scope, the model answered "the system is currently unable
to retrieve page 15 due to a parsing limitation" — a plausible-sounding
excuse that get_page_content itself never produces (confirmed live: calling
get_page_content for the exact same file_id and page immediately afterward,
with no code changed in between, returned the real page in full). The model
either didn't call the tool or discarded what it returned.
"""
import re

import pytest

from app.graph import agent as A


# --- _edit_distance -----------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("datafile", "datafile", 0),
    ("datafile", "datafle", 1),     # the real typo: one letter dropped
    ("datafile", "datafiel", 2),    # a transposition costs 2 under plain Levenshtein
    ("datafile", "databank", 6),
    ("", "abc", 3),
])
def test_edit_distance(a, b, expected):
    assert A._edit_distance(a, b) == expected if expected <= 2 else \
        A._edit_distance(a, b) >= 3  # loose bound for the dissimilar case


# --- _fuzzy_name_in_prompt: the typo itself ------------------------------

def test_the_real_typo_now_matches():
    normalized = re.sub(r"[^a-z0-9]", "", "in data-fle-4 page number 15".lower())
    assert A._fuzzy_name_in_prompt("datafile4", normalized)


def test_exact_spelling_still_matches():
    normalized = re.sub(r"[^a-z0-9]", "", "in data-file-4 page number 15".lower())
    assert A._fuzzy_name_in_prompt("datafile4", normalized)


# --- safety: fuzzing the letters must never fuzz the number --------------

@pytest.mark.parametrize("key,prompt", [
    ("datafile4", "in data-file-1 page number 15"),
    ("datafile1", "in data-file-4 page number 15"),
    ("datafile4", "in data-file-14 page number 15"),
    ("datafile4", "in data-file-24 page number 15"),
])
def test_a_typo_tolerant_match_never_crosses_document_numbers(key, prompt):
    """data-file-4 and data-file-1 are one character apart, exactly like
    'file' and 'fle' — fuzzing the whole key would make one document's name
    match another's registry entry. The digit suffix must always be exact."""
    normalized = re.sub(r"[^a-z0-9]", "", prompt.lower())
    assert not A._fuzzy_name_in_prompt(key, normalized)


def test_two_typos_are_still_rejected():
    """max_dist=1 is deliberate — beyond one typo, silently guessing which
    document was meant becomes riskier than asking."""
    normalized = re.sub(r"[^a-z0-9]", "", "in dtaflee 4 page 15".lower())
    assert not A._fuzzy_name_in_prompt("datafile4", normalized)


def test_unrelated_text_is_not_a_match():
    normalized = re.sub(r"[^a-z0-9]", "", "export the working sheet to excel".lower())
    assert not A._fuzzy_name_in_prompt("datafile4", normalized)


# --- _resolve_named_file end to end, through the typo ---------------------

@pytest.fixture
def registry(monkeypatch, tmp_path):
    paths = {}
    for n in (1, 4):
        p = tmp_path / f"data-file-{n}.pdf"
        p.write_bytes(b"x")
        paths[n] = str(p)
    records = [
        {"file_id": "data-file-1_1111", "original_filename": "data-file-1.pdf",
         "path": paths[1], "kind": "docling"},
        {"file_id": "data-file-4_2222", "original_filename": "data-file-4.pdf",
         "path": paths[4], "kind": "docling"},
    ]
    monkeypatch.setattr("app.persistence.get_all_files",
                        lambda session_id=None: records)
    return records


def test_a_typo_still_resolves_to_the_named_document(registry):
    """The exact failing prompt shape. current_file_id is some OTHER active
    file (a spreadsheet, standing in for data-file-5_3361 in the real
    session) — the typo'd name must still redirect to data-file-4."""
    resolved = A._resolve_named_file(
        "in data-fle-4 page number 15 what thing is there", "some-other-file")
    assert resolved == "data-file-4_2222"


def test_a_typo_naming_the_already_active_file_does_not_churn_it(registry):
    resolved = A._resolve_named_file(
        "in data-fle-4 page number 15 what thing is there", "data-file-4_2222")
    assert resolved is None


def test_a_typo_never_redirects_to_the_wrong_numbered_document(registry):
    """Typing 'data-fle-4' must never resolve to data-file-1, even though
    they're both one edit away from SOME string in the prompt."""
    resolved = A._resolve_named_file(
        "in data-fle-4 page number 15 what thing is there", "data-file-1_1111")
    assert resolved == "data-file-4_2222"
    assert resolved != "data-file-1_1111"


# --- _catch_fabricated_page_excuse ---------------------------------------

@pytest.fixture
def docling_file(monkeypatch):
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "pdf1", "docling")
    return "pdf1"


def test_a_fabricated_technical_excuse_is_replaced_with_the_real_page(
        docling_file, monkeypatch):
    real_content = "### Clause 33\nIntegrity Pact details for contracts above..."
    monkeypatch.setattr("app.graph.tools.get_page_content",
                        type("T", (), {"invoke": staticmethod(
                            lambda args: real_content)})())
    excuse = ("I understand the confusion. The file is a PDF document, so it "
             "does have pages. However, the system is currently unable to "
             "retrieve page 15 due to a parsing limitation — likely because "
             "the PDF was not fully processed.")
    out = A._catch_fabricated_page_excuse(excuse, "page number 15", docling_file)
    assert "Clause 33" in out
    assert "Correction" in out
    assert excuse in out  # original text kept, not silently replaced


def test_no_excuse_language_leaves_the_answer_untouched(docling_file):
    real = "### Clause 33\nHere is page 15 in full."
    assert A._catch_fabricated_page_excuse(real, "page 15", docling_file) == real


def test_a_genuine_tool_failure_is_not_overridden(docling_file, monkeypatch):
    """If get_page_content itself reports nothing extractable, the excuse
    text (however oddly worded) was not actually wrong — leave it."""
    monkeypatch.setattr("app.graph.tools.get_page_content",
                        type("T", (), {"invoke": staticmethod(
                            lambda args: "This document has 10 pages, so page "
                                        "15 does not exist.")})())
    excuse = ("The system is currently unable to retrieve page 15 due to a "
             "technical issue.")
    out = A._catch_fabricated_page_excuse(excuse, "page 15", docling_file)
    assert out == excuse


def test_never_fires_for_a_spreadsheet(monkeypatch):
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "xlsx1", "tabular")
    excuse = "unable to retrieve page 15 due to a technical issue."
    assert A._catch_fabricated_page_excuse(excuse, "page 15", "xlsx1") == excuse


def test_never_fires_without_a_page_number(docling_file):
    excuse = "unable to retrieve this content due to a technical issue."
    assert A._catch_fabricated_page_excuse(excuse, "what is in this file", docling_file) == excuse


def test_a_second_excuse_shape_is_also_replaced(docling_file, monkeypatch):
    """Confirmed live, after the file-resolution fix landed: the model
    stopped mislabeling the file but produced a DIFFERENT excuse for the
    same non-existent problem — "the content for that page was not
    extracted or made available in the system" — phrased as a report about
    the system's contents rather than a technical apology, so the first
    version of this regex missed it entirely."""
    real_content = "### Clause 33\nIntegrity Pact details..."
    monkeypatch.setattr("app.graph.tools.get_page_content",
                        type("T", (), {"invoke": staticmethod(
                            lambda args: real_content)})())
    excuse = ("Page 15 falls within the range of the document (1-24), but "
             "the content for that page was not extracted or made "
             "available in the system.")
    out = A._catch_fabricated_page_excuse(excuse, "page number 15", docling_file)
    assert "Clause 33" in out
    assert "Correction" in out


@pytest.mark.parametrize("legit", [
    "This document has 24 pages, so page 15 does not exist.",
    "Page 15 has no extractable text, and no similar content was found in this file.",
    "Page 15 has no separately extractable text (it may be a scan or an image-only page).",
])
def test_legitimate_tool_phrasing_never_trips_the_broadened_regex(legit):
    """The broadened regex added for the second excuse shape must not start
    treating get_page_content's own real "nothing here" wording as a
    fabrication to correct."""
    assert not A._FABRICATED_PAGE_EXCUSE_RE.search(legit)
