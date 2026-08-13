"""Tests for the three defects in interaction_log.jsonl, session 477b296a,
2026-08-13 08:39–09:10 — all of them places where the code discarded something
the conversation had already established.

1. 6914de87 / 91ac7565 — "export THAT page number started table", "THIS table
   data available on page number 6 to 10". Both ran against data-file-5.xlsx,
   uploaded at 08:56 and left selected by the UI, while the two turns before
   them were about data-file-2.pdf. Both reported the pages did not exist.
2. 5529d728 / c94da052 — successful exports whose answer opened with "I need to
   correct my previous response" and printed the verified result twice: two
   backstops each ran the same recovery export.
3. f8068c73 — "remind last 2 chats" was answered with a pandas sandbox error,
   because the fabricated-claim backstop replaced the whole response and ran an
   export for a message that asked for no file.
"""
import pytest

from app.graph import agent
from app.graph.agent import (_asks_for_a_file, _catch_unverified_export_claim,
                             _extract_requested_pages, _is_recall_request,
                             _recall_count, _refocus_on_conversation,
                             recent_exchanges, run_export_backstops)

SELECTED = "data-file-5_3361"     # the spreadsheet the UI had selected
FOCUS = "data-file-2_0060"        # the document the conversation was about


# --- Step 1: the document follows the conversation ---------------------------

@pytest.mark.parametrize("prompt,expected", [
    # Verbatim from log id=6914de87. The range straddles the anchor word: the
    # start sits before "page", the end after it, and reading only forwards
    # turned a five-page export into a one-page one.
    ("this table is spread 6 to page number 10 so i need to export this data",
     [6, 7, 8, 9, 10]),
    ("data available 8 to page 10", [8, 9, 10]),
    # Non-range separators before the word do not open a range.
    ("combine 2 and page 5", [5]),
    # Every earlier reading still holds.
    ("page number 6 to 10 we have table", [6, 7, 8, 9, 10]),
    ("check for this data is present on 9 ,10 page", [9, 10]),
    ("export page 3, 5 and 9", [3, 5, 9]),
    ("in data-file -1 page number 8", [8]),
    ("what is on page 3 of 31", [3]),
    ("export the table on page 8 to excel", [8]),
])
def test_a_page_range_can_straddle_the_word_page(prompt, expected):
    assert _extract_requested_pages(prompt) == expected


def _answerable(mapping):
    """Stand in for reading the real documents: file_id -> pages it has."""
    return lambda fid, pages: len(set(mapping.get(fid, [])) & set(pages))


def test_the_conversations_document_wins_when_the_selected_one_has_no_such_page(monkeypatch):
    """The production failure: pages 6–10 exist in data-file-2, not in the
    spreadsheet the UI still had selected."""
    monkeypatch.setattr(agent, "_pages_answerable",
                        _answerable({FOCUS: [7, 8, 9, 10]}))

    prompt = ("this table data available on page number 6 to 10 we need to "
              "export all this table data also i don't need table header on it")

    assert _refocus_on_conversation(prompt, SELECTED, FOCUS) == FOCUS


def test_a_selection_that_can_answer_is_never_churned(monkeypatch):
    """The chip is a deliberate act. A document that has the pages keeps the
    turn, whatever the conversation was about a moment ago."""
    monkeypatch.setattr(agent, "_pages_answerable",
                        _answerable({SELECTED: [6, 7], FOCUS: [6, 7]}))

    assert _refocus_on_conversation("export page 6 and 7", SELECTED, FOCUS) is None


def test_no_switch_when_neither_document_has_the_pages(monkeypatch):
    """An honest "page 40 has no table here" beats silently answering from
    another document that cannot answer either."""
    monkeypatch.setattr(agent, "_pages_answerable", _answerable({}))

    assert _refocus_on_conversation("export page 40", SELECTED, FOCUS) is None


def test_with_nothing_selected_the_conversations_document_is_used(monkeypatch):
    monkeypatch.setattr(agent, "_pages_answerable", _answerable({FOCUS: [8]}))
    monkeypatch.setattr(agent, "_restore_file_if_needed", lambda fid: None)

    assert _refocus_on_conversation("what is on this table", None, FOCUS) == FOCUS


def test_a_stale_selection_loses_even_with_no_page_to_test(monkeypatch):
    """"still we get only one page data we need all this table data" — verbatim
    from log id=34abf85d. No page number to check either document against, and
    read as belonging to the selected spreadsheet it exported nothing; the
    model then invented a file rather than come back empty."""
    monkeypatch.setattr(agent, "_pages_answerable", _answerable({}))
    monkeypatch.setattr(agent, "_restore_file_if_needed", lambda fid: None)

    prompt = ("still we get only one page data we need all this table data we "
              "need to export , export into f1-14.xlsx do it")

    assert _refocus_on_conversation(prompt, SELECTED, FOCUS) == FOCUS


def test_a_selection_the_user_just_changed_wins(monkeypatch):
    """Clicking a chip is a deliberate act about THIS message. Only a chip
    that has sat unchanged while the conversation moved on is stale."""
    monkeypatch.setattr(agent, "_pages_answerable", _answerable({FOCUS: [8]}))
    monkeypatch.setattr(agent, "_restore_file_if_needed", lambda fid: None)

    assert _refocus_on_conversation("how many rows", SELECTED, FOCUS,
                                    selection_changed=True) is None
    assert _refocus_on_conversation("export page 8", SELECTED, FOCUS,
                                    selection_changed=True) is None


def test_an_unloadable_focus_is_not_switched_to(monkeypatch):
    monkeypatch.setattr(agent, "_restore_file_if_needed",
                        lambda fid: "could not be restored")

    assert _refocus_on_conversation("all this table data", SELECTED, FOCUS) is None


def test_focus_equal_to_the_selection_is_not_a_switch(monkeypatch):
    monkeypatch.setattr(agent, "_pages_answerable", _answerable({SELECTED: [8]}))

    assert _refocus_on_conversation("export page 8", SELECTED, SELECTED) is None
    assert _refocus_on_conversation("export page 8", SELECTED, None) is None


@pytest.mark.parametrize("named,refocused,existing,response,expected", [
    # The failure that poisoned everything after it: a failed export against
    # the stale spreadsheet recorded the spreadsheet as the conversation's
    # subject, so the next message had nothing left to correct it.
    (False, False, FOCUS, "Could not extract data: {'Cover_Sheet_Raw': …}", False),
    (False, False, FOCUS, "None of the requested page(s) [6,7] have no "
                          "extractable table in 'data-file-5.xlsx'", False),
    # A document the user named outright counts even when the turn failed —
    # they said which one they meant.
    (True, False, FOCUS, "Could not extract data", True),
    (False, True, FOCUS, "Could not extract data", True),
    # Nothing to protect yet, and ordinary successful turns.
    (False, False, None, "Could not extract data", True),
    (False, False, FOCUS, "✓ Verified: 64 rows saved to f1.xlsx", True),
])
def test_only_a_turn_that_read_the_document_defines_it(named, refocused,
                                                       existing, response,
                                                       expected):
    from app.graph.agent import _turn_defines_focus

    assert _turn_defines_focus(named, refocused, existing, response) is expected


def test_focus_is_remembered_per_session():
    from app import state

    state.SESSION_FOCUS.clear()
    state.set_session_focus("chat-a", FOCUS)
    state.set_session_focus("chat-b", SELECTED)

    assert state.get_session_focus("chat-a") == FOCUS
    assert state.get_session_focus("chat-b") == SELECTED
    assert state.get_session_focus("chat-c") is None
    # Neither half of a missing pair may write a stray entry.
    state.set_session_focus(None, FOCUS)
    state.set_session_focus("chat-d", None)
    assert state.get_session_focus("chat-d") is None


# --- Step 2: one recovery per turn -------------------------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("create excel file ... file name give f1-011.xlsx", True),
    ("i need to export this data to f1-f01.xlsx file", True),
    # Verbatim from log id=f8068c73 — names no file, asks for none.
    ("still you not able to givw currect data to me like file i did not get "
     "in last chat remind last 2 chats", False),
    ("what is on page 6", False),
    ("f1-011.xlsx looks wrong", False),          # a filename alone is not a request
])
def test_whether_this_message_asks_for_a_file(prompt, expected):
    assert _asks_for_a_file(prompt) is expected


@pytest.mark.parametrize("prompt,expected", [
    # Verbatim from log id=3fd484e8. The user had named f1-14.xlsx one message
    # earlier; read as asking for no file, nothing was exported and the model
    # announced 9 rows and 6,145 bytes for a file that was never written.
    ("we need to export data page number 6 columns and page number 7,8,9,10 "
     "all data of this table", True),
    ("export all this table data, i don't need the header", True),
    # The verb is what makes it a request. A complaint is not one, however
    # recently a filename was mentioned.
    ("i not able to download last given file like it show 404 not found", False),
    ("still not able to download", False),
    ("what is on page 6", False),
])
def test_a_filename_named_earlier_still_counts_as_asking(prompt, expected):
    assert _asks_for_a_file(prompt, "f1-14.xlsx") is expected


@pytest.mark.parametrize("prompt,expected", [
    # Verbatim from log id=34abf85d. Said twice over: "do not add header" and
    # "right now i don't need" — but "if i need header" sits nearer a "header"
    # than either negation does, and the band was written anyway.
    ("also do not add header on table if i need header than i will tell but "
     "right now i don't need", True),
    ("do not add header on table", True),
    # A conditional must not silently cancel a real instruction either.
    ("add header if it is missing", False),
    ("add the header details to that excel", False),
])
def test_a_conditional_clause_is_not_an_instruction(prompt, expected):
    """"if i need header than i will tell" describes a future message, not
    this one."""
    from app.graph.agent import _wants_bare_table

    assert _wants_bare_table(prompt) is expected


def test_a_stated_band_preference_holds_for_later_exports(monkeypatch):
    """"if i need header than i will tell but right now i don't need" says the
    setting stands until it is changed. The band was back on the very next
    export, which is the third time the same complaint was filed."""
    from app import state
    from app.graph.agent import _wants_bare_table

    state.BAND_PREFERENCE.clear()
    monkeypatch.setattr(state, "ACTIVE_SESSION_ID", "s")

    # Nothing said yet: the band is on, as it has always been by default.
    assert _wants_bare_table("export pages 7 to 10 into f1.xlsx") is False

    state.set_band_preference("s", False)
    assert _wants_bare_table("export pages 7 to 10 into f1.xlsx") is True
    # …and the user can still say so explicitly in either direction.
    assert _wants_bare_table("add the header details to that excel") is False

    state.set_band_preference("s", True)
    assert _wants_bare_table("export pages 7 to 10 into f1.xlsx") is False
    state.BAND_PREFERENCE.clear()


def test_an_invented_download_link_is_removed():
    """The model wrote a link to a host this app has never served from, for a
    file that did not exist, and the user chased the 404 for three turns."""
    from app.graph.agent import _strip_invented_links

    text = ("📥 **Download Ready**:\n"
            "👉 [Click here to download `f1-14.xlsx`](https://example.com/f1-14.xlsx)")

    cleaned = _strip_invented_links(text)

    assert "example.com" not in cleaned
    assert "http" not in cleaned
    assert "Click here to download `f1-14.xlsx`" in cleaned


def test_a_real_download_path_is_left_alone():
    from app.graph.agent import _strip_invented_links

    text = "Saved. [f1-14.xlsx](/files/download/f1-14.xlsx) is ready."

    assert _strip_invented_links(text) == text


class _Tool:
    def __init__(self, name, content):
        self.name, self.content = name, content


def _count_recoveries(monkeypatch, result):
    calls = []

    def fake(prompt, file_id, fallback_filename=None):
        calls.append((prompt, file_id, fallback_filename))
        return result

    monkeypatch.setattr(agent, "_attempt_recovery_export", fake)
    return calls


def test_the_same_export_is_never_recovered_twice(monkeypatch):
    """Both backstops fired on one turn: the first recovered outside the graph,
    leaving no tool message, so the second read that as "no export attempted"
    and ran it again — two writes, one result shown twice."""
    calls = _count_recoveries(monkeypatch, "✓ Verified: 64 rows saved to f1.xlsx")
    claim = "The file is ready — f1.xlsx has been created."

    text, tool = run_export_backstops(
        "export page 8 to f1.xlsx", claim, [], "data-file-1_1580")

    assert len(calls) == 1
    assert tool == "export_data"
    assert text.count("Verified:") == 1


def test_the_verified_result_leads_and_the_apology_does_not(monkeypatch):
    _count_recoveries(monkeypatch, "✓ Verified: 64 rows saved to f1.xlsx")

    text, _tool = run_export_backstops(
        "export page 8 to f1.xlsx", "The file is ready — f1.xlsx created.",
        [], "data-file-1_1580")

    assert text.startswith("✓ Verified:")
    assert "I need to correct my previous response" not in text


def test_a_skipped_export_still_gets_recovered(monkeypatch):
    """The other backstop must keep working: the model described the page and
    offered to export instead of exporting."""
    calls = _count_recoveries(monkeypatch, "✓ Verified: 27 rows saved to f1.xlsx")
    essay = "Page 8 contains a schedule of valves. Let me know if you'd like me to export it."

    text, tool = run_export_backstops(
        "create excel from page 8, file name f1.xlsx", essay, [], "data-file-1_1580")

    assert len(calls) == 1
    assert tool == "export_data"
    assert "Verified:" in text


def test_a_real_tool_success_is_left_alone(monkeypatch):
    calls = _count_recoveries(monkeypatch, "should not be called")
    tool_msg = _Tool("export_data", "✓ Verified: 64 rows saved to f1.xlsx")
    answer = "Done — f1.xlsx has been created with 64 rows."

    text, tool = run_export_backstops(
        "export page 8 to f1.xlsx", answer, [tool_msg], "data-file-1_1580")

    assert calls == []
    assert text == answer
    assert tool is None


def test_a_stale_file_claim_does_not_trigger_an_export(monkeypatch):
    """Verbatim shape of log id=f8068c73: the answer still carried a filename
    from an earlier turn, so the claim guard fired, discarded the whole reply
    and ran an export — for a message that asked for no file at all. The user's
    actual question was never answered."""
    calls = _count_recoveries(monkeypatch, "should not be called")
    answer = "Earlier I saved f1-f01.xlsx for you; it was created successfully."
    prompt = ("still you not able to givw currect data to me like file i did "
              "not get in last chat remind last 2 chats")

    text, tool, recovered = _catch_unverified_export_claim(
        answer, [], prompt, "data-file-5_3361")

    assert calls == []
    assert recovered is False
    assert tool is None
    assert answer in text                   # the answer survives
    # …and the correction leads, where it will actually be read. Buried under
    # sixty lines of confident markdown it was scrolled past for three turns.
    assert text.startswith("**No file was created this turn.**")


def test_a_fabricated_export_is_labelled_before_the_essay(monkeypatch):
    """Log ids 3fd484e8 / 1988acf6 / 8a3be4d5: three answers in a row reported
    9 rows, 6,145 bytes and a download link for f1-14.xlsx, which was never
    written. Whatever else the answer says, it must not open as a success."""
    _count_recoveries(monkeypatch, "None of the requested page(s) have an "
                                   "extractable table — no file was created.")
    essay = ("✅ **Exported to `f1-14.xlsx`** — 9 rows, 6,145 bytes, verified "
             "and complete.\n\n👉 [Download](https://example.com/f1-14.xlsx)")

    text, tool = run_export_backstops(
        "export all this table data", essay, [], "data-file-5_3361", "f1-14.xlsx")

    assert tool is None
    assert text.startswith("**No file was created.**")
    assert "example.com" not in text


# --- Step 3: the assistant can read its own conversation ---------------------

@pytest.mark.parametrize("prompt,expected", [
    ("still you not able to givw currect data to me like file i did not get "
     "in last chat remind last 2 chats", True),
    ("remind last 3 chats", True),
    ("what did i ask before", True),
    ("what was my last request", True),
    ("recap", True),
    ("export page 8 to f1.xlsx", False),
    ("how many rows are in this table", False),
])
def test_a_question_about_the_conversation_is_recognised(prompt, expected):
    assert _is_recall_request(prompt) is expected


@pytest.mark.parametrize("prompt,expected", [
    ("remind last 2 chats", 2),
    ("remind me the last three messages", 3),
    ("what did i ask in the last 10 messages", 10),
    ("remind me", 3),          # no number named
    ("recap the last 500 chats", 20),   # capped
])
def test_how_many_exchanges_the_user_asked_for(prompt, expected):
    assert _recall_count(prompt) == expected


def test_the_last_exchanges_pair_each_question_with_its_answer(monkeypatch):
    monkeypatch.setattr(agent, "thread_messages", lambda sid: [
        {"role": "user", "content": "what is on page 6"},
        {"role": "assistant", "content": "Page 6 covers Make in India."},
        {"role": "user", "content": "export page 6 and 7"},
        {"role": "assistant", "content": "17 rows saved."},
        {"role": "user", "content": "export pages 6 to 10"},
        {"role": "assistant", "content": "No table on those pages."},
    ])

    last_two = recent_exchanges("s", 2)

    assert [p["asked"] for p in last_two] == ["export page 6 and 7",
                                              "export pages 6 to 10"]
    assert last_two[0]["answered"] == "17 rows saved."


def test_a_question_with_no_answer_yet_still_appears(monkeypatch):
    monkeypatch.setattr(agent, "thread_messages", lambda sid: [
        {"role": "user", "content": "remind last chat"},
    ])

    assert recent_exchanges("s", 3) == [{"asked": "remind last chat",
                                         "answered": ""}]
