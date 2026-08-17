# Fix: `create_excel_template` gave two different results for one request

**Date:** 2026-08-14
**Session affected:** `7e43a023-14b8-4055-8b87-cdd6bb4ea5c3` (6 files uploaded: `data-file-1` … `5`, `sbs-f2`)
**Files changed:** `app/graph/tools.py`, `app/graph/agent.py`
**Tests added:** `tests/test_create_excel_template.py::test_omitting_file_id_resolves_the_same_document_as_passing_it_explicitly`

## What the user saw

One request —

> create excel file based on this columsn = 1.Sr No 2.Group … 31.Total Value
> for All Shipset, export in w1-02.xlsx

— produced this reply:

> **No file was created.** I claimed one was; nothing had written it, and
> running the export directly did not succeed either:
> ✓ Created w1-02.xlsx — 31 columns, 17 row(s). Filled from
> 'data-file-2…pdf': Category, Rating. Left blank …
> The export attempt this turn returned:
> ✓ Created w1-02.xlsx — 31 columns, 0 row(s). Left blank …

Two different files, both claiming success, pasted back-to-back with the
model narrating its own confusion. The user (reasonably) read this as a
failure/lie, having already been burned by fabricated success claims
earlier in the same investigation.

## Root cause

`_build_template_export` (`app/graph/tools.py`) resolves which uploaded
document to pull matching columns from. The old logic:

```python
target = None
if file_id and _is_known_file(file_id):
    target = file_id
elif not file_id and len(app_state.FILE_ORDER) == 1:
    target = app_state.FILE_ORDER[0]
```

This only auto-resolves a document when **exactly one** file is loaded.
Every other tool in this file that needs the same thing calls
`resolve_target_file(file_id, question)`, which additionally checks a file
named in the prompt and the session's active file — `_build_template_export`
never did.

With 6 files uploaded, `len(FILE_ORDER) == 1` was always false. So the tool's
result depended entirely on the model passing `file_id` on *every* call. It
did on the first call (resolved to `data-file-2_7210`, matched `Category`/
`Rating` in one of that document's tables, 17 rows). On a retry within the
same turn it omitted `file_id` — plausible ReAct-loop variance, not a
one-off — and `target` silently fell to `None`, producing a fully blank
0-row template. Same request, same session, two contradictory files, and no
error surfaced anywhere to explain why.

The confusing reply itself is a secondary effect: Rule 11 of the system
prompt told the model to "say exactly that" when attempts disagree, which it
interpreted as quoting both raw tool outputs verbatim with first-person
narration ("I claimed one was...").

## Fix

1. **`app/graph/tools.py`** — `_build_template_export` now resolves an
   omitted `file_id` via the existing `resolve_target_file(None, prompt)`
   helper (file named in the prompt → active file → lone loaded file), the
   same precedence `export_data` already uses. This makes repeated calls in
   one session resolve to the same document regardless of whether the model
   remembers to pass `file_id` every time.
2. Updated the `create_excel_template` docstring to describe the new
   resolution order and warn the model that an inconsistent `file_id` across
   retries of the same request can resolve to a different document.
3. **`app/graph/agent.py`**, system prompt Rule 11 — added an explicit
   instruction: report only the *last* tool call's result, in plain words;
   never quote or compare raw tool-output text from more than one attempt in
   the same reply.

## Verification

- New regression test reproduces the exact shape of the bug: 3 files
  loaded, one active, a table with `Category`/`Rating`; calls
  `create_excel_template` once with `file_id` and once without, asserts
  both write identical data. Fails against the old code (second call comes
  back blank), passes against the fix.
- Full suite: 372 passed, 35 skipped, 0 failed (`/tmp/full_test_run10.log`).
