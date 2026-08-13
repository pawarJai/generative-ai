"""
The orchestrator: takes a QueryPlan and executes it. This is the one
function that should ever grow a new `if plan.intent == ...:` branch —
everything it calls lives in its own module.
"""
from typing import List, Optional
from app import state
from app.config import llm, vector_db
from app.models import QueryPlan
from app.query.planner import plan_query
from app.query.code_exec import run_code_on_files
from app.query.data_query import WIDE_TABLE_COLUMN_THRESHOLD, format_as_record_cards
from app.tables.helpers import get_all_real_tables, get_page_markdown
from app.export.exporters import EXPORTERS
from app.export.text_export import save_text_as, clean_llm_csv
from app.agent.tools import agent_executor
from app.logging_utils import log_interaction
import re
import time
import io
import pandas as pd



def _record_turn(session_id, fid, prompt, plan, resp, success, latency, error=None) -> None:
    """Log the turn, append it to session history, and remember it.

    History is appended HERE for every intent rather than inside individual
    branches: previously only qa/page_lookup/complex wrote to history, so
    `chat_history` reported "no previous messages" after a data_query turn.
    Branches that need prior context read history during dispatch, which
    still sees the pre-turn state because this runs afterwards."""
    history = state.get_session_history(session_id)
    history.add_user_message(prompt)
    history.add_ai_message(resp if isinstance(resp, str) else str(resp))

    state.LAST_INTERACTION[session_id] = log_interaction(
        session_id, fid, prompt, plan, resp, success=success, error=error, latency=latency)
    state.LAST_TURN[session_id] = {
        "prompt": prompt,
        "intent": getattr(plan, "intent", None),
        "success": success,
        "error": error,
        "latency_sec": round(latency, 2) if latency is not None else None,
        "response": resp,
    }


def _resolve_target_files(explicit_fid: Optional[str], plan: QueryPlan) -> List[str]:
    """Which file_id(s) a multi-file-aware intent (currently data_query) should
    actually load. Priority:
      1. file_id explicitly passed in the API request
      2. files named/referenced in the prompt text (plan.file_scope, set by
         resolve_file_scope in planner.py)
      3. every ingested file

    Never narrows to "whatever was uploaded last" — that silent narrowing
    (ACTIVE_FILE_ID overwritten on every /ingest call) was the single-file
    lock-in bug: upload 5 files, only the most recent one was queryable."""
    if explicit_fid:
        return [explicit_fid]
    if plan.file_scope:
        return plan.file_scope
    return list(state.FILE_ORDER)


def _chat_dispatch(prompt: str, session_id: str, fid: Optional[str], plan: QueryPlan,
                   explicit_fid: Optional[str] = None) -> str:
    """Always returns a string response. Wide-table results are formatted as record cards."""
    if plan.intent == "list_files":
        if not state.FILE_ORDER:
            return "You haven't uploaded any files yet."
        lines = [f"You've uploaded {len(state.FILE_ORDER)} file(s):"]
        for i, f in enumerate(state.FILE_ORDER, 1):
            name = state.FILE_ORIGINAL_NAME.get(f, f)
            kind = state.FILE_KIND.get(f, "unknown")
            n_tables = len(get_all_real_tables(f))
            lines.append(f"{i}. {name} — file_id={f}, kind={kind}, {n_tables} table(s)")
        return "\n".join(lines)

    if plan.intent == "general":
        # No connection to any uploaded file — must never touch vector search
        # or file state. Route through the tool-equipped agent (web search,
        # weather, calculator) instead of guessing or hedging about "the
        # uploaded document".
        history = state.get_session_history(session_id)
        try:
            result = agent_executor.invoke({"input": prompt, "chat_history": history.messages})
            return result["output"]
        except Exception:
            return llm.invoke(prompt).content

    if plan.intent == "chat_history":
        history = state.get_session_history(session_id)
        if not history.messages:
            return "No previous messages in this session yet."
        recent = history.messages[-6:]
        lines = [f"{'You' if m.type == 'human' else 'Assistant'}: "
                 f"{m.content[:300]}{'…' if len(m.content) > 300 else ''}"
                 for m in recent]
        return "Here's what was said recently:\n\n" + "\n\n".join(lines)

    if plan.intent == "ai_meta":
        # Never answer a question about the assistant's own behaviour by
        # searching the document — that produced confidently-wrong answers
        # assembled from unrelated document content.
        last = state.LAST_TURN.get(session_id)
        if last and not last.get("success", True):
            return (f"The previous request did fail. Error: {last.get('error')}\n"
                    f"Prompt was: {last.get('prompt')}")
        if last:
            return ("The previous request completed without an error on my side "
                    f"(intent={last.get('intent')}, {last.get('latency_sec')}s). "
                    "If the result was wrong, tell me what you expected instead and "
                    "I'll re-run it differently.")
        return ("I don't have a recorded failure for this session. Tell me what "
                "you expected to happen and I'll look into it.")

    if plan.intent == "data_query":
        # ONE tool handles every factual/tabular question — sheet names, counts,
        # filters, column values — by writing real pandas against the real schema.
        # No per-phrasing intent category needed. Multi-file aware: loads every
        # file the prompt could plausibly mean, not just the single active one.
        target_files = _resolve_target_files(explicit_fid, plan)
        if not target_files:
            return "No files are uploaded yet, so there's no data to query."
        result = run_code_on_files(target_files, prompt)

        if plan.sink and result.get("table"):
            df = pd.DataFrame(result["table"]["rows"], columns=result["table"]["columns"])
            export_plan = QueryPlan(intent="export", filename=plan.filename, sink=plan.sink)
            # verify_export inside EXPORTERS confirms the file really landed on disk.
            # Label reflects every source file when the export spans more than one.
            export_label = "+".join(target_files) if len(target_files) > 1 else target_files[0]
            return EXPORTERS[plan.sink](export_label, export_plan, tables=[df])

        tbl = result.get("table")
        if tbl:
            df = pd.DataFrame(tbl["rows"], columns=tbl["columns"])
            if len(tbl["columns"]) > WIDE_TABLE_COLUMN_THRESHOLD:
                return f"{result['text']}\n\n{format_as_record_cards(df)}"
            return f"{result['text']}\n\n{df.to_string(index=False)}"
        return result["text"]

    if plan.intent == "generate":
        raw = llm.invoke(prompt).content if plan.sink != "csv" else llm.invoke(
            f"{prompt}\n\n(Output ONLY the CSV content — a header row plus data rows. "
            f"No explanation, no preamble, no markdown fences, nothing else.)"
        ).content
        if not plan.sink:
            return raw
        if plan.sink == "csv":
            from app.config import OUTPUT_DIR
            from app.export.exporters import verify_export
            import os
            cleaned = clean_llm_csv(raw)
            out = plan.filename or "generated.csv"
            try:
                df = pd.read_csv(io.StringIO(cleaned))
                path = os.path.join(OUTPUT_DIR, out)
                df.to_csv(path, index=False)
                expected = len(df)
                # Verify the file was actually written
                success, msg = verify_export(path, expected)
                return msg
            except Exception as e:
                path = os.path.join(OUTPUT_DIR, out)
                with open(path, "w") as f:
                    f.write(cleaned)
                return f"Model output wasn't valid CSV ({e}); wrote raw text -> {path}. Inspect it."
        if plan.sink in ("excel", "docx", "pptx", "chart"):
            try:
                df = pd.read_csv(io.StringIO(clean_llm_csv(raw)))
            except Exception as e:
                return f"Couldn't parse generated content as tabular data ({e})."
            fake_plan = QueryPlan(intent="generate", filename=plan.filename)
            return EXPORTERS[plan.sink](fid or "generated", fake_plan, tables=[df])
        return raw

    if plan.intent == "overview":
        scope = plan.file_scope or [fid]
        if len(scope) == 1:
            meta = state.FILE_META.get(scope[0], {})
            text = meta.get("summary") or "No cached overview yet — re-run ingest for this file."
        else:
            parts = []
            for f in scope:
                name = state.FILE_ORIGINAL_NAME.get(f, f)
                summ = state.FILE_META.get(f, {}).get("summary") or "(no cached overview)"
                parts.append(f"--- {name} ({f}) ---\n{summ}")
            text = "\n\n".join(parts)
        return save_text_as(text, plan.sink, plan.filename, f"{fid}_overview", f"Overview — {fid}") \
            if plan.sink else text

    if plan.intent == "table_of_contents":
        toc = state.FILE_META.get(fid, {}).get("toc", [])
        text = ("\n".join(f"- {t['text']} (p.{t['page']})" for t in toc) if toc else
                "No headings/sections detected.")
        return save_text_as(text, plan.sink, plan.filename, f"{fid}_toc", f"Table of Contents — {fid}") \
            if plan.sink else text

    if plan.intent == "export":
        fmt = plan.sink or "csv"
        return EXPORTERS[fmt](fid, plan)

    if plan.intent == "list_columns":
        tables = get_all_real_tables(fid)
        if not tables:
            return "No data tables found in this document."
        lines = [f"Found {len(tables)} data table(s) in this document.\n"]
        for i, df in enumerate(tables, 1):
            lines.append(f"Table {i} (p.{df.attrs.get('page')}) — {df.shape[0]} rows x {df.shape[1]} cols")
            lines.append(f"  Columns: {list(df.columns)}")
            lines.append(f"  Sample:\n{df.head(2).to_string(index=False)}\n")
        text = "\n".join(lines)
        return save_text_as(text, plan.sink, plan.filename, f"{fid}_tables", f"Tables — {fid}") \
            if plan.sink else text

    if plan.intent == "page_lookup" and plan.page_number:
        md = get_page_markdown(fid, plan.page_number)
        resp = llm.invoke(
            f"PAGE {plan.page_number} CONTENT (authoritative — answer only from this):\n{md}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer concisely. Do not add headers or bullet points unless the user asked for them."
        ).content
        return save_text_as(resp, plan.sink, plan.filename, f"{fid}_page{plan.page_number}",
                             f"Page {plan.page_number}") if plan.sink else resp

    if plan.intent == "complex":
        history = state.get_session_history(session_id)
        result = agent_executor.invoke({"input": prompt, "chat_history": history.messages})
        return result["output"]

    # default: qa — vector search + LLM, for narrative documents only.
    # Tabular fact questions should already have been routed to data_query
    # by the planner; this remains the fallback for prose PDFs.

    # Guard: if the question is about the AI's own behavior/errors (not document content),
    # don't pretend document chunks are relevant answers.
    ai_behavior_question = bool(re.search(
        r"\b(you (fail|failed|failing|wrong|error|crashed|broke|messed up|suck|stupid)|"
        r"why (did|do|can't|cannot) you|what.?s wrong with you|what.?s the problem with you|"
        r"fix (yourself|your|your mistake)|you made (a )?mistake|why (are|is) you)\b", prompt.lower()))

    if ai_behavior_question:
        return "I can't answer questions about my own behavior or errors by searching the document. " \
               "If you encountered a specific error with this file, please describe what happened " \
               "and I'll try to help troubleshoot."

    results = vector_db.similarity_search(prompt, k=6, filter={"file_id": fid})
    context = "\n\n".join(d.page_content for d in results) or "(no matching content found)"

    resp = llm.invoke(
        "Answer using ONLY the CONTEXT if it addresses the question. If it doesn't but "
        "you can answer from general knowledge, answer and prefix with '(Not from the "
        "uploaded document — general knowledge)'. If the question needs live/real-time "
        "data, say so instead of guessing. Never state something as fact if unsure.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"
    ).content
    return resp


def _available_files() -> List[dict]:
    return [
        {"file_id": f, "name": state.FILE_ORIGINAL_NAME.get(f, f),
         "kind": state.FILE_KIND.get(f, "unknown")}
        for f in state.FILE_ORDER
    ]


def chat(prompt: str, session_id: str = "default", file_id: Optional[str] = None) -> dict:
    fid = file_id or state.get_active_file_id()
    t0 = time.time()
    plan = None
    try:
        plan = plan_query(prompt, session_id=session_id, fid=fid)

        # General chat mode (no file ever ingested) - use web search tools
        if fid is None or fid == "general_chat":
            history = state.get_session_history(session_id)
            try:
                result = agent_executor.invoke({"input": prompt, "chat_history": history.messages})
                resp = result["output"]
            except Exception:
                # Fallback if agent fails
                resp = llm.invoke(prompt).content
            _record_turn(session_id, fid or "general_chat", prompt,
                         plan or QueryPlan(intent="qa"), resp,
                         success=True, latency=time.time() - t0)
            return {"response": resp, "intent": "qa", "sink": None,
                    "filename": None, "file_id": fid or "general_chat",
                    "available_files": _available_files(), "target_files": []}

        raw = _chat_dispatch(prompt, session_id, fid, plan, explicit_fid=file_id)
        if isinstance(raw, dict):
            resp, table, sql = raw.get("text", ""), raw.get("table"), raw.get("sql")
        else:
            resp, table, sql = raw, None, None
        _record_turn(session_id, fid, prompt, plan, resp,
                     success=True, latency=time.time() - t0)
        target_files = (_resolve_target_files(file_id, plan) if plan.intent == "data_query"
                        else ([fid] if fid else []))
        return {"response": resp, "intent": plan.intent, "sink": plan.sink,
                "filename": plan.filename, "file_id": fid, "table": table, "sql": sql,
                "available_files": _available_files(), "target_files": target_files}
    except Exception as e:
        import traceback; traceback.print_exc()
        resp = f"Something went wrong ({type(e).__name__}: {e})."
        _record_turn(session_id, fid, prompt, plan, resp, success=False,
                     error=str(e), latency=time.time() - t0)
        return {"response": resp, "intent": getattr(plan, "intent", "unknown"),
                "sink": None, "filename": None, "file_id": fid, "table": None, "sql": None,
                "available_files": _available_files(), "target_files": []}
