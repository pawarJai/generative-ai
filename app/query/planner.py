"""Intent routing. Deterministic regex/keyword checks fire first for cases a
small LLM planner reliably misroutes; the LLM planner is the fallback, with
guardrails afterward to correct known failure patterns."""
import re
from typing import Dict, List, Optional
from app.config import llm
from app.models import QueryPlan
from app import state
from app.tables.helpers import resolve_file_scope


def extract_rename_clause(prompt: str) -> Dict[str, str]:
    out = {}
    for m in re.finditer(
        r"rename\s+['\"]?([^'\",]+?)['\"]?\s+(?:to|as)\s+['\"]?([^'\",.\n]+?)['\"]?(?:[,.\n]|$)",
        prompt, re.IGNORECASE
    ):
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def extract_columns_clause(prompt: str) -> Optional[List[str]]:
    m = re.search(r"columns?\s*[:=]\s*(.+)$", prompt, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    parts = re.split(r"\t+|,\s*|\n|\s{2,}", m.group(1).strip())
    cols = [p.strip() for p in parts if p.strip()]
    return cols or None


def _plan_query_raw(prompt: str, fid: Optional[str]) -> QueryPlan:
    p = prompt.lower()
    filename_match = re.search(r"([\w\-./]+\.(csv|xlsx?|docx?|pptx?|png|jpe?g))", p)
    filename = filename_match.group(1) if filename_match else None
    sink_from_ext = {"csv": "csv", "xlsx": "excel", "xls": "excel", "docx": "docx",
                      "doc": "docx", "pptx": "pptx", "ppt": "pptx",
                      "png": "chart", "jpg": "chart", "jpeg": "chart"}
    detected_sink = sink_from_ext.get(filename_match.group(2)) if filename_match else None
    if not detected_sink:
        for kw, fmt in [("csv", "csv"), ("excel", "excel"), ("xlsx", "excel"),
                         ("word", "docx"), ("docx", "docx"), ("pptx", "pptx"),
                         ("ppt", "pptx"), ("slide", "pptx"), ("chart", "chart"),
                         ("image", "chart"), ("png", "chart")]:
            if kw in p:
                detected_sink = fmt
                break

    # ---------------------------------------------------------------
    # NON-DATA fast paths MUST come first. These are not questions about
    # the document at all, so they must not be shadowed by the broad
    # data_fact_cues check below. (Regression this ordering fixes: a
    # `list_sheets` path placed after data_fact_cues was unreachable,
    # because data_fact_cues matches any prompt containing "sheet".)
    # ---------------------------------------------------------------
    has_action_verb = bool(re.search(r"\b(export|generate|create|save|write|make)\b", p))

    # Conversation-history requests — answered from session history, never
    # from the document or the data.
    if not has_action_verb and re.search(
        r"\b(remind me|remind (the |last )|what did (i|you|we) (say|tell|ask|do)|"
        r"last (chat|message|output|conversation|query|prompt|response|answer)|"
        r"previous (chat|message|request|query|prompt|response|answer)|"
        r"what (i|did i) (told|tell|said|say)|what was (my|the) last|"
        r"recap|chat history|conversation history)\b", p
    ):
        return QueryPlan(intent="chat_history")

    # Questions about the assistant's own behaviour/errors. Searching the
    # document for these produced confidently-wrong answers built from
    # unrelated document content (confirmed in production logs).
    if not has_action_verb and re.search(
        r"\b(you (fail|failed|failing|fails|lie|lied|lying|wrong|broke|broken|"
        r"crashed|suck|stupid)|wh(y|ay) (you|are you|did you|do you|does it|is it)|"
        r"what.?s wrong with you|you (gave|give) (me )?(wrong|bad|sql)|"
        r"not working|doesn.?t work|didn.?t work)\b", p
    ):
        return QueryPlan(intent="ai_meta")

    # Meta question about which files exist — must not fall into data_query
    # (there's no "data" to run code against, it's a question about state.py's
    # FILE_ORDER) and must not fall into 'general' (it IS about the files).
    if re.search(
        r"\b(how many files|how many documents|what files|which files|"
        r"list (of )?(my |the |uploaded )?files|files (have i|did i) (upload|ingest)|"
        r"what (have i|did i) upload(ed)?)\b", p
    ):
        return QueryPlan(intent="list_files")

    # ---------------------------------------------------------------
    # GENERAL knowledge questions — no connection to any uploaded file.
    # These must never touch vector search or file state: "what's the
    # weather", "what is 25*4", casual conversation. Doing so previously
    # ran a vector search against whatever file happened to be active and
    # produced a hedged non-answer instead of a real one.
    # ---------------------------------------------------------------
    file_language = re.search(
        r"\b(file|document|sheet|table|upload|data|column|row|csv|excel|xlsx|xls|"
        r"pdf|export|working.?sheet|cover.?sheet|record|dataset|report|page\s*\d+|"
        r"generate|create|save|summary|summarize|overview|toc|chapter|section)\b", p)
    if not file_language:
        # Genuine general-knowledge cues (math, weather, trivia) always win, even
        # with a file active — "what is 25 * 4" must never search the document.
        generic_knowledge = re.search(
            r"(\d+\s*[\+\-\*/]\s*\d+)|"
            r"\b(weather|forecast|temperature in|fun fact|trivia|who is|who was|"
            r"capital of|define|meaning of)\b", p)
        if generic_knowledge or not fid:
            return QueryPlan(intent="general")
        # No file/data keywords, but a file IS actively selected and this isn't a
        # generic-knowledge question — fall through to the DATA questions logic
        # below instead of dead-ending in the general agent, which has no tool to
        # read the uploaded file at all (e.g. "what is the total quantity").

    # ---------------------------------------------------------------
    # DATA questions. One intent, no per-phrasing categories: the
    # code-execution tool writes real pandas against the real schema, so
    # "list sheet names", "how many rows", "filter by X", "give me column
    # Y" all resolve without a dedicated regex for each. Multi-file aware:
    # checks every file the prompt could plausibly target (named files, or
    # every ingested file if none is named), not just the single active one
    # — a prompt mentioning a tabular file must route to data_query even if
    # a different (non-tabular) file happens to be ACTIVE_FILE_ID.
    # ---------------------------------------------------------------
    scope_candidates = resolve_file_scope(prompt) or ([fid] if fid else list(state.FILE_ORDER))
    is_tabular_scope = any(state.FILE_KIND.get(f) == "tabular" for f in scope_candidates)
    if not is_tabular_scope and re.search(r"\b(sheets?|csv|xlsx?)\b", p):
        is_tabular_scope = True
    data_fact_cues = re.search(
        r"\b(rows?|records?|how many|total|count|find|filter|value|list|show|give|"
        r"user_id|user[_ ]name|columns?|fields?|sheets?|tables?|data\b|combine|merge)\b", p)
    non_data_intent = re.search(r"\b(overview|summary|summarize|generate|create)\b", p)
    if is_tabular_scope and data_fact_cues and not re.search(r"columns?\s*[:=]", p) \
       and not non_data_intent:
        return QueryPlan(intent="data_query", sink=detected_sink, filename=filename)

    generate_cues = re.search(
        r"\b(generate|create)\b.{0,40}\b(rows?|data|records?|dataset|sample|dummy|"
        r"fake|synthetic|realistic)\b", p)
    references_doc = re.search(r"\b(this (file|document|pdf)|the (file|document|pdf)|"
                                r"uploaded|extracted|provided file)\b", p)
    if generate_cues and not references_doc:
        return QueryPlan(intent="generate", sink=detected_sink, filename=filename)

    if re.search(r"\b(overview|summary|summarize|what.?s this (document|file|pdf) about)\b", p) \
       and not re.search(r"\bcolumns?|headers?|fields?\b", p):
        return QueryPlan(intent="overview", sink=detected_sink, filename=filename)
    if re.search(r"\b(chapters?|table of contents|toc|sections?|headings?)\b", p) \
       and not re.search(r"\bpage\s*\d+\b", p):
        return QueryPlan(intent="table_of_contents", sink=detected_sink, filename=filename)
    if re.search(r"\b(how many tables?|what tables?|table data|tables? (are|is) (there|present|found)|"
                 r"describe the tables?|what.?s in the tables?|"
                 r"(what|list|show|get|give( me)?|all) (the )?(columns?|headers?|fields?))\b", p) \
       and not re.search(r"\bpage\s*\d+\b", p) \
       and not re.search(r"columns?\s*[:=]", p):
        return QueryPlan(intent="list_columns", sink=detected_sink, filename=filename)

    wants_file = bool(detected_sink) or bool(re.search(
        r"\b(export|save|store|download|write|create.*file)\b", p))
    try:
        planner = llm.with_structured_output(QueryPlan)
        sys = ("Plan the user's request. Rules:\n"
               "- intent='generate' if they want NEW content created from a spec/schema/"
               "example they gave — never pulls from the uploaded document.\n"
               "- intent='export' ONLY if they want EXISTING data pulled FROM the uploaded "
               "document and saved.\n"
               "- intent='data_query' for any factual or tabular questions (filters, counts, values, lookup).\n"
               "- 'sink' = file format to ALSO save as, if any. A plain question has sink=null.\n"
               "- If they mention a specific page number -> page_lookup.\n"
               "- If they ask what columns/headers/fields exist -> list_columns (do NOT use for filtering or querying by column).\n"
               "- If the request needs several different outputs/steps -> complex.\n"
               "- Only fill filename/rename/columns if the user actually said them.")
        plan = planner.invoke([("system", sys), ("human", prompt)])
    except Exception as e:
        print(f"(planner fallback: {e})")
        plan = QueryPlan(intent="qa")

    if plan.intent == "generate" and not re.search(
        r"\b(sample|fake|dummy|synthetic|realistic|mock)\b", p):
        plan.intent = "export"

    if plan.intent == "export" and not wants_file:
        m = re.search(r"\bpage\s*(\d+)", p)
        if m:
            return QueryPlan(intent="page_lookup", page_number=int(m.group(1)))
        if re.search(r"\bcolumns?|schema|headers?|fields?\b", p):
            return QueryPlan(intent="list_columns")
        return QueryPlan(intent="qa")
    return plan


def plan_query(prompt: str, session_id: str = "default", fid: Optional[str] = None) -> QueryPlan:
    plan = _plan_query_raw(prompt, fid)

    det_rename = extract_rename_clause(prompt)
    if det_rename:
        plan.rename = {**(plan.rename or {}), **det_rename}
    det_columns = extract_columns_clause(prompt)
    if det_columns:
        plan.columns = det_columns

    scope = resolve_file_scope(prompt)
    if scope:
        plan.file_scope = scope

    if plan.intent in ("export", "generate") and not plan.columns and not plan.rename:
        if re.search(r"\b(as per my requirement|like (i said|before)|same columns|"
                      r"as (mentioned|discussed) (before|earlier))\b", prompt.lower()):
            remembered = state.LAST_EXPORT_SPEC.get(session_id)
            if remembered:
                plan.columns = plan.columns or remembered.get("columns")
                plan.rename = plan.rename or remembered.get("rename")

    if plan.intent == "export" and (plan.columns or plan.rename):
        state.LAST_EXPORT_SPEC[session_id] = {"columns": plan.columns, "rename": plan.rename}

    return plan
