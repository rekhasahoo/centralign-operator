"""
LLM-backed Planner. Every method returns a *proposal* validated against typed schemas
(with one repair retry); the Engine still decides what actually happens.
Task-agnostic by construction: nothing here mentions invoices. Company context arrives as
procedure documents; capabilities arrive as the tool catalog.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Union

from pydantic import ValidationError

from .context import render_history
from .llm import LLMClient
from .planner import AskHuman, Done, ProposedFact, RecordFacts
from .registry import ToolRegistry
from .schemas import Event, Fact, FactRef, Plan, RunState, TaskSpec, ToolCall, ToolResult

SYSTEM = ("You are the planning brain of an autonomous enterprise operator. You never execute anything: "
          "you PROPOSE; a deterministic runtime validates, gates (permissions, human approvals), executes, and "
          "independently verifies. Be precise and never invent values. Reply with ONE JSON object and nothing else.")

UNDERSTAND_EXAMPLE = """{
  "goal": "Customer Priya Rao's phone number in the CRM is 98765 43210",
  "success_criteria": [{"id": "phone", "description": "CRM record shows the new phone number",
    "check": {"tool": "crm_get_customer", "args": {"name": "Priya Rao"},
              "expect": {"op": "eq", "path": "$.phone", "value": {"fact": "customer.phone"}}}}],
  "constraints": ["customer records may only be edited with the requester's approval"],
  "assumptions": ["'new number' means the number in the attached email, not the one on file"],
  "open_questions": []
}"""

ACTION_FORMAT = """Reply with exactly one of:
{"action": "tool", "tool": "<name>", "args": {...}, "rationale": "<why this step, one sentence>", "step_id": "<plan step id or null>"}
{"action": "record_facts", "facts": [{"key": "<fact key>", "value": <value>, "source_call_id": "<call_id from history>", "excerpt": "<VERBATIM text copied from that tool result>", "confidence": 0.0-1.0}]}
{"action": "ask_human", "question": "<specific question>", "blocked_on": "<what you cannot safely decide>"}
{"action": "done", "summary": "<what you did and the evidence>"}
The value of "action" must be exactly one of: tool, record_facts, ask_human, done. To use a tool, set "action" to "tool" and put the tool's name in "tool" (never put a tool name in "action")."""


class PlannerError(RuntimeError):
    pass


def _fact_keys(spec: TaskSpec) -> list[str]:
    keys: list[str] = []

    def walk(x: Any):
        if isinstance(x, FactRef): keys.append(x.fact)
        elif isinstance(x, dict):
            if set(x) == {"fact"}: keys.append(x["fact"])
            elif set(x) == {"$fact"}: keys.append(x["$fact"])
            else: [walk(v) for v in x.values()]
        elif isinstance(x, list): [walk(v) for v in x]
    for c in spec.success_criteria:
        walk(c.check.args); walk(c.check.expect.value)
    return list(dict.fromkeys(keys))


class LLMPlanner:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, knowledge_dir: str | None = None):
        self.llm, self.registry = llm, registry
        docs = sorted(pathlib.Path(knowledge_dir).glob("*.md")) if knowledge_dir else []
        # Prototype: all procedures fit in context. At scale this becomes retrieval (vector store).
        self.procedures = "\n\n".join(d.read_text() for d in docs) or "(no company procedures provided)"

    # ------------------------------------------------------------------ plumbing
    def _ask(self, stage: str, body: str, parse: Callable[[dict], Any]) -> Any:
        prompt = f"STAGE: {stage}\n{body}"
        err = ""
        for _ in range(2):
            try:
                return parse(self.llm.complete_json(SYSTEM, prompt + (f"\n\nYour previous reply was invalid ({err}). Reply again with corrected JSON only." if err else "")))
            except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                err = str(e)[:300]
        raise PlannerError(f"{stage}: model output invalid after retry: {err}")

    def _tools(self, read_only: bool = False) -> str:
        lines = []
        for t in self.registry.catalog():
            if read_only and t["risk"] != "read":
                continue
            props = t["input_schema"].get("properties", {})
            req = set(t["input_schema"].get("required", []))
            args = ", ".join(f"{k}{'*' if k in req else ''}:{v.get('type', 'any')}" for k, v in props.items())
            lines.append(f"- {t['name']} [{t['risk']}] ({args}): {t['description']}")
        return "\n".join(lines)

    def _env(self) -> str:
        return f"ENVIRONMENT NOTES (where things live):\n{self.registry.notes}\n\n" if self.registry.notes else ""

    def _task(self, s: RunState) -> str:
        sp = s.spec
        crit = "\n".join(f"  - {c.id}: {c.description}" for c in sp.success_criteria)
        have = {k for k in s.facts}
        need = _fact_keys(sp)
        facts = "\n".join(f"  - {k} = {f.value!r}" for k, f in s.facts.items()) or "  (none yet)"
        clar = "".join(f"\n  Q: {c['question']}\n  A: {c['answer']}" for c in s.clarifications)
        return (f"GOAL: {sp.goal}\nSUCCESS CRITERIA (verified independently afterwards):\n{crit}\n"
                f"CONSTRAINTS: {sp.constraints}\nASSUMPTIONS: {sp.assumptions}\n"
                f"FACTS RECORDED:\n{facts}\nFACTS STILL MISSING: {[k for k in need if k not in have] or 'none'}"
                + (f"\nCLARIFICATIONS FROM HUMAN:{clar}" if clar else ""))

    # ------------------------------------------------------------------ Planner API
    def understand(self, state: RunState, events: list[Event]) -> TaskSpec:
        clar = "".join(f"\nQ: {c['question']}\nA: {c['answer']}" for c in state.clarifications) or " (none)"
        body = f"""REQUEST: {state.request}
CLARIFICATIONS FROM HUMAN:{clar}

{self._env()}COMPANY PROCEDURES:
{self.procedures}

TOOLS YOU WILL BE ABLE TO USE (read-only ones are usable in success checks):
{self._tools()}

Turn the request into a checkable task. Reply with JSON:
{{"goal": str, "success_criteria": [{{"id": str, "description": str, "check": {{"tool": str, "args": {{...}}, "expect": {{"op": "eq|contains|exists|matches", "path": "$.field", "value": ...}}}}}}], "constraints": [str], "assumptions": [str], "open_questions": [str]}}

Rules:
- goal: restate the desired OUTCOME, not the steps.
- success_criteria: 1-4 checks proving the outcome in the REAL system. Each uses exactly one READ-ONLY tool. `path` points into that tool's output.
- Values that are only known from source material (amounts, dates, ids) must be written as {{"fact": "<key>"}} in expect.value, and as {{"$fact": "<key>"}} inside check.args. Use dotted lowercase keys. The agent will record these facts later from evidence. Define each fact in the format the system of record stores (follow the procedures).
- constraints: limits from the request or procedures. assumptions: interpretations you made (say what ambiguous words like "latest" mean).
- open_questions: ONLY ambiguity that cannot be resolved from the request, procedures, or by inspecting files/systems with tools. Prefer [].

Example of the expected JSON shape (different domain):
{UNDERSTAND_EXAMPLE}"""
        return self._ask("understand", body, lambda d: TaskSpec.model_validate(d))

    def plan(self, state: RunState, events: list[Event]) -> Plan:
        body = f"""{self._task(state)}

{self._env()}COMPANY PROCEDURES:
{self.procedures}

TOOLS:
{self._tools()}

Write a short high-level plan (3-8 steps: intent only, not tool calls). Reply with JSON:
{{"steps": [{{"id": "s1", "intent": str, "depends_on": [str]}}], "rationale": str}}"""
        return self._ask("plan", body, lambda d: Plan.model_validate({**d, "version": 1}))

    def replan(self, state: RunState, events: list[Event], reason: str) -> Plan:
        body = f"""{self._task(state)}

WHY WE ARE REPLANNING: {reason}

HISTORY:
{render_history(events, budget=12000)}

{self._env()}COMPANY PROCEDURES:
{self.procedures}

TOOLS:
{self._tools()}

Revise the plan to fix what went wrong. Do not repeat an approach that already failed. Reply with JSON:
{{"steps": [{{"id": "s1", "intent": str, "depends_on": [str]}}], "rationale": str}}"""
        v = (state.plan.version if state.plan else 0) + 1
        return self._ask("replan", body, lambda d: Plan.model_validate({**d, "version": v}))

    def next_action(self, state: RunState, events: list[Event]) -> Union[ToolCall, Done, AskHuman, RecordFacts]:
        plan = "\n".join(f"  {st.id}: {st.intent}" for st in state.plan.steps) if state.plan else "(none)"
        body = f"""{self._task(state)}

PLAN (v{state.plan.version if state.plan else 0}):
{plan}

{self._env()}COMPANY PROCEDURES:
{self.procedures}

TOOLS:
{self._tools()}

BUDGET: {state.tool_calls_made} tool calls used so far.

HISTORY (oldest to newest; [call_id] labels let you cite evidence):
{render_history(events)}

Decide the single next action.
Rules:
- One action per turn. Never repeat a call that just failed in the same way; adapt (re-read the page, fix arguments, try another tool) or ask the human.
- Observe before acting: browser element refs (e1, e2...) come from the latest page snapshot only and reset on navigation.
- SELECTING AMONG CANDIDATES ("latest", "largest", "the one for X"): inspect ALL candidate sources before choosing. Do not exclude a source because of its name or metadata; names can be uninformative or misleading. Choose using the evidence inside them, and say which ones you compared.
- Facts: record every fact in FACTS STILL MISSING once you have evidence. Compare candidates first if the task needs selecting among several sources (e.g. "latest"), then record from the chosen one. Each fact needs source_call_id and an excerpt copied VERBATIM from that tool result; the runtime rejects anything it cannot find. Store values in the format the target system expects (see procedures).
- Follow COMPANY PROCEDURES exactly. Some actions pause for human approval automatically; just propose them.
- Do not act on a fact that is missing or was rejected. Do not invent values.
- Answer "done" when you believe the goal is met; an independent check runs afterwards and will tell you if you were wrong.
- DISCOVER BEFORE ASKING: if you do not know where something is, look with your tools first (e.g. list_files '.'). Never guess paths or URLs; use ENVIRONMENT NOTES.
- ask_human only for what tools cannot reveal: a decision, a permission, or information that exists only in the human's head.

{ACTION_FORMAT}"""
        try:
            return self._ask("next_action", body, self._parse_action)
        except PlannerError as e:
            return AskHuman(question=f"I could not produce a valid next step ({e}). How should I proceed?", blocked_on="planner_error")

    def extract_facts(self, state, call, result: ToolResult, source_event_id: str) -> list[Fact]:
        return []   # facts are proposed explicitly via `record_facts` and grounded by the runtime

    # ------------------------------------------------------------------
    def _parse_action(self, d: dict) -> Union[ToolCall, Done, AskHuman, RecordFacts]:
        a = d.get("action")
        if a not in ("tool", "record_facts", "ask_human", "done") and self.registry.get(str(a)):
            # tolerate {"action": "browser_select", "ref": ...}: the model put the tool name in "action"
            args = d.get("args") if isinstance(d.get("args"), dict) else {k: v for k, v in d.items()
                                                                           if k not in ("action", "rationale", "step_id", "tool")}
            return ToolCall(tool=str(a), args=args, rationale=d.get("rationale", ""), step_id=d.get("step_id"))
        if a == "tool":
            if not isinstance(d.get("args", {}), dict):
                raise ValueError("args must be an object")
            return ToolCall(tool=d["tool"], args=d.get("args", {}), rationale=d.get("rationale", ""), step_id=d.get("step_id"))
        if a == "record_facts":
            return RecordFacts([ProposedFact(key=f["key"], value=f["value"], source_call_id=f["source_call_id"],
                                             excerpt=f["excerpt"], confidence=float(f.get("confidence", 1.0)))
                                for f in d["facts"]])
        if a == "ask_human":
            return AskHuman(question=d["question"], blocked_on=d.get("blocked_on", "unspecified"))
        if a == "done":
            return Done(summary=d.get("summary", ""))
        raise ValueError(f"unknown action {a!r}: action must be one of tool|record_facts|ask_human|done; "
                         "to call a tool use {\"action\": \"tool\", \"tool\": \"<name>\", \"args\": {...}}")
