"""
The run engine. Every iteration: fold the log into state, then take exactly one step.
Nothing lives in memory between iterations, so a run can park for hours (approval/input)
or survive a crash and resume from the log.
"""
from __future__ import annotations

import posixpath
import time
from typing import Callable, Optional
from uuid import uuid4

from pydantic import BaseModel

from . import recovery
from . import grounding
from .planner import AskHuman, Done, Planner, RecordFacts
from .policy import PolicyEngine
from .registry import ToolRegistry
from .schemas import (ApprovalRequested, ApprovalResolved, ErrorKind, EventType, FactRecorded,
                      Fact, FactRejected, InputProvided, InputRequested, PlanEvent, RecoveryChosen, RunCreated,
                      RunFinished, RunState, TaskUnderstood, ToolOutcome, ToolRequested,
                      ToolResult, VerificationResult, fold)
from .store import EventStore
from .verifier import Verifier

TERMINAL = {"completed", "failed", "escalated"}
ACTIONABLE = {"requested", "allowed", "approved", "denied", "rejected"}


class Engine:
    def __init__(self, store: EventStore, registry: ToolRegistry, policy: PolicyEngine,
                 planner: Planner, *, max_tool_calls: int = 40, max_replans: int = 3,
                 max_verify_failures: int = 2, max_iterations: int = 500, backoff_s: float = 0.5,
                 max_fact_rejections: int = 12, on_event: Optional[Callable] = None):
        self.store, self.registry, self.policy, self.planner = store, registry, policy, planner
        self.verifier = Verifier(registry)
        self.max_tool_calls, self.max_replans = max_tool_calls, max_replans
        self.max_verify_failures, self.max_iterations, self.backoff_s = max_verify_failures, max_iterations, backoff_s
        self.max_fact_rejections, self.on_event = max_fact_rejections, on_event

    # ---- public API --------------------------------------------------------------------
    def start(self, request: str, requester: str = "user") -> str:
        run_id = f"run_{uuid4().hex[:8]}"
        self._emit(run_id, EventType.RUN_CREATED, RunCreated(request=request, requester=requester))
        return run_id

    def resolve_approval(self, run_id: str, approval_id: str, approved: bool,
                         by: str = "user", note: str = "") -> RunState:
        s = self.state(run_id)
        if not s.pending_approval or s.pending_approval.approval_id != approval_id:
            raise ValueError("no such pending approval")
        self._emit(run_id, EventType.APPROVAL_RESOLVED,
                   ApprovalResolved(approval_id=approval_id, approved=approved, by=by, note=note))
        return self.run(run_id)

    def provide_input(self, run_id: str, request_id: str, answer: str) -> RunState:
        s = self.state(run_id)
        if not s.pending_input or s.pending_input.request_id != request_id:
            raise ValueError("no such pending input request")
        self._emit(run_id, EventType.INPUT_PROVIDED, InputProvided(request_id=request_id, answer=answer))
        return self.run(run_id)

    def state(self, run_id: str) -> RunState:
        return fold(run_id, self.store.read(run_id))

    # ---- the loop ----------------------------------------------------------------------
    def run(self, run_id: str) -> RunState:
        for _ in range(self.max_iterations):
            events = self.store.read(run_id)
            s = fold(run_id, events)
            if s.status in TERMINAL or s.pending_approval or s.pending_input:
                return s  # finished, or parked waiting for a human

            if s.tool_calls_made >= self.max_tool_calls:
                self._finish(run_id, "escalated", f"step budget ({self.max_tool_calls}) exhausted"); continue
            if s.fact_rejections > self.max_fact_rejections:
                self._finish(run_id, "escalated", "agent repeatedly proposed facts the evidence does not support"); continue
            if s.replans > self.max_replans:
                self._finish(run_id, "escalated", f"replan budget ({self.max_replans}) exhausted"); continue

            if s.spec is None:
                self._emit(run_id, EventType.TASK_UNDERSTOOD, TaskUnderstood(spec=self.planner.understand(s, events)))
                continue
            if s.spec.open_questions:
                if not s.clarifications:  # ask once, before acting on an ambiguous goal
                    self._emit(run_id, EventType.INPUT_REQUESTED, InputRequested(
                        question="; ".join(s.spec.open_questions), blocked_on="task_ambiguity"))
                else:                     # got answers: re-understand, proceed with stated assumptions
                    spec = self.planner.understand(s, events)
                    spec.open_questions = []
                    self._emit(run_id, EventType.TASK_UNDERSTOOD, TaskUnderstood(spec=spec))
                continue
            if s.plan is None:
                self._emit(run_id, EventType.PLAN_CREATED, PlanEvent(plan=self.planner.plan(s, events)))
                continue

            pending = next(((cid, st) for cid, st in s.call_status.items() if st in ACTIONABLE), None)
            if pending:
                self._handle_call(run_id, s, *pending)
                continue

            action = self.planner.next_action(s, events)
            if isinstance(action, AskHuman):
                self._emit(run_id, EventType.INPUT_REQUESTED,
                           InputRequested(question=action.question, blocked_on=action.blocked_on))
            elif isinstance(action, RecordFacts):
                self._record_facts(run_id, events, action)
            elif isinstance(action, Done):
                self._verify_and_finish(run_id, s, action.summary)
            else:
                self._emit(run_id, EventType.TOOL_REQUESTED, ToolRequested(call=action))
        self._finish(run_id, "escalated", "iteration limit hit")
        return self.state(run_id)

    # ---- call lifecycle ----------------------------------------------------------------
    def _handle_call(self, run_id: str, s: RunState, call_id: str, status: str) -> None:
        call = s.calls[call_id]
        if status == "requested":
            tool = self.registry.get(call.tool)
            effect = self.registry.preview(call)
            if tool and tool.spec.risk.value in ("write_irreversible", "external_comms"):
                unread = self._unread_sources(self.store.read(run_id))
                if unread:
                    effect["_unread_sources"] = unread
            d = self.policy.decide(call, tool.spec if tool else None, s, effect)
            self._emit(run_id, EventType.POLICY_DECIDED, d)
            if d.decision == "require_approval":
                shown = f" Concretely: { {k: v for k, v in effect.items() if k != '_page'} }." if effect else ""
                if s.facts:
                    shown += " Facts relied on: " + "; ".join(
                        f"{k}={f.value!r} (evidence: \"{f.source_excerpt[:70]}\")" for k, f in s.facts.items()) + "."
                self._emit(run_id, EventType.APPROVAL_REQUESTED, ApprovalRequested(
                    call_id=call_id, risk=tool.spec.risk,
                    summary=f"About to run {call.tool}({call.args}).{shown} Why: {call.rationale}. Needs approval because: {d.reason}"))
        elif status in ("allowed", "approved"):
            self._execute(run_id, call)
        else:  # denied by policy, or rejected by a human: never route around it, replan instead
            why = self._why_blocked(run_id, call_id, status)
            self._emit(run_id, EventType.RECOVERY_CHOSEN, RecoveryChosen(
                call_id=call_id, error_kind=ErrorKind.PERMISSION_DENIED, strategy="replan", detail=why))
            self._replan(run_id, why)

    def _unread_sources(self, events, max_listing: int = 25) -> list[str]:
        """Items that were enumerated (e.g. files listed) but never opened. Lets policy refuse an
        irreversible action that rests on a selection ("latest", "largest") made without seeing every candidate."""
        norm = lambda p: posixpath.normpath(str(p).replace("\\", "/")).lstrip("./") or "."  # noqa: E731
        calls = {e.payload["call"]["call_id"]: e.payload["call"] for e in events if e.type == EventType.TOOL_REQUESTED}
        listed, opened = set(), set()
        for e in events:
            if e.type != EventType.TOOL_SUCCEEDED:
                continue
            r = e.payload["result"]; call = calls.get(r["call_id"]); t = self.registry.get(call["tool"]) if call else None
            if not t:
                continue
            if t.spec.enumerates and isinstance(r.get("output"), list):
                base = call["args"].get("directory", ".")
                listed |= {norm(posixpath.join(str(base), row["name"])) for row in r["output"] if isinstance(row, dict) and "name" in row}
            if t.spec.inspects and t.spec.inspects in call["args"]:
                opened.add(norm(call["args"][t.spec.inspects]))
        return sorted(listed - opened) if 0 < len(listed) <= max_listing else []

    def _why_blocked(self, run_id: str, call_id: str, status: str) -> str:
        evs = self.store.read(run_id)
        call = self.state(run_id).calls[call_id]
        if status == "rejected":
            note = next((e.payload.get("note", "") for e in reversed(evs)
                         if e.type == EventType.APPROVAL_RESOLVED), "")
            return f"human rejected {call.tool}({call.args})" + (f": {note}" if note else "")
        reason = next((e.payload["reason"] for e in reversed(evs)
                       if e.type == EventType.POLICY_DECIDED and e.payload["call_id"] == call_id), "")
        return f"policy denied {call.tool}: {reason}"

    def _execute(self, run_id: str, call) -> None:
        tool = self.registry.get(call.tool)
        attempt = self.state(run_id).attempts.get(call.call_id, 0)
        while True:
            attempt += 1
            result = self.registry.execute(call)
            if result.ok:
                ev = self._emit(run_id, EventType.TOOL_SUCCEEDED, ToolOutcome(result=result, attempt=attempt))
                s = self.state(run_id)
                for f in self.planner.extract_facts(s, call, result, ev.event_id):
                    self._emit(run_id, EventType.FACT_RECORDED, FactRecorded(fact=f))
                return
            self._emit(run_id, EventType.TOOL_FAILED, ToolOutcome(result=result, attempt=attempt))
            landed = None
            if tool and tool.confirm and result.error.kind == ErrorKind.TRANSIENT:
                try:
                    landed = bool(tool.confirm(call.args))
                except Exception:  # noqa: BLE001 - unknown means unknown
                    landed = None
            strategy, detail = recovery.choose(result.error, tool, attempt, landed)
            self._emit(run_id, EventType.RECOVERY_CHOSEN, RecoveryChosen(
                call_id=call.call_id, error_kind=result.error.kind, strategy=strategy, detail=detail))
            if strategy == "retry":
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
                continue
            if strategy == "accept":
                ok = ToolResult(call_id=call.call_id, ok=True, duration_ms=result.duration_ms,
                                summary=f"{call.tool}: confirmed applied despite reported error")
                self._emit(run_id, EventType.TOOL_SUCCEEDED, ToolOutcome(result=ok, attempt=attempt))
            elif strategy == "replan":
                self._replan(run_id, f"{call.tool} failed: {detail}")
            elif strategy == "ask_human":
                self._emit(run_id, EventType.INPUT_REQUESTED, InputRequested(
                    question=f"{call.tool} failed and I can't safely continue: {detail}. How should I proceed?",
                    blocked_on=f"{call.tool} failure"))
            else:
                self._finish(run_id, "failed", detail)
            return

    # ---- verification ------------------------------------------------------------------
    def _verify_and_finish(self, run_id: str, s: RunState, claimed: str) -> None:
        if not s.spec.success_criteria:
            self._finish(run_id, "escalated", "no verifiable success criteria; refusing to claim completion")
            return
        results = self.verifier.run(s)
        for r in results:
            self._emit(run_id, EventType.VERIFICATION_RESULT, r)
        failed = [r for r in results if not r.passed]
        evidence = [a for r in results for a in r.evidence]
        if not failed:
            self._finish(run_id, "completed", claimed, evidence)
            return
        detail = "; ".join(f"{r.criterion_id}: expected {r.expected!r}, actual {r.actual!r}" for r in failed)
        if s.verify_failures >= self.max_verify_failures:
            self._finish(run_id, "escalated", f"verification still failing: {detail}", evidence)
            return
        self._emit(run_id, EventType.RECOVERY_CHOSEN, RecoveryChosen(
            call_id="verify", error_kind=ErrorKind.STATE_MISMATCH, strategy="replan", detail=detail))
        self._replan(run_id, f"verification failed: {detail}")

    def _record_facts(self, run_id: str, events, action: RecordFacts) -> None:
        """Facts enter working memory only if the cited evidence really contains them."""
        for pf in action.facts:
            ok, why, src_event_id = grounding.check(pf, events)
            if ok:
                self._emit(run_id, EventType.FACT_RECORDED, FactRecorded(fact=Fact(
                    key=pf.key, value=pf.value, source_event_id=src_event_id,
                    source_excerpt=pf.excerpt, confidence=pf.confidence)))
            else:
                self._emit(run_id, EventType.FACT_REJECTED, FactRejected(key=pf.key, reason=why, excerpt=pf.excerpt))

    # ---- helpers -----------------------------------------------------------------------
    def _replan(self, run_id: str, reason: str) -> None:
        events = self.store.read(run_id)
        plan = self.planner.replan(fold(run_id, events), events, reason)
        self._emit(run_id, EventType.PLAN_REVISED, PlanEvent(plan=plan, reason=reason))

    def _finish(self, run_id: str, status: str, summary: str, evidence: Optional[list] = None) -> None:
        self._emit(run_id, EventType.RUN_FINISHED,
                   RunFinished(status=status, summary=summary, evidence=evidence or []))

    def _emit(self, run_id: str, type: EventType, payload: BaseModel):
        ev = self.store.append(run_id, type, payload)
        if self.on_event:
            self.on_event(ev)
        return ev
