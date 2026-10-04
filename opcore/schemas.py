"""
Contracts for the AI-operator runtime.

Design rule: the LLM only ever produces *proposals* (TaskSpec, Plan, ToolCall).
Everything that actually happens is recorded as an append-only Event, and the
current run state is a pure fold over those events (see `reduce`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class Risk(str, Enum):
    READ = "read"                              # no side effects
    WRITE_REVERSIBLE = "write_reversible"      # undoable (draft, form fill)
    WRITE_IRREVERSIBLE = "write_irreversible"  # submit, delete, pay
    EXTERNAL = "external_comms"                # email/message leaves the org


class ErrorKind(str, Enum):
    TRANSIENT = "transient"                # timeout, flaky page -> retry
    INVALID_INPUT = "invalid_input"        # bad args -> fix args, replan
    NOT_FOUND = "not_found"                # element/file missing -> alternate
    PERMISSION_DENIED = "permission_denied"  # policy/auth -> escalate
    STATE_MISMATCH = "state_mismatch"      # world differs from plan -> replan
    UNKNOWN = "unknown"


class ToolSpec(BaseModel):
    """Registered once; the planner sees name/description/input_schema only."""
    name: str
    description: str
    input_schema: dict[str, Any]           # JSON Schema for args
    risk: Risk
    idempotent: bool = False               # safe to blindly retry?
    timeout_s: float = 30.0
    max_retries: int = 2
    confirm_with: Optional[str] = None     # read-tool that checks this write took effect
    enumerates: bool = False               # output is a list of items ({"name": ...}) the agent could inspect
    inspects: Optional[str] = None         # name of the arg that identifies the item this tool opens


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: _id("call"))
    step_id: Optional[str] = None
    tool: str
    args: dict[str, Any]
    rationale: str                         # why the agent chose this; shown in audit/UI


class ArtifactRef(BaseModel):
    """Large outputs (screenshots, PDFs, DOM snapshots) live on disk; events hold refs."""
    artifact_id: str = Field(default_factory=lambda: _id("art"))
    kind: Literal["screenshot", "file", "snapshot", "text"]
    path: str
    caption: str = ""


class ToolError(BaseModel):
    kind: ErrorKind
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    call_id: str
    ok: bool
    summary: str                           # short, what the LLM sees next turn
    output: Any = None                     # structured payload, may be truncated for the LLM
    error: Optional[ToolError] = None
    artifacts: list[ArtifactRef] = []
    duration_ms: int = 0


# --------------------------------------------------------------------------- #
# Task understanding: the goal compiled into something checkable
# --------------------------------------------------------------------------- #
class FactRef(BaseModel):
    """Expected value is taken from a fact gathered earlier (with provenance)."""
    fact: str


class Expectation(BaseModel):
    op: Literal["eq", "contains", "exists", "matches"]
    path: str = "$"                        # JSONPath-ish into the check tool's output
    value: Any | FactRef = None


class Check(BaseModel):
    """Run by the Verifier with a READ tool, never by the acting agent."""
    tool: str
    args: dict[str, Any]
    expect: Expectation


class Criterion(BaseModel):
    id: str
    description: str
    check: Check


class TaskSpec(BaseModel):
    goal: str                              # restated outcome, not steps
    success_criteria: list[Criterion]
    constraints: list[str] = []            # e.g. "never email external parties"
    assumptions: list[str] = []            # surfaced to the user, not hidden
    open_questions: list[str] = []         # non-empty -> ask before acting


class PlanStep(BaseModel):
    id: str
    intent: str                            # what, not how
    depends_on: list[str] = []
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"


class Plan(BaseModel):
    version: int = 1
    steps: list[PlanStep]
    rationale: str = ""


class Fact(BaseModel):
    """Working memory entry. Provenance is what lets the verifier and a human audit it."""
    key: str                               # e.g. "invoice.amount"
    value: Any
    source_event_id: str
    source_excerpt: str = ""
    confidence: float = 1.0                # < threshold -> ask_human before using


# --------------------------------------------------------------------------- #
# Events (append-only log = single source of truth)
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    RUN_CREATED = "run.created"
    TASK_UNDERSTOOD = "task.understood"
    PLAN_CREATED = "plan.created"
    PLAN_REVISED = "plan.revised"
    TOOL_REQUESTED = "tool.requested"
    POLICY_DECIDED = "policy.decided"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    INPUT_REQUESTED = "input.requested"
    INPUT_PROVIDED = "input.provided"
    TOOL_SUCCEEDED = "tool.succeeded"
    TOOL_FAILED = "tool.failed"
    FACT_RECORDED = "fact.recorded"
    FACT_REJECTED = "fact.rejected"
    RECOVERY_CHOSEN = "recovery.chosen"
    VERIFICATION_RESULT = "verification.result"
    RUN_FINISHED = "run.finished"


class RunCreated(BaseModel):
    request: str
    requester: str = "user"


class TaskUnderstood(BaseModel):
    spec: TaskSpec


class PlanEvent(BaseModel):                # PLAN_CREATED and PLAN_REVISED
    plan: Plan
    reason: str = ""


class ToolRequested(BaseModel):
    call: ToolCall


class PolicyDecided(BaseModel):
    call_id: str
    decision: Literal["allow", "require_approval", "deny"]
    rule: str                              # which rule fired, for audit
    reason: str


class ApprovalRequested(BaseModel):
    approval_id: str = Field(default_factory=lambda: _id("apr"))
    call_id: str
    summary: str                           # plain-language "I am about to ..."
    risk: Risk


class ApprovalResolved(BaseModel):
    approval_id: str
    approved: bool
    by: str
    note: str = ""


class InputRequested(BaseModel):
    request_id: str = Field(default_factory=lambda: _id("req"))
    question: str
    blocked_on: str                        # what the agent cannot safely decide


class InputProvided(BaseModel):
    request_id: str
    answer: str


class ToolOutcome(BaseModel):              # TOOL_SUCCEEDED and TOOL_FAILED
    result: ToolResult
    attempt: int = 1


class FactRecorded(BaseModel):
    fact: Fact


class FactRejected(BaseModel):
    key: str
    reason: str
    excerpt: str = ""


class RecoveryChosen(BaseModel):
    call_id: str
    error_kind: ErrorKind
    strategy: Literal["retry", "accept", "replan", "ask_human", "abort"]
    detail: str = ""


class VerificationResult(BaseModel):
    criterion_id: str
    passed: bool
    expected: Any = None
    actual: Any = None
    evidence: list[ArtifactRef] = []


class RunFinished(BaseModel):
    status: Literal["completed", "failed", "escalated"]
    summary: str
    evidence: list[ArtifactRef] = []


PAYLOAD_MODELS: dict[EventType, type[BaseModel]] = {
    EventType.RUN_CREATED: RunCreated,
    EventType.TASK_UNDERSTOOD: TaskUnderstood,
    EventType.PLAN_CREATED: PlanEvent,
    EventType.PLAN_REVISED: PlanEvent,
    EventType.TOOL_REQUESTED: ToolRequested,
    EventType.POLICY_DECIDED: PolicyDecided,
    EventType.APPROVAL_REQUESTED: ApprovalRequested,
    EventType.APPROVAL_RESOLVED: ApprovalResolved,
    EventType.INPUT_REQUESTED: InputRequested,
    EventType.INPUT_PROVIDED: InputProvided,
    EventType.TOOL_SUCCEEDED: ToolOutcome,
    EventType.TOOL_FAILED: ToolOutcome,
    EventType.FACT_RECORDED: FactRecorded,
    EventType.FACT_REJECTED: FactRejected,
    EventType.RECOVERY_CHOSEN: RecoveryChosen,
    EventType.VERIFICATION_RESULT: VerificationResult,
    EventType.RUN_FINISHED: RunFinished,
}


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: _id("evt"))
    run_id: str
    seq: int                               # gapless per run; enforced by the store
    ts: datetime = Field(default_factory=_now)
    type: EventType
    payload: dict[str, Any]

    @classmethod
    def make(cls, run_id: str, seq: int, p: BaseModel, type: EventType) -> "Event":
        assert isinstance(p, PAYLOAD_MODELS[type]), f"{type} expects {PAYLOAD_MODELS[type]}"
        return cls(run_id=run_id, seq=seq, type=type, payload=p.model_dump(mode="json"))

    def typed(self) -> BaseModel:
        return PAYLOAD_MODELS[self.type].model_validate(self.payload)


# --------------------------------------------------------------------------- #
# State = fold(events). No hidden in-memory state, so runs resume after a crash.
# --------------------------------------------------------------------------- #
class RunState(BaseModel):
    run_id: str
    status: Literal[
        "created", "planning", "executing", "waiting_approval",
        "waiting_input", "verifying", "completed", "failed", "escalated",
    ] = "created"
    request: str = ""
    spec: Optional[TaskSpec] = None
    plan: Optional[Plan] = None
    facts: dict[str, Fact] = {}
    attempts: dict[str, int] = {}                  # call_id -> tries so far
    pending_approval: Optional[ApprovalRequested] = None
    pending_input: Optional[InputRequested] = None
    verification: dict[str, VerificationResult] = {}
    summary: str = ""
    # tool-call lifecycle: requested -> allowed|needs_approval|denied -> approved|rejected
    #   -> succeeded|failed -> handled
    calls: dict[str, ToolCall] = {}
    call_status: dict[str, str] = {}
    tool_calls_made: int = 0
    replans: int = 0
    verify_failures: int = 0
    fact_rejections: int = 0
    clarifications: list[dict[str, str]] = []


_DECISION_TO_STATUS = {"allow": "allowed", "require_approval": "needs_approval", "deny": "denied"}


def reduce(s: RunState, e: Event) -> RunState:
    p, t = e.typed(), e.type
    if t == EventType.RUN_CREATED:
        s.request, s.status = p.request, "planning"
    elif t == EventType.TASK_UNDERSTOOD:
        s.spec = p.spec
        s.status = "waiting_input" if p.spec.open_questions else "planning"
    elif t == EventType.PLAN_CREATED:
        s.plan, s.status = p.plan, "executing"
    elif t == EventType.PLAN_REVISED:
        s.plan, s.status = p.plan, "executing"
        s.replans += 1
        s.verification = {}
    elif t == EventType.TOOL_REQUESTED:
        s.calls[p.call.call_id] = p.call
        s.call_status[p.call.call_id] = "requested"
        s.tool_calls_made += 1
    elif t == EventType.POLICY_DECIDED:
        s.call_status[p.call_id] = _DECISION_TO_STATUS[p.decision]
    elif t == EventType.APPROVAL_REQUESTED:
        s.pending_approval, s.status = p, "waiting_approval"
    elif t == EventType.APPROVAL_RESOLVED:
        if s.pending_approval:
            s.call_status[s.pending_approval.call_id] = "approved" if p.approved else "rejected"
        s.pending_approval, s.status = None, "executing"
    elif t == EventType.INPUT_REQUESTED:
        s.pending_input, s.status = p, "waiting_input"
    elif t == EventType.INPUT_PROVIDED:
        if s.pending_input:
            s.clarifications.append({"question": s.pending_input.question, "answer": p.answer})
        s.pending_input, s.status = None, "executing"
    elif t in (EventType.TOOL_SUCCEEDED, EventType.TOOL_FAILED):
        s.attempts[p.result.call_id] = p.attempt
        s.call_status[p.result.call_id] = "succeeded" if t == EventType.TOOL_SUCCEEDED else "failed"
    elif t == EventType.FACT_RECORDED:
        s.facts[p.fact.key] = p.fact
    elif t == EventType.FACT_REJECTED:
        s.fact_rejections += 1
    elif t == EventType.RECOVERY_CHOSEN:
        if p.call_id == "verify":
            s.verify_failures += 1
        elif p.strategy in ("replan", "ask_human", "abort") and p.call_id in s.call_status:
            s.call_status[p.call_id] = "handled"
    elif t == EventType.VERIFICATION_RESULT:
        s.status = "verifying"
        s.verification[p.criterion_id] = p
    elif t == EventType.RUN_FINISHED:
        s.status, s.summary = p.status, p.summary
    return s


def fold(run_id: str, events: list[Event]) -> RunState:
    s = RunState(run_id=run_id)
    for e in sorted(events, key=lambda x: x.seq):
        s = reduce(s, e)
    return s


# --------------------------------------------------------------------------- #
# Storage (SQLite). Append-only; UNIQUE(run_id, seq) gives optimistic concurrency.
# --------------------------------------------------------------------------- #
DDL = """
CREATE TABLE IF NOT EXISTS events (
  run_id   TEXT NOT NULL,
  seq      INTEGER NOT NULL,
  event_id TEXT NOT NULL UNIQUE,
  ts       TEXT NOT NULL,
  type     TEXT NOT NULL,
  payload  TEXT NOT NULL,           -- JSON
  PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, type);
CREATE TABLE IF NOT EXISTS runs (   -- denormalised for listing only; events are truth
  run_id TEXT PRIMARY KEY, request TEXT, status TEXT, created_at TEXT, updated_at TEXT
);
"""
