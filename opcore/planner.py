"""The only LLM-facing interface. Every method returns a *proposal*; the Engine decides what happens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

from .schemas import Event, Fact, Plan, RunState, TaskSpec, ToolCall, ToolResult


@dataclass
class Done:
    summary: str                       # the planner *believes* the goal is met; the Verifier decides


@dataclass
class AskHuman:
    question: str
    blocked_on: str


@dataclass
class ProposedFact:
    key: str
    value: object
    source_call_id: str                # which tool result this came from
    excerpt: str                       # VERBATIM text from that result; the runtime checks it exists
    confidence: float = 1.0


@dataclass
class RecordFacts:
    facts: list[ProposedFact]


class Planner(Protocol):
    def understand(self, state: RunState, events: list[Event]) -> TaskSpec: ...
    def plan(self, state: RunState, events: list[Event]) -> Plan: ...
    def replan(self, state: RunState, events: list[Event], reason: str) -> Plan: ...
    def next_action(self, state: RunState, events: list[Event]) -> Union[ToolCall, Done, AskHuman, RecordFacts]: ...
    def extract_facts(self, state: RunState, call: ToolCall, result: ToolResult,
                      source_event_id: str) -> list[Fact]: ...
