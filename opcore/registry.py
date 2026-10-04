"""Typed tools with risk tiers. Tools hold no task-specific logic and never decide permissions."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jsonschema

from .schemas import ArtifactRef, ErrorKind, ToolCall, ToolError, ToolResult, ToolSpec

_POOL = ThreadPoolExecutor(max_workers=8)  # NB: a timed-out thread keeps running; use process isolation in prod


class ToolFailure(Exception):
    """Raise from a tool to report a *typed* failure the recovery layer can reason about."""

    def __init__(self, kind: ErrorKind, message: str, retryable: Optional[bool] = None):
        super().__init__(message)
        self.kind, self.message = kind, message
        self.retryable = (kind == ErrorKind.TRANSIENT) if retryable is None else retryable


@dataclass
class ToolOutput:
    summary: str                      # what the LLM sees next turn: keep it short
    output: Any = None                # full structured payload (verifier / fact extraction)
    artifacts: list[ArtifactRef] = field(default_factory=list)


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[..., Any]
    confirm: Optional[Callable[[dict], bool]] = None  # "did this write actually land?"
    preview: Optional[Callable[[dict], dict]] = None  # "what exactly is about to happen?" (read-only)


def _short(x: Any, n: int = 400) -> str:
    t = repr(x)
    return t if len(t) <= n else t[: n - 3] + "..."


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.notes: str = ""   # environment facts the planner needs (where things live), set by the scenario

    def tool(self, spec: ToolSpec, confirm: Optional[Callable[[dict], bool]] = None,
             preview: Optional[Callable[[dict], dict]] = None):
        def deco(fn):
            self._tools[spec.name] = Tool(spec, fn, confirm, preview)
            return fn
        return deco

    def preview(self, call: ToolCall) -> dict:
        """Concrete effect of a call (e.g. the form values about to be submitted). Never raises."""
        t = self._tools.get(call.tool)
        if not t or not t.preview:
            return {}
        try:
            return dict(t.preview(call.args))
        except Exception:  # noqa: BLE001
            return {}

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def catalog(self) -> list[dict]:
        """What the planner is shown. Deliberately excludes callables and internals."""
        return [
            {"name": t.spec.name, "description": t.spec.description,
             "input_schema": t.spec.input_schema, "risk": t.spec.risk.value}
            for t in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        t0 = time.monotonic()

        def fail(kind: ErrorKind, msg: str, retryable: bool = False) -> ToolResult:
            return ToolResult(call_id=call.call_id, ok=False, summary=f"{call.tool} failed: {msg}",
                              error=ToolError(kind=kind, message=msg, retryable=retryable),
                              duration_ms=int((time.monotonic() - t0) * 1000))

        tool = self._tools.get(call.tool)
        if tool is None:
            return fail(ErrorKind.INVALID_INPUT, f"unknown tool '{call.tool}'")
        try:
            jsonschema.validate(call.args, tool.spec.input_schema)
        except jsonschema.ValidationError as e:
            return fail(ErrorKind.INVALID_INPUT, f"bad arguments: {e.message}")

        fut = _POOL.submit(tool.fn, **call.args)
        try:
            out = fut.result(timeout=tool.spec.timeout_s)
        except FutureTimeout:
            fut.cancel()
            return fail(ErrorKind.TRANSIENT, f"timed out after {tool.spec.timeout_s}s", True)
        except ToolFailure as e:
            return fail(e.kind, e.message, e.retryable)
        except Exception as e:  # noqa: BLE001 - tools are untrusted
            return fail(ErrorKind.UNKNOWN, f"{type(e).__name__}: {e}")

        if not isinstance(out, ToolOutput):
            out = ToolOutput(summary=_short(out), output=out)
        return ToolResult(call_id=call.call_id, ok=True, summary=out.summary, output=out.output,
                          artifacts=out.artifacts, duration_ms=int((time.monotonic() - t0) * 1000))
