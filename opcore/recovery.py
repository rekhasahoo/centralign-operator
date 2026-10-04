"""Maps a tool failure to a strategy. Pure function: easy to test, easy to explain."""
from __future__ import annotations

from typing import Optional

from .registry import Tool
from .schemas import ErrorKind, Risk, ToolError


def choose(err: ToolError, tool: Tool, attempt: int, landed: Optional[bool]) -> tuple[str, str]:
    """
    attempt: tries so far (>=1).  landed: result of the tool's `confirm` check, if it has one
    (True = the write actually took effect despite the error, False = it did not, None = unknown).
    """
    k, spec = err.kind, tool.spec
    safe_to_retry = spec.risk == Risk.READ or spec.idempotent or landed is False

    if k == ErrorKind.TRANSIENT:
        if landed is True:
            return "accept", "error reported but confirm-check shows the write landed"
        if attempt > spec.max_retries:
            return "ask_human", f"still failing after {attempt} attempts: {err.message}"
        if safe_to_retry:
            return "retry", f"transient error, attempt {attempt}/{spec.max_retries + 1}"
        return "ask_human", "non-idempotent write failed and cannot confirm it did not land"
    if k in (ErrorKind.NOT_FOUND, ErrorKind.INVALID_INPUT, ErrorKind.STATE_MISMATCH):
        return "replan", f"{k.value}: {err.message}"
    if k == ErrorKind.PERMISSION_DENIED:
        return "ask_human", f"permission denied: {err.message}"
    # UNKNOWN: one cautious retry for safe calls, otherwise a human
    if attempt <= 1 and (spec.risk == Risk.READ or spec.idempotent):
        return "retry", f"unknown error, one retry: {err.message}"
    return "ask_human", f"unexpected error: {err.message}"
