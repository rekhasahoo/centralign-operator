"""
Independent verification. Runs each criterion's Check with a READ tool against the real
system and compares to the expectation. The acting agent's own claim of success is never an input.
"""
from __future__ import annotations

import re
from typing import Any

from .registry import ToolRegistry
from .schemas import Criterion, FactRef, Risk, RunState, ToolCall, VerificationResult

_MISSING = object()
_TOKEN = re.compile(r"\.?([A-Za-z_][\w-]*)|\[(\d+)\]")


def get_path(obj: Any, path: str) -> Any:
    """Tiny JSONPath subset: $, $.a.b, $.items[0].x"""
    if path in ("$", ""):
        return obj
    cur = obj
    for m in _TOKEN.finditer(path[1:] if path.startswith("$") else path):
        key, idx = m.groups()
        try:
            cur = cur[key] if key is not None else cur[int(idx)]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    return cur


def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().replace(",", "")
        try:
            return float(t)
        except ValueError:
            return v.strip().lower()
    return v


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if actual is _MISSING:
        return False
    if op == "exists":
        return actual is not None
    if op == "eq":
        return _norm(actual) == _norm(expected)
    if op == "contains":
        if isinstance(actual, (list, tuple, set)):
            return any(_norm(a) == _norm(expected) for a in actual)
        return str(expected).lower() in str(actual).lower()
    if op == "matches":
        return re.search(str(expected), str(actual)) is not None
    return False


def _is_fact_ref(v: Any) -> bool:
    return isinstance(v, FactRef) or (isinstance(v, dict) and set(v) == {"fact"})


def _resolve(v: Any, facts: dict) -> Any:
    """Replace {"$fact": key} placeholders anywhere in args."""
    if isinstance(v, dict):
        if set(v) == {"$fact"}:
            f = facts.get(v["$fact"])
            return f.value if f else None
        return {k: _resolve(x, facts) for k, x in v.items()}
    if isinstance(v, list):
        return [_resolve(x, facts) for x in v]
    return v


class Verifier:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, state: RunState) -> list[VerificationResult]:
        return [self._check(c, state) for c in (state.spec.success_criteria if state.spec else [])]

    def _check(self, c: Criterion, state: RunState) -> VerificationResult:
        chk = c.check
        tool = self.registry.get(chk.tool)
        if tool is None or tool.spec.risk != Risk.READ:
            return VerificationResult(criterion_id=c.id, passed=False,
                                      actual=f"check tool '{chk.tool}' missing or not read-only")
        expected = chk.expect.value
        if _is_fact_ref(expected):
            key = expected.fact if isinstance(expected, FactRef) else expected["fact"]
            fact = state.facts.get(key)
            if fact is None:
                return VerificationResult(criterion_id=c.id, passed=False,
                                          actual=f"no recorded fact '{key}' to compare against")
            expected = fact.value
        res = self.registry.execute(ToolCall(tool=chk.tool, args=_resolve(chk.args, state.facts),
                                             rationale=f"verify {c.id}: {c.description}"))
        if not res.ok:
            return VerificationResult(criterion_id=c.id, passed=False, expected=expected,
                                      actual=f"check failed: {res.error.message}", evidence=res.artifacts)
        actual = get_path(res.output, chk.expect.path)
        shown = None if actual is _MISSING else actual
        return VerificationResult(criterion_id=c.id, passed=_compare(chk.expect.op, actual, expected),
                                  expected=expected, actual=shown, evidence=res.artifacts)
