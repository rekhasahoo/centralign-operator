"""
Deterministic permission layer. The LLM can ask for anything; this decides what runs.

Precedence: deny > require_approval > allow. Tier defaults are the baseline; an explicit
`allow` rule can pre-authorise a tier's approval requirement, but never overrides a deny,
an explicit require_approval rule, or the low-confidence-fact guard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Optional

from .schemas import PolicyDecided, Risk, RunState, ToolCall, ToolSpec

_TIER_DEFAULT = {
    Risk.READ: "allow",
    Risk.WRITE_REVERSIBLE: "allow",
    Risk.WRITE_IRREVERSIBLE: "require_approval",
    Risk.EXTERNAL: "require_approval",
}
_OPS = {
    "gt": lambda a, b: float(a) > float(b), "gte": lambda a, b: float(a) >= float(b),
    "lt": lambda a, b: float(a) < float(b), "lte": lambda a, b: float(a) <= float(b),
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "in": lambda a, b: a in b, "contains": lambda a, b: b in str(a),
    "matches": lambda a, b: re.search(b, str(a)) is not None,
    "nonempty": lambda a, b: bool(a),
}


@dataclass
class Rule:
    name: str
    decision: str                          # allow | require_approval | deny
    reason: str
    tool: str = "*"                        # fnmatch pattern
    arg: Optional[str] = None              # if set, rule only fires when this field satisfies op/value
    source: str = "args"                   # "args" (tool arguments) or "effect" (previewed real effect)
    op: Optional[str] = None
    value: Any = None

    def why(self, effect: Optional[dict]) -> str:
        """Reason text may embed effect values, e.g. 'never opened: {_unread_sources}'."""
        out = self.reason
        for k, v in (effect or {}).items():
            out = out.replace("{" + k + "}", str(v))
        return out

    def matches(self, call: ToolCall, effect: Optional[dict] = None) -> bool:
        if not fnmatch(call.tool, self.tool):
            return False
        if self.arg is None:
            return True
        data = (effect or {}) if self.source == "effect" else call.args
        if self.arg not in data:
            return False
        try:
            return bool(_OPS[self.op](data[self.arg], self.value))
        except (TypeError, ValueError, KeyError):
            return False


class PolicyEngine:
    def __init__(self, rules: Optional[list[Rule]] = None, min_fact_confidence: float = 0.8):
        self.rules = rules or []
        self.min_conf = min_fact_confidence

    @classmethod
    def from_dicts(cls, rules: list[dict], **kw) -> "PolicyEngine":
        """Rules are data (YAML/JSON/DB), so policy changes need no code change."""
        return cls([Rule(**r) for r in rules], **kw)

    def decide(self, call: ToolCall, spec: Optional[ToolSpec], state: RunState,
               effect: Optional[dict] = None) -> PolicyDecided:
        def out(d: str, rule: str, why: str) -> PolicyDecided:
            return PolicyDecided(call_id=call.call_id, decision=d, rule=rule, reason=why)

        if spec is None:
            return out("deny", "unknown_tool", f"tool '{call.tool}' is not registered")

        hits = [r for r in self.rules if r.matches(call, effect)]
        for r in hits:
            if r.decision == "deny":
                return out("deny", r.name, r.why(effect))
        for r in hits:
            if r.decision == "require_approval":
                return out("require_approval", r.name, r.why(effect))

        risky = spec.risk in (Risk.WRITE_IRREVERSIBLE, Risk.EXTERNAL)
        low = [f"{f.key} (conf {f.confidence:.2f})" for f in state.facts.values()
               if f.confidence < self.min_conf]
        if risky and low:
            return out("require_approval", "low_confidence_facts",
                       "acting on uncertain extracted values: " + ", ".join(low))

        for r in hits:
            if r.decision == "allow":
                return out("allow", r.name, r.why(effect))
        base = _TIER_DEFAULT[spec.risk]
        return out(base, f"tier:{spec.risk.value}", f"default for {spec.risk.value} tools")
