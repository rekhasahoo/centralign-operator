"""
Grounding check: a fact the agent wants to rely on must cite a tool result and quote it verbatim,
and the claimed value must be supported by that quote. This is what stops "plausible but invented"
numbers from ever reaching an irreversible action. (Verification later checks the *outcome*;
grounding checks the *input*.)
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Optional

from dateutil import parser as dtparse

from .schemas import Event, EventType

_NUM = re.compile(r"\d[\d,]*\.?\d*")
_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}",
    r"\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}",
    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}",
]
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _bare(call_id: Any) -> str:
    """Models often drop the 'call_' prefix or wrap ids in brackets; match on the bare id."""
    x = str(call_id).strip().strip("[]()'\" ")
    return x[5:] if x.startswith("call_") else x


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _corpus(result: dict) -> str:
    parts = [result.get("summary", "")]

    def walk(x: Any):
        if isinstance(x, str): parts.append(x)
        elif isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, list): [walk(v) for v in x]
        elif x is not None: parts.append(json.dumps(x))
    walk(result.get("output"))
    return "\n".join(parts)


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try: out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError: pass
    return out


def _dates(text: str) -> set[date]:
    found: set[date] = set()
    for pat in _DATE_PATTERNS:
        for m in re.findall(pat, text):
            try: found.add(dtparse.parse(m, dayfirst=True).date())  # India: day/month/year
            except (ValueError, OverflowError): pass
    return found


def _supports(value: Any, excerpt: str) -> bool:
    if isinstance(value, bool):
        return str(value).lower() in excerpt.lower()
    try:
        num = float(str(value).replace(",", ""))
        if not _ISO.match(str(value)):
            return any(abs(n - num) < 0.005 for n in _numbers(excerpt))
    except ValueError:
        pass
    if _ISO.match(str(value)):
        return date.fromisoformat(str(value)) in _dates(excerpt)
    return _norm(str(value)) in _norm(excerpt)


def check(pf, events: list[Event]) -> tuple[bool, str, Optional[str]]:
    """-> (ok, reason, source_event_id)"""
    src, valid = None, []
    for e in events:
        if e.type == EventType.TOOL_SUCCEEDED:
            cid = e.payload["result"]["call_id"]
            valid.append(cid)
            if _bare(cid) == _bare(pf.source_call_id):
                src = e
    if src is None:
        return False, (f"source_call_id '{pf.source_call_id}' matches no successful tool result. "
                       f"Valid ids: {valid[-8:]}. Copy the id exactly as shown in HISTORY"), None
    if len(pf.excerpt.strip()) < 3:
        return False, "excerpt is empty; quote the exact text that shows the value", None
    if _norm(pf.excerpt) not in _norm(_corpus(src.payload["result"])):
        return False, "excerpt does not appear verbatim in that tool result; copy the text exactly", None
    if not _supports(pf.value, pf.excerpt):
        return False, f"value {pf.value!r} is not supported by the quoted excerpt", None
    return True, "ok", src.event_id
