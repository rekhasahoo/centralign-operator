"""Renders the event log into a compact, budgeted history for the planner prompt."""
from __future__ import annotations

import json

from .schemas import Event, EventType as ET


def _one_line(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 3] + "..."


def render_history(events: list[Event], *, doc_chars: int = 1800, latest_browser_chars: int = 4500,
                   budget: int = 24000) -> str:
    # the newest browser observation is shown in full; older ones are compressed to a headline
    last_browser = max((e.seq for e in events if e.type == ET.TOOL_SUCCEEDED
                        and e.payload["result"]["summary"].lstrip().startswith(("Opened", "Snapshot", "Clicked", "Submitted"))), default=-1)
    tool_of: dict[str, str] = {}
    blocks: list[str] = []
    for e in events:
        p, t = e.payload, e.type
        if t == ET.TOOL_REQUESTED:
            c = p["call"]; tool_of[c["call_id"]] = c["tool"]
            blocks.append(f"[{c['call_id']}] CALL {c['tool']}({json.dumps(c['args'])}): {c['rationale']}")
        elif t == ET.POLICY_DECIDED and p["decision"] != "allow":
            blocks.append(f"  policy: {p['decision']} ({p['rule']}): {p['reason']}")
        elif t == ET.APPROVAL_REQUESTED:
            blocks.append("  (paused: waiting for human approval)")
        elif t == ET.APPROVAL_RESOLVED:
            blocks.append(f"  human {'APPROVED' if p['approved'] else 'REJECTED'} the action. {p.get('note', '')}")
        elif t == ET.TOOL_SUCCEEDED:
            r = p["result"]; text = r["summary"]
            if e.seq == last_browser:
                text = text[:latest_browser_chars]
            elif text.lstrip().startswith(("Opened", "Snapshot", "Clicked", "Submitted")):
                text = _one_line(text.split("Page text:")[0], 300)       # old page: headline + alerts only
            else:
                text = text[:doc_chars]
            blocks.append("  RESULT:\n    " + text.replace("\n", "\n    "))
        elif t == ET.TOOL_FAILED:
            err = p["result"]["error"]
            blocks.append(f"  FAILED (attempt {p['attempt']}, {err['kind']}): {err['message']}")
        elif t == ET.RECOVERY_CHOSEN:
            blocks.append(f"  runtime recovery: {p['strategy']}: {p['detail']}")
        elif t == ET.INPUT_REQUESTED:
            blocks.append(f"ASKED HUMAN: {p['question']}")
        elif t == ET.INPUT_PROVIDED:
            blocks.append(f"HUMAN ANSWERED: {p['answer']}")
        elif t == ET.FACT_RECORDED:
            f = p["fact"]; blocks.append(f"FACT RECORDED: {f['key']} = {f['value']!r}")
        elif t == ET.FACT_REJECTED:
            blocks.append(f"FACT REJECTED: {p['key']}: {p['reason']}")
        elif t == ET.VERIFICATION_RESULT:
            blocks.append(f"INDEPENDENT VERIFICATION of '{p['criterion_id']}': {'PASSED' if p['passed'] else 'FAILED'} "
                          f"(expected {p['expected']!r}, actual {p['actual']!r})")
        elif t == ET.PLAN_REVISED:
            blocks.append(f"PLAN REVISED: {p['reason']}")
    out, size = [], 0
    for b in reversed(blocks):                 # keep the most recent history when over budget
        size += len(b) + 1
        if size > budget:
            out.append("(earlier history truncated)"); break
        out.append(b)
    return "\n".join(reversed(out)) or "(nothing has happened yet)"
