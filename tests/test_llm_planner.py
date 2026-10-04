"""Full run through LLMPlanner + Engine + real browser, with a scripted *fake LLM* (no network)."""
import re
from test_env import POLICY, TRUTH, refs  # noqa: F401
from opcore import *  # noqa
from opcore.schemas import EventType as ET
from opcore.context import render_history
import httpx

KNOW = "mockco/knowledge"
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, handlers): self.h, self.prompts = handlers, {}
    def complete_json(self, system, user):
        stage = user.split("\n", 1)[0].replace("STAGE: ", "").strip()
        self.prompts.setdefault(stage, []).append(user)
        r = self.h[stage]
        r = r.pop(0) if isinstance(r, list) else r
        return r(user) if callable(r) else r


UNDERSTAND = {"goal": "Acme's latest invoice is recorded in AP",
    "success_criteria": [
        {"id": k, "description": f"AP {k} equals the invoice's", "check": {"tool": "ap_api_get_invoice",
         "args": {"invoice_number": {"$fact": "invoice.number"}},
         "expect": {"op": "eq", "path": f"$.{p}", "value": {"fact": f"invoice.{k}"}}}}
        for k, p in [("amount", "amount"), ("due_date", "due_date")]],
    "constraints": ["over 50,000 needs approval"], "assumptions": ["latest = newest invoice date"], "open_questions": []}
PLAN = {"steps": [{"id": "s1", "intent": "find and read invoice", "depends_on": []}, {"id": "s2", "intent": "enter it", "depends_on": ["s1"]}]}


def last_call(prompt, tool): return re.findall(rf"\[(call_\w+)\] CALL {tool}", prompt)[-1]
def line(prompt, key): return next(l.strip() for l in prompt.splitlines() if key in l)


def test_full_run_with_grounding_rejection_then_recovery(env, base_url):
    reg, _ = env
    r = refs(reg, base_url)
    T = lambda tool, **a: {"action": "tool", "tool": tool, "args": a, "rationale": "step"}  # noqa: E731

    def bad_then_good_facts(p):
        cid = last_call(p, "read_document")
        return {"action": "record_facts", "facts": [
            {"key": "invoice.number", "value": "INV-2026-0455", "source_call_id": cid, "excerpt": line(p, "Invoice No."), "confidence": 0.99},
            {"key": "invoice.amount", "value": 64200, "source_call_id": cid, "excerpt": "TOTAL DUE (INR) 64,200.00", "confidence": 0.9},  # hallucinated
            {"key": "invoice.due_date", "value": "2026-10-28", "source_call_id": cid, "excerpt": line(p, "Payment Due"), "confidence": 0.99}]}

    def fix_amount(p):
        assert "FACT REJECTED: invoice.amount" in p            # the agent is told WHY, and can adapt
        return {"action": "record_facts", "facts": [{"key": "invoice.amount", "value": 62400,
                "source_call_id": last_call(p, "read_document"), "excerpt": line(p, "TOTAL DUE"), "confidence": 0.99}]}

    fake = FakeLLM({"understand": UNDERSTAND, "plan": PLAN, "next_action": [
        T("list_files", directory="."), T("read_document", path="scan_0931.pdf"), bad_then_good_facts, fix_amount,
        T("browser_open", url=f"{base_url}/invoices/new"),
        T("browser_select", ref=r["Vendor"], option="Acme Corp"),
        T("browser_fill", ref=r["Invoice number"], value="INV-2026-0455"),
        T("browser_fill", ref=r["Amount (INR)"], value="62400"),
        T("browser_fill", ref=r["Due date (YYYY-MM-DD)"], value="2026-10-28"),
        T("browser_submit", ref=r["Submit invoice"]),
        {"action": "done", "summary": "Entered INV-2026-0455"}]})

    planner = LLMPlanner(fake, reg, str(ROOT / KNOW))
    e = Engine(EventStore(), reg, PolicyEngine.from_dicts(POLICY), planner, backoff_s=0)
    rid = e.start("Find the latest invoice from Acme Corp and enter it into AP")
    s = e.run(rid)
    assert s.status == "waiting_approval"
    assert httpx.get(f"{base_url}/api/invoices").json() == []
    s = e.resolve_approval(rid, s.pending_approval.approval_id, True)

    assert s.status == "completed"
    assert set(s.facts) == {"invoice.number", "invoice.amount", "invoice.due_date"}
    assert s.facts["invoice.amount"].value == 62400
    assert sum(1 for x in e.store.read(rid) if x.type == ET.FACT_REJECTED) == 1
    assert httpx.get(f"{base_url}/api/invoices").json()[0]["amount"] == 62400.0
    # the prompt really carries company context + tool catalog, and nothing is hard-coded in the planner
    p = fake.prompts["next_action"][0]
    assert "ENVIRONMENT NOTES" in p and "list_files with directory '.'" in p
    assert "Never submit an invoice number that already exists" in p and "browser_submit [write_irreversible]" in p


def test_history_is_budgeted_and_keeps_only_latest_page_in_full():
    big = "Opened x\nURL: x\nPage text:\n" + "Z" * 3000
    mk = lambda i, s: [Event.make("r", 2 * i + 1, ToolRequested(call=ToolCall(call_id=f"c{i}", tool="browser_open", args={}, rationale="r")), ET.TOOL_REQUESTED),
                       Event.make("r", 2 * i + 2, ToolOutcome(result=ToolResult(call_id=f"c{i}", ok=True, summary=s)), ET.TOOL_SUCCEEDED)]
    evs = mk(0, big) + mk(1, big)
    h = render_history(evs)
    assert h.count("Z" * 3000) == 1                                  # old page compressed, newest kept whole
    assert "(earlier history truncated)" in render_history(evs * 30, budget=2000)


def test_planner_accepts_tool_name_used_as_action_and_rejects_nonsense():
    from opcore.llm_planner import LLMPlanner as P
    reg, _ = None, None
    class R:  # minimal registry stand-in
        def get(self, n): return object() if n == "browser_select" else None
    pl = P.__new__(P); pl.registry = R()
    c = pl._parse_action({"action": "browser_select", "ref": "e2", "option": "Acme Corp", "rationale": "pick vendor"})
    assert c.tool == "browser_select" and c.args == {"ref": "e2", "option": "Acme Corp"}
    import pytest
    with pytest.raises(ValueError, match="must be one of"):
        pl._parse_action({"action": "dance"})
