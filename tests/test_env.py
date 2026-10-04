"""
Real browser + real mock app + real PDFs, driven by a scripted planner (no LLM).
Proves the *tools and runtime* hold up against an imperfect system before any model is involved.
"""
import json, pathlib, socket, sys, threading, time

import httpx, pytest, uvicorn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opcore import *  # noqa: F401,F403
from opcore.schemas import EventType as ET
from envs.invoice_env import build_invoice_env
from mockco.app import app

TRUTH = json.loads((ROOT / "mockco/eval/ground_truth.json").read_text())["latest_by_vendor"]["Acme Corp"]


def call(reg, tool, **args):
    return reg.execute(ToolCall(tool=tool, args=args, rationale="test"))


def refs(reg, base_url):
    """Open the form and map field names -> refs (refs are deterministic for a fixed layout)."""
    snap = call(reg, "browser_open", url=f"{base_url}/invoices/new").output["elements"]
    return {e["name"]: e["ref"] for e in snap}


# ---------------------------------------------------------------- tools on their own
def test_pdf_tools_read_the_real_documents_and_filenames_lie(env):
    reg, _ = env
    assert call(reg, "read_document", path="scan_0931.pdf").output["text"].count("62,400.00") == 1
    assert "INV-2026-0388" in call(reg, "read_document", path="acme_invoice_latest.pdf").output["text"]  # "latest" is the OLDEST


def test_file_sandbox_blocks_path_traversal(env):
    r = call(env[0], "read_document", path="../../../etc/passwd")
    assert not r.ok and r.error.kind == ErrorKind.PERMISSION_DENIED


def test_click_refuses_submit_controls_so_policy_cannot_be_bypassed(env, base_url):
    reg, _ = env
    r = refs(reg, base_url)
    res = call(reg, "browser_click", ref=r["Submit invoice"])
    assert not res.ok and res.error.kind == ErrorKind.INVALID_INPUT and "browser_submit" in res.error.message


def test_validation_error_is_an_observation_not_a_crash(env, base_url):
    reg, _ = env
    r = refs(reg, base_url)
    call(reg, "browser_select", ref=r["Vendor"], option="Acme Corp")
    call(reg, "browser_fill", ref=r["Invoice number"], value="INV-X")
    call(reg, "browser_fill", ref=r["Amount (INR)"], value="62,400")           # the comma trap
    call(reg, "browser_fill", ref=r["Due date (YYYY-MM-DD)"], value="2026-10-28")
    res = call(reg, "browser_submit", ref=r["Submit invoice"])
    assert res.ok and "plain positive number" in res.summary and "HTTP 422" in res.summary
    assert httpx.get(f"{base_url}/api/invoices").json() == []


# ---------------------------------------------------------------- full runs through the engine
SPEC = TaskSpec(goal="Record Acme's latest invoice in AP", success_criteria=[
    Criterion(id=k, description=f"AP {k} matches the invoice",
              check=Check(tool="ap_api_get_invoice", args={"invoice_number": TRUTH["invoice_number"]},
                          expect=Expectation(op="eq", path=f"$.{path}", value=FactRef(fact=f"invoice.{k}"))))
    for k, path in [("amount", "amount"), ("due_date", "due_date")]])


class Scripted:
    def __init__(self, queue, replan_queue=None):
        self.queue, self.replan_queue, self.reasons = queue, replan_queue or [], []
    def understand(self, s, ev): return SPEC.model_copy(deep=True)
    def plan(self, s, ev): return Plan(steps=[PlanStep(id="s1", intent="enter invoice")])
    def replan(self, s, ev, reason):
        self.reasons.append(reason); self.queue = list(self.replan_queue)
        return Plan(version=s.plan.version + 1, steps=s.plan.steps)
    def next_action(self, s, ev): return self.queue.pop(0) if self.queue else Done("entered the invoice")
    def extract_facts(self, s, call, result, src):
        if call.tool != "read_document": return []
        return [Fact(key=f"invoice.{k}", value=TRUTH[k], source_event_id=src, source_excerpt="scan_0931.pdf")
                for k in ("amount", "due_date")]


def fill_calls(r, amount="62400"):
    def c(tool, **a): return ToolCall(tool=tool, args=a, rationale="enter invoice")
    return [c("browser_select", ref=r["Vendor"], option="Acme Corp"),
            c("browser_fill", ref=r["Invoice number"], value=TRUTH["invoice_number"]),
            c("browser_fill", ref=r["Amount (INR)"], value=amount),
            c("browser_fill", ref=r["Due date (YYYY-MM-DD)"], value=TRUTH["due_date"]),
            c("browser_fill", ref=r["PO number (optional)"], value=TRUTH["po_number"]),
            c("browser_submit", ref=r["Submit invoice"])]


POLICY = [{"name": "big_invoice", "decision": "require_approval", "reason": "invoice total over INR 50,000",
           "tool": "browser_submit", "source": "effect", "arg": "amount", "op": "gt", "value": 50000}]


def run_engine(reg, planner, base_url, **kw):
    e = Engine(EventStore(), reg, PolicyEngine.from_dicts(POLICY), planner, backoff_s=0, **kw)
    rid = e.start("Enter Acme's latest invoice into AP")
    return e, rid, e.run(rid)


def head(reg, base_url, amount="62400"):
    r = refs(reg, base_url)   # leaves the browser on the form
    pre = [ToolCall(tool="read_document", args={"path": "scan_0931.pdf"}, rationale="read invoice")]
    return pre + fill_calls(r, amount)


def test_policy_sees_the_REAL_form_values_and_approval_summary_shows_them(env, base_url):
    reg, _ = env
    e, rid, s = run_engine(reg, Scripted(head(reg, base_url)), base_url)
    assert s.status == "waiting_approval"
    assert "62400" in s.pending_approval.summary and "Acme Corp" in s.pending_approval.summary  # not just "e9"
    assert httpx.get(f"{base_url}/api/invoices").json() == []                                   # nothing written yet


@pytest.mark.parametrize("mode,expect_accept", [("before_write", False), ("after_write", True)])
def test_ambiguous_failures_never_duplicate_or_lose_the_write(env, base_url, mode, expect_accept):
    reg, _ = env
    q = head(reg, base_url)
    httpx.post(f"{base_url}/__admin/chaos", data={"mode": mode, "count": 1})
    e, rid, s = run_engine(reg, Scripted(q), base_url)
    s = e.resolve_approval(rid, s.pending_approval.approval_id, True, by="rashmi")
    rows = httpx.get(f"{base_url}/api/invoices").json()
    strategies = [x.payload["strategy"] for x in e.store.read(rid) if x.type == ET.RECOVERY_CHOSEN]
    assert s.status == "completed" and len(rows) == 1 and rows[0]["amount"] == 62400.0
    assert strategies == (["accept"] if expect_accept else ["retry"])     # landed -> accept; not landed -> retry


def test_agent_recovers_from_validation_error_and_verifier_confirms(env, base_url):
    reg, _ = env
    bad = head(reg, base_url, amount="62,400")                              # comma trap, app rejects with 422
    fixed = fill_calls(refs(reg, base_url), "62400")                        # replan: plain number
    planner = Scripted(bad, replan_queue=fixed)
    e, rid, s = run_engine(reg, planner, base_url)
    s = e.resolve_approval(rid, s.pending_approval.approval_id, True)        # approve the (bad) submit
    assert s.status == "waiting_approval"                                    # verifier failed -> replanned -> new submit needs approval
    s = e.resolve_approval(rid, s.pending_approval.approval_id, True)
    assert s.status == "completed" and s.verify_failures == 1
    assert httpx.get(f"{base_url}/api/invoices").json()[0]["amount"] == 62400.0


def test_cannot_submit_a_selection_made_without_opening_every_candidate(env, base_url):
    """Regression for a REAL live-run failure: the agent read 2 of 3 Acme invoices, picked the wrong
    'latest', and verification passed (it only compared the entry against the agent's own facts)."""
    reg, _ = env
    r = refs(reg, base_url)
    c = lambda tool, **a: ToolCall(tool=tool, args=a, rationale="x")  # noqa: E731
    form = fill_calls(r)
    fill, submit = form[:-1], form[-1]
    others = ["globex_oct.pdf", "initech_sep.pdf", "INV-0412_acme.pdf", "scan_0931.pdf"]
    planner = Scripted([c("list_files", directory="."), c("read_document", path="acme_invoice_latest.pdf")] + fill + [submit],
                       replan_queue=[c("read_document", path=f) for f in others] + [c("browser_submit", ref=r["Submit invoice"])])
    rules = POLICY + [{"name": "inspect_all", "decision": "deny", "tool": "browser_submit", "source": "effect",
                       "arg": "_unread_sources", "op": "nonempty", "reason": "never opened: {_unread_sources}"}]
    e = Engine(EventStore(), reg, PolicyEngine.from_dicts(rules), planner, backoff_s=0)
    rid = e.start("x"); s = e.run(rid)
    assert "scan_0931.pdf" in planner.reasons[0] and "policy denied" in planner.reasons[0]   # agent is told exactly what it skipped
    assert s.status == "waiting_approval"                                                     # only after reading everything
    assert "invoice.amount=" in s.pending_approval.summary and "evidence" in s.pending_approval.summary  # human sees sources
    assert httpx.get(f"{base_url}/api/invoices").json() == []
    assert e.resolve_approval(rid, s.pending_approval.approval_id, True).status == "completed"
