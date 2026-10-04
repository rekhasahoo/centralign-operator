"""
End-to-end tests of the deterministic runtime with a *scripted* planner (no LLM).
They prove the safety/reliability properties independent of model quality.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opcore import *  # noqa: F401,F403
from opcore.schemas import EventType as ET

INVOICE = {"vendor": "Acme", "amount": 62400, "due": "2026-10-28"}


# ---- a tiny fake company system + tools ---------------------------------------------------
def make_env(fail_first_submit=False):
    ap, calls = {}, {"submit": 0}
    reg = ToolRegistry()

    @reg.tool(ToolSpec(name="read_invoice", description="Read an invoice file", risk=Risk.READ,
                       input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}))
    def read_invoice(path):
        return ToolOutput("read invoice", dict(INVOICE))

    @reg.tool(ToolSpec(name="submit_ap_record", description="Create AP record", risk=Risk.WRITE_IRREVERSIBLE,
                       input_schema={"type": "object", "required": ["vendor", "amount", "due"],
                                     "properties": {"vendor": {"type": "string"}, "amount": {"type": "number"}, "due": {"type": "string"}}}),
              confirm=lambda a: a["vendor"] in ap and ap[a["vendor"]]["amount"] == a["amount"])
    def submit_ap_record(vendor, amount, due):
        calls["submit"] += 1
        if fail_first_submit and calls["submit"] == 1:
            raise ToolFailure(ErrorKind.TRANSIENT, "form timeout")   # fails BEFORE writing
        ap[vendor] = {"amount": amount, "due": due}
        return "submitted"

    @reg.tool(ToolSpec(name="get_ap_record", description="Look up AP record", risk=Risk.READ,
                       input_schema={"type": "object", "required": ["vendor"], "properties": {"vendor": {"type": "string"}}}))
    def get_ap_record(vendor):
        if vendor not in ap:
            raise ToolFailure(ErrorKind.NOT_FOUND, "no record")
        return ap[vendor]

    return reg, ap, calls


SPEC = TaskSpec(goal="Record Acme invoice in AP", success_criteria=[
    Criterion(id="amount", description="AP amount matches invoice",
              check=Check(tool="get_ap_record", args={"vendor": "Acme"},
                          expect=Expectation(op="eq", path="$.amount", value=FactRef(fact="invoice.amount")))),
    Criterion(id="due", description="AP due date matches invoice",
              check=Check(tool="get_ap_record", args={"vendor": "Acme"},
                          expect=Expectation(op="eq", path="$.due", value=FactRef(fact="invoice.due")))),
])


def read_call():  return ToolCall(tool="read_invoice", args={"path": "acme.pdf"}, rationale="need invoice data")
def submit_call(amount=62400): return ToolCall(tool="submit_ap_record",
                    args={"vendor": "Acme", "amount": amount, "due": "2026-10-28"}, rationale="enter into AP")


class Scripted:
    def __init__(self, queue, replan_queue=None, confidence=1.0):
        self.queue, self.replan_queue, self.conf, self.reasons = queue, replan_queue or [], confidence, []
    def understand(self, s, ev): return SPEC.model_copy(deep=True)
    def plan(self, s, ev): return Plan(steps=[PlanStep(id="s1", intent="do it")])
    def replan(self, s, ev, reason):
        self.reasons.append(reason); self.queue = list(self.replan_queue)
        return Plan(version=s.plan.version + 1, steps=s.plan.steps)
    def next_action(self, s, ev): return self.queue.pop(0) if self.queue else Done("finished")
    def extract_facts(self, s, call, result, src):
        if call.tool != "read_invoice": return []
        return [Fact(key=f"invoice.{k}", value=v, source_event_id=src, confidence=self.conf)
                for k, v in result.output.items()]


def engine(reg, planner, policy=None, store=None):
    return Engine(store or EventStore(), reg, policy or PolicyEngine.from_dicts([
        {"name": "big_invoice", "decision": "require_approval", "reason": "amount over 50,000",
         "tool": "submit_ap_record", "arg": "amount", "op": "gt", "value": 50000}]), planner, backoff_s=0)


def types(eng, rid): return [e.type for e in eng.store.read(rid)]


# ---- tests -------------------------------------------------------------------------------
def test_approval_gate_retry_and_verification_with_resume_on_fresh_engine():
    reg, ap, calls = make_env(fail_first_submit=True)
    planner = Scripted([read_call(), submit_call()])
    store = EventStore()
    e1 = engine(reg, planner, store=store)
    rid = e1.start("Enter Acme's latest invoice into AP")
    s = e1.run(rid)
    assert s.status == "waiting_approval" and ap == {}            # nothing written before approval

    e2 = engine(reg, planner, store=store)                         # "process restart": new engine, same log
    s = e2.resolve_approval(rid, s.pending_approval.approval_id, True, by="rashmi")
    assert s.status == "completed"
    assert ap["Acme"]["amount"] == 62400 and calls["submit"] == 2  # failed once, retried once
    t = types(e2, rid)
    assert ET.TOOL_FAILED in t and ET.RECOVERY_CHOSEN in t
    assert [e.payload["passed"] for e in store.read(rid) if e.type == ET.VERIFICATION_RESULT] == [True, True]


def test_rejection_means_nothing_is_written_and_run_escalates_honestly():
    reg, ap, _ = make_env()
    planner = Scripted([read_call(), submit_call()], replan_queue=[])   # replan finds nothing else to do
    e = engine(reg, planner)
    rid = e.start("x"); s = e.run(rid)
    s = e.resolve_approval(rid, s.pending_approval.approval_id, False, note="amount looks wrong")
    assert ap == {} and s.status == "escalated"                    # never claims success it didn't achieve
    assert "rejected" in planner.reasons[0] and "amount looks wrong" in planner.reasons[0]


def test_verifier_catches_a_wrong_write_that_the_agent_thinks_is_fine():
    reg, ap, _ = make_env()
    allow = PolicyEngine.from_dicts([{"name": "preauth", "decision": "allow", "reason": "pre-authorised",
                                      "tool": "submit_ap_record"}])
    planner = Scripted([read_call(), submit_call(amount=62040)], replan_queue=[submit_call(62400)])  # typo first
    e = engine(reg, planner, policy=allow)
    rid = e.start("x"); s = e.run(rid)
    assert s.status == "completed" and ap["Acme"]["amount"] == 62400
    assert s.verify_failures == 1 and "amount" in planner.reasons[0]


def test_low_confidence_extraction_forces_approval_even_if_preauthorised():
    reg, ap, _ = make_env()
    allow = PolicyEngine.from_dicts([{"name": "preauth", "decision": "allow", "reason": "ok", "tool": "submit_ap_record"}])
    e = engine(reg, Scripted([read_call(), submit_call()], confidence=0.5), policy=allow)
    rid = e.start("x"); s = e.run(rid)
    assert s.status == "waiting_approval" and ap == {}
    assert "low_confidence" in [x.payload.get("rule") for x in e.store.read(rid) if x.type == ET.POLICY_DECIDED][-1]


def test_unknown_tool_is_denied_not_executed():
    reg, ap, _ = make_env()
    bad = ToolCall(tool="wire_money", args={}, rationale="yolo")
    e = engine(reg, Scripted([bad], replan_queue=[]))
    rid = e.start("x"); s = e.run(rid)
    assert s.call_status[bad.call_id] == "handled" and ap == {}


def test_run_without_verifiable_criteria_refuses_to_claim_completion():
    reg, _, _ = make_env()
    p = Scripted([])
    p.understand = lambda s, ev: TaskSpec(goal="vibes", success_criteria=[])
    e = engine(reg, p)
    rid = e.start("x")
    assert e.run(rid).status == "escalated"
