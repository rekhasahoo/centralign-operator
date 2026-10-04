"""
Run the operator on a task from the terminal.   python run_task.py --task "..." [--headed] [--chaos after_write]
Starts the mock AP portal in the background, wires tools + policy + LLM planner, streams the event log live,
and asks YOU for approvals / clarifications in the terminal.
"""
import argparse, json, os, pathlib, socket, sys, threading, time

import httpx, uvicorn

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from envs.invoice_env import build_invoice_env  # noqa: E402
from mockco.app import app  # noqa: E402
from opcore import AzureOpenAIClient, Engine, EventStore, LLMPlanner, PolicyEngine  # noqa: E402
from opcore.schemas import EventType as ET  # noqa: E402

DEFAULT_TASK = "Find the latest invoice from Acme Corp, enter its amount and due date into the AP portal, and tell me once it is done."


def load_dotenv():
    f = ROOT / ".env"
    if f.exists():
        for ln in f.read_text().splitlines():
            if ln.strip() and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


def gist(ev) -> str:
    p, t = ev.payload, ev.type
    if t == ET.TOOL_REQUESTED: return f"{p['call']['tool']}({json.dumps(p['call']['args'])[:90]})  # {p['call']['rationale'][:80]}"
    if t == ET.POLICY_DECIDED: return f"{p['decision']} [{p['rule']}] {p['reason'][:80]}"
    if t in (ET.TOOL_SUCCEEDED, ET.TOOL_FAILED):
        r = p["result"]; return (r["summary"].splitlines() or [""])[0][:100] + (f"  <{r['error']['kind']}>" if r.get("error") else "")
    if t == ET.RECOVERY_CHOSEN: return f"{p['strategy']}: {p['detail'][:90]}"
    if t == ET.FACT_RECORDED: return f"{p['fact']['key']} = {p['fact']['value']!r}"
    if t == ET.FACT_REJECTED: return f"{p['key']}: {p['reason']}"
    if t == ET.VERIFICATION_RESULT: return f"{p['criterion_id']}: {'PASS' if p['passed'] else 'FAIL'} expected={p['expected']!r} actual={p['actual']!r}"
    if t == ET.TASK_UNDERSTOOD: return p["spec"]["goal"]
    if t == ET.PLAN_CREATED or t == ET.PLAN_REVISED: return " > ".join(s["intent"] for s in p["plan"]["steps"])[:140]
    if t == ET.RUN_FINISHED: return f"{p['status'].upper()}: {p['summary'][:120]}"
    if t == ET.INPUT_REQUESTED: return p["question"][:120]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--chaos", choices=["before_write", "after_write"], help="inject a portal failure on the first submit")
    ap.add_argument("--db", default=":memory:", help="SQLite path for the event log (persist to inspect/replay)")
    a = ap.parse_args()
    load_dotenv()

    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    while True:
        try: httpx.get(base); break
        except httpx.HTTPError: time.sleep(0.05)
    if a.chaos: httpx.post(f"{base}/__admin/chaos", data={"mode": a.chaos, "count": 1})

    reg, session = build_invoice_env(base, str(ROOT / "mockco/data/invoices"), str(ROOT / "artifacts"), headless=not a.headed)
    llm = AzureOpenAIClient()
    planner = LLMPlanner(llm, reg, str(ROOT / "mockco/knowledge"))
    policy = PolicyEngine.from_dicts(json.loads((ROOT / "envs/invoice_policy.json").read_text()))
    engine = Engine(EventStore(a.db), reg, policy, planner,
                    on_event=lambda ev: print(f"{ev.seq:>3}  {ev.type.value:<20} {gist(ev)}"))
    rid = engine.start(a.task)
    print(f"Portal: {base}   run: {rid}\nTask: {a.task}\n")
    try:
        s = engine.run(rid)
        while s.status in ("waiting_approval", "waiting_input"):
            if s.pending_approval:
                print(f"\n APPROVAL NEEDED\n {s.pending_approval.summary}")
                ok = input(" Approve? [y/N] ").strip().lower() == "y"
                s = engine.resolve_approval(rid, s.pending_approval.approval_id, ok, by="cli", note="" if ok else "rejected in CLI")
            else:
                print(f"\n AGENT ASKS: {s.pending_input.question}")
                s = engine.provide_input(rid, s.pending_input.request_id, input(" > "))
    finally:
        print(f"\nFinal status: {s.status}\n{s.summary}")
        print("AP records now:", httpx.get(f"{base}/api/invoices").json())
        print(f"LLM usage: {llm.calls} calls, {llm.prompt_tokens} prompt + {llm.completion_tokens} completion tokens")
        session.close()


if __name__ == "__main__":
    main()
