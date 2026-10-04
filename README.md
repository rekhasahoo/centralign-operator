# CentrAlign Operator: an autonomous AI operator that completes company tasks and proves it

Founding Engineer take-home. The prototype turns a plain-language company request into **completed work in a real
(sandboxed) web app**, pausing for human approval where policy requires it and **independently verifying** the outcome.

> *"Find the latest invoice from Acme Corp, enter its amount and due date into the AP portal, and tell me once it is done."*

## The idea in one paragraph
Most agent demos are an LLM in a loop that grades itself. This one is built around **trust**: the LLM only *proposes*
(a task spec, a plan, the next action); a deterministic runtime decides what is *permitted* (policy + human approval),
*executes*, *records everything* in an append-only event log, and *verifies* the outcome against the real system with
read-only checks the agent cannot influence. Facts the agent relies on must quote their evidence verbatim or they are rejected.

## What is real vs. mocked (honest)
| Real | Mocked / simulated |
|---|---|
| LLM planning (Azure OpenAI `gpt-4.1-mini`) | The "company": a small FastAPI Accounts Payable portal (`mockco/app.py`) |
| Real Chromium driven by Playwright | Invoice PDFs are generated (with deliberate traps) |
| Real PDF text extraction, real form submission | Company procedures are a markdown file |
| Real policy gate, approvals, retries, verification | Failures are injected on demand (`--chaos`) |

## Quick start
```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env        # then fill in endpoint, key, deployment name (never commit .env)
python -m pytest tests -q   # 21 tests; no network or LLM needed
python run_task.py --headed --db run.db
python run_task.py --headed --chaos after_write     # portal saves the record, then returns 503
python run_task.py --task "Find the latest invoice from Globex Ltd and enter it into the AP portal"
```
The CLI streams the event log live and asks *you* to approve risky actions. `--db run.db` keeps the log for replay.

## Architecture
```
request ─▶ Understand ─▶ Plan ─▶ [ Decide ▶ Gate ▶ Execute ▶ Observe ▶ Adapt ]* ─▶ Verify ─▶ Complete
              LLM         LLM        LLM     policy   tools    log       LLM/recovery     verifier
```
- `opcore/engine.py`: the loop. Each iteration folds the event log into state and takes one step; nothing lives only in memory, so runs can pause for hours (approval) or resume after a crash.
- `opcore/policy.py`: permissions as data (`envs/invoice_policy.json`). Tier defaults (read / reversible / irreversible / external) plus rules on tool args *or on the previewed real effect*. deny > require_approval > allow.
- `opcore/registry.py`, `tools_browser.py`, `tools_files.py`: typed tools (JSON-schema args, timeouts, typed failures). Browser control uses accessibility-style snapshots, not screenshots.
- `opcore/recovery.py`: failure kind to strategy (retry / accept / replan / ask human / abort), with retry budgets.
- `opcore/verifier.py`: success criteria are *machine-checkable* read-tool checks produced at the Understand step and run after the agent says "done".
- `opcore/grounding.py`: facts must cite a tool result and quote it verbatim; the value must be supported by the quote.
- `opcore/llm_planner.py`, `llm.py`: the only LLM-facing code (Azure OpenAI is isolated in `llm.py`).
- `envs/` + `mockco/`: a *scenario* (tools, policy, company docs). `opcore/` contains nothing invoice-specific.

Deeper write-up: `ARCHITECTURE.md`. Typed contracts: `opcore/schemas.py`.

## Key design decisions (and why)
1. **LLM proposes, runtime disposes.** Safety and success are not decided by the model, so they do not depend on prompt wording.
2. **Event sourcing.** State is a pure fold over an append-only log: audit trail, resumability, replay for debugging/evals, and "why did the agent do that?" answered from data.
3. **The DOM decides what is irreversible.** `browser_click` refuses submit buttons; `browser_submit` is a separate WRITE_IRREVERSIBLE tool that policy can gate. The model cannot sneak a submission through a generic click.
4. **Effect preview.** Before a risky call the runtime reads the real form values, so policy rules can match on them (amount > 50,000) and approvals show *what will actually happen*, with the evidence behind each fact.
5. **Grounded facts.** Stops plausible-but-invented values from reaching an irreversible action.
6. **Ambiguous failures are handled with confirm-checks.** After a 503 the runtime asks the system of record whether the write landed: landed means accept (no duplicate), not landed means retry. Non-idempotent writes are never blindly retried.
7. **Coverage guard (added after a real failure, see below).** An irreversible action is denied if the agent enumerated candidates (listed files) but never opened them all.

## What testing taught me (a real failure and the fix)
In a live run the agent read only 2 of the 3 Acme invoices, entered the **wrong "latest" invoice**, and independent
verification still **passed**, because the checks compared the portal record with the agent's *own* facts. A prompt
instruction ("inspect all candidates") did not stop a small model from skipping one. The fix is a rule enforced by the
runtime: submit is denied with the exact list of unread documents, the agent replans, reads them, and corrects itself.
Final live run: the guard fired, the agent read the remaining files, recorded `INV-2026-0455 / 62400 / 2026-10-28` with
verbatim evidence, the over-INR-50,000 rule paused for approval, and verification passed (28 LLM calls, about 82k prompt tokens).
A regression test reproduces the failure (`tests/test_env.py`).

## Generalization
To support a new task, add a folder under `envs/` (tools + policy JSON) and company documents under `knowledge/`.
The engine, policy engine, recovery, verifier, grounding and event schema are unchanged. See limitations for how much
of this has actually been exercised.

## Models, APIs, frameworks used
Azure OpenAI chat deployment `gpt-4.1-mini` (JSON mode; `AzureOpenAI` client, api-version 2024-12-01-preview);
Playwright (Chromium); FastAPI + Uvicorn (mock portal); Pydantic v2; SQLite; pypdf; jsonschema; httpx; python-dateutil;
reportlab (generating test PDFs); pytest. AI coding assistance (Claude) was used for parts of the implementation; I can
walk through and modify any module.

## Known limitations
- **One scenario exercised live end to end** (Acme invoice). Other tasks run through the same code but are less tested.
- **The model is small.** `gpt-4.1-mini` needed guardrails (parsing leniency, coverage guard). A stronger model would likely need fewer.
- **Coverage guard proves the agent *looked*, not that it *judged correctly*.** It could still misread a date; grounding and the human approval (which shows the evidence) are the remaining defences.
- **Optional fields can be skipped.** In the final live run the agent left the PO number blank (`PO-88123` was on the invoice). Verification did not check it because the model did not include it as a success criterion. I tightened the procedure text; a stronger fix is a required-field checklist derived from the form.
- Single process / single tenant; thread-based timeouts (a hung tool thread is not killed); no secrets vault; no auth in the mock portal.
- Company knowledge is a handful of documents injected into the prompt; no retrieval/vector store yet.
- Browser and files only: no desktop-application control.
- No web UI: the interface is a streaming CLI timeline.
- Policy rules are hand-written; the agent does not yet learn from approvals or rejections.

## Assumptions
"Latest" means the most recent *invoice date*, not file name or modified time; slash dates are day/month/year (India);
the amount is the final payable total including GST; invoices over INR 50,000 need requester approval; the mock portal and
PDFs stand in for real systems (no real credentials or third-party systems are used).

## What I would build next
1. Required-field and "unchanged side effects" verification (derive checks from the form schema; assert nothing else changed).
2. Retrieval over company memory, and learning from approved/rejected runs (distil into procedures).
3. Durable queue + workers with leases; sandboxed per-tenant browser containers; credential vault.
4. Desktop control (OS accessibility APIs / computer-use models) behind the same tool + policy interfaces.
5. An eval harness that replays logged runs against changed prompts/models and scores outcome, safety and cost.
6. Web UI: live timeline, approve/reject, evidence viewer.

## Repository layout
```
run_task.py            CLI entry point
opcore/                task-agnostic runtime (engine, policy, tools, verifier, grounding, planner)
envs/                  scenario wiring: tools + policy JSON
mockco/                mock company: AP portal, invoice PDFs, procedures, ground truth
tests/                 21 tests (engine, real browser end-to-end, grounding, LLM planner with a fake LLM)
ARCHITECTURE.md        design write-up
```
