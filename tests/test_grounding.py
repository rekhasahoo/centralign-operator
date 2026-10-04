import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from opcore import *  # noqa
from opcore import grounding
from opcore.planner import ProposedFact
from opcore.schemas import EventType as ET

TEXT = "ACME\nInvoice No.: INV-1\nIssued: 11/09/2026\nPayment Due: 28 Oct 2026\nTOTAL DUE (INR)      62,400.00\nTotal payable: INR 1,25,000.00"


def evs():
    r = ToolResult(call_id="c1", ok=True, summary=TEXT[:50], output={"text": TEXT})
    return [Event.make("r", 1, ToolOutcome(result=r), ET.TOOL_SUCCEEDED)]


def pf(value, excerpt, call="c1"): return ProposedFact(key="k", value=value, source_call_id=call, excerpt=excerpt)
def ok(*a, **k): return grounding.check(pf(*a, **k), evs())[0]


def test_accepts_supported_values_in_the_formats_real_documents_use():
    assert ok(62400, "TOTAL DUE (INR) 62,400.00")                    # whitespace-insensitive
    assert ok(125000, "Total payable: INR 1,25,000.00")              # Indian digit grouping
    assert ok("2026-10-28", "Payment Due: 28 Oct 2026")              # date normalisation
    assert ok("2026-09-11", "Issued: 11/09/2026")                    # day/month/year (India)
    assert ok("INV-1", "Invoice No.: INV-1")


def test_rejects_invention_and_misreadings():
    assert not ok(64200, "TOTAL DUE (INR) 62,400.00")                # value not in excerpt
    assert not ok(64200, "TOTAL DUE (INR) 64,200.00")                # excerpt itself invented
    assert not ok("2026-11-09", "Issued: 11/09/2026")                # wrong day/month reading
    assert not ok(62400, "TOTAL DUE (INR) 62,400.00", call="nope")   # unknown evidence
    assert not ok(62400, "")                                         # no quote


def test_tolerates_call_id_formatting_and_lists_valid_ids_on_miss():
    e = [Event.make("r", 1, ToolOutcome(result=ToolResult(call_id="call_ab12", ok=True, summary="x", output={"text": TEXT})), ET.TOOL_SUCCEEDED)]
    for cid in ("call_ab12", "ab12", "[call_ab12]"):                      # real LLMs drop the prefix / add brackets
        assert grounding.check(pf(62400, "TOTAL DUE (INR) 62,400.00", call=cid), e)[0]
    ok_, why, _ = grounding.check(pf(62400, "TOTAL DUE (INR) 62,400.00", call="zzzz"), e)
    assert not ok_ and "call_ab12" in why                                  # error tells the model the valid ids
