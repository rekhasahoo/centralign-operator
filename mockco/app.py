"""
"Northwind AP Portal": a deliberately imperfect internal system for the agent to operate.

Realistic traps (each one a reason the runtime needs verification/recovery):
  * strict validation: plain-number amounts, YYYY-MM-DD dates          -> 422 with an alert banner
  * duplicate invoice numbers rejected                                  -> blind retries are NOT harmless
  * chaos mode: 503 *before* the write, or 503 *after* the write        -> ambiguous failures
  * a read-only JSON API the verifier uses independently of the browser
"""
from __future__ import annotations

import html
import re
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

app = FastAPI(title="Northwind AP Portal")

VENDORS = ["Acme Corp", "Globex Ltd", "Initech"]
RECORDS: dict[str, dict] = {}
CHAOS = {"mode": None, "count": 0}   # mode: "before_write" | "after_write"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def page(title: str, body: str, status: int = 200) -> HTMLResponse:
    nav = '<nav><a href="/">Home</a> | <a href="/invoices">Invoices</a> | <a href="/invoices/new">New invoice</a></nav>'
    return HTMLResponse(f"<!doctype html><html><head><title>{html.escape(title)}</title>"
                        f"<style>body{{font-family:sans-serif;margin:2rem}}label{{display:block;margin:.6rem 0}}"
                        f".err{{color:#b00020}}.ok{{color:#0a7d33}}td,th{{border:1px solid #ccc;padding:.3rem .6rem}}</style>"
                        f"</head><body>{nav}<h1>{html.escape(title)}</h1>{body}</body></html>", status_code=status)


@app.get("/", response_class=HTMLResponse)
def home():
    return page("Northwind Accounts Payable", "<p>Welcome. Use the menu to view or record vendor invoices.</p>")


@app.get("/invoices", response_class=HTMLResponse)
def invoices(created: Optional[str] = None):
    banner = f'<p role="status" class="ok">Invoice {html.escape(created)} saved.</p>' if created else ""
    rows = "".join(
        f"<tr><td>{html.escape(r['invoice_number'])}</td><td>{html.escape(r['vendor'])}</td>"
        f"<td>{r['amount']:.2f}</td><td>{r['due_date']}</td><td>{html.escape(r['po_number'])}</td></tr>"
        for r in RECORDS.values())
    table = ("<table><tr><th>Invoice number</th><th>Vendor</th><th>Amount</th><th>Due date</th><th>PO</th></tr>"
             f"{rows}</table>") if RECORDS else "<p>No invoices recorded yet.</p>"
    return page("Accounts Payable: Invoices", banner + table)


def _form(values: dict, error: str = "", status: int = 200) -> HTMLResponse:
    v = lambda k: html.escape(values.get(k, ""))  # noqa: E731
    opts = "".join(f'<option value="{html.escape(x)}"{" selected" if x == values.get("vendor") else ""}>{html.escape(x)}</option>'
                   for x in [""] + VENDORS)
    err = f'<p role="alert" class="err">{html.escape(error)}</p>' if error else ""
    body = (f'{err}<form method="post" action="/invoices">'
            f'<label for="vendor">Vendor</label><select id="vendor" name="vendor">{opts}</select>'
            f'<label for="invoice_number">Invoice number</label><input id="invoice_number" name="invoice_number" value="{v("invoice_number")}">'
            f'<label for="amount">Amount (INR)</label><input id="amount" name="amount" value="{v("amount")}">'
            f'<label for="due_date">Due date (YYYY-MM-DD)</label><input id="due_date" name="due_date" value="{v("due_date")}">'
            f'<label for="po_number">PO number (optional)</label><input id="po_number" name="po_number" value="{v("po_number")}">'
            f'<button type="submit">Submit invoice</button> <a href="/invoices">Cancel</a></form>')
    return page("Record vendor invoice", body, status)


@app.get("/invoices/new", response_class=HTMLResponse)
def new_invoice():
    return _form({})


@app.post("/invoices")
def create_invoice(vendor: str = Form(""), invoice_number: str = Form(""), amount: str = Form(""),
                   due_date: str = Form(""), po_number: str = Form("")):
    vals = dict(vendor=vendor, invoice_number=invoice_number.strip(), amount=amount.strip(),
                due_date=due_date.strip(), po_number=po_number.strip())
    if CHAOS["count"] > 0 and CHAOS["mode"] == "before_write":
        CHAOS["count"] -= 1
        return page("Service unavailable", '<p role="alert" class="err">503: upstream timeout, nothing was saved.</p>', 503)
    if vendor not in VENDORS:
        return _form(vals, "Please select a vendor.", 422)
    if not vals["invoice_number"]:
        return _form(vals, "Invoice number is required.", 422)
    if vals["invoice_number"] in RECORDS:
        return _form(vals, f"Invoice number {vals['invoice_number']} already exists.", 409)
    try:
        amt = float(vals["amount"])
        if amt <= 0:
            raise ValueError
    except ValueError:
        return _form(vals, "Amount must be a plain positive number (no commas or currency symbols).", 422)
    if not DATE_RE.match(vals["due_date"]):
        return _form(vals, "Due date must use the format YYYY-MM-DD.", 422)
    RECORDS[vals["invoice_number"]] = dict(vendor=vendor, invoice_number=vals["invoice_number"], amount=amt,
                                           due_date=vals["due_date"], po_number=vals["po_number"])
    if CHAOS["count"] > 0 and CHAOS["mode"] == "after_write":
        CHAOS["count"] -= 1  # the record IS saved, but the user sees an error: the nasty case
        return page("Service unavailable", '<p role="alert" class="err">503: gateway error while confirming.</p>', 503)
    return RedirectResponse(f"/invoices?created={vals['invoice_number']}", status_code=303)


@app.get("/api/invoices")
def api_invoices(invoice_number: Optional[str] = None, vendor: Optional[str] = None):
    """Read-only system-of-record API, used by the verifier (independent of the browser)."""
    rows = [r for r in RECORDS.values()
            if (invoice_number is None or r["invoice_number"] == invoice_number)
            and (vendor is None or r["vendor"] == vendor)]
    return JSONResponse(rows)


# --- test/demo controls (not part of the "company system" the agent is told about) ---
@app.post("/__admin/reset")
def reset():
    RECORDS.clear(); CHAOS.update(mode=None, count=0)
    return {"ok": True}


@app.post("/__admin/chaos")
def chaos(mode: str = Form(...), count: int = Form(1)):
    CHAOS.update(mode=mode, count=count)
    return {"ok": True, **CHAOS}
