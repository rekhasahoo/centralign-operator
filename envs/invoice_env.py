"""Wires the invoice scenario: file tools + browser tools + a read-only AP API tool."""
from __future__ import annotations

import httpx

from opcore.registry import ToolFailure, ToolOutput, ToolRegistry
from opcore.schemas import ErrorKind, Risk, ToolSpec
from opcore.tools_browser import BrowserSession, register_browser_tools
from opcore.tools_files import register_file_tools


def build_invoice_env(base_url: str, invoice_dir: str, artifacts_dir: str, headless: bool = True):
    reg, session = ToolRegistry(), BrowserSession(artifacts_dir, headless)
    http = httpx.Client(base_url=base_url, timeout=5)

    def lookup(invoice_number: str):
        r = http.get("/api/invoices", params={"invoice_number": invoice_number})
        r.raise_for_status()
        return r.json()

    def confirm_submit(form: dict) -> bool:
        """Did the last submitted form actually create a record? Asked of the system of record, not the browser."""
        rows = lookup(form.get("invoice_number", ""))
        return bool(rows) and abs(rows[0]["amount"] - float(form.get("amount") or -1)) < 0.005

    register_file_tools(reg, invoice_dir)
    register_browser_tools(reg, session, confirm_submit)

    @reg.tool(ToolSpec(
        name="ap_api_get_invoice", risk=Risk.READ, idempotent=True,
        description="Look up an invoice record in the AP system of record by invoice number (read-only API, independent of the browser UI).",
        input_schema={"type": "object", "required": ["invoice_number"], "properties": {"invoice_number": {"type": "string"}}}))
    def ap_api_get_invoice(invoice_number: str):
        try:
            rows = lookup(invoice_number)
        except httpx.HTTPError as e:
            raise ToolFailure(ErrorKind.TRANSIENT, f"AP API unreachable: {type(e).__name__}") from e
        if not rows:
            return ToolOutput(f"No AP record exists for {invoice_number}.", None)   # an answer, not an error
        return ToolOutput(f"AP record: {rows[0]}", rows[0])

    reg.notes = (f"- Documents: you can read files only inside one sandboxed document folder. Call list_files with directory '.' to see them; paths are relative to it.\n"
                 f"- The internal Accounts Payable (AP) portal is a web app at {base_url} (pages: /invoices lists recorded invoices, /invoices/new is the entry form). Use browser_open with full URLs.")
    return reg, session
