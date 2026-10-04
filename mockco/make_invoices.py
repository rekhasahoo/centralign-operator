"""Generates messy, realistic invoice PDFs + a ground-truth file for evals. Run: python -m mockco.make_invoices"""
import json
import pathlib

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = pathlib.Path(__file__).parent / "data" / "invoices"
OUT.mkdir(parents=True, exist_ok=True)


def pdf(name, lines):
    c = canvas.Canvas(str(OUT / name), pagesize=A4)
    y = 800
    for size, text in lines:
        c.setFont("Helvetica-Bold" if size > 12 else "Helvetica", size)
        c.drawString(60, y, text)
        y -= size + 10
    c.save()


# Acme, the NEWEST invoice, but with a meaningless filename. Total includes GST (subtotal != total).
pdf("scan_0931.pdf", [
    (20, "ACME CORP"), (14, "TAX INVOICE"), (11, ""),
    (11, "Invoice No.: INV-2026-0455"), (11, "Invoice Date: 02 Oct 2026"), (11, "Payment Due: 28 Oct 2026"),
    (11, "PO Ref: PO-88123"), (11, "Bill To: Northwind Traders Pvt Ltd"), (11, ""),
    (11, "Industrial sensors (x40)            38,400.00"), (11, "Installation & calibration      14,481.36"),
    (11, "Subtotal                            52,881.36"), (11, "GST @ 18%                          9,518.64"),
    (13, "TOTAL DUE (INR)                    62,400.00"),
])
# Acme, OLDER, slash dates in dd/mm/yyyy (India), different labels.
pdf("INV-0412_acme.pdf", [
    (20, "Acme Corp."), (14, "Commercial Bill"), (11, ""),
    (11, "Bill #: INV-2026-0412"), (11, "Issued: 11/09/2026"), (11, "Terms: Net 30  (due 11/10/2026)"),
    (11, "Customer PO: PO-87650"), (11, ""),
    (11, "Maintenance contract Q3             41,440.68"), (11, "CGST 9%                             3,729.66"),
    (11, "SGST 9%                             3,729.66"), (13, "Amount Payable: INR 48,900.00"),
])
# Acme, OLDEST, but the filename claims "latest" (filenames lie).
pdf("acme_invoice_latest.pdf", [
    (20, "ACME CORP"), (14, "INVOICE"), (11, ""),
    (11, "Document ID: INV-2026-0388"), (11, "Invoice date: 2026-08-14"), (11, "Due date: 2026-09-13"),
    (13, "Grand Total (INR): 31,250.00"),
])
# Globex, Indian digit grouping.
pdf("globex_oct.pdf", [
    (20, "GLOBEX LTD"), (14, "TAX INVOICE"), (11, ""),
    (11, "Invoice No.: INV-G-7781"), (11, "Date: 05 Oct 2026"), (11, "Due: 04 Nov 2026"),
    (13, "Total payable: INR 1,25,000.00"),
])
pdf("initech_sep.pdf", [
    (20, "Initech"), (14, "Invoice"), (11, ""),
    (11, "Invoice #: INV-IN-220"), (11, "Date: 19 Sep 2026"), (11, "Due: 19 Oct 2026"),
    (13, "Amount due: INR 9,800.00"),
])

truth = {
    "latest_by_vendor": {
        "Acme Corp": {"invoice_number": "INV-2026-0455", "amount": 62400.0, "due_date": "2026-10-28", "po_number": "PO-88123", "file": "scan_0931.pdf"},
        "Globex Ltd": {"invoice_number": "INV-G-7781", "amount": 125000.0, "due_date": "2026-11-04", "po_number": "", "file": "globex_oct.pdf"},
        "Initech": {"invoice_number": "INV-IN-220", "amount": 9800.0, "due_date": "2026-10-19", "po_number": "", "file": "initech_sep.pdf"},
    }
}
(pathlib.Path(__file__).parent / "eval" / "ground_truth.json").write_text(json.dumps(truth, indent=2))
print("generated", sorted(p.name for p in OUT.glob("*.pdf")))
