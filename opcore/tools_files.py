"""Sandboxed file tools. All paths are resolved inside `root`; escaping it is a permission error."""
from __future__ import annotations

import datetime as dt
import pathlib

from pypdf import PdfReader

from .registry import ToolFailure, ToolOutput, ToolRegistry
from .schemas import ErrorKind, Risk, ToolSpec


def register_file_tools(reg: ToolRegistry, root: str) -> None:
    base = pathlib.Path(root).resolve()

    def safe(rel: str) -> pathlib.Path:
        p = (base / rel).resolve()
        if base != p and base not in p.parents:
            raise ToolFailure(ErrorKind.PERMISSION_DENIED, f"'{rel}' is outside the allowed folder")
        return p

    @reg.tool(ToolSpec(
        name="list_files", description="List files in the document folder, which is the ONLY folder you can access. Call it with directory '.' first to discover what exists; do not guess folder names.",
        risk=Risk.READ, idempotent=True, enumerates=True,
        input_schema={"type": "object", "properties": {"directory": {"type": "string", "default": "."}}}))
    def list_files(directory: str = "."):
        d = safe(directory.replace("\\", "/"))
        if not d.is_dir():
            raise ToolFailure(ErrorKind.NOT_FOUND, f"folder '{directory}' not found. The only accessible folder is the document root '.'; call list_files with directory '.' to see what is there")
        rows = [{"name": f.name, "size": f.stat().st_size,
                 "modified": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")}
                for f in sorted(d.iterdir()) if f.is_file()]
        return ToolOutput(f"{len(rows)} files: " + ", ".join(r["name"] for r in rows), rows)

    @reg.tool(ToolSpec(
        name="read_document", description="Extract the text of a PDF or text file. `path` is relative to the document folder (use names returned by list_files).",
        risk=Risk.READ, idempotent=True, inspects="path",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}))
    def read_document(path: str):
        f = safe(path.replace("\\", "/"))
        if not f.is_file():
            raise ToolFailure(ErrorKind.NOT_FOUND, f"file '{path}' not found; use list_files with directory '.' to see available files")
        text = ("\n".join(pg.extract_text() or "" for pg in PdfReader(str(f)).pages)
                if f.suffix.lower() == ".pdf" else f.read_text(errors="replace"))
        if not text.strip():
            raise ToolFailure(ErrorKind.STATE_MISMATCH, f"'{path}' has no extractable text (scanned image?)")
        return ToolOutput(f"[{path}]\n{text[:3000]}", {"path": path, "text": text})
