"""
Browser tools over Playwright, driven by an accessibility-style *snapshot* (numbered interactive
elements + visible text) rather than screenshots: cheaper, deterministic, and checkable.

Design decisions:
  * One dedicated browser thread (Playwright's sync API is thread-affine).
  * `browser_click` refuses submit controls; `browser_submit` is a separate WRITE_IRREVERSIBLE
    tool. The *DOM*, not the LLM, decides what counts as irreversible, so policy can gate it.
  * `browser_submit` has a `preview` (real form values) and a `confirm` hook (did it land?).
  * A 5xx response is a TRANSIENT failure; a 4xx re-rendered form is an *observation* (the
    planner sees the alert text and adapts).
"""
from __future__ import annotations

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .registry import ToolFailure, ToolOutput, ToolRegistry
from .schemas import ArtifactRef, ErrorKind, Risk, ToolSpec

_SNAPSHOT_JS = """() => {
  const sel = 'a[href],button,input,select,textarea,[role=button]';
  const vis = e => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length) && getComputedStyle(e).visibility !== 'hidden';
  document.querySelectorAll('[data-ref]').forEach(e => e.removeAttribute('data-ref'));
  let n = 0; const els = [];
  for (const e of document.querySelectorAll(sel)) {
    if (!vis(e) || e.type === 'hidden') continue;
    const ref = 'e' + (++n); e.setAttribute('data-ref', ref);
    const label = (e.labels && e.labels[0] ? e.labels[0].innerText : '') || e.getAttribute('aria-label') ||
                  (['BUTTON', 'A'].includes(e.tagName) ? e.innerText : '') || e.placeholder || e.name || '';
    const d = {ref, tag: e.tagName.toLowerCase(), type: e.type || null, name: label.trim().slice(0, 80)};
    if ('value' in e && e.tagName !== 'BUTTON') d.value = e.value;
    if (e.tagName === 'SELECT') d.options = [...e.options].map(o => o.text);
    if (e.tagName === 'A') d.href = e.getAttribute('href');
    if (e.disabled) d.disabled = true;
    els.push(d);
  }
  const alerts = [...document.querySelectorAll('[role=alert],[role=status]')].map(e => e.innerText.trim()).filter(Boolean);
  return {url: location.href, title: document.title, alerts, text: document.body.innerText.slice(0, 2500), elements: els};
}"""
_FORM_JS = "el => el.form ? Object.fromEntries(new FormData(el.form).entries()) : {}"
_RESTORE_JS = "vals => { for (const [k, v] of Object.entries(vals)) { const el = document.querySelector(`[name=\"${k}\"]`); if (el) el.value = v; } }"


def _fmt(s: dict, status: Optional[int]) -> str:
    out = [f"URL: {s['url']}  (HTTP {status})" if status else f"URL: {s['url']}", f"Title: {s['title']}"]
    if s["alerts"]:
        out.append("ALERTS: " + " | ".join(s["alerts"]))
    out.append("Interactive elements:")
    for e in s["elements"]:
        extra = (f" value={e['value']!r}" if e.get("value") else "") + (f" options={e['options']}" if e.get("options") else "") \
            + (f" href={e['href']}" if e.get("href") else "") + (" [disabled]" if e.get("disabled") else "")
        kind = e["tag"] + (f"({e['type']})" if e.get("type") and e["tag"] == "input" else "")
        out.append(f"  [{e['ref']}] {kind} {e['name']!r}{extra}")
    out.append("Page text:\n" + s["text"])
    return "\n".join(out)[:4500]


class BrowserSession:
    def __init__(self, artifacts_dir: str, headless: bool = True):
        self.artifacts = pathlib.Path(artifacts_dir); self.artifacts.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.last_form: dict = {}          # form values captured at the last submit (used by confirm hooks)
        self.last_status: Optional[int] = None
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._pw = self._browser = self._page = None

    def run(self, fn: Callable, *a):
        return self._ex.submit(fn, *a).result()

    def _page_(self):
        if self._page is None:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless, args=["--no-sandbox"])
            self._page = self._browser.new_page()
            self._page.set_default_timeout(8000)
            self._page.on("response", self._on_response)
        return self._page

    def _on_response(self, r):
        if r.request.resource_type == "document" and r.frame == self._page.main_frame:
            self.last_status = r.status

    def snapshot(self) -> dict:
        return self._page_().evaluate(_SNAPSHOT_JS)

    def loc(self, ref: str):
        l = self._page_().locator(f'[data-ref="{ref}"]')
        if l.count() == 0:
            raise ToolFailure(ErrorKind.NOT_FOUND, f"element {ref} not found; take a fresh browser_snapshot (refs reset on navigation)")
        return l.first

    def restore_form(self) -> None:
        """After a transient failure, put the browser back in its pre-submit state (form visible, values
        refilled, refs reassigned) so that *retrying the same call* is meaningful. Best-effort."""
        try:
            p = self._page_()
            p.go_back(wait_until="load")
            p.evaluate(_RESTORE_JS, self.last_form)
            self.snapshot()
        except Exception:  # noqa: BLE001
            pass

    def screenshot(self, name: str) -> ArtifactRef:
        path = self.artifacts / f"{name}_{int(time.time() * 1000)}.png"
        self._page_().screenshot(path=str(path), full_page=True)
        return ArtifactRef(kind="screenshot", path=str(path), caption=name)

    def close(self):
        def _c():
            for fn in (lambda: self._browser and self._browser.close(), lambda: self._pw and self._pw.stop()):
                try: fn()
                except Exception:  # noqa: BLE001 - driver already gone (window closed / Ctrl+C): nothing to clean up
                    pass
        if self._page is not None:
            try: self.run(_c)
            except Exception:  # noqa: BLE001
                pass
        self._ex.shutdown(wait=False)


def register_browser_tools(reg: ToolRegistry, s: BrowserSession,
                           confirm_submit: Optional[Callable[[dict], bool]] = None) -> None:
    ref_schema = {"type": "object", "required": ["ref"], "properties": {"ref": {"type": "string", "pattern": "^e\\d+$"}}}

    def observe(prefix: str, shot: Optional[str] = None) -> ToolOutput:
        snap = s.snapshot()
        arts = [s.screenshot(shot)] if shot else []
        return ToolOutput(f"{prefix}\n{_fmt(snap, s.last_status)}", {**snap, "status": s.last_status}, arts)

    @reg.tool(ToolSpec(name="browser_open", description="Navigate the browser to a URL and return a snapshot.",
                       risk=Risk.READ, idempotent=True,
                       input_schema={"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}}))
    def browser_open(url: str):
        def go():
            try:
                s._page_().goto(url, wait_until="load")
            except Exception as e:  # noqa: BLE001
                raise ToolFailure(ErrorKind.TRANSIENT, f"could not load {url}: {type(e).__name__}") from e
            return observe(f"Opened {url}")
        return s.run(go)

    @reg.tool(ToolSpec(name="browser_snapshot", description="Re-read the current page: numbered interactive elements and visible text.",
                       risk=Risk.READ, idempotent=True, input_schema={"type": "object", "properties": {}}))
    def browser_snapshot():
        return s.run(lambda: observe("Snapshot"))

    @reg.tool(ToolSpec(name="browser_fill", description="Type text into an input/textarea (replaces its value). Reversible until submitted.",
                       risk=Risk.WRITE_REVERSIBLE, idempotent=True,
                       input_schema={"type": "object", "required": ["ref", "value"],
                                     "properties": {"ref": {"type": "string", "pattern": "^e\\d+$"}, "value": {"type": "string"}}}))
    def browser_fill(ref: str, value: str):
        def go():
            l = s.loc(ref)
            if l.evaluate("e => e.tagName") not in ("INPUT", "TEXTAREA"):
                raise ToolFailure(ErrorKind.INVALID_INPUT, f"{ref} is not a text field (use browser_select for dropdowns)")
            l.fill(value)
            return ToolOutput(f"Filled {ref} with {value!r}", {"ref": ref, "value": value})
        return s.run(go)

    @reg.tool(ToolSpec(name="browser_select", description="Choose an option in a dropdown by its visible text.",
                       risk=Risk.WRITE_REVERSIBLE, idempotent=True,
                       input_schema={"type": "object", "required": ["ref", "option"],
                                     "properties": {"ref": {"type": "string", "pattern": "^e\\d+$"}, "option": {"type": "string"}}}))
    def browser_select(ref: str, option: str):
        def go():
            l = s.loc(ref)
            try:
                l.select_option(label=option)
            except Exception as e:  # noqa: BLE001
                opts = l.evaluate("e => e.tagName === 'SELECT' ? [...e.options].map(o => o.text) : null")
                raise ToolFailure(ErrorKind.INVALID_INPUT, f"cannot select {option!r}; available: {opts}") from e
            return ToolOutput(f"Selected {option!r} in {ref}", {"ref": ref, "option": option})
        return s.run(go)

    @reg.tool(ToolSpec(name="browser_click", description="Click a link or ordinary button. NOT for submit buttons (use browser_submit).",
                       risk=Risk.WRITE_REVERSIBLE, input_schema=ref_schema))
    def browser_click(ref: str):
        def go():
            l = s.loc(ref)
            if l.evaluate("e => e.type === 'submit'"):
                raise ToolFailure(ErrorKind.INVALID_INPUT, f"{ref} submits a form; use browser_submit so the action can be reviewed")
            l.click()
            s._page_().wait_for_load_state("load")
            return observe(f"Clicked {ref}")
        return s.run(go)

    def _submit_preview(args: dict) -> dict:
        def go():
            l = s.loc(args["ref"])
            return {**l.evaluate(_FORM_JS), "_page": s._page_().url}
        return s.run(go)

    @reg.tool(ToolSpec(name="browser_submit", description="Click a form's submit button. IRREVERSIBLE: creates/changes records in the target system.",
                       risk=Risk.WRITE_IRREVERSIBLE, idempotent=False, max_retries=2, input_schema=ref_schema),
              confirm=(lambda _args: confirm_submit(s.last_form)) if confirm_submit else None,
              preview=_submit_preview)
    def browser_submit(ref: str):
        def go():
            l = s.loc(ref)
            if not l.evaluate("e => e.type === 'submit'"):
                raise ToolFailure(ErrorKind.INVALID_INPUT, f"{ref} is not a submit control; use browser_click")
            s.last_form = l.evaluate(_FORM_JS)
            s.last_status = None
            l.click()
            s._page_().wait_for_load_state("load")
            if s.last_status and s.last_status >= 500:
                s.restore_form()
                raise ToolFailure(ErrorKind.TRANSIENT, f"server returned HTTP {s.last_status} after submit")
            return observe(f"Submitted form (HTTP {s.last_status}). Check ALERTS below for validation errors.", "after_submit")
        return s.run(go)

    @reg.tool(ToolSpec(name="browser_screenshot", description="Capture a screenshot of the current page as evidence.",
                       risk=Risk.READ, idempotent=True, input_schema={"type": "object", "properties": {}}))
    def browser_screenshot():
        return s.run(lambda: ToolOutput("Screenshot captured", None, [s.screenshot("evidence")]))
