"""Browser smoke tests for the context panel's page (issue #687).

Seeds a channel's window on the server, logs in, opens the panel in headless
Chrome or Chromium, picks the channel in the ``<select>`` and asserts on the
rendered DOM: untrusted text stays text (R11), each category gets its class
(R15), carets open and close entries (R16), and the tail and resync paths
reach the page (R7).

Like tests/test_web_render_smoke.py it is skipped without Playwright, and the
``web-smoke`` CI job fails if it is skipped. To run it locally::

    uv run --with playwright pytest tests/test_context_panel/test_render_smoke.py
"""

from __future__ import annotations

import re

import pytest

# Skip the module cleanly where Playwright isn't installed.
pytest.importorskip("playwright.async_api")

from context_panel import ContextPanelPlugin  # noqa: E402
from playwright.async_api import Error as PlaywrightError  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402

from corvidae.context import ContextWindow, MessageType  # noqa: E402
from corvidae.outbound import OutboundProgress, OutboundToolStatus  # noqa: E402
from corvidae.tool_policy import ToolView  # noqa: E402
from tests.test_context_panel.test_context_panel import log_row  # noqa: E402
from tests.test_web_plugin import TOKEN, _no_env_token, make_env  # noqa: E402,F401
from tests.test_web_render_smoke import HOSTILE, _launch  # noqa: E402

CID = "web:main"

# Every entry in the panel: its label, classes, open state and body text.
ENTRIES_JS = """
() => [...document.querySelectorAll("#panel details.entry")].map(d => ({
  label: (d.querySelector(":scope > summary") || {}).textContent || "",
  cls: [...d.classList],
  open: d.open,
  text: (d.querySelector(":scope > pre.body") || {}).textContent || "",
}))
"""


async def seeded_env(make_env, build) -> tuple:
    """A panel-enabled console whose web:main window `build(win)` fills."""
    env = await make_env(extra_plugins={"context_panel": ContextPanelPlugin()},
                         extra_config={"context_panel": {}})
    channel = env.registry.get_or_create("web", "main")
    win = ContextWindow(channel.id)
    channel.conversation = win
    build(win)
    return env, win


async def open_panel(browser, env, live: str | None = None, timezone_id: str | None = None):
    """Log in, open the panel, pick web:main in the channel list, and wait
    for the live event whose text is ``live``, if given (it arrives on the poll
    after the open)."""
    page = await (await browser.new_context(timezone_id=timezone_id)).new_page()
    await page.goto(f"{env.base}/?token={TOKEN}")
    await page.wait_for_url(f"{env.base}/")
    await page.goto(f"{env.base}/panel/context_panel/")
    try:
        await page.select_option("#panel select", value=CID, timeout=5000)
        await page.wait_for_selector("#panel details.entry", timeout=5000)
        if live is not None:
            await page.wait_for_function(
                "(t) => [...document.querySelectorAll('#panel pre.body')].some(p => p.textContent === t)",
                arg=live, timeout=5000)
    except PlaywrightError as exc:
        pytest.fail(f"panel never showed {CID}'s entries: {exc}")
    return page


def by_text(entries: list[dict], text: str) -> dict:
    found = [e for e in entries if e["text"] == text]
    assert len(found) == 1, (text, [e["text"] for e in entries])
    return found[0]


def fill_every_category(win: ContextWindow) -> None:
    """One row per category, in the order the assertions name them."""
    win.system_prompt = "sys-prompt-text"
    win.tool_view = ToolView(schemas=[{"type": "function", "function": {"name": "lookup"}}])
    win.append({"role": "system", "content": "context-text"}, MessageType.CONTEXT)
    win.append({"role": "user", "content": "summary-text"}, MessageType.SUMMARY)
    win.append({"role": "system", "content": "notification-text"})
    win.append({"role": "user", "content": "user-text"})
    win.append({"role": "assistant", "content": "answer-text",
                "reasoning_content": "thinking-text"})
    win.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "lookup", "arguments": '{"q": "x"}'}}]})
    win.append({"role": "tool", "tool_call_id": "c1", "content": "tool-output-text"})
    win.append({"role": "assistant", "content": "unsent-text"}, MessageType.UNSENT)


async def send_live(env, event) -> None:
    await env.pm.ahook.send(channel=env.registry.get(CID), event=event)


@pytest.mark.timeout(60)
async def test_hostile_content_renders_as_text(make_env):
    """R11: markup in any untrusted field shows literally and never runs."""
    tool_markup = "<img src=x onerror=\"window.__pwned = true\">"
    prompt_markup = "<script>window.__pwned = true</script>"
    live_markup = "<b onclick=\"window.__pwned = true\">live</b><style>body{display:none}</style>"

    def build(win):
        win.system_prompt = prompt_markup
        win.append({"role": "user", "content": HOSTILE})
        win.append({"role": "tool", "tool_call_id": "c1", "content": tool_markup})

    env, _ = await seeded_env(make_env, build)
    await send_live(env, OutboundProgress(text=live_markup))
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, live_markup)
            entries = await page.evaluate(ENTRIES_JS)
            found = await page.evaluate("""() => ({
              bad: document.querySelectorAll("#panel script, #panel img, #panel style").length,
              pwned: window.__pwned === true,
            })""")
        finally:
            await browser.close()

    texts = {e["text"] for e in entries}
    for markup in (HOSTILE, tool_markup, prompt_markup, live_markup):
        assert markup in texts
    assert found == {"bad": 0, "pwned": False}


@pytest.mark.timeout(60)
async def test_category_classes(make_env):
    """R15: each category gets its class; UNSENT is chat marked withheld."""
    env, _ = await seeded_env(make_env, fill_every_category)
    await send_live(env, OutboundProgress(text="live-text"))
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, "live-text")
            entries = await page.evaluate(ENTRIES_JS)
        finally:
            await browser.close()

    expected = {
        "sys-prompt-text": "cat-system-prompt",
        "context-text": "cat-inserted",
        "summary-text": "cat-inserted",
        "notification-text": "cat-inserted",
        "thinking-text": "cat-thinking",
        "answer-text": "cat-chat",
        'lookup({"q": "x"})': "cat-tool-call",
        "tool-output-text": "cat-tool-output",
        "user-text": "cat-chat",
        "unsent-text": "cat-chat",
        "live-text": "cat-chat",
    }
    for text, cls in expected.items():
        assert cls in by_text(entries, text)["cls"], text
    tools = [e for e in entries if "tools" in e["label"] and "cat-system-prompt" in e["cls"]]
    assert len(tools) == 1
    unsent = by_text(entries, "unsent-text")
    assert "withheld" in unsent["cls"]
    assert "withheld (not sent)" in unsent["label"]
    assert all("withheld" not in e["cls"] for e in entries if e["text"] != "unsent-text")


@pytest.mark.timeout(60)
async def test_only_chat_entries_start_open(make_env):
    """R1: an entry starts open iff it is chat; clicking a summary toggles it."""
    env, _ = await seeded_env(make_env, fill_every_category)
    await send_live(env, OutboundProgress(text="live-text"))
    await send_live(env, OutboundToolStatus(tool_name="live_tool", status="dispatched",
                                            args_summary="live-args"))
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, live="live-args")
            entries = await page.evaluate(ENTRIES_JS)
            live_open = await page.evaluate("document.querySelector('#live').open")
            summary = page.locator("#panel details.cat-thinking summary").first
            await summary.click()
            opened = await page.locator("#panel details.cat-thinking").first.evaluate("d => d.open")
        finally:
            await browser.close()

    for e in entries:
        assert e["open"] is ("cat-chat" in e["cls"]), (e["label"], e["cls"])
    assert opened is True
    assert live_open is True


@pytest.mark.timeout(60)
async def test_tool_name_in_call_and_output_summaries(make_env):
    """R2: a tool call's summary and its output's both name the tool."""
    def build(win):
        win.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        win.append({"role": "tool", "tool_call_id": "c1", "content": "out"})

    env, _ = await seeded_env(make_env, build)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            entries = await (await open_panel(browser, env)).evaluate(ENTRIES_JS)
        finally:
            await browser.close()

    for cls in ("cat-tool-call", "cat-tool-output"):
        assert all("read_file" in e["label"] for e in entries if cls in e["cls"])


@pytest.mark.timeout(60)
async def test_row_stamp_is_browser_local_time(make_env):
    """R3: a ts shows as local YYYY-MM-DD HH:MM; a row without ts shows no stamp."""
    env, win = await seeded_env(make_env, lambda w: [
        w.append({"role": "user", "content": "stamped"}),
        w.append({"role": "user", "content": "bare"})])
    win.messages[0]["_db_id"] = await log_row(env, CID, 1772618820)  # 10:07 UTC
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, timezone_id="America/New_York")
            entries = await page.evaluate(ENTRIES_JS)
        finally:
            await browser.close()

    label = by_text(entries, "stamped")["label"]
    assert "2026-03-04 05:07" in label and "10:07" not in label
    assert "NaN" not in by_text(entries, "bare")["label"]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", by_text(entries, "bare")["label"])


@pytest.mark.timeout(60)
async def test_tail_and_resync_in_page(make_env):
    """R7: an appended row shows up and keeps toggles; compaction reloads."""
    env, win = await seeded_env(make_env, fill_every_category)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            thinking = page.locator("#panel details.cat-thinking").first
            await thinking.locator("summary").first.click()
            assert await thinking.evaluate("d => d.open") is True

            win.append({"role": "user", "content": "appended-text"})
            await page.wait_for_function(
                "() => document.querySelector('#entries').textContent.includes('appended-text')",
                timeout=3000)
            entries = await page.evaluate(ENTRIES_JS)
            assert by_text(entries, "thinking-text")["open"] is True

            win.replace_with_summary({"role": "user", "content": "compacted-text"}, 1)
            await page.wait_for_function(
                "() => { const t = document.querySelector('#entries').textContent;"
                " return t.includes('compacted-text') && !t.includes('user-text'); }",
                timeout=5000)
            entries = await page.evaluate(ENTRIES_JS)
        finally:
            await browser.close()

    assert "cat-inserted" in by_text(entries, "compacted-text")["cls"]
    assert "thinking-text" not in {e["text"] for e in entries}
