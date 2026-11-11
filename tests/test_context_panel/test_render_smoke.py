"""Browser smoke tests for the context tabs in the chat page (issue #687).

Seeds a channel's window on the server, logs in, clicks the channel's tab in
the chat sidebar in headless Chrome or Chromium and asserts on the rendered
DOM: untrusted text stays text, each category gets its class, carets open and
close entries, the tail and resync paths reach the page, and the tabs, hash
routing, scroll rules, colors and chat hand-off of the context section.

Like tests/test_web_render_smoke.py it is skipped without Playwright, and the
``web-smoke`` CI job fails if it is skipped. To run it locally::

    uv run --with playwright pytest tests/test_context_panel/test_render_smoke.py
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import quote

import pytest

# Skip the module cleanly where Playwright isn't installed.
pytest.importorskip("playwright.async_api")

from context_panel import ContextPanelPlugin  # noqa: E402
from playwright.async_api import Error as PlaywrightError  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402

from corvidae.context import ContextWindow, MessageType  # noqa: E402
from corvidae.outbound import (  # noqa: E402
    OutboundMessage,
    OutboundProgress,
    OutboundToolStatus,
)
from corvidae.tool_policy import ToolView  # noqa: E402
from tests.test_context_panel.test_context_panel import log_row  # noqa: E402
from tests.test_web_plugin import TOKEN, _no_env_token, _seed, make_env  # noqa: E402,F401
from tests.test_web_render_smoke import HOSTILE, _launch, _wait_subscribed  # noqa: E402

CID = "web:main"

# Every entry in the panel: its label, classes, open state and body text.
ENTRIES_JS = """
() => [...document.querySelectorAll("#context-view details.entry")].map(d => ({
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


def enc(channel_id: str) -> str:
    """encodeURIComponent, as the tab hrefs use."""
    return quote(channel_id, safe="!*'()")


def tab(channel_id: str = CID) -> str:
    return f'a[href="#context_panel/{enc(channel_id)}"]'


async def login(browser, env, **context_args):
    """A page logged in and sitting on the chat page."""
    page = await (await browser.new_context(**context_args)).new_page()
    await page.goto(f"{env.base}/?token={TOKEN}")
    await page.wait_for_url(f"{env.base}/")
    return page


async def wait_tab(page, channel_id: str = CID) -> None:
    try:
        await page.wait_for_selector(tab(channel_id), timeout=5000)
    except PlaywrightError as exc:
        pytest.fail(f"the Context section never showed a tab for {channel_id}: {exc}")


async def open_panel(browser, env, live: str | None = None, cid: str = CID, **context_args):
    """Log in, click the channel's tab in the chat sidebar, and wait for the
    view's entries and for the live event whose text is ``live``, if given (it
    arrives on the poll after the open)."""
    page = await login(browser, env, **context_args)
    await wait_tab(page, cid)
    await page.click(tab(cid))
    try:
        await page.wait_for_selector("#context-view details.entry", timeout=5000)
        if live is not None:
            await page.wait_for_function(
                "(t) => [...document.querySelectorAll('#context-view pre.body')].some(p => p.textContent === t)",
                arg=live, timeout=5000)
    except PlaywrightError as exc:
        pytest.fail(f"the context view never showed {cid}'s entries: {exc}")
    return page


async def wait_view_text(page, text: str) -> None:
    await page.wait_for_function(
        "(t) => { const v = document.getElementById('context-view');"
        " return !!v && !v.classList.contains('hidden') && v.textContent.includes(t); }",
        arg=text, timeout=5000)


async def wait_items_text(page, text: str) -> None:
    await page.wait_for_function(
        "(t) => { const i = document.getElementById('items');"
        " return i.offsetParent !== null && i.textContent.includes(t); }",
        arg=text, timeout=5000)


async def seed_chat(env, channel_id: str, lines: list[str]) -> None:
    """Persist alternating user and assistant rows: the chat's history."""
    await _seed(env.persistence.db, channel_id, [
        ({"role": "user" if i % 2 == 0 else "assistant", "content": line}, "message", float(i + 1))
        for i, line in enumerate(lines)])


def tall(win: ContextWindow, rows: int = 60) -> None:
    """Enough user rows to overflow the viewport."""
    for i in range(rows):
        win.append({"role": "user", "content": f"row-{i}\nsecond line\nthird line"})


SCROLL_JS = """
() => {
  const v = document.getElementById("context-view");
  return {top: v.scrollTop, client: v.clientHeight, height: v.scrollHeight};
}
"""

STYLES_JS = """
(texts) => Object.fromEntries(texts.map((t) => {
  const d = [...document.querySelectorAll("#context-view details.entry")]
    .find((e) => ((e.querySelector(":scope > pre.body") || {}).textContent) === t);
  const cs = getComputedStyle(d);
  return [t, {border: cs.borderLeftColor, bg: cs.backgroundColor,
              style: cs.borderLeftStyle, opacity: Number(cs.opacity)}];
}))
"""


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
    """Markup in any untrusted field shows literally and never runs."""
    tool_markup = "<img src=x onerror=\"window.__pwned = true\">"
    prompt_markup = "<script>window.__pwned = true</script>"
    live_markup = "<b onclick=\"window.__pwned = true\">live</b><style>body{display:none}</style>"

    def build(win):
        win.system_prompt = prompt_markup
        win.append({"role": "user", "content": HOSTILE})
        win.append({"role": "tool", "tool_call_id": "c1", "content": tool_markup})

    env, _ = await seeded_env(make_env, build)
    name_markup = "<img src=x onerror=\"window.__pwned = true\">"
    hostile_named = env.registry.get_or_create("web", "hostile")
    hostile_named.name = name_markup
    hostile_named.conversation = ContextWindow(hostile_named.id)
    await send_live(env, OutboundProgress(text=live_markup))
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, live_markup)
            entries = await page.evaluate(ENTRIES_JS)
            tabs = await page.eval_on_selector_all(
                '.panel-section[data-panel="context_panel"] a', "as => as.map(a => a.textContent)")
            found = await page.evaluate("""() => ({
              bad: document.querySelectorAll(
                "#context-view script, #context-view img, #context-view style, "
                + ".panel-section script, .panel-section img, .panel-section style").length,
              pwned: window.__pwned === true,
            })""")
        finally:
            await browser.close()

    texts = {e["text"] for e in entries}
    for markup in (HOSTILE, tool_markup, prompt_markup, live_markup):
        assert markup in texts
    assert any(name_markup in t for t in tabs)
    assert found == {"bad": 0, "pwned": False}


@pytest.mark.timeout(60)
async def test_category_classes(make_env):
    """Each category gets its class; UNSENT is assistant marked withheld."""
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
        "answer-text": "cat-assistant",
        'lookup({"q": "x"})': "cat-tool-call",
        "tool-output-text": "cat-tool-output",
        "user-text": "cat-user",
        "unsent-text": "cat-assistant",
        "live-text": "cat-assistant",
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
async def test_only_user_and_assistant_entries_start_open(make_env):
    """An entry starts open iff it is a user or assistant one; clicking a summary toggles it."""
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
            summary = page.locator("#context-view details.cat-thinking summary").first
            await summary.click()
            opened = await page.locator("#context-view details.cat-thinking").first.evaluate("d => d.open")
        finally:
            await browser.close()

    for e in entries:
        assert e["open"] is bool({"cat-user", "cat-assistant"} & set(e["cls"])), (e["label"], e["cls"])
    assert opened is True
    assert live_open is True


@pytest.mark.timeout(60)
async def test_tool_name_in_call_and_output_summaries(make_env):
    """A tool call's summary and its output's both name the tool."""
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
    """A ts shows as local YYYY-MM-DD HH:MM; a row without ts shows no stamp."""
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
    """An appended row shows up and keeps toggles; compaction reloads."""
    env, win = await seeded_env(make_env, fill_every_category)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            thinking = page.locator("#context-view details.cat-thinking").first
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


@pytest.mark.timeout(60)
async def test_tabs_one_per_channel_labelled_name_before_id(make_env):
    """A1.2: one tab per listed channel; a named one leads with the name, an unnamed one shows its id."""
    env, _ = await seeded_env(make_env, lambda w: None)
    home = env.registry.get_or_create("web", "home")
    home.name = "#home"
    home.conversation = ContextWindow(home.id)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await wait_tab(page, home.id)
            channels = (await page.evaluate(
                "fetch('/panel/context_panel/channels').then(r => r.json())"))["channels"]
            tabs = await page.eval_on_selector_all(
                'a[href^="#context_panel/"]',
                "as => as.map(a => ({href: a.getAttribute('href'), text: a.textContent}))")
        finally:
            await browser.close()

    assert [t["href"] for t in tabs] == [f"#context_panel/{enc(c['id'])}" for c in channels]
    assert {c["id"] for c in channels} == {CID, home.id}
    for t, c in zip(tabs, channels):
        assert f"{c['messages']} msgs, {c['tokens']} / {c['budget']} tokens" in t["text"]
    named = next(t for t, c in zip(tabs, channels) if c["id"] == home.id)
    unnamed = next(t for t, c in zip(tabs, channels) if c["id"] == CID)
    assert "#home (web:home)" in named["text"]
    assert CID in unnamed["text"] and f"({CID})" not in unnamed["text"]


@pytest.mark.timeout(60)
async def test_conversation_list_is_above_context_section(make_env):
    """A1.3: at desktop width the conversation list's box is above the Context section's."""
    env, _ = await seeded_env(make_env, lambda w: None)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await wait_tab(page)
            conversations = await page.locator("#conversations").bounding_box()
            section = await page.locator('.panel-section[data-panel="context_panel"]').bounding_box()
        finally:
            await browser.close()

    assert conversations["y"] + conversations["height"] <= section["y"] + 1


@pytest.mark.timeout(60)
async def test_tab_click_sets_fragment_and_keeps_path(make_env):
    """A2.1: a tab click sets the fragment to #context_panel/<encoded id>; the path stays /."""
    env, _ = await seeded_env(make_env, lambda w: None)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            url = page.url
        finally:
            await browser.close()

    assert url == f"{env.base}/#context_panel/{enc(CID)}"


@pytest.mark.timeout(60)
async def test_tab_hides_chat_chrome_and_shows_window(make_env):
    """A2.2: a selected tab hides the chat chrome, shows the window, and leaves no visible input."""
    env, _ = await seeded_env(make_env, lambda w: w.append({"role": "user", "content": "window-row"}))
    chrome = ("composer", "banner", "unconfirmed", "items", "working")
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await _wait_subscribed(page, env, "main")
            await wait_tab(page)
            env.plugin._set_working(env.channel("main"), True)
            await page.wait_for_selector("#working", state="visible", timeout=5000)
            await page.evaluate(
                "() => { document.getElementById('banner').textContent = 'notice';"
                " for (const id of ['banner', 'unconfirmed'])"
                " document.getElementById(id).classList.remove('hidden'); }")
            before = [await page.locator(f"#{i}").is_visible() for i in chrome]
            await page.click(tab())
            await wait_view_text(page, "window-row")
            after = [await page.locator(f"#{i}").is_visible() for i in chrome]
            view_visible = await page.locator("#context-view").is_visible()
            inputs = await page.locator("#main input:visible, #main textarea:visible").count()
            composers = await page.locator("#composer").count()
        finally:
            await browser.close()

    assert before == [True] * len(chrome)
    assert after == [False] * len(chrome)
    assert view_visible
    assert inputs == 0
    assert composers == 1


@pytest.mark.timeout(60)
async def test_conversation_click_returns_to_chat_history(make_env):
    """A2.3: a conversation click sets the fragment to its name and shows that conversation's history."""
    env, _ = await seeded_env(make_env, lambda w: None)
    await seed_chat(env, CID, ["hello-chat", "reply-chat"])
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            await page.locator("#conversations li", has_text="main").click()
            await wait_items_text(page, "reply-chat")
            hash_ = await page.evaluate("location.hash")
            composer = await page.locator("#composer").is_visible()
            view = await page.locator("#context-view").is_visible()
            items = await page.locator("#items").text_content()
        finally:
            await browser.close()

    assert hash_ == "#main"
    assert composer and not view
    assert "hello-chat" in items and "reply-chat" in items


def two_windows(env) -> None:
    """web:main and web:home, each with a window row and a chat history of its own."""
    home = env.registry.get_or_create("web", "home")
    home.conversation = ContextWindow(home.id)
    home.conversation.append({"role": "user", "content": "home-ctx"})


@pytest.mark.timeout(60)
async def test_reload_on_tab_reopens_same_window(make_env):
    """A2.4: reloading on a tab fragment reopens that channel's window with its tab current."""
    env, _ = await seeded_env(make_env, lambda w: w.append({"role": "user", "content": "main-ctx"}))
    two_windows(env)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, cid="web:home")
            await wait_view_text(page, "home-ctx")
            await page.reload()
            await wait_view_text(page, "home-ctx")
            text = await page.locator("#context-view").text_content()
            current = await page.eval_on_selector_all(
                'a[href^="#context_panel/"].current', "as => as.map(a => a.getAttribute('href'))")
        finally:
            await browser.close()

    assert "main-ctx" not in text
    assert current == [f"#context_panel/{enc('web:home')}"]


@pytest.mark.timeout(60)
async def test_back_and_forward_restore_chat_and_tab(make_env):
    """A2.4: back from a tab restores the previous conversation, forward restores the tab."""
    env, _ = await seeded_env(make_env, lambda w: w.append({"role": "user", "content": "main-ctx"}))
    two_windows(env)
    await seed_chat(env, "web:home", ["home-chat"])
    tab_hash = f"#context_panel/{enc(CID)}"
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await page.locator("#conversations li", has_text="home").click()
            await wait_items_text(page, "home-chat")
            await wait_tab(page)
            await page.click(tab())
            await wait_view_text(page, "main-ctx")

            await page.go_back()
            await page.wait_for_function("() => location.hash === '#home'", timeout=5000)
            await wait_items_text(page, "home-chat")
            back = {"view": await page.locator("#context-view").is_visible(),
                    "composer": await page.locator("#composer").is_visible()}

            await page.go_forward()
            await page.wait_for_function("(h) => location.hash === h", arg=tab_hash, timeout=5000)
            await wait_view_text(page, "main-ctx")
            forward = {"items": await page.locator("#items").is_visible(),
                       "composer": await page.locator("#composer").is_visible()}
        finally:
            await browser.close()

    assert back == {"view": False, "composer": True}
    assert forward == {"items": False, "composer": False}


@pytest.mark.timeout(60)
async def test_reload_on_tab_then_current_conversation_is_scrolled_to_bottom(make_env):
    """A2.4: after a reload on a tab, clicking the current conversation shows its history at the bottom."""
    env, _ = await seeded_env(make_env, lambda w: None)
    lines = [f"chat-line-{i}" for i in range(80)]
    await seed_chat(env, CID, lines)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            await page.reload()
            await wait_view_text(page, "system prompt")
            await page.locator("#conversations li", has_text="main").click()
            await wait_items_text(page, lines[-1])
            metrics = await page.evaluate(
                "() => { const i = document.getElementById('items');"
                " return {top: i.scrollTop, client: i.clientHeight, height: i.scrollHeight,"
                " text: i.textContent}; }")
        finally:
            await browser.close()

    assert metrics["height"] > metrics["client"]
    assert "chat-line-0" in metrics["text"] and lines[-1] in metrics["text"]
    assert abs(metrics["top"] + metrics["client"] - metrics["height"]) <= 1


@pytest.mark.timeout(60)
async def test_phone_tab_selection_closes_sidebar(make_env):
    """A2.5: on a phone-width page, selecting a tab closes the sidebar."""
    env, _ = await seeded_env(make_env, lambda w: None)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env, viewport={"width": 390, "height": 800})
            await page.click("#sidebar-toggle")
            await wait_tab(page)
            opened = await page.evaluate("document.getElementById('sidebar').classList.contains('open')")
            await page.click(tab())
            await page.wait_for_function(
                "() => !document.getElementById('sidebar').classList.contains('open')", timeout=5000)
            await wait_view_text(page, "system prompt")
        finally:
            await browser.close()

    assert opened


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.timeout(60)
async def test_user_and_assistant_entries_have_distinct_colors(make_env, scheme):
    """A5.1: user and assistant entries differ in border and background, and from every other category."""
    env, _ = await seeded_env(make_env, fill_every_category)
    others = ["sys-prompt-text", "context-text", "thinking-text", 'lookup({"q": "x"})',
              "tool-output-text"]
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, color_scheme=scheme)
            styles = await page.evaluate(STYLES_JS, ["user-text", "answer-text", *others])
        finally:
            await browser.close()

    user, assistant = styles["user-text"], styles["answer-text"]
    for prop in ("border", "bg"):
        assert user[prop] != assistant[prop], prop
        for text in others:
            assert styles[text][prop] not in (user[prop], assistant[prop]), (text, prop)


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.timeout(60)
async def test_unsent_assistant_entry_is_dashed_dimmed_in_assistant_color(make_env, scheme):
    """A5.2: an UNSENT assistant entry keeps the dashed, dimmed style and the assistant color."""
    env, _ = await seeded_env(make_env, fill_every_category)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, color_scheme=scheme)
            styles = await page.evaluate(STYLES_JS, ["unsent-text", "answer-text", "user-text"])
        finally:
            await browser.close()

    unsent, assistant = styles["unsent-text"], styles["answer-text"]
    assert unsent["style"] == "dashed"
    assert unsent["opacity"] < 1
    assert (unsent["border"], unsent["bg"]) == (assistant["border"], assistant["bg"])
    assert (unsent["border"], unsent["bg"]) != (styles["user-text"]["border"], styles["user-text"]["bg"])


@pytest.mark.timeout(60)
async def test_live_progress_entry_takes_assistant_color(make_env):
    """A5.3: a live progress entry is colored as an assistant entry, not as a user one."""
    env, _ = await seeded_env(make_env, fill_every_category)
    await send_live(env, OutboundProgress(text="live-text"))
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, "live-text")
            styles = await page.evaluate(STYLES_JS, ["live-text", "answer-text", "user-text"])
        finally:
            await browser.close()

    live, assistant = styles["live-text"], styles["answer-text"]
    assert (live["border"], live["bg"]) == (assistant["border"], assistant["bg"])
    assert (live["border"], live["bg"]) != (styles["user-text"]["border"], styles["user-text"]["bg"])


@pytest.mark.timeout(60)
async def test_first_render_scrolls_view_to_bottom(make_env):
    """A6.1: after a tab's first render the view's scroller sits at its bottom."""
    env, _ = await seeded_env(make_env, tall)
    await send_live(env, OutboundProgress(text="live-text"))
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env, "live-text")
            metrics = await page.evaluate(SCROLL_JS)
        finally:
            await browser.close()

    assert metrics["height"] > metrics["client"]
    assert abs(metrics["top"] + metrics["client"] - metrics["height"]) <= 1


@pytest.mark.timeout(60)
async def test_switching_tabs_mid_load_still_scrolls_to_bottom(make_env):
    """A6.1: picking a second tab while the first is still loading ends at the second's bottom."""
    env, _ = await seeded_env(make_env, tall)
    other = env.registry.get_or_create("web", "other")
    other.conversation = ContextWindow(other.id)
    other.conversation.append({"role": "user", "content": "other-text"})

    async def slow_window(route):
        await asyncio.sleep(1.5 if "channel=web%3Amain" in route.request.url else 0.3)
        await route.continue_()

    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await wait_tab(page)
            await wait_tab(page, "web:other")
            await page.route("**/panel/context_panel/window*", slow_window)
            await page.click(tab("web:other"))
            await page.click(tab())
            await wait_view_text(page, "row-59")
            await page.wait_for_timeout(3500)
            metrics = await page.evaluate(SCROLL_JS)
        finally:
            await browser.close()

    assert metrics["height"] > metrics["client"]
    assert abs(metrics["top"] + metrics["client"] - metrics["height"]) <= 1


@pytest.mark.timeout(60)
async def test_resync_during_first_open_still_scrolls_to_bottom(make_env):
    """A6.1: a tab's first poll answering resync still ends at the bottom."""
    env, _ = await seeded_env(make_env, tall)
    answered = []

    async def resync_once(route):
        if answered:
            await route.continue_()
            return
        answered.append(True)
        await route.fulfill(json={"loaded": True, "resync": True})

    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await wait_tab(page)
            await page.route("**/panel/context_panel/poll*", resync_once)
            await page.click(tab())
            await wait_view_text(page, "row-59")
            await page.wait_for_timeout(1500)
            metrics = await page.evaluate(SCROLL_JS)
        finally:
            await browser.close()

    assert answered
    assert metrics["height"] > metrics["client"]
    assert abs(metrics["top"] + metrics["client"] - metrics["height"]) <= 1


@pytest.mark.timeout(60)
async def test_resync_answered_before_first_poll_still_scrolls_to_bottom(make_env):
    """A6.1: a tick's resync that overtakes the open's first poll still ends at the bottom."""
    env, _ = await seeded_env(make_env, tall)
    await send_live(env, OutboundProgress(text="live-text"))
    windows, polls = [], []

    async def window_route(route):
        windows.append(True)
        if len(windows) == 2:
            await asyncio.sleep(1.5)
        await route.continue_()

    async def poll_route(route):
        polls.append(True)
        if len(polls) == 1:
            await asyncio.sleep(1.5)
            await route.continue_()
        elif len(polls) == 2:
            await route.fulfill(json={"loaded": True, "resync": True})
        else:
            await route.continue_()

    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await wait_tab(page)
            await page.route("**/panel/context_panel/window*", window_route)
            await page.route("**/panel/context_panel/poll*", poll_route)
            await page.click(tab())
            await wait_view_text(page, "live-text")
            await page.wait_for_timeout(1500)
            metrics = await page.evaluate(SCROLL_JS)
        finally:
            await browser.close()

    assert len(windows) >= 2
    assert metrics["height"] > metrics["client"]
    assert abs(metrics["top"] + metrics["client"] - metrics["height"]) <= 1


@pytest.mark.timeout(60)
async def test_poll_appending_rows_keeps_scroll_position(make_env):
    """A6.2: a poll that appends rows leaves a mid-view scroll position where it was."""
    env, win = await seeded_env(make_env, tall)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            await page.wait_for_function(
                "() => { const v = document.getElementById('context-view');"
                " return v.scrollHeight > v.clientHeight"
                " && Math.abs(v.scrollTop + v.clientHeight - v.scrollHeight) <= 1; }",
                timeout=5000)
            before = await page.evaluate(
                "() => { const v = document.getElementById('context-view');"
                " v.scrollTop = Math.floor((v.scrollHeight - v.clientHeight) / 2); return v.scrollTop; }")
            for i in range(5):
                win.append({"role": "user", "content": f"late-row-{i}"})
            await page.wait_for_function(
                "() => document.querySelector('#entries').textContent.includes('late-row-4')",
                timeout=3000)
            after = await page.evaluate(SCROLL_JS)
        finally:
            await browser.close()

    assert 0 < before < after["height"] - after["client"]
    assert after["top"] == before


@pytest.mark.timeout(60)
async def test_new_channel_gets_a_tab_only_after_refresh(make_env):
    """A7.1: a window loaded after the page opened has no tab until refresh; the active tab stays active."""
    env, _ = await seeded_env(make_env, lambda w: None)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await open_panel(browser, env)
            late = env.registry.get_or_create("web", "late")
            late.conversation = ContextWindow(late.id)
            await page.wait_for_timeout(1500)
            before = await page.locator(tab(late.id)).count()
            await page.locator('.panel-section[data-panel="context_panel"] button',
                               has_text="refresh").click()
            await wait_tab(page, late.id)
            current = await page.eval_on_selector_all(
                'a[href^="#context_panel/"].current', "as => as.map(a => a.getAttribute('href'))")
        finally:
            await browser.close()

    assert before == 0
    assert current == [f"#context_panel/{enc(CID)}"]


@pytest.mark.timeout(60)
async def test_tab_is_present_at_load_without_a_click(make_env):
    """A7.2: a page opened after the window loaded shows its tab with no click."""
    env, _ = await seeded_env(make_env, lambda w: None)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await wait_tab(page)
            url = page.url
        finally:
            await browser.close()

    assert url == f"{env.base}/"


@pytest.mark.timeout(60)
async def test_chat_frame_and_working_while_tab_shown_are_in_chat_after_return(make_env):
    """N3: a chat frame and the working state from while a tab was shown are in the chat after returning."""
    env, _ = await seeded_env(make_env, lambda w: None)
    channel = env.channel("main")
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await login(browser, env)
            await _wait_subscribed(page, env, "main")
            await wait_tab(page)
            await page.click(tab())
            await wait_view_text(page, "system prompt")

            await seed_chat(env, CID, ["away-reply"])
            await env.pm.ahook.send(channel=channel, event=OutboundMessage(text="away-reply"))
            env.plugin._set_working(channel, True)
            await page.wait_for_function(
                "() => document.getElementById('items').textContent.includes('away-reply')", timeout=3000)

            await page.locator("#conversations li", has_text="main").click()
            await wait_items_text(page, "away-reply")
            working = await page.locator("#working").is_visible()
        finally:
            await browser.close()

    assert working
