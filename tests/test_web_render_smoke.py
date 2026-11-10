"""Browser smoke tests for the web UI: Markdown sanitising (issue #90) and
restoring unacknowledged sends when the socket closes (issue #89).

Starts a real WebPlugin on 127.0.0.1 and opens the page in headless Chrome or
Chromium. For the sanitising check, the "model" sends a hostile reply plus an
ordinary Markdown fixture, and the test asserts on the rendered DOM: no
``class``, ``id`` or ``data-*`` attribute inside a message, the older
protections (no style, no script, no ``on*``, safe links) still hold, and
ordinary Markdown still renders. For the restore checks, a WebSocket route
drops the page's ``send`` frames and then closes the socket, and the test
asserts that each unconfirmed message appears in the banner labelled with its
conversation, in order, and that the composer draft is untouched.

It needs Playwright and a browser, so the whole module is skipped when
``playwright`` isn't importable. The ``web-smoke`` CI job
(``.github/workflows/tests.yml``) installs a pinned Playwright and Chromium and
fails if this module is skipped, so CI guards the sanitize options in
``web_static/app.js``. To run it locally::

    uv run --with playwright pytest tests/test_web_render_smoke.py

It uses ``channel="chrome"`` (the system Google Chrome) and falls back to
Playwright's own Chromium.
"""

from __future__ import annotations

import json

import pytest

# Skip the module cleanly where Playwright isn't installed (plain dev sync; the
# web-smoke CI job treats a skip as a failure).
pytest.importorskip("playwright.async_api")

from playwright.async_api import async_playwright  # noqa: E402

from corvidae.outbound import OutboundMessage  # noqa: E402

# Reuse the web plugin test harness: the env factory and the token fixture.
from tests.test_web_plugin import TOKEN, _no_env_token, make_env  # noqa: E402,F401

# Hostile reply: borrowed app classes, id shadowing, data attributes, plus
# the older XSS vectors (style attr/tag, script, on* handler, link).
HOSTILE = (
    '<div class="item user" id="send" data-id="x" data-foo="y">spoof</div>\n\n'
    '<p style="color:red" onclick="alert(1)">styled</p>\n\n'
    "<style>body{display:none}</style>\n\n"
    "<script>window.__pwned = true</script>\n\n"
    '<img src="x" onerror="window.__pwned = true">\n\n'
    "[a link](https://example.com/)\n"
)

# Ordinary Markdown: fenced code with a language, lists, table, link.
FIXTURE = (
    "```python\nprint('hi')\n```\n\n"
    "- one\n- two\n\n"
    "1. first\n2. second\n\n"
    "| h1 | h2 |\n|----|----|\n| c1 | c2 |\n\n"
    "[ok](https://example.com/ok)\n"
)

# In-page assertions: returns a dict of findings for the .md element at
# index `i` of the message list.
INSPECT_JS = """
(i) => {
  const md = document.querySelectorAll(".item.message .md")[i];
  const bad = [];
  for (const n of md.querySelectorAll("*")) {
    for (const a of n.attributes) {
      if (a.name === "class" || a.name === "id" || a.name.startsWith("data-")) {
        bad.push(n.tagName + "[" + a.name + "]");
      }
    }
  }
  const links = [...md.querySelectorAll("a[href]")];
  return {
    bad,
    styleAttr: md.querySelectorAll("[style]").length,
    styleTag: md.querySelectorAll("style").length,
    script: md.querySelectorAll("script").length,
    handlers: [...md.querySelectorAll("*")].flatMap(
      (n) => [...n.attributes].filter((a) => a.name.startsWith("on")).map((a) => a.name)),
    pwned: !!window.__pwned,
    sendIsButton: document.getElementById("send").tagName === "BUTTON",
    links: links.map((a) => [a.getAttribute("target"), a.getAttribute("rel")]),
    preCode: md.querySelectorAll("pre > code").length,
    ul: md.querySelectorAll("ul > li").length,
    ol: md.querySelectorAll("ol > li").length,
    th: md.querySelectorAll("table th").length,
    td: md.querySelectorAll("table td").length,
    mdClass: md.className,
    boxClass: md.parentElement.className,
  };
}
"""


async def _launch(pw):
    """Headless system Chrome, else Playwright's Chromium."""
    # --no-sandbox: the test container runs as root.
    try:
        return await pw.chromium.launch(channel="chrome", args=["--no-sandbox"])
    except Exception as exc:  # noqa: BLE001 - logged via skip reason below
        try:
            return await pw.chromium.launch(args=["--no-sandbox"])
        except Exception as exc2:
            pytest.skip(f"no usable browser (chrome: {exc}; chromium: {exc2})")


@pytest.mark.timeout(60)
async def test_rendered_model_output_is_sanitised(make_env):
    """R1-R3: hostile attributes stripped, protections and Markdown intact."""
    env = await make_env()
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page = await browser.new_page()
            # Log in (the token trades for the session cookie). The login
            # answer meta-refreshes to "/"; wait for that navigation instead
            # of starting a second goto, which races it and flakes.
            await page.goto(f"{env.base}/?token={TOKEN}")
            await page.wait_for_url(f"{env.base}/")
            # Wait until the server has finished the page's subscribe/history.
            for _ in range(200):
                if any(c.conversation == "web:main" and c.buffer is None
                       for c in env.plugin._clients):
                    break
                await page.wait_for_timeout(50)
            else:
                pytest.fail("page never subscribed to web:main")
            # Deliver the hostile reply, then the ordinary Markdown.
            channel = env.channel("main")
            for text in (HOSTILE, FIXTURE):
                await env.pm.ahook.send(channel=channel, event=OutboundMessage(text=text))
            await page.wait_for_function(
                "document.querySelectorAll('.item.message .md').length >= 2")
            hostile = await page.evaluate(INSPECT_JS, 0)
            ordinary = await page.evaluate(INSPECT_JS, 1)
        finally:
            await browser.close()

    # R1: nothing inside a message carries class, id or data-*.
    assert hostile["bad"] == [], f"attributes survived: {hostile['bad']}"
    assert ordinary["bad"] == [], f"attributes survived: {ordinary['bad']}"
    # The wrapper keeps its own classes (set outside the sanitizer).
    assert hostile["mdClass"] == "md" and hostile["boxClass"] == "item message"
    # An injected id="send" must not shadow the real send button.
    assert hostile["sendIsButton"]

    # R2: older protections hold.
    assert hostile["styleAttr"] == 0 and hostile["styleTag"] == 0
    assert hostile["script"] == 0 and hostile["handlers"] == []
    assert not hostile["pwned"]
    assert hostile["links"] == [["_blank", "noopener noreferrer"]]

    # R3: ordinary Markdown still renders.
    assert ordinary["preCode"] == 1
    assert ordinary["ul"] == 2 and ordinary["ol"] == 2
    assert ordinary["th"] == 2 and ordinary["td"] == 2
    assert ordinary["links"] == [["_blank", "noopener noreferrer"]]


async def _open_page(browser, env, *, drop_sends=False):
    """Log in, open the page behind a forwarding WebSocket route, and wait
    for its subscribe to finish.

    Returns (page, state). While state["drop"] is true, client `send`
    frames are swallowed (not forwarded) and appended to state["dropped"];
    await state["routes"][-1].close() closes the page's current socket.
    """
    state = {"drop": drop_sends, "dropped": [], "routes": []}

    def handle(route):
        # Forward both ways, except for swallowed sends.
        server = route.connect_to_server()
        state["routes"].append(route)

        def from_page(message):
            if state["drop"] and json.loads(message).get("type") == "send":
                state["dropped"].append(message)
            else:
                server.send(message)

        route.on_message(from_page)
        server.on_message(lambda message: route.send(message))

    page = await browser.new_page()
    await page.route_web_socket("**/ws", handle)
    await page.goto(f"{env.base}/?token={TOKEN}")
    await page.wait_for_url(f"{env.base}/")
    await _wait_subscribed(page, env, "main")
    return page, state


async def _wait_subscribed(page, env, name):
    """Wait until the server has finished the page's subscribe/history."""
    for _ in range(200):
        if any(c.conversation == f"web:{name}" and c.buffer is None
               for c in env.plugin._clients):
            return
        await page.wait_for_timeout(50)
    pytest.fail(f"page never subscribed to web:{name}")


async def _send_dropped(page, state, text):
    """Send `text` from the composer and wait until the route swallowed it."""
    n = len(state["dropped"])
    await page.fill("#input", text)
    await page.click("#send")
    for _ in range(100):
        if len(state["dropped"]) == n + 1:
            return
        await page.wait_for_timeout(50)
    pytest.fail(f"send of {text!r} never reached the route")


async def _unconfirmed(page):
    """The banner's entries as (label, text) pairs."""
    await page.wait_for_selector("#unconfirmed:not(.hidden)", timeout=3000)
    return await page.eval_on_selector_all(
        "#unconfirmed-list li",
        "els => els.map(e => [e.querySelector('.conv').textContent,"
        " e.querySelector('.text').textContent])")


@pytest.mark.timeout(60)
async def test_unanswered_send_restored_on_close(make_env):
    """S2: a send with no answer before the socket closes comes back."""
    env = await make_env()
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page, state = await _open_page(browser, env, drop_sends=True)
            await _send_dropped(page, state, "lost one")
            await state["routes"][-1].close()
            entries = await _unconfirmed(page)
        finally:
            await browser.close()
    assert entries == [["#main", "lost one"]]


@pytest.mark.timeout(60)
async def test_unanswered_send_restored_after_switching(make_env):
    """S2: the restored send keeps the conversation it was sent to."""
    env = await make_env()
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page, state = await _open_page(browser, env, drop_sends=True)
            await _send_dropped(page, state, "lost in main")
            # Switch conversation before the socket closes.
            await page.evaluate("location.hash = 'other'")
            await _wait_subscribed(page, env, "other")
            await state["routes"][-1].close()
            entries = await _unconfirmed(page)
        finally:
            await browser.close()
    assert entries == [["#main", "lost in main"]]


@pytest.mark.timeout(60)
async def test_dropped_sends_and_draft_survive_close(make_env):
    """S2: two lost sends are restored in order; the typed draft is kept."""
    env = await make_env()
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            page, state = await _open_page(browser, env, drop_sends=True)
            await _send_dropped(page, state, "first lost")
            await _send_dropped(page, state, "second lost")
            await page.fill("#input", "my draft")
            await state["routes"][-1].close()
            entries = await _unconfirmed(page)
            draft = await page.input_value("#input")
        finally:
            await browser.close()
    assert entries == [["#main", "first lost"], ["#main", "second lost"]]
    assert draft == "my draft"
