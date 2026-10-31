"""Browser smoke test for the web UI's Markdown sanitising (issue #90).

Starts a real WebPlugin on 127.0.0.1, opens the page in headless Chrome or
Chromium, has the "model" send a hostile reply plus an ordinary Markdown
fixture, and asserts on the rendered DOM: no ``class``, ``id`` or
``data-*`` attribute inside a message, the older protections (no style, no
script, no ``on*``, safe links) still hold, and ordinary Markdown still
renders.

It needs Playwright and a browser, so the whole module is skipped when
``playwright`` isn't importable. CI doesn't install it, which means CI does
NOT guard the sanitize options in ``web_static/app.js``: after editing them,
run this by hand::

    uv run --with playwright pytest tests/test_web_render_smoke.py

It uses ``channel="chrome"`` (the system Google Chrome) and falls back to
Playwright's own Chromium.
"""

from __future__ import annotations

import pytest

# Skip the module cleanly where Playwright isn't installed (CI, plain dev sync).
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
            # Log in (the token trades for the session cookie), then load the app.
            await page.goto(f"{env.base}/?token={TOKEN}")
            await page.goto(f"{env.base}/")
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
