# Corvidae Plugin Guide

Corvidae plugins extend the agent daemon using [apluggy](https://pypi.org/project/apluggy/) (async pluggy). A plugin subclasses `CorvidaePlugin` and decorates methods with `@hookimpl` for each hook it handles.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.outbound import OutboundMessage

class GreetPlugin(CorvidaePlugin):
    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        # An impl may take any subset of the spec's args; trace_id is omitted here.
        if text.strip().lower() == "hello":
            await self.pm.ahook.send(
                channel=channel, event=OutboundMessage(text=f"Hello, {sender}!")
            )
```

`CorvidaePlugin.on_init` stores `pm` and `config` as instance attributes. Subclasses that need extra initialization override `on_init` and call `super().on_init(pm, config)` first.

## Plugin anatomy

A plugin subclasses `CorvidaePlugin` from `corvidae.hooks`. It only needs to implement the hooks it cares about — all hooks are optional. The `@hookimpl` decorator marks each implementation. Plugins use no-argument constructors; `pm` and `config` are received via the `on_init` hook.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl

class MyPlugin(CorvidaePlugin):
    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.setting = config.get("my_plugin", {}).get("setting", "default")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # open runtime resources (connections, file handles, asyncio tasks)
        pass

    @hookimpl
    async def on_stop(self) -> None:
        # clean up resources
        pass
```

## Registering a plugin

Plugins are loaded via setuptools entry points. Declare them in `pyproject.toml`:

```toml
[project.entry-points.corvidae]
my_plugin = "my_package:MyPlugin"
```

`Runtime.start()` calls `pm.load_setuptools_entrypoints("corvidae")`, which instantiates each entry point class with no arguments and registers it. Then `install_plugin_hookspecs(pm)` installs [plugin-declared hooks](#declaring-hooks-in-a-plugin), the arg-binding guard checks every hookimpl, `validate_dependencies(pm)` runs, hookimpls with no spec are reported, and `pm.ahook.on_init(pm=pm, config=config)` broadcasts to all registered plugins.

All plugin classes must be instantiable with no arguments. `pm` and `config` are delivered via `on_init`.

**ChannelRegistry** is the exception: it is not an entry-point plugin. It is a plain class with no `@hookimpl` decorators, constructed explicitly in `Runtime.start()` and populated from config before the entry point plugins are loaded.

## Writing replacement plugins

To replace a built-in hook handler, implement the hook at default priority. Built-in handlers use `trylast=True` and serve as fallbacks.

```python
class FancyCompactionPlugin(CorvidaePlugin):
    @hookimpl
    async def resolve_compaction(self, channel, conversation, max_tokens):
        # Custom strategy
        await self.my_compaction(conversation, max_tokens)
        return True  # non-None stops the chain; built-in never runs
```

To observe without replacing, use `tryfirst=True` and return None:

```python
class CompactionObserver(CorvidaePlugin):
    @hookimpl(tryfirst=True)
    async def resolve_compaction(self, channel, conversation, max_tokens):
        self.record_metrics(conversation)
        return None  # chain continues to the actual compaction handler
```

This pattern applies to any `firstresult=True` hook. For broadcast hooks (where all handlers run regardless), `tryfirst` and `trylast` affect only execution order, not whether subsequent handlers run.

## Registering CLI subcommands

Plugins can add subcommands to the `corvidae` CLI by registering entries in the `corvidae.commands` entry point group. Each entry point maps a subcommand name to a `click.Command` (or `click.Group` for nested subcommands).

```toml
[project.entry-points."corvidae.commands"]
scaffold = "corvidae_scaffold:scaffold_command"
```

The entry point value must be a `click.Command` or `click.Group` object. `corvidae.main` discovers all registered entries at import time and adds them to the top-level `corvidae` group. If an entry point fails to load, a warning is logged and the subcommand is skipped — other subcommands are unaffected.

Subcommands are standard click commands managed by the `corvidae` click Group. Subcommands that need the agent runtime construct a `Runtime` instance and call `asyncio.run(runtime.run())`:

```python
import asyncio
import click
from corvidae.runtime import Runtime

@click.command("myplugin")
@click.option("--config", default="agent.yaml", help="Path to config file")
def myplugin_command(config):
    """Start corvidae with myplugin active."""
    runtime = Runtime(
        config_path=config,
        overrides={"channels": {"cli:local": {}}},
    )
    asyncio.run(runtime.run())
```

Utility subcommands that do not need the agent runtime omit the `Runtime` import entirely. They can perform their work synchronously without booting the plugin system.

Entry point modules in `corvidae.commands` are imported at dispatcher load time, before `Runtime.start()` configures logging or creates an event loop. These modules must not have module-level side effects that depend on logging configuration or a running event loop. Standard library imports (`asyncio`, `click`) and `corvidae.runtime` are safe at module level. Avoid importing modules with side effects (network connections, file I/O, logging configuration) at the top level — use lazy imports inside the command function for those.

The existing `corvidae` entry point group (for plugins), the `corvidae.commands` group (for subcommands) and the `corvidae.redact` group (for redact scrubbers, below) are distinct setuptools groups. Registering in one does not affect the others.

The built-in subcommand `scaffold` is registered by corvidae itself under `corvidae.commands`. Do not register an entry point named `scaffold` in the `corvidae.commands` group — a collision will silently overwrite the built-in command, causing unpredictable behavior.

Built-in `corvidae acp` (from `corvidae.channels.acp:acp_command`) starts an ACP v1 agent on stdio. It boots `Runtime` with `overrides={"_acp_mode": True}` so `AcpPlugin` owns the connection; without that flag the plugin stays inert. Install the optional extra first: add `--extra acp` to your `uv sync` line.

## Registering a redact scrubber

`corvidae redact` (see "Operator redaction" in [design.md](design.md)) tombstones `message_log` rows and cascades to memory. A plugin package that copies conversation text into a table of its own registers a scrubber under the `corvidae.redact` entry-point group, so a redaction reaches that copy too. The kernel never names the table.

```toml
[project.entry-points."corvidae.redact"]
exchange_log = "corvidae_cognition.outcome_log:redact_exchange_log"
```

The entry-point name labels the surface in the CLI output (`2 exchange_log rows`) and in skip notices. The object is an async function:

```python
async def redact_my_table(
    db: aiosqlite.Connection,  # the CLI's own connection (WAL, busy_timeout 5000)
    channel_id: str,           # one channel at a time
    msg_ids: list[int],        # the message_log ids just tombstoned in it
    dry_run: bool,             # count what would change, write nothing
    tombstone: str,            # the operator tombstone text to write
) -> int | None: ...
```

Every redact form (`message`, `range`, `memory`) calls each scrubber once per channel, after the message tombstones and the memory cascade have committed. Return the number of rows changed (or that would be, under `--dry-run`). Return `None` if your table isn't in this database (the daemon may never have loaded your plugin). The count is then 0, and one `SKIP` line naming your scrubber goes into the `notices` list a caller of `redact_messages` passes. The CLI doesn't print notices, so an operator sees only `0 <name> rows`. The scrubber commits its own writes. Probe for your table rather than creating it, because schema is daemon-owned. An exception propagates and aborts the command after the earlier steps have committed, so make the scrub idempotent: a re-run must finish the job and count nothing already scrubbed. The module is imported each time a redact form runs, inside its event loop, so keep it free of import-time side effects.

## Hook grammar

A hook's name tells you how its results are resolved. Every hookspec name
starts with a prefix that fixes its pluggy options, or is a bare imperative
verb on a short allowlist. `tests/test_hook_grammar.py` checks names against
pluggy options. Hooks a plugin declares itself also carry the owner's
plugin name after the prefix, can't be bare imperatives, and are checked
at startup; see [Declaring hooks in a plugin](#declaring-hooks-in-a-plugin).

| Prefix | pluggy spec | Resolution | Core uses result | Implementations |
|---|---|---|---|---|
| `on_`, `before_`, `after_` | broadcast | none | no | any number; side effects allowed, including on objects passed in |
| `should_` | broadcast | `resolve_reject_wins` | yes, `bool` | any number; `False` from any one rejects; `None` abstains |
| `resolve_` | `firstresult=True` | first non-`None` | yes | any number; first to answer wins, so order matters and the docstring says who answers |
| `transform_` | `firstresult=True` wrapper chain with a seed in `_SeedHooksPlugin` | chain | yes | any number, LIFO |
| bare imperative | broadcast | caller-defined over the result list | sometimes | any number; the docstring states what the caller does with results |

The observe class has three prefixes because `before_agent_turn`,
`before_compaction` and `after_persist_assistant` are observations placed
relative to a stage. All three mean the same thing to pluggy.

An observe hook may mutate an object it was handed (`after_persist_assistant`
strips `reasoning_content` in place; `before_agent_turn` appends to
`channel.conversation`), but its return value is never read. A transform
returns the replacement and the core uses it.

`resolve_channel` shows the other safe shape for `resolve_`: each
implementer answers only for ids under its own `transport:` prefix and
returns `None` for the rest, so the order pluggy calls them in cannot change
the answer.

The bare imperative allowlist is `send` (transports receive it through
`TransportPlugin`; see [Writing a transport](#writing-a-transport)),
`register_tools` (implementers append to the list passed in),
`register_llm_stages` (implementers add to the dict passed in) and
`persist_message` (every implementation runs; the caller resolves the list
with `resolve_single_result`).

### Adding a hook

A hookspec earns a name when it is a decision point (veto, resolve,
transform), when it is an observation with at least two plausible
implementers, or when it is an observability hook that pairs with an
existing one (`on_llm_request` with `on_llm_response`). A single-consumer
observation does not get a hookspec; it gets an entry in `annotations`, a return
value on an existing hook, or waits for its second consumer. Pick the name
from the table above. The test checks the prefix against the pluggy
options; whether the core reads a hook's return value is not something
pluggy exposes, so that half is checked in review.

### Declaring hooks in a plugin

A hook that only one plugin calls doesn't belong in the kernel's
`AgentSpec`. The plugin that calls it owns it instead. Set `hookspecs` on
the plugin class to a class of `@hookspec` methods:

```python
from corvidae.hooks import TransportPlugin, hookimpl, hookspec

class WebConsoleSpec:
    @hookspec
    async def on_web_console_init(self, panels: list[Panel]) -> None:
        """Append one Panel per implementing plugin."""

class WebPlugin(TransportPlugin):
    hookspecs = WebConsoleSpec

    @hookimpl
    async def on_init(self, pm, config):
        ...  # parse and store the web block
        panels: list[Panel] = []
        await self.pm.ahook.on_web_console_init(panels=panels)
        ...  # validate and store the panels
```

`Runtime.start()` installs every declared spec right after entry points
load (`install_plugin_hookspecs`), before the arg-binding guard and
before `on_init`, so any plugin's `on_init` can already call the hook.
Startup fails, naming the hook, the plugin and the fix, when a
declaration breaks one of these rules:

- **Grammar, with the owner named.** The name starts with a grammar prefix
  from the table above, then the owning plugin's registered name and an
  underscore: `on_web_…`, `should_web_…`, `resolve_web_…`,
  `transform_web_…`. Bare imperatives stay kernel-only. The prefix still
  fixes the pluggy options (`resolve_`/`transform_` need
  `firstresult=True`; the others must not have it).
- **One owner per hook.** A name declared by two plugins, or by a plugin
  and `AgentSpec`, is an error in either registration order. To share a
  hook, depend on its owner and call it; any plugin can call any hook.
- **Transform seeds come from the owner.** A `transform_` hook needs an
  `@hookimpl(trylast=True)` on the owner itself that returns its input
  unchanged, which is what `_SeedHooksPlugin` provides for kernel transforms.

The owner segment is the plugin's registered name: its entry-point name,
or the `name=` passed to `pm.register`. Renaming an entry point renames
its hooks. A plugin registered without a name, or under a name that can't
appear in a Python identifier (`web-console`), can't own hooks. The
declaration is read from the class, so a base class can declare for a
subclass, but only one plugin registered from that class hierarchy can
own the hooks. A plugin that is a module rather than a class instance
can't declare hooks.

A plugin hook still has to earn its name under the admission rule in
[Adding a hook](#adding-a-hook). Owning it yourself changes where the spec
lives, not whether it should exist.

Implementing a plugin's hook is the same as implementing a kernel hook:
write a `@hookimpl` with the same name, and declare the dependency (see
[Plugin dependencies](#plugin-dependencies)). The hook's contract is
documented by its owner, not in [Available hooks](#available-hooks).

At startup, a hookimpl whose name matches no spec is logged at WARNING
(`plugin 'x' implements hook 'on_web_consol_init', which no loaded plugin
declares`), since it will never be called. Usually that's a typo. The
exception is an impl for a hook owned by one of your `optional_depends_on`
plugins that isn't loaded: that's expected, and is logged at DEBUG.

### Implementing `on_funnel_ingest`

The funnel (cognition extra) owns `on_funnel_ingest`, the hook a source uses
to hand it entries ahead of a turn. Use it when your entries are not ready
when the funnel asks: you submit a promise, and the entries are admitted
once it resolves, through the same dedupe, budget and framing as `admit()`.
If you have the entries now, call `admit()` or `ingest()` instead. The funnel's `ponder` tool waits for
the channel's in-flight promises, up to 5 s, so a slow recall can land before
the model answers; a promise that resolves to `None` or no entries counts as
finished with nothing. The full
contract is under `FunnelPlugin` in
[cognition.md](cognition.md#funnelplugin-packagescorvidae-cognitioncorvidae_cognitionfunnelpy).

```python
import asyncio

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.funnel import Ingest  # cognition extra

class RecallPlugin(CorvidaePlugin):
    optional_depends_on = frozenset({"funnel"})

    def __init__(self) -> None:
        # The promise this plugin last submitted, per channel id.
        self._promises: dict[str, asyncio.Future] = {}

    @hookimpl
    async def on_funnel_ingest(self, channel, trace_id, annotations, submit) -> None:
        # Do not submit again while this channel's promise is pending: a newer submit cancels it.
        old = self._promises.get(channel.id)
        if old is not None and not old.done():
            return
        # Start the slow work in the background and hand the funnel the future.
        promise = asyncio.ensure_future(self._recall(channel))
        self._promises[channel.id] = promise
        submit("recall", promise)

    async def _recall(self, channel) -> Ingest | None:
        # Resolve to the lines to admit, or to None when there is nothing to add.
        lines = await search_notes(channel)  # your own lookup
        return Ingest(entries=lines) if lines else None
```

- The hook must be `async`. It submits and returns; it never awaits a promise.
- `submit(source, promise)` takes a source label, which becomes the frame
  label, and an `asyncio.Future`. The future resolves to
  `Ingest(entries, on_admitted=None)` or to `None`. `entries` are
  pre-formatted lines, admitted as one frame in the order given.
- One pending promise per `(channel, source)`. A newer submit cancels the
  older, which is why the example checks before it submits.
- A promise that raised is logged with its source and channel and dropped.
  A promise its provider cancelled is dropped quietly. `None` admits nothing.
- `on_admitted`, if given, is a sync callable. The funnel calls it with the
  `Admission` (`admitted` and `duplicate` lists) after each admission attempt
  that completes. A raise is logged and does not stop the drain.
- A promise that is done when the turn starts is admitted that turn. One
  still pending is admitted at the channel's next turn after it resolves,
  whatever that turn's origin. Nothing wakes the channel for it.
- `funnel.ingest_ttl`, when set, drops a queued stub-less payload (`ingest`) or a pending promise
  older than it; an unresolved promise is cancelled, so your work should
  tolerate cancellation. Shutdown cancels pending promises.
- A promise that finishes during a turn, after the funnel's sweep, is lost
  if you submit again in that same turn. The funnel does not admit such a
  result on re-submit yet.

### Writing a recall source

If your plugin looks something up for each user turn (a wiki, a folder of
notes) and wants the result in the window as CONTEXT, don't write the hook
above yourself. `RecallSourceMixin` in `corvidae_cognition.recall` (cognition
extra) already does the parts every such source repeats: it checks that the
turn is a user turn, builds the query, keeps a cooldown and a single search
in flight per channel, runs your search in the background under a timeout,
catches and logs its failures, and submits the promise. You supply two methods.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.recall import RecallContext, RecallSourceMixin


class NotesRecall(RecallSourceMixin, CorvidaePlugin):
    # The funnel's frame label, and the key of funnel.budgets.<source_name>.
    source_name = "notes"
    # Your own agent.yaml block; the mixin reads cooldown and timeout from it.
    config_key = "notes_recall"

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        # Raises ValueError on a malformed block, which aborts startup.
        self.init_recall(config)

    async def search(self, query: str, ctx: RecallContext) -> list[dict] | None:
        # Hits in rank order, or None or [] for none. May raise.
        return await lookup(query)  # your own lookup

    def render(self, hit: dict) -> str:
        # One line per hit; render it the same way every time (see below).
        return f"{hit['title']}: {hit['text']}"
```

The mixin is a helper, not a plugin: it has no entry point. Your class does,
and `plugins.disabled` switches your source off by that name. The rest of the
contract:

- **When it searches.** Only when the window ends in a plain user message
  with text. CONTEXT rows below that message are looked through, because the
  funnel runs this hook last, after it may have appended CONTEXT under the
  user's words. A turn that a tool result or an agent reply ends is skipped. The
  query is memory's `retrieval_query` with `previous_reply` context: the
  unanswered user messages, preceded by the agent's last reply.
- **Settings.** `<config_key>.cooldown` (seconds, 0 or more, default 5) and
  `<config_key>.timeout` (seconds, greater than 0, default 10). Put your own
  keys in the same block; the mixin ignores them. No block gives the defaults and
  logs nothing. A bad value raises `ValueError` naming the key, the value and
  what is accepted. A top-level `cooldown` is not read.
- **Cooldown and single-flight.** At most one search is in flight per channel,
  and a channel is not searched again until `cooldown` seconds after the
  last search started. A search that failed or timed out still counts as one.
  While a search is pending the mixin does not submit again, because a newer
  submit would cancel it. A turn skipped by either rule is not retried: the
  next turn asks again.
- **Fail-soft.** A search that raises, or runs past `timeout`, is logged at
  WARNING with your source name, the channel and the traceback, and the turn goes
  on without it. The funnel cancelling a promise (`funnel.ingest_ttl`,
  shutdown) is not a failure and logs nothing. A hit whose `render` raises is
  dropped with a warning and the others stand. `search` must not swallow
  `CancelledError`.
- **Size and freshness.** The mixin has no `k`, no bands and no score
  threshold. What fits is `funnel.budgets.<source_name>`, an entry bigger than
  the whole budget is dropped with a warning, and `funnel.ingest_ttl` ages out
  a result nobody collected. Whether a hit is relevant enough to show is not
  decided here (#674).
- **Render stably.** The funnel dedupes on a whole line for now (#725), so a
  line that carries something that changes (an age, a rank) re-admits the same
  hit every turn.
- **Without the funnel** nothing calls the hook and the source does nothing.
  The mixin lists `funnel` in `optional_depends_on`, so that is logged at DEBUG.
- **One timing limit** (#726). A result is admitted at the turn after the one
  that started its search, never the same turn. The mixin never submits while its
  own search is pending, so the funnel's lost-on-resubmit case needs a search that
  finishes between the funnel's sweep and its collect in a single turn.
- **Tests.** Pass a fake clock as `init_recall(config, now=clock)`. See
  `tests/cognition/test_recall_mixin.py` for a source on a real plugin
  manager with the real funnel.

## Subscribing to webhooks

The `webhook` plugin (`corvidae/channels/webhook.py`) owns `on_webhook_subscribe(subscriptions)`. Implement it to receive the JSON objects POSTed to `/hook/<endpoint>`; the listener, auth and endpoint names are the operator's `webhook:` config ([configuration.md](configuration.md#webhook--webhook-ingress)).

```python
from corvidae.channels.webhook import WebhookDelivery
from corvidae.hooks import CorvidaePlugin, hookimpl

class GithubEvents(CorvidaePlugin):
    optional_depends_on = frozenset({"webhook"})

    @hookimpl
    async def on_webhook_subscribe(self, subscriptions) -> None:
        subscriptions.setdefault("github", []).append(self.handle)

    async def handle(self, delivery: WebhookDelivery) -> None:
        ...  # delivery.payload, delivery.delivery_id, delivery.origin
```

Declare `optional_depends_on = frozenset({"webhook"})` so the impl is not reported as specless when the ingress is disabled. The handler contract:

- It runs in a worker task owned by the ingress, one delivery at a time per handler, in arrival order. It may run before your own `on_start` finishes, so resolve dependencies in `on_init` or lazily.
- `delivery.payload` is the parsed JSON object, shared with every other subscriber: do not mutate it. It is untrusted text. `delivery.origin` is `webhook:<endpoint>`, minted by the ingress.
- To let the model see an event, call `on_notify` yourself; the ingress never does.
- An exception is logged at `ERROR` and the next delivery proceeds. A handler that falls 100 deliveries behind loses its oldest ones, with a `WARNING`.

## Adding a web console panel

The `web` plugin (`corvidae/channels/web.py`) owns `on_web_console_init(panels)`. Implement it to add a read-only panel to the console: a section at the bottom of the chat page's sidebar, with its own script, styles and JSON endpoints. A panel has no page of its own; its script runs in the chat page. The hook fires at the end of `WebPlugin.on_init`, only when agent.yaml has a `web` block.

```python
from importlib.resources import files
from pathlib import Path

from aiohttp import web

from corvidae.channel import ChannelRegistry
from corvidae.channels.web import Panel
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

class ChannelsPanel(CorvidaePlugin):
    optional_depends_on = frozenset({"web"})

    @hookimpl
    async def on_web_console_init(self, panels: list[Panel]) -> None:
        panels.append(Panel(
            plugin=self,
            title="Channels",
            static_dir=Path(str(files("my_pkg") / "panel")),
            routes=[web.get("/data", self.data)],
        ))

    async def data(self, request: web.Request) -> web.Response:
        registry = get_dependency(self.pm, "registry", ChannelRegistry)
        rows = [
            {"id": c.id, "messages": len(c.conversation.messages) if c.conversation else 0}
            for c in registry.all()
        ]
        return web.json_response(rows)
```

`Panel` has four fields:

- `plugin`: the registering plugin; pass `self`. Its registered name becomes the panel's URL segment.
- `title`: the heading of the panel's sidebar section.
- `static_dir`: a directory holding `panel.js` and `panel.css`, which the chat page loads, plus anything else they load.
- `routes`: data endpoints, as aiohttp route definitions (`web.get(path, handler)`). Handlers are ordinary aiohttp handlers.

A plugin named `channels_panel` gets these URLs:

| URL | Serves |
|---|---|
| `/panel/channels_panel/static/<file>` | files from `static_dir` |
| `/panel/channels_panel/data` | each route, under the panel's prefix |

There is no `/panel/channels_panel/` page. The chat page, rendered once at startup, carries for each panel a `<link>` for `panel.css`, a deferred `<script data-panel="channels_panel">` for `panel.js` (after the console's own script), and an empty `<section class="panel-section" data-panel="channels_panel"><h2>Channels</h2></section>` at the bottom of the sidebar. Panels appear in name order.

The page and your script share a DOM contract and nothing else; there is no JS API, global or event:

- **Find your section** with `document.currentScript.dataset.panel` (read it at load, not later) and fill it. Sidebar content, such as a list of links, goes there.
- **Put a view in the main pane.** Append your own `<div class="panel-view hidden">` to `#main` and remove `hidden` only while the hash is your route. While any `#main > .panel-view` is shown, the console's CSS hides the rest of the chat pane (composer, banner, unconfirmed-sends list, conversation items, working indicator) and the view fills the pane as the scroller. You don't hide or restore anything yourself.
- **Route by hash.** `#<slug>` is a conversation, `#<panel name>/<arg>` is a panel route, and anything else is `main`. `<arg>` is yours: percent-encode it (`encodeURIComponent`) and decode it with a `try`, since a hand-edited hash can be malformed. A link in your section such as `<a href="#channels_panel/web%3Ahome">` selects the route; listen to `hashchange` and check the hash once at load to follow it. The console recognizes a route by your panel's name, so the hash must start with `<name>/`. It closes the phone sidebar and unmarks the conversation list on your routes.
- **Scope your CSS** under your view or section (`#channels-view ...`, `.panel-section[data-panel="channels_panel"] ...`). The stylesheet is global to the chat page, and the console's theme variables (`--border`, `--panel`, `--accent`, `--muted`) are available.
- **Wrap your script in an IIFE** so it adds no globals to the chat page, and fetch with absolute URLs (`/panel/channels_panel/data`).

An example `panel.js` for the plugin above:

```js
(function () {
  const NAME = document.currentScript.dataset.panel;
  const section = document.querySelector(`.panel-section[data-panel="${NAME}"]`);
  const list = document.createElement("ul");
  const view = document.createElement("div");
  view.id = "channels-view";
  view.className = "panel-view hidden";
  section.append(list);
  document.getElementById("main").append(view);

  async function load() {
    const resp = await fetch(`/panel/${NAME}/data`, { cache: "no-store" });
    if (resp.status === 401) {
      list.textContent = "Session expired; reload to log in.";
      return;
    }
    const rows = await resp.json();
    list.replaceChildren(...rows.map((row) => {
      const a = document.createElement("a");
      a.href = `#${NAME}/${encodeURIComponent(row.id)}`;
      a.textContent = `${row.id}: ${row.messages} messages`;
      const li = document.createElement("li");
      li.append(a);
      return li;
    }));
  }

  function route() {
    const prefix = `#${NAME}/`;
    const shown = location.hash.startsWith(prefix);
    view.classList.toggle("hidden", !shown);
    if (!shown) return;
    let id;
    try { id = decodeURIComponent(location.hash.slice(prefix.length)); } catch (err) { id = "?"; }
    view.textContent = `Channel ${id}`;
  }

  window.addEventListener("hashchange", route);
  load();
  route();
})();
```

The contract:

- **Read-only.** Every route must be a GET route made with `web.get`. Anything else (`web.post`, `web.view`, `web.route("*", ...)`, a static definition) fails startup with a `PluginValidationError` that names your plugin. It fails in `on_init`, so the restart pre-flight refuses the restart too. GET is a floor on the HTTP method, not a sandbox: your handler runs in-process, so it's on you not to change state from it.
- **One panel per plugin.** Register a single `Panel`; a second one from the same plugin is not supported.
- **Resolve dependencies lazily.** When the hook fires, each plugin's `on_init` has run up to its first suspending `await`, so a panel may decide whether to register from its own config, read synchronously in `on_init`. That holds only while no plugin wraps `on_init` or `on_web_console_init` with `@hookimpl(wrapper=True)`: apluggy then awaits the impls one by one. Look up other plugins in the handlers, not at registration. A handler can be called before your `on_start` or before a channel has a conversation, so check for `None`.
- **Read synchronously.** Handlers run on the event loop. They may read kernel objects this guide documents for plugins (`channel.conversation`, the channel registry), as long as they read without awaiting in between, so the view is consistent.
- **Text, never markup.** A panel often shows untrusted text: prompts, tool output, other people's messages. Render it with `textContent` or `createTextNode`, never `innerHTML`.
- **The console's floor applies.** Panel routes sit behind the console's token cookie, `Host` check and security headers, and the CSP (`'self'` only, no inline script or style) covers the chat page your script runs in. A handler can't loosen them: headers are applied to every response just before it is sent, streamed ones included. A WebSocket upgrade in a handler needs an allowed `Origin`, like the chat socket. Your script runs in the chat page's document, with the same origin and the same trust as the console's own script.

Declare `optional_depends_on = frozenset({"web"})` so the impl is not reported as specless when the console isn't installed. With no `web` block, the hook never fires and the panel plugin does nothing.

## Available hooks

### Lifecycle

| Hook | Type | When |
|------|------|------|
| `on_init(pm, config: dict)` | async broadcast | After all plugins are registered, before `on_start`. Use to store `pm`, read config values, and resolve references to other plugins. Do not create runtime resources here: `on_init` also runs in the restart pre-flight, a second process alongside the live daemon, so it must only validate and store. |
| `on_start(config: dict)` | async broadcast | Once at startup, after `on_init`. Use to open runtime resources: DB connections, network clients, file handles, asyncio tasks. |
| `on_stop()` | async broadcast | On SIGINT/SIGTERM, before process exits (a SIGINT during startup is held until `on_start` finishes; a SIGTERM then ends the process without `on_stop`). Also after a failed startup, before the error propagates, so it may run when this plugin's `on_start` (or `on_init`) never ran or didn't finish: guard each resource with a None check. It also runs in the restart pre-flight process, so keep it free of effects outside the process. |
| `on_config_reload(config: dict)` | async broadcast | When `agent.yaml` is modified while the daemon is running. The config dict has been re-parsed, merged with CLI overrides, and validated before dispatch. |

`config` is the full parsed `agent.yaml` dict. The key `_base_dir` (a `Path`) is set by `Runtime.start()` pointing to the config file's directory.

`CorvidaePlugin.on_init` stores `pm` and `config` as instance attributes. Subclasses that override `on_init` must call `await super().on_init(pm, config)` to preserve this behavior.

#### Supporting hot-reload in plugins

Implement `on_config_reload` to re-read configuration values when `agent.yaml` changes at runtime. The hook fires on the event loop thread; implementations must not block.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl

class MyPlugin(CorvidaePlugin):
    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.threshold = config.get("my_plugin", {}).get("threshold", 0.8)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        self.threshold = config.get("my_plugin", {}).get("threshold", 0.8)
```

`on_config_reload` receives the full re-parsed config dict, already merged with CLI overrides. Per-plugin errors are caught and logged by `ConfigWatcherPlugin`; a failing `on_config_reload` in one plugin does not prevent other plugins from receiving the reload.

The `config` parameter is optional in `on_config_reload` hookimpls. Pluggy forwards only the arguments declared in the hookimpl signature, so a hookimpl that omits `config` is called with no arguments.

### Messaging

| Hook | Type | When |
|------|------|------|
| `on_message(channel, sender: str, text: str, trace_id: str)` | async broadcast | Inbound message arrives; the transport mints and passes the id (see [Traces](#traces)) |
| `should_process_message(channel, sender: str, text: str, trace_id: str \| None)` | async broadcast, reject-wins | Gate: decide whether to process a message. Return False to reject, True to accept, None for no opinion. |
| `on_message_admitted(channel, trace_id: str, sender: str, text: str)` | async broadcast | After should_process_message admits a USER message |
| `on_message_rejected(channel, trace_id: str, sender: str, text: str)` | async broadcast | After should_process_message vetoes a USER message |
| `should_send_response(channel, text: str, emission: str, trace_id: str \| None, annotations: dict)` | async broadcast, reject-wins | Outbound gate, before each emission (`final`, `progress`, `thinking`, `error`). Return False to withhold, None for no opinion. See [Output gating](#output-gating). |
| `on_message_persisted(channel, trace_id: str, rowid: int \| None, text: str, annotations: dict)` | async broadcast | After the trace-originating message is persisted to the database. Fires once per originating item regardless of persistence outcome (`rowid` may be None). |
| `send(channel, event: Outbound)` | async broadcast | Outbound delivery: `event` is an `OutboundMessage`, `OutboundThinking`, `OutboundToolStatus` or `OutboundProgress` (`corvidae/outbound.py`). Transports receive it through `TransportPlugin` (see [Writing a transport](#writing-a-transport)). A non-transport plugin may implement `send` to observe every channel's events, as the context panel does |
| `on_notify(channel, source: str, text: str, tool_call_id: str \| None, annotations: dict \| None)` | async broadcast | Notification injected into channel |

Any plugin can send: `await self.pm.ahook.send(channel=channel, event=OutboundMessage(text="..."))`. `send` is broadcast to every plugin, and `TransportPlugin` drops the events for channels a transport does not own, so a sender never addresses a transport by name.

### Extension points

| Hook | Type | When |
|------|------|------|
| `register_tools(tool_registry: list)` | sync broadcast | During `on_start`, to collect tools |
| `register_llm_stages(stages: dict[str, str])` | sync broadcast | At the top of `LLMPlugin.on_start`, to collect the model-calling stages each plugin owns: `stages[name] = kind`, with kind `"chat"` or `"embedding"` |
| `on_agent_response(channel, request_text: str, response_text: str, trace_id: str \| None, annotations: dict, logprobs: dict \| None)` | async broadcast | After agent produces a response. Fires for a final the output gate vetoed (or an empty one) too, after its assistant row is in the window and before the transport send; `final_was_unsent(channel)` tells the two apart |
| `transform_system_prompt(channel, text: str) -> str \| None` | async `firstresult=True` wrapper chain | When a conversation's system prompt is composed (identity, then register): a new conversation, a restart, a compaction boundary, and the background views. Return the prompt with your section appended; see below |
| `before_agent_turn(channel, trace_id: str \| None, annotations: dict)` | async broadcast | Before each LLM invocation |
| `after_persist_assistant(channel, message: dict)` | async broadcast | After assistant message is written to DB; plugins may mutate the in-memory dict |
| `persist_message(channel, message: dict, message_type: MessageType) -> int \| None` | async broadcast | After every `conv.append()` call; message is the untagged dict (no `_message_type` key). Persistence returns the inserted `message_log` rowid; **exactly one implementation may return non-None** (all others return None). Callers resolve with `resolve_single_result` and attach the rowid to the in-window message as `_db_id`. |
| `before_compaction(channel, compacted_ids: list[int], cut: int)` | async broadcast, awaited | After the compaction summary succeeded and before `replace_with_summary`. The window still holds the messages about to be dropped (`channel.conversation.messages[:cut]`; `cut` is moved off tool cycles where compaction can afford it, see design.md), so this is where to fork over that prefix (see "Forking the window"). Runs on the response path; failures are logged and never block compaction. |
| `on_compaction(channel, summary_msg: dict, retain_count: int, compacted_ids: list[int])` | async broadcast | After compaction replaces older messages with a summary; `summary_msg` is the untagged summary dict. `compacted_ids` carries the `message_log` rowids of exactly the removed messages (empty when unknown) — the consolidation range consumed by `MemoryPlugin`. |
| `resolve_channel(channel_id: str) -> Channel \| None` | async firstresult | A caller holding only a persisted `transport:scope` id (`SchedulerPlugin` firing a row after a restart) asks transports for the live channel. Return it for ids your transport owns, else None. Return None for your own id only when it cannot be built yet; the scheduler then retries. See [Channels](#channels). |

`after_persist_assistant` — the DB row is already written when this
hook fires. Mutations to `message` affect in-memory prompt construction
only; they do not update the persisted record.

`transform_system_prompt` — `text` is the identity and register as composed, or the identity alone when the register is empty. It runs on the composed prompt whatever the register's source, so a channel with its own `register_prompt`, or with `register_prompt: ""`, still goes through it. A wrapper that raises is logged at WARNING with the channel and the untransformed prompt is used. A transform that modifies the identity+register prefix is rejected with a WARNING; the untransformed prompt is used. Appending is the expected use. The result is re-resolved at compaction boundaries, so a plugin's section persists across compactions as long as the wrapper remains registered. To act only on chats, find the transport that owns the channel and read its `conversational` (`SilencePlugin` does this).

`before_agent_turn` — messages injected via `channel.conversation.append()` inside this hook are passed through `persist_message` and persisted to the DB, unless they already carry a `_db_id` (i.e. their producer persisted them itself).

Window messages carry internal `_`-prefixed tags (`_message_type`, `_db_id`). Every serialization boundary (prompt build, compaction summarizer prep, persistence, JSONL log) strips all `_`-prefixed keys — internal tags never reach the LLM or the DB.

### Traces

A trace is everything one inbound item causes: its turns, its tool tasks, follow-up turns from their completions, and every LLM call in them. One exception: when several user messages are queued back to back, they coalesce into one turn (PRD §3.1.19), and that turn belongs to the last message's trace. Each earlier message's trace gets `on_message_admitted` and `on_message_persisted` and nothing else — no `before_agent_turn`, no `should_send_response`, no `on_agent_response`. Per-trace state opened at admission or persistence must not wait on a turn hook to close (bound it, as the cognition plugins do with LRUs). A `before_agent_turn` implementation that needs every message the turn answers can read them from the window with `corvidae.context.unanswered_user_messages(channel.conversation.messages)`. Each trace has an opaque `trace_id`. One exists for every inbound user message and for every notification-born queue item that isn't itself a continuation of an existing trace. The transport mints a user message's id with `corvidae.trace.mint_trace_id()` and passes it to `on_message`; the Agent mints at dequeue for a notification-born item with no inherited id. It travels alongside the `annotations` dict — extensible, plugin-owned and opaque to the framework — through the messaging and extension-point hooks above, into `Task.trace_id`/`Task.annotations`, and back out through the completion notification's `annotations` (`TaskPlugin._on_task_complete` merges `{"task_id", "trace_id", **task.annotations}`). This is how a tool cycle several hops deep still carries the trace id and annotations its originating message started with.

**Transport contract.** A transport mints a fresh, unique id per inbound message, passes it as `trace_id` to `on_message`, and matches `should_process_message`, `on_message_admitted`, `on_message_rejected` and `on_message_persisted` back to its message by that id, never by ordering or context inheritance. Import `mint_trace_id` from `corvidae.trace` (stdlib only), not from `corvidae.agent`. The Agent uses the id verbatim. Omitting the argument raises `pluggy.HookCallError`; a falsy id raises `ValueError`. The lifecycle hooks broadcast to every plugin, and unlike `send` they get no filtering from `TransportPlugin`, so each of a transport's lifecycle hookimpls opens with an ownership check (`if not self.owns(channel): return`; Discord and Signal spell it `channel.matches_transport(...)`, which is the same test for them) and ignores ids it never minted.

`on_message_admitted` or `on_message_rejected` fires before `on_message` returns, unless the agent drops the message without either (for example, it is not ready yet). State a transport took before the call (a typing slot, a pending read receipt, an unsent-message entry) can therefore be released in a `finally` when the id was not admitted:

```python
tid = mint_trace_id()
self._pending[tid] = state          # taken before the call
try:
    await self.pm.ahook.on_message(
        channel=channel, sender=sender, text=text, trace_id=tid,
    )
finally:
    if tid not in self._admitted:   # set by on_message_admitted
        self._pending.pop(tid, None)
    self._admitted.discard(tid)

@hookimpl
async def on_message_admitted(self, channel, trace_id, sender, text):
    if not self.owns(channel):
        return
    self._admitted.add(trace_id)
```

The framework itself never stamps meaning into `annotations` — it is a pass-through. Cognition plugins (see below) use the convention `annotations["origin"]` to record where a queue item came from ("user" recognized by subscribing to `on_message_admitted`, or an explicit string like `"reminder"` stamped by a producer); that convention is plugin-side, not a core concept.

`annotations` never reach the model on their own. A plugin that wants a key shown declares it on the class:

```python
class HeartbeatPlugin(CorvidaePlugin):
    model_visible_annotations = frozenset({"origin"})
```

The Agent collects these once at startup (a plugin loaded later is not read) and, for a notification carrying a declared key, stores that key's value with the row and renders it in a kernel-built `[annotations: origin=heartbeat]` line. Only `origin` is declared today (scheduler producers, heartbeat, funnel, cross-channel send, restart). Declare a key only for short, plugin-chosen text, never payload: the value is sanitized but otherwise shown as given. The header's wording and the payload fence are the kernel's, and a plugin cannot change them ([design.md](design.md#provenance-header-and-fence)).

`SchedulerPlugin` is the producer that stamps it for clock-driven items. A due reminder fires as

```python
await self.pm.ahook.on_notify(
    channel=channel, source="reminder", text=framed_text, tool_call_id=None,
    annotations={"origin": "reminder", "schedule_id": row_id},
)
```

and core carries `annotations` untouched to `on_message_persisted`, `before_agent_turn`, `on_agent_response`, every tool Task in the cycle, and the usage-log attribution. A notification like this, with no inherited `annotations["trace_id"]`, originates its own trace, so it also gets a fresh `max_turns` budget (reset only when no tool cycle is in flight).

Producers that start a background task in `on_start` and inject notifications should wait for `pm.get_plugin("agent").started` before calling `on_notify`: `on_start` is broadcast to every plugin concurrently, and `Agent` starts after the broadcast.

### Observability

| Hook | Type | When |
|------|------|------|
| `on_llm_request(profile, model, request_id, message_count, tool_count, attribution)` | async broadcast | Immediately before an LLM chat-completion call |
| `on_llm_response(profile, model, request_id, usage, latency_ms, attribution, error, timings, started_at, total_ms, attempts, message_count, tool_count)` | async broadcast | Exactly once per LLM call, after success or terminal failure |
| `on_metrics(name, value, tags)` | async broadcast | Whenever any plugin emits a metric event |

`on_llm_request` / `on_llm_response` fire from the `LLMClient` chokepoint
(via an observer injected by `LLMPlugin`), so **every** LLM call in the
system is covered — turn-loop, compaction, and subagent calls alike.
`request_id` pairs a request with its response. Retried transient attempts
do not fire `on_llm_response`; one request pairs with exactly one response.
`model` is the model the request was sent with: the profile's configured
model unless an `extra_body` (the profile's or the call's) overrode it.
`usage` is the response envelope's `usage` field verbatim (or `None`);
`timings` is the response's `timings` field verbatim, which llama-server
sends (`cache_n` prompt tokens reused from cache, `prompt_n` prefilled)
and other providers omit (`None`, as on terminal failure);
`latency_ms` is the final attempt's duration; `started_at` is the call's
wall-clock start (epoch seconds), `total_ms` its monotonic duration from
the first attempt to the end including retries and the sleeps between,
and `attempts` how many attempts were made (1 = no retry; transient-status
and connection retries and model-not-loaded waits each count); `message_count` and `tool_count` are the request's counts
(embeddings: the number of texts, and 0);
`error` is `None` on success and the exception string on terminal failure.
The request payload is summarized (message/tool counts), never shipped
wholesale. Observer failures are logged and never break the LLM call.

`attribution` is a snapshot of `corvidae.attribution.get_attribution()` —
a contextvar dict describing what the current code path is doing on whose
behalf. Built-in callers set `stage` (`"turn"`, `"compaction"`,
`"subagent"`, `"consolidation"`) and `channel_id`; the dict is open for
plugin-defined fields. `MemoryPlugin`'s consolidation stage adds `trigger`
(`"compaction"`, `"idle"` or `"turn"`; `"eval"` when
`scripts/eval_memory.py` drives it), naming which consolidation trigger
is driving the current `on_llm_request`/`on_llm_response` pair —
a plugin consuming the attribution snapshot on a `stage="consolidation"`
call can rely on `trigger` being present. Set it around a logical
operation and always reset in a `finally`:

```python
from corvidae.attribution import reset_attribution, set_attribution

token = set_attribution(
    stage="consolidation", channel_id=channel.id, trigger="idle"
)
try:
    await client.chat(messages)
finally:
    reset_attribution(token)
```

Contextvars snapshot at `asyncio.create_task` time, so attribution does
not automatically reach long-lived workers. `Task` (corvidae/task.py)
captures `contextvars.copy_context()` at creation and the TaskQueue worker
runs the work inside that context — attribution set by the enqueuer is
visible in the task body.

`on_metrics` carries a named numeric value with dimensional tags
(StatsD/OpenTelemetry common denominator; dotted names like
`llm.tokens.prompt`, `llm.latency_ms`). Emit metrics from any hook or tool
with `await self.pm.ahook.on_metrics(name=..., value=..., tags=...)`.
**Reentrancy constraint:** never emit from inside an `on_metrics`
implementation — pluggy dispatches back into the same implementation and
recurses forever.

### Hook result resolution

These hooks return a value. Hooks marked `firstresult=True` (sequential) stop at the first non-None return. Wrapper chain hooks use `firstresult=True` with an identity seed; implementations use `@hookimpl(wrapper=True)` to compose transforms. Broadcast gate hooks use `resolve_reject_wins` for result resolution.

| Hook | Strategy | Returns | Behavior |
|------|----------|---------|----------|
| `should_process_message(channel, sender, text, trace_id)` | reject-wins (broadcast) | `bool \| None` | Any `False` vetoes the message; any `True` (with no `False`) accepts; `None` if all defer |
| `should_send_response(channel, text, emission, trace_id, annotations)` | reject-wins (broadcast) | `bool \| None` | Any `False` withholds the emission; anything else sends it |
| `resolve_llm_error(channel, error)` | `firstresult=True` | `str \| None` | First non-None string wins; chain stops. If all return None, the default message names the failure class (timeout, unreachable server, or an HTTP status) and gives its remedy: a timeout or an HTTP error asks the user to try again, while an unreachable backend the daemon has scheduled a retry for says the answer will come when the model server is back. When no retry could be scheduled (no kernel clock registered), the unreachable message falls back to asking the user to try again rather than promising a retry that will not happen. `LLMPlugin` registers a `trylast` handler that explains an unloaded router model, any non-2xx `LLMHTTPError` (naming the status, an actionable hint, and a short provider body), and connection/timeout failures that point at `llm.main.base_url`; it defers (`None`) for anything else. To inspect HTTP failures yourself, check `isinstance(error, corvidae.llm.LLMHTTPError)` and read `error.status`, `error.body`, and `error.model`. |
| `resolve_compaction(channel, conversation, max_tokens)` | `firstresult=True` | `bool \| None` | First non-None return stops the chain. `CompactionPlugin` uses `trylast` (returns True). Third-party plugins run at default priority. |
| `transform_tool_result(tool_name, result, channel)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose transforms in LIFO order. The seed returns the input unchanged. Non-wrapper hookimpls short-circuit the seed via firstresult. |
| `transform_display_text(channel, text, result_message)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose transforms in LIFO order. The seed returns the input text unchanged. Non-wrapper hookimpls short-circuit the seed via firstresult. |
| `transform_system_prompt(channel, text)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose in LIFO order, so a later wrapper's text lands after an earlier one's. The seed returns the input text unchanged. |
| `resolve_conversation(channel)` | `firstresult=True` | `list[dict] \| None` | First non-None result wins. `PersistencePlugin` uses `trylast` as fallback. Called once when a channel's conversation is first initialized, and by `AcpPlugin` for `session/load`. |
| `persist_message(channel, message, message_type)` | `resolve_single_result` (broadcast) | `int \| None` | Every implementation runs. `PersistencePlugin` returns the `message_log` rowid; others return None. More than one non-None is a configuration error: logged, and the first is used. |

`resolve_compaction` — sequential hook (`firstresult=True`). The first handler returning a non-None value stops the chain. `CompactionPlugin` runs last (`trylast=True`) as the default strategy and returns True. A third-party plugin at default priority runs before it — if it returns non-None, `CompactionPlugin` is never called.

`transform_tool_result` fires for all tool calls — it is invoked from `dispatch_tool_call` in `corvidae/tool.py`, which is used by both the main agent loop and background subagent loops. It does not fire for pre-dispatch errors (JSON parse failure or unknown tool name).

`transform_display_text` — `result_message` is the raw assistant message dict from the LLM response. It may contain `reasoning_content` if the model produces thinking tokens. `text` is the string content extracted from that message. Wrapper implementations receive the chain result (the output of inner wrappers or the seed) via `yield`. Non-wrapper implementations that return `None` defer to the seed, which returns `text` unchanged.

Example hook returning a value:

```python
@hookimpl
async def should_process_message(self, channel, sender: str, text: str, trace_id: str | None) -> bool | None:
    if sender in self.blocklist:
        return False
    return None  # no opinion
```

## Tool registration

Tools **must** be coroutine functions (`async def`) with type-annotated parameters. `Tool.from_function()` enforces this at registration time and raises `TypeError` if given a sync callable. The docstring's first line becomes the tool description for the LLM. See [Avoid blocking the event loop](#avoid-blocking-the-event-loop) for what `async def` means in practice — the check rejects sync functions, but it cannot detect blocking I/O called from inside an async one. Register tools in `register_tools`:

```python
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

class WeatherPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def get_weather(city: str, units: str = "metric") -> str:
            """Get current weather for a city."""
            return await fetch_weather(city, units)

        tool_registry.append(Tool.from_function(get_weather))
```

`Tool.from_function()` infers the tool name from `fn.__name__` and generates a JSON schema from the type annotations. Parameters prefixed with `_` are excluded from the schema — they are injected at call time, not supplied by the LLM.

### Timeouts and cancellation

Every tool call is bounded by `tools.call_timeout` (default 600 s, per-tool
`tools.timeouts`). A tool that has its own timeout can declare it so dispatch
skips the outer bound when the tool's own is not longer:

```python
get_weather.own_timeout = lambda: self._timeout   # a number, or a zero-arg callable
```

Point it at the value the tool really uses; it is read at every call. Set it
on the registered function, partial or instance itself (a class-level property
is not read). A callable that raises, or a non-number, is ignored with one
`WARNING` and the outer bound applies.

On timeout the tool is cancelled. Clean up, then let `CancelledError`
propagate: a tool that swallows it has its result discarded and a `WARNING`
logged. `shell` (`corvidae/tools/shell.py`) is the model for subprocess
cleanup: on `CancelledError` it kills and reaps its process, then re-raises.
Work run through `asyncio.to_thread` and blocking code cannot be cancelled
(a cancelled `write_file` may still write the file).

## Context injection (ToolContext)

Tools that need channel context, the task queue, or the tool call ID declare a `_ctx: ToolContext` parameter. Corvidae injects it automatically; the LLM never sees it.

```python
from corvidae.tool import Tool, ToolContext

class MyPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def enqueue_work(instructions: str, _ctx: ToolContext) -> str:
            """Enqueue background work."""
            from corvidae.task import Task

            async def work():
                return await do_something(instructions)

            task = Task(
                work=work,
                channel=_ctx.channel,
                tool_call_id=_ctx.tool_call_id,
                description=instructions[:80],
            )
            await _ctx.task_queue.enqueue(task)
            return f"Enqueued task {task.task_id}"

        tool_registry.append(Tool.from_function(enqueue_work))
```

`ToolContext` attributes:
- `channel: Channel | None` — the channel this tool call is executing on
- `tool_call_id: str` — the LLM-assigned call ID for this invocation
- `task_queue: TaskQueue | None` — the task queue (None if TaskPlugin is not registered)

## Avoid blocking the event loop

Every tool runs on the single main asyncio event loop. A synchronous blocking call inside a tool — `requests.get`, `time.sleep`, a slow `open(...).read()`, a sync database driver — stalls the entire process: the main agent, every task-queue worker, and every channel queue all wait until the call returns.

The `Tool.from_function()` async check rejects functions defined with `def` instead of `async def`, but it cannot see what your `async def` does internally. If your code calls a blocking library, you are responsible for moving that call off the event loop with `asyncio.to_thread()`. The pattern used by the built-in file tools (`corvidae/tools/files.py`) is the model: a small sync helper does the I/O, the async tool awaits it via `to_thread`.

```python
import asyncio
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

def _lookup_sync(symbol: str) -> str:
    # Blocking call — sync HTTP client, sync DB driver, slow file read, etc.
    import requests
    return requests.get(f"https://api.example.com/{symbol}", timeout=5).text

class QuotePlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def lookup(symbol: str) -> str:
            """Look up a stock quote."""
            return await asyncio.to_thread(_lookup_sync, symbol)

        tool_registry.append(Tool.from_function(lookup))
```

For common cases the standard library and ecosystem offer native async alternatives — prefer them when available:

- HTTP — use `aiohttp` (see `corvidae/tools/web.py`).
- Subprocesses — use `asyncio.create_subprocess_shell` / `create_subprocess_exec` (see `corvidae/tools/shell.py`).

As a last-resort safety net, `execute_tool_call` detects a sync callable registered as a bare `Tool(name=..., fn=sync_fn, schema=...)` (bypassing `Tool.from_function()`) and wraps the call in `asyncio.to_thread()`, logging a warning. Treat that warning as a bug to fix in the plugin, not a supported registration path.

## What the kernel exports to plugins

A plugin outside the `corvidae` package (the `corvidae-cognition` plugins,
`cross-channel-send`, or your own) should import only these kernel modules.
This is the API the bundled plugins use, and
`tests/test_kernel_boundary.py` fails if a bundled plugin imports any other
kernel module. Changing a signature here is a plugin-API change and needs a
line in this guide.

| Module | Names |
|---|---|
| `corvidae.hooks` | `CorvidaePlugin`, `TransportPlugin`, `hookimpl`, `get_dependency`, `resolve_single_result`, `final_was_unsent` |
| `corvidae.fork` | `fork_agent`, `quote`, `window_db_ids`, `ViewInputs`, `view_agent`, `view_call`, `ForkRefused`, `build_fork_prompt`, `snap_cut_forward`, `snap_cut_to_tool_boundary` (see [Forking the window](#forking-the-window) and [The view call](#the-view-call)) |
| `corvidae.attribution` | `get_attribution`, `set_attribution`, `reset_attribution` |
| `corvidae.tuning` | `config_value`, `resolve_tunable` |
| `corvidae.channel` | `ChannelRegistry`, `resolve_persisted_channel`, `ChannelNotReady` |
| `corvidae.channels.webhook` | `WebhookDelivery`, `WebhookHandler` (for subscribers to the webhook ingress) |
| `corvidae.channels.web` | `Panel` (see [Adding a web console panel](#adding-a-web-console-panel)) |
| `corvidae.context` | `MessageType`, `ContextWindow`, `count_tokens`, `message_tokens` |
| `corvidae.task` | `Task` |
| `corvidae.tool` | `Tool`, `ToolContext`, `tool_to_schema` |
| `corvidae.persistence` | `PersistencePlugin`, `add_column_if_missing`, `view_rows`, `decodable_summaries` |
| `corvidae.llm_plugin` | `LLMPlugin` |
| `corvidae.clock` | `Clock`, `Timer` (see [Acting on time and quiet](#acting-on-time-and-quiet)) |
| `corvidae.idle` | `IdleTracker`, `IdleWatch` |
| `corvidae.spill` | `NotificationSpill` (see [Spilling a long notification](#spilling-a-long-notification)) |
| `corvidae.runtime` | `Runtime`: `load_config`, `setup_logging` and `load_plugins`, the startup steps the daemon and the restart pre-flight share, plus `arm_exit_bound()` — the exit watchdog with the configured `daemon.exit_timeout`, for a process whose exit is not `run()`'s to bound (the restart pre-flight, #757). For tools that check a config the way the daemon boots, not for ordinary plugins |
| `corvidae.outbound` | `OutboundMessage`, `OutboundThinking`, `OutboundToolStatus`, `OutboundProgress` (for senders and transports) |

The kernel never imports a plugin outside it. A plugin that needs another
plugin resolves it at runtime with `get_dependency(self.pm, "<name>", Type)`.

## Acting on time and quiet

To run something later, or when a channel or the system has been quiet,
use the kernel clock and idle tracker. They are not hooks. Declare both
dependencies and resolve them in `on_init`:

```python
from corvidae.clock import Clock
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.idle import IdleTracker


class Tidy(CorvidaePlugin):
    depends_on = frozenset({"clock", "idle"})

    @hookimpl
    async def on_init(self, pm, config):
        self.pm = pm
        self.clock = get_dependency(pm, "clock", Clock)
        self.idle = get_dependency(pm, "idle", IdleTracker)

    @hookimpl
    async def on_start(self, config):
        self.watch = self.idle.watch_channels(600, self.on_quiet, name="tidy")
        self.timer = self.clock.call_later(3600, self.hourly, name="tidy.hourly")

    @hookimpl
    async def on_stop(self):
        self.watch.cancel()
        self.timer.cancel()

    async def on_quiet(self, channel_id: str) -> None: ...
    async def hourly(self) -> None: ...
```

`call_later` and `call_at` are one-shot; re-arm from the callback for a
periodic job. `call_soon(cb, name=...)` runs once on the next loop step.
`watch_channels` repeats every `repeat` seconds (default: the period) while
the channel stays quiet, and `watch_system` does the same for the whole
system. `repeat` must be a positive finite number, so `watch_channels(0, cb,
name=...)` without `repeat=` (or a negative period) raises `ValueError`. To
learn that a channel just drained, as a transport does for a working
indicator, watch with `watch_channels(0, cb, name=..., repeat=3600)`: a zero
period fires at settle, and the callback sees every channel, so filter on
the id. A zero-period callback that awaits something outside the clock
blocks `FakeClock.settle()` in tests until it finishes. Cancel watches and
timers in `on_stop`; the Runtime cancels stragglers afterwards. Callbacks
must let `asyncio.CancelledError` through. An idle or timer callback is
upkeep: it does plugin-owned work and must never start a turn or enqueue work on a channel (principle 13), so it
never calls `on_notify`. Starting a turn on time is agency, which is the
scheduler's job: its firings go through `on_notify` and run a full turn. In
tests, `tests/fake_clock.py`'s `install_kernel_clock(pm)` registers
a fake clock and a real tracker, so time moves only when the test says so.
Full semantics: [design.md](design.md#clock-and-idle-tracker-clockpy-idlepy).

## Spilling a long notification

A plugin that cuts text before notifying a channel calls the spill service
instead of `on_notify`. The service cuts the text to your limit, writes the
full text to a file where a channel's tools can read it, and has the kernel
say so in the header; you write no marker and keep no annotation for it.

```python
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.spill import NotificationSpill

class MyPlugin(CorvidaePlugin):
    depends_on = frozenset({"registry", "spill"})

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._spill = get_dependency(pm, "spill", NotificationSpill)

    async def _deliver(self, channels, text: str) -> None:
        # One call for all channels: one file, however many are notified.
        await self._spill.notify(
            channels, source="my_plugin:thing", text=text,
            annotations={"origin": "my_plugin:thing"}, limit=8000,
        )
```

`notify` takes the resolved channels, so resolve them first. Text at or under
`limit` goes out unchanged. Each channel gets its own `on_notify` with
`tool_call_id=None` and a fresh copy of `annotations`. A channel gets the
file's path in its header only if an enabled tool there declares it reads
spill files; a tool does so by setting an attribute on the function:

```python
async def read_log(path: str) -> str:
    "Read a log file."
    ...

read_log.reads_spill_files = True
```

Set it only on a tool that can open an arbitrary path (`read_file` and
`shell` do). Declare a tool that cannot, and the agent is told about a file it
cannot reach.
Tuning (directory, size cap, age) is in
[configuration.md](configuration.md#daemon--daemon-level-settings), and the
mechanism in [design.md](design.md#notification-spill-spillpy).

### Declaring untrusted output

A tool whose result carries text its author doesn't control (a fetched page,
a file's contents, a search snippet, a third-party service's reply) declares
it the same way, on the registered callable:

```python
async def fetch_ticket(id: str) -> str:
    "Fetch a ticket body."
    ...

fetch_ticket.returns_untrusted = True
```

The kernel then renders each result of that tool in the nonce fence under a
header it builds, naming the tool and the call id and labelling the content
untrusted outside content, so the text can't imitate a header, a role or an
instruction. Your tool returns plain text and does no framing of its own; the
stored result is unchanged and the fence is applied when the prompt is built.
The declaration is read when the call is dispatched and the answer is stored
on the result row, so changing it later does not re-render history, and a
tool unregistered while it runs stays fenced. It is strict: only `True` counts
(`1` or a callable does not). Without it a result reaches the model as before.
MCP tools are declared by the client for every server, with no setting. See
[design.md](design.md#untrusted-tool-results).

## Plugin dependencies

Declare `depends_on` as a class attribute (a set of plugin names). `validate_dependencies()` raises `RuntimeError` at startup if any declared dependency is not registered, or if the dependency graph contains a cycle. The error message includes the full cycle path, e.g. `Dependency cycle detected: a -> b -> a`.

```python
class MyPlugin(CorvidaePlugin):
    depends_on = {"agent", "task"}
```

To get a typed reference to a dependency:

```python
from corvidae.hooks import get_dependency
from corvidae.tool_collection import ToolCollectionPlugin

tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
registry = tools_plugin.get_registry()
```

`get_dependency` raises `RuntimeError` if the plugin is not found and `TypeError` if it is the wrong type.

A plugin that uses another plugin when it's there and works without it
declares `optional_depends_on` instead. A missing optional dependency is
not an error, and optional dependencies aren't part of cycle detection.
Look the plugin up with `pm.get_plugin(name)` and treat `None` as absent
(it's `None` whether the plugin is disabled or not installed). The declaration is documentary: it only changes the log level of a hookimpl with no spec (above) and doesn't order loading, so look the plugin up when you use it, not in `__init__`:

```python
class ProfilerPanel(CorvidaePlugin):
    optional_depends_on = frozenset({"web"})

    @hookimpl
    async def on_web_console_init(self, panels):
        panels.append(...)  # only called when web is loaded
```

If the owner is absent, the impl stays registered with no spec and is
never called. See [Declaring hooks in a plugin](#declaring-hooks-in-a-plugin).

## Injecting context before agent turns

`before_agent_turn` fires before every LLM call. Use it to inject contextual information (memory retrieval, current state, etc.) into the conversation. The hook receives the current turn's trace id and annotations dict:

```python
from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, hookimpl

class MemoryPlugin(CorvidaePlugin):
    @hookimpl
    async def before_agent_turn(self, channel, trace_id: str | None, annotations: dict) -> None:
        notes = await self.fetch_relevant_notes(channel.id, trace_id)
        if notes:
            channel.conversation.append(
                {"role": "user", "content": f"[Context]\n{notes}"},
                message_type=MessageType.CONTEXT,
            )
```

Compaction summarizes only `MESSAGE` entries, but a `MessageType.CONTEXT` row in the range it drops leaves the window with it; CONTEXT in the retained tail stays.

Prefer routing CONTEXT through the admission funnel instead of appending directly: `pm.get_plugin("funnel").admit(channel, conv, "<source>", entries)` gives you window dedupe, per-source token budgets (`admit_detailed(...)` takes the same arguments and also reports which entries were dropped as duplicates), and the mandatory data-not-instructions framing for free (see [FunnelPlugin](cognition.md#funnelplugin-packagescorvidae-cognitioncorvidae_cognitionfunnelpy) in cognition.md; the funnel is part of the cognition extra, so check that `pm.get_plugin("funnel")` is not None). Direct appends bypass all three. Dedupe matches each entry's whole text against every CONTEXT row by default; if your lines carry something that changes from turn to turn (an age, a score), pass `keys=[...]`, one stable string per entry, and an entry is a duplicate when its key appears in a CONTEXT row framed for your source. Memory passes `memory <id>)`. Choose keys that can't occur inside one another, which is why memory's ends in a paren, and pass exactly one per entry: a length mismatch raises `ValueError`. Operators can also cap what your source holds in a window with `funnel.window_budgets.<source>` (see [configuration.md](configuration.md#funnel--context-admission-funnel)); entries the cap drops come back like budget drops, in neither `admitted` nor `duplicate`.

## Forking the window

A background process that wants the agent's own context and voice, and on
a single-slot llama-server wants to reuse the cached prompt instead of
evicting it, can run as a **fork** of the channel's window (GitHub issue
#51, [design.md](design.md#forks)). A fork is one LLM call: the main
turn's render of the window, up to a cut, plus one `user` instruction.

```python
from corvidae.fork import fork_agent, window_db_ids

class MyPlugin(CorvidaePlugin):
    @hookimpl
    def register_llm_stages(self, stages: dict[str, str]) -> None:
        stages["review"] = "chat"              # the stage the fork below names

    async def review(self, channel, rowid: int) -> str | None:
        agent = fork_agent(self.pm, "review")   # None unless agent.forks enables "review"
        conv = channel.conversation
        if agent is None or conv is None or rowid not in window_db_ids(conv.messages):
            # Live-window work skips on a miss, with a warning naming the
            # stage, channel, trace and what was missing (PRD §3.8.5).
            logger.warning("review skipped on channel %s: no fork or row %s not in the window", channel.id, rowid)
            return None
        try:
            result = await agent.fork(
                channel, "Pause here and review your last answer. Reply in JSON.",
                stage="review",
                extra_body={"response_format": {...}},
            )
        except Exception as exc:
            logger.warning("review skipped on channel %s: fork failed (%s)", channel.id, type(exc).__name__, exc_info=True)
            return None
        return result.text
```

- `Agent.fork(channel, instruction, *, stage, cut=None, extra_body=None)` returns
  `ForkResult(text, message, usage, model)`. `model` is the model the fork
  request was sent with (`LLMClient.request_model(body)`, so an
  `extra_body["model"]` override shows), for plugins that record how their
  output was made; `Agent.fork` always sets it, so it is None only when a
  `ForkResult` is built without it. `cut=None` forks
  the whole window;
  `cut=k` forks `messages[:k]` (use `snap_cut_to_tool_boundary` if `k`
  might split a tool cycle, or the prefix won't match the cache).
- `stage` is required, and must be a stage you registered with
  `register_llm_stages` (an unregistered name raises `ValueError`). The call
  goes to that stage's profile: the one the operator maps it to under
  `llm.stages`, else the default chat profile. Name the stage, never a
  profile.
- It sends the agent's tool schemas with `tool_choice: "none"` and the
  channel's runtime overrides, so the render matches the main turn; the
  prefix is cached only if the stage's profile is the main turn's, or
  differs from it in sampler-side keys. Don't pass `reasoning_effort` or
  anything else the chat template renders into the prompt in your own
  `extra_body`: that breaks the cached prefix. There is no fork budget key:
  the operator bounds a stage's reasoning and output with
  `reasoning_budget_tokens` and `max_tokens` in its profile's `extra_body`.
  Where a runtime override and the stage profile set the same key, the
  profile wins on a fork (the override is not sent); your `extra_body` beats
  the profile, and `tool_choice: "none"` beats everything (design.md,
  "Body precedence").
- It mutates nothing, persists nothing, fires no hooks and never runs
  tools: a reply with `tool_calls` (or a call written out as text) raises `ForkRefused`; no window or no
  client for the stage raises `ForkUnavailable`. Treat these as a miss: skip the live-window work and log it. Keep a non-fork path only for windowless work (seeding, imports), and for a range that has left the window or a stage whose output can't be skipped, as compaction does: for those use the view call below, not a prompt that describes the conversation.
- Set your own `stage` attribution around the call; `Agent.fork` adds
  `mode="fork"`. Tag any standalone call you keep `mode="standalone"`.
- Check containment yourself: a fork only makes sense while the rows your
  process is about are still in the window (`window_db_ids`).
- Read the instruction from static config via `resolve_prompt` (#411),
  not `resolve_tunable`, unless you want the agent to be able to rewrite it.
  Prompt keys accept a string (inline text), `{"file": path}`, a list of
  paths, or absent (built-in default). Resolve in `on_init` and
  `on_config_reload`; a failed reload keeps the previous value.
- Stage names are opaque strings. List yours in `agent.forks.stages` (or
  leave it null for all stages).

### The view call

When the work is about a range that is no longer in the live window, send the
agent's rendered view of it (PRD §3.8) rather than a standalone prompt. That
is a window (live, or rebuilt from `message_log`) rendered with
`build_fork_prompt` under the channel's system prompt, plus your instruction
as the last `user` message, sent with `view_call(client, channel_id,
messages, extra_body=...)` on a client you bind once
(`llm.get_client("<your stage>")`, a stage you registered with `register_llm_stages`). It sends no tools, raises `ForkRefused` on
a reply with tool calls (structured or written out), and tags the call
`mode="standalone"`; set your own `stage`, channel and trigger attribution
around it. Get the channel's system prompt, budget (`max_tokens`),
mid-conversation role and `reasoning_history` policy from
`view_agent(self.pm).view_inputs(channel_id)`, which returns a `ViewInputs`
(`view_agent` is None when no agent can supply one; it ignores
`agent.forks`). `view_rows` and `decodable_summaries` in
`corvidae.persistence` read the rows a reload would load under a summary.
Trim the rebuilt window to `max_tokens` with `ContextWindow.fit_to_budget`
and apply the policy, as the live loop does. Do not use the view for judgment
of an exchange: that must fork the live window (PRD §3.8.2). Memory's
`build_catchup_segments` is the worked example.

## Message gating and lifecycle

The `should_process_message` gate fires before a message enters the conversation. Implement it to filter messages by sender, rate limit, or apply channel muting:

```python
class FilterPlugin(CorvidaePlugin):
    @hookimpl
    async def should_process_message(self, channel, sender: str, text: str, trace_id: str | None) -> bool | None:
        if sender in self.muted_users:
            return False  # Reject
        return None  # No opinion; other plugins decide
```

When a message is admitted or rejected, plugins are notified via `on_message_admitted` or `on_message_rejected`. These fire exactly once per message. The `on_message_persisted` hook fires after the trace-originating message (the initial USER message or a standalone notification) has been written to the database, regardless of persistence outcome; it fires once per message in a coalesced burst, in arrival order, though only the last message's trace gets the turn (see Traces) — it receives the database rowid (nullable), the originating text, and the item's annotations dict:

```python
class LogPlugin(CorvidaePlugin):
    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        await self.log_event("message_admitted", trace_id, sender)

    @hookimpl
    async def on_message_persisted(self, channel, trace_id: str, rowid: int | None, text: str, annotations: dict) -> None:
        await self.log_event("message_persisted", trace_id, rowid, annotations)
```

## Channels

A channel is a `transport:scope` pair like `irc:#general` or `cli:local`. Each channel has its own `ContextWindow` initialized on first message.

```python
channel.id              # "irc:#general"
channel.transport       # "irc"
channel.scope           # "#general"
channel.conversation    # ContextWindow | None
channel.matches_transport("irc")  # True
```

`id` is the key and never derives from a name. A transport that knows a name a
reader could use sets it, for the model to read and for nothing else:

```python
channel.name            # "#general", or "thread merge-order in #general", or None
channel.sender_names    # {"408709225498542081": "schuyler"} — ids to display names
channel.addressed       # True / False: did this message address the agent? None: not known
```

All are optional. The names are data; `addressed` renders as the fixed words
`addressed: yes` or `addressed: no`, and None renders nothing. A transport
that sets `addressed` assigns it before every `on_message` call on that
channel and dispatches that channel's messages one at a time; core reads it
once, at admission. When any is set, the prompt carries a
system message `[metadata: sender: schuyler (id 408...); channel: #general (id irc:#general); addressed: yes]`
before the person's message, which is left exactly as the person typed it.
`build_prompt` renders it per person row and drops it when it repeats the
header of the row before, so a burst from one sender carries one. A transport
that sets none gets no header. `sender_names` is filled by the transport
before it calls `on_message`; core reads it once, at admission, and stores it
in the row's provenance record ([design.md](design.md#provenance-header-and-fence)).

Channels are created on-demand when messages arrive, or pre-registered in `agent.yaml`:

```yaml
channels:
  irc:#general:
    register_prompt: "This is #general; keep it brief."
    max_context_tokens: 16000
```

A transport can ship a default register for its channels by implementing `resolve_register_prompt(channel)`: open with `if not self.owns(channel): return None` and return the register text. `corvidae.channels._registers.shipped_register(*names)` joins fragments from `corvidae/channels/registers/`. The Agent asks only when no `register_prompt` is configured for the channel, and appends the answer after `agent.system_prompt`.

To add a section to the system prompt on every channel, or only on some, wrap `transform_system_prompt` instead; it sees the register whatever its source (see [Extension points](#extension-points)).

The `irc` transport is provided by `IRCPlugin` (registered via the `corvidae` entry point group). Its config block:

```yaml
irc:
  host: irc.libera.chat       # IRC server (default irc.libera.chat)
  port: 6667                   # IRC port (default 6667)
  nick: corvidae               # Bot nickname (default corvidae)
  tls: false                   # Use TLS (default false)
  channels: ["#general"]       # Channels to join
  message_chunk_size: 400      # Max UTF-8 bytes per IRC message (default 400)
```

The `signal` transport is provided by `SignalPlugin` (same entry point group). It speaks JSON-RPC to a `signal-cli daemon` that is already running — corvidae never starts one. Its config block:

```yaml
signal:
  socket: /run/corvidae/signal.sock   # signal-cli daemon socket (required)
  account: "+15550001111"             # the bot's own number (required)
  allow:                              # default-deny; E.164 or ACI
    - "+15551234567"
  message_chunk_size: 2000            # Max UTF-8 bytes per message (default 2000)
```

Omitting the block leaves the plugin loaded but inert. Signal channel scopes are always the sender's ACI (`signal:8f2c1c9a-…`), never a phone number — `sourceNumber` is absent under phone-number privacy, and a scope that can vanish would fork the channel's persisted history. Numbers in config are resolved forward to ACIs after the transport connects. See [signal-ops.md](signal-ops.md) for provisioning and [configuration.md](configuration.md) for the full key reference.

### `resolve_channel` for transport authors

Something other than an inbound message sometimes needs a channel: `SchedulerPlugin` fires a reminder after a restart, holding only the persisted id. It calls `registry.get(channel_id)`, then the `resolve_channel` hook, then `registry.get_or_create(transport, scope)` from the split id. If your transport creates channels with config a bare `get_or_create` would miss, implement the hook, and keep it default-free (the startup arg-binding guard, `tests/test_hook_arg_binding.py`, rejects a defaulted `channel_id`: pluggy would silently pass your default instead of the caller's id):

```python
@hookimpl
async def resolve_channel(self, channel_id: str) -> Channel | None:
    transport, sep, scope = channel_id.partition(":")
    if transport != "mytransport" or not sep or not scope:
        return None          # not ours: defer to the next implementation
    return self._channel_for(scope)   # the same path inbound messages take
```

`SignalPlugin` implements it so an ACI it has not seen since startup still gets its E.164 alias's `ChannelConfig`. When your transport owns the id but cannot apply its config yet (Signal before its first alias resolution), return None. The scheduler sees that a plugin with `transport` equal to the id's prefix implements the hook, so it holds the row and retries every few seconds rather than falling back to a default-configured channel. Don't return None for an id you own and will never be able to build: the row would retry forever. IRC and CLI don't need it; the registry fallback is already correct for them. Channels no transport owns (`internal:heartbeat`) also take the fallback, and since `TransportPlugin.send` delivers only to the transport that owns a channel, replies there go only to the log and `sessions.db`.

### Writing a transport

Subclass `TransportPlugin` (in `corvidae.hooks`), set `transport` to the name your channel ids start with, set `conversational = True` if a person on the other end can let an exchange wind down (a chat), and override the outbound methods your transport can display. They are plain methods, not hookimpls, and each takes the channel and an event from `corvidae.outbound`:

| Method | Event | Fields |
|---|---|---|
| `send_message` | `OutboundMessage` | `text`, `latency_ms` (None if unknown) |
| `send_thinking` | `OutboundThinking` | `text` |
| `send_tool_status` | `OutboundToolStatus` | `tool_name`, `status` (`"dispatched"` or `"completed"`), `args_summary`, `result_summary`, `tool_call_id` (None if unknown) |
| `send_progress` | `OutboundProgress` | `text` |

The base class implements the `send` hook once: it drops events for channels the transport does not own, then dispatches by event type. Methods you don't override are no-ops, which is how a chat transport skips reasoning and tool status. `TransportPlugin` also declares `depends_on = frozenset({"registry"})`.

```python
from corvidae.hooks import TransportPlugin
from corvidae.outbound import OutboundMessage

class MyTransport(TransportPlugin):
    transport = "mytransport"

    async def send_message(self, channel, event: OutboundMessage) -> None:
        await self.client.post(channel.scope, event.text)
```

`conversational` is a plain fact about the transport, default `False`. Plugins that tailor the system prompt to chats read it: with the `silence` plugin loaded, a conversational transport's channels get the silence instruction, and a transport left at `False`, third-party ones included, never does. IRC, Discord, Signal and Zulip set it; ACP, web and the CLI don't.

The filter is `owns(channel)`, which by default is `channel.matches_transport(self.transport)`. Override it only to narrow it: `WebPlugin` also requires that it is enabled. Inbound hooks are not filtered for you; open each inbound hookimpl with `if not self.owns(channel): return`.

### Transports with per-turn state

A transport that holds state for the length of a turn (Discord's and Signal's typing slots, ACP's pending prompt, the CLI's input prompt) releases it in `send_message`, which runs when a final is sent. Discord and Signal release every outstanding slot there, since one final can answer a coalesced burst (PRD §3.1.19). A final the gate vetoed, or an empty one, never gets there, so the transport also implements `on_agent_response` and releases the same state when `corvidae.hooks.final_was_unsent(channel)` is true. A sent final's row is a normal message, so it releases in `send_message` only: once per final either way. Three rules:

- Open with `if not self.owns(channel): return`. `on_agent_response` reaches every transport.
- Do the ownership check, `final_was_unsent` and the release synchronously, before the first `await`. Implementations run under `asyncio.gather` and the kernel may move on when a sibling raises.
- If `final_was_unsent` raises, log a warning and release nothing. A stuck indicator is better than a double release.

The `IRCPlugin` and web transports keep nothing per turn, so they don't need it.

### ACP transport

`AcpPlugin` (`corvidae` entry point `acp`) speaks Agent Client Protocol v1
over stdio when launched via `corvidae acp`. It does **not** connect under
`serve` / `cli` unless `config["_acp_mode"]` is set (only the `acp`
command does that). Startup misconfiguration — a missing config file, an
invalid YAML document, or an invalid config value — exits non-zero with a
single line naming the config path (`Config error:` from Runtime for an
unreadable or malformed YAML file, `Error:` otherwise), so an editor shows
the diagnosis instead of a traceback.

Optional YAML block (all keys optional today; presence is enough to
document intent):

```yaml
# acp:
#   # Reserved for future agentCapabilities / logging hints.
#   # Mode is not enabled by YAML alone — use `corvidae acp`.
```

For step-by-step setup in Zed and bb, see [acp-clients.md](acp-clients.md).

**bb smoke (`customAcpAgents`):**

1. `uv sync --extra acp --extra dev --extra cognition` (add `--extra discord` if you use it)
2. Point bb at Corvidae, e.g. command `uv` with args
   `run --directory /path/to/corvidae corvidae acp` (or a wrapper script).
3. `bb thread spawn --provider acp-corvidae --prompt "…" ` (provider name
   per your bb config).
4. Confirm a reply and any tool rows in the timeline.

Channel ids are `acp:<sessionId>` and stay stable across `session/load`
and `session/resume` (history is keyed on that id in `sessions.db`).
Advertised capabilities: `loadSession`, plus `sessionCapabilities.list` /
`resume` / `close`. `session/delete` is not advertised. Disable the
plugin with `plugins.disabled: [acp]` if needed.

**Client input this transport rejects or ignores.** `session/new`,
`session/load`, and `session/resume` require `cwd` to be an existing
directory; anything else returns JSON-RPC `invalidParams`. The `cwd` is
recorded for `session/list`, but tools still run in the process working
directory — a differing `cwd` is logged at WARNING rather than acting as a
chdir. `mcpServers` and `additionalDirectories` are accepted and ignored
with a WARNING; configure `mcp:` in `agent.yaml` for MCP servers. Prompts
must carry at least one text block: image, audio, and resource blocks are
logged and dropped, and a prompt with no text at all returns
`invalidParams` instead of an empty turn. An unknown session id is
`resourceNotFound` for `prompt`, `load`, `resume`, and `close` alike.

A prompt always ends. If the client connection dies while the final
message is being delivered, the turn still completes with `end_turn` and
the failed delivery is logged at WARNING: the reply is in `message_log`
and a later `session/load` replays it (#99). If the turn never starts
(the enqueue fails), its future is cleared so a later prompt cannot
inherit it. A second `session/prompt` on a session that already has one
in flight is `invalidRequest`; the first prompt is left untouched, and a
client that wants to steer sends `session/cancel` first (#93; the policy
and its revisit trigger are in `docs/prd.md`).

The ACP loop owns stdin/stdout, so the process stops when the loop ends:
a client disconnect initiates a normal shutdown, and a loop crash is
logged and then exits non-zero, so a client sees a failed provider
instead of a live process with a dead transport.

## Registration order

Plugins are loaded via `pm.load_setuptools_entrypoints("corvidae")`. The entry point loading order is non-deterministic — it is not guaranteed to match any specific sequence. The following plugins are registered:

```
registry          (ChannelRegistry)     — explicit, before entry points load
persistence       (PersistencePlugin)   — entry point
jsonl_log         (JsonlLogPlugin)      — entry point
shell_tools       (ShellToolPlugin)     — entry point
file_tools        (FileToolsPlugin)     — entry point
web_tools         (WebToolsPlugin)      — entry point
cli               (CLIPlugin)           — entry point
irc               (IRCPlugin)           — entry point
signal            (SignalPlugin)        — entry point
acp               (AcpPlugin)           — entry point (inert unless `_acp_mode`)
task              (TaskPlugin)          — entry point
subagent          (SubagentPlugin)      — entry point
mcp               (McpClientPlugin)     — entry point
llm               (LLMPlugin)           — entry point
compaction        (CompactionPlugin)    — entry point
thinking          (ThinkingPlugin)      — entry point
runtime_settings  (RuntimeSettingsPlugin) — entry point
tools             (ToolCollectionPlugin) — entry point
memory            (MemoryPlugin)        — entry point (cognition extra)
memory_tools      (MemoryToolsPlugin)   — entry point (cognition extra)
funnel            (FunnelPlugin)        — entry point (cognition extra)
goals             (GoalsPlugin)         — entry point (cognition extra)
agent             (Agent)               — entry point
config_watcher    (ConfigWatcherPlugin) — entry point
```

`ChannelRegistry` is constructed and registered explicitly before entry points load. Its `agent_defaults` attribute and channel config are populated from config before the `on_init` broadcast runs.

**Tool collection:** `ToolCollectionPlugin.on_start` uses `@hookimpl(trylast=True)` so it fires after all other `on_start` hooks, then calls `register_tools` to collect tools from every registered plugin. The non-deterministic loading order does not affect tool collection because `trylast=True` ensures `ToolCollectionPlugin.on_start` always runs last in the broadcast.

**Startup order:** `Runtime.start()` broadcasts `on_init`, then `on_start`. `ToolCollectionPlugin.on_start` runs last (trylast) to collect tools. `Runtime.start()` then calls `agent.on_start(config=config)` explicitly after the broadcast completes. `Agent.on_start` does not have `@hookimpl` — it is called only by `Runtime.start()`. A transport may deliver inbound messages from its own `on_start`: `Agent.on_message` holds them until the Agent has started (the awaiting `on_message` call simply takes longer to return) rather than dropping them. If the Agent stops before starting, the held call returns without admission and a WARNING names the channel and trace id. Adding `@hookimpl` to `Agent.on_start` would cause double initialization.

**Shutdown order:** `Runtime.stop()` calls `agent.on_stop()` first (drains queues), then `pm.ahook.on_stop()` to tear down all other plugins, then stops the idle tracker and closes the kernel clock, which cancels any timers left running.

**Failed startup:** if `Runtime.start()` raises an exception (a plugin's `on_init` or `on_start`, say; a config error exits with one `Config error:` line and no traceback, but goes through the same cleanup once plugins have loaded), `Runtime.run` runs the `on_stop` broadcast, stops the idle tracker and closes the clock, under the same bound as a normal shutdown (`daemon.shutdown_timeout`, 3 s by default), then re-raises so the process exits non-zero. `agent.on_stop()` is skipped. A plugin's `on_stop` must therefore cope with resources its `on_start` never opened.

`ChannelRegistry`, the `Clock` (`"clock"`) and the `IdleTracker` (`"idle"`) are registered by `Runtime.start()` before the entry points load, so any plugin can depend on them. See [Acting on time and quiet](#acting-on-time-and-quiet).

## Plugin disable

Disable entry-point plugins via the `plugins.disabled` config key:

```yaml
plugins:
  disabled:
    - compaction          # disable the built-in compaction plugin
```

Names must match the entry-point name from `[project.entry-points.corvidae]` in `pyproject.toml`. `Runtime.start()` calls `pm.set_blocked(name)` for each entry before loading entry points. Blocked plugins are not instantiated.

Only entry-point plugins can be disabled this way. Manually registered plugins (e.g., `ChannelRegistry`) cannot be blocked via config.

This is a startup-time mechanism. It does not interact with hot-reload.

Right after entry points load, `Runtime.start()` logs a `"plugins loaded"`
line with `loaded` (the sorted names that actually registered, not the
blocked ones) and
`blocked` (the `plugins.disabled` list from config, verbatim). Compare the
two to confirm a disable took effect. A name that matches no installed
plugin (a typo, a retired name, a plugin from an extra that isn't
installed) blocks nothing, and `Runtime.start()` logs one `WARNING` for it
naming `plugins.disabled`, the value, and the installed entry-point names.

## Hook exception safety

Broadcast hook calls from `Agent` are wrapped in `try/except`. If a plugin raises an exception from `before_agent_turn`, `after_persist_assistant`, `on_agent_response`, or `send`, the exception is logged at WARNING or ERROR level and processing continues. Plugins do not need to catch their own exceptions to protect the queue consumer.

For `firstresult=True` hooks (`resolve_compaction`, `resolve_conversation`, `resolve_llm_error`), pluggy propagates the first handler's exception immediately — the chain stops and no further handlers run. The call sites in `Agent` handle these as follows:

- `resolve_compaction` — wrapped in try/except; failure is logged at WARNING and the turn continues without compaction. Whatever an implementation returns or raises, the kernel then enforces `max_context_tokens` itself, dropping the oldest rows from the in-memory window if it is still over budget.
- `resolve_conversation` — not wrapped; an exception propagates to `_process_queue_item`, which will fail the queue item.
- `resolve_llm_error` — wrapped in its own try/except inside `_run_turn`'s error handler; failure is logged at WARNING and the default error message is sent instead.

For `should_process_message` (broadcast, resolved with `resolve_reject_wins`), exceptions propagate to the call site.

For `should_send_response`, the Agent catches any exception, logs it at WARNING and sends the emission anyway (fail open). apluggy gathers a broadcast's implementations concurrently without `return_exceptions`, so one raising implementation discards every sibling's verdict for that emission; once any plugin registers a `wrapper=True` implementation on the hook (as `outbound_cap` does, to count after the combined verdict), apluggy awaits the implementations one after another instead, and a raising one stops the rest from running. Either way a gate that can fail should catch and log its own errors and return None, and a slow gate adds its latency to every emission when the hook is wrapped.

For `transform_display_text`, `transform_system_prompt` and `transform_tool_result` (wrapper chain hooks), pluggy propagates exceptions from wrapper implementations to the call site. `transform_display_text` is wrapped in try/except in `_resolve_display_text`, and `transform_system_prompt` in `channel_system_prompt`.

## Output gating

`should_send_response(channel, text, emission, trace_id, annotations)` is the outbound mirror of `should_process_message`: return `False` to withhold, `None` for no opinion. `emission` is `final` (the reply), `progress` (text alongside dispatching tool calls), `thinking` (reasoning) or `error` (the LLM-error fallback); treat anything else as no opinion. `text` is exactly what would be sent. Where each fires and what a veto does to the stored row and the window is in [design.md](design.md#the-output-gate); in short, a vetoed final is kept in the window as `MessageType.UNSENT` and never sent, and any other veto suppresses only that send.

Origin is not a kernel parameter: read `annotations["origin"]` if your policy depends on it, and record the traces you see in `on_message_admitted` if you need to recognise user exchanges.

```python
class QuietHoursPlugin(CorvidaePlugin):
    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        # Hold back self-initiated finals overnight; never a reply to a person.
        if emission == "final" and annotations.get("origin") == "heartbeat" and self.is_night():
            return False
        return None
```

If you write a transport with history replay, skip `message_type == "unsent"` rows, and use `corvidae.context.unsent_marker_emission(content)` to hold back progress or thinking that a marker says was suppressed. Cognition's `GatePlugin` is the reference implementation ([cognition.md](cognition.md#gateplugin)).

## Async considerations

All broadcast hooks are `async`. Corvidae uses apluggy's `pm.ahook.*` for async dispatch.

For hooks with `firstresult=True` (`resolve_compaction`, `resolve_conversation`, `resolve_llm_error`), `pm.ahook.<hook>(...)` returns a single value (or None) directly:

```python
result = await pm.ahook.resolve_conversation(channel=channel)
```

For `transform_display_text` and `transform_tool_result`, use `@hookimpl(wrapper=True)` to participate in the wrapper chain. The seed plugin returns the input value unchanged as the innermost result; each wrapper receives that result, optionally transforms it, and returns the modified value:

```python
from corvidae.hooks import hookimpl

@hookimpl(wrapper=True)
def transform_display_text(self, **kwargs):
    result = yield
    if result is not None:
        return my_transform(result)
    return result
```

For the `should_process_message` broadcast hook, call `pm.ahook.<hook>(...)` and pass the result list to `resolve_reject_wins`:

```python
from corvidae.hooks import resolve_reject_wins

results = await pm.ahook.should_process_message(channel=channel, ...)
result = resolve_reject_wins(results)
```

`@hookimpl(tryfirst=True)` and `@hookimpl(trylast=True)` markers are respected by apluggy's dispatch and affect execution order. For `firstresult=True` hooks, `tryfirst` handlers run before default-priority handlers, which run before `trylast` handlers. The chain stops at the first non-None return.

The async-only rule extends to **tool functions** as well as hooks: `Tool.from_function()` enforces it at registration time, and any blocking I/O inside a tool must be wrapped with `asyncio.to_thread()`. See [Avoid blocking the event loop](#avoid-blocking-the-event-loop).

## Stock tools

These tools are registered by built-in plugins. They are available to the LLM in every standard Corvidae deployment.

A tool returns its whole result: `execute_tool_call` does not cut it. A result longer than `tools.max_result_chars` (default 100,000 characters) is cut when the Agent receives it, and its full text is spilled to a file through the kernel spill service. The model sees the head behind a kernel-built cut line naming the file, where the channel has a tool that declares `reads_spill_files`. A tool that can read spill files sets `reads_spill_files = True` on its function. Override the cap via config:

```yaml
tools:
  max_result_chars: 100000  # read by ToolCollectionPlugin during on_start
```

The legacy key `agent.max_tool_result_chars` is still accepted but deprecated; a warning is logged at startup when it is present.

### Built-in tool plugins

Three plugins register the built-in tools, each from its own entry point:

| Plugin | Entry point | Tools |
|--------|-------------|-------|
| `ShellToolPlugin` | `shell_tools` | `shell`, `task_pipeline` (one capability: PRD §3.5.3) |
| `FileToolsPlugin` | `file_tools` | `read_file`, `write_file` |
| `WebToolsPlugin` | `web_tools` | `web_fetch`, `web_search` |

Each can be disabled by name in `plugins.disabled` and the others keep
working. They replaced `CoreToolsPlugin` (`core_tools`), with no alias.

| Tool | Parameters | What it does |
|------|------------|--------------|
| `shell` | `command: str` | Runs a shell command and returns combined stdout/stderr. Times out after `tools.shell_timeout` seconds (default 30). Returns `"(no output)"` if the command produces none. Non-zero exit codes are appended to the output. |
| `read_file` | `path: str`, `offset: int \| None`, `length: int \| None`, `start_line: int \| None`, `end_line: int \| None` | Reads a file and returns its text content. With `offset` or `length` (counted in characters) it returns that range behind a header naming the range, the file's total length and the next offset, e.g. `[characters 0-99932 of 300000 in /path/f.json; next offset 99932]`. With `start_line` or `end_line` (1-based, inclusive) it returns those lines behind a header naming them, the file's line count and the next start line, e.g. `[lines 10-40 of 900 in /path/f.log; next start_line 41]`; one call takes one kind of range, never both. Either range is clamped to a page sized to the calling channel's budget (`tools.read_file_window_fraction`, never above `tools.max_result_chars`) and counted against it, so a page of dense text is smaller than the same page of prose; a line range backs up to the last whole line, and only a first line longer than a page is cut, with the header giving the character offset to continue from and the start line where line paging resumes. An unranged read of a file longer than a page returns the first page with its header. Returns an error string for missing files, directories, unreadable files, an offset at or past the end, a `start_line` past the last line, a negative offset, non-positive length or line number, an `end_line` before `start_line`, or an unranged read of a file larger than `tools.max_file_read_bytes` bytes (default 1 MB). |
| `write_file` | `path: str`, `content: str` | Writes `content` to `path`, creating parent directories as needed. Returns a confirmation with the byte count, or an error string on failure. |
| `web_fetch` | `url: str` | Fetches a URL via HTTP GET and returns the response body as text. Times out after `tools.web_fetch_timeout` seconds (default 15). Truncates responses at `tools.web_max_response_bytes` characters (default 50,000, independent of `MAX_TOOL_RESULT_CHARS`). A network failure names the URL and the failure class. |
| `web_search` | `query: str`, `max_results: int` (optional) | Searches the web via DuckDuckGo and returns formatted results with titles, URLs, and snippets. Defaults to 8 results per page. |
| `task_pipeline` | `definition: str` | Executes a task graph defined in YAML or JSON. The definition must contain a `tasks` key with a list of objects, each having `name`, `command`, and optionally `depends_on`. Tasks run in topological order; failed tasks block their dependents. Returns a status summary; a task failure names the task and the command. The whole pipeline is cut at `tools.call_timeout` (600 s by default); set `tools.timeouts.task_pipeline` for longer runs. |

**`tools` config** (the three tool plugins each read only their own keys; `call_timeout` and `timeouts` are read by `ToolCollectionPlugin`):

```yaml
tools:
  call_timeout: 600                  # seconds bound on every tool call (0 or null disables)
  timeouts: {task_pipeline: 3600}    # per-tool override of call_timeout
  shell_timeout: 30                  # seconds before shell command is killed
  web_fetch_timeout: 15              # seconds before web request is aborted
  web_max_response_bytes: 50000      # response body truncation limit
  web_search_max_results: 8          # max results returned per search query
  max_file_read_bytes: 1048576   # file size limit for unranged reads (1 MB)
```

### SubagentPlugin tools

Registered by `SubagentPlugin` (entry point name: `subagent`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `subagent` | `instructions: str`, `description: str`, `_ctx: ToolContext` | Enqueues a background task that runs a full agent loop with `instructions` as its prompt. Returns immediately with the enqueued task ID. Results are delivered via `on_notify` when the task completes. |

`subagent` requires `ToolContext` (injected automatically — not an LLM parameter). It excludes itself from the subagent's tool registry; nesting through other tools is bounded by `tools.max_agent_depth`. The subagent calls the model as the `subagent` stage, a `chat` stage, so it uses `llm.stages.subagent`, else the default chat profile (`llm.main` by default).

### RuntimeSettingsPlugin tools

Registered by `RuntimeSettingsPlugin` (entry point name: `runtime_settings`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `set_settings` | `settings: dict` | Updates per-channel runtime settings. Accepts LLM inference parameters, framework parameters and dotted plugin tunables; any other key rejects the call. Pass `null` for a key to revert it to the static config value. Returns the current overrides after the update. |

### SchedulerPlugin tools

Registered by `SchedulerPlugin` (entry point name: `scheduler`). All three act on `_ctx.channel` only and take no channel argument, so a speaker on one channel cannot schedule a message to another ([threat-model.md](threat-model.md)).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `remind_me` | `when: str`, `text: str` | Creates a one-shot reminder on the calling channel. `when` is a relative duration (`20m`, `2h`, `3d`, `1h30m`), an ISO 8601 time (no offset means `scheduler.timezone`), or `HH:MM` for the next occurrence; a bare number is refused as ambiguous. The tool does the date arithmetic and returns the resolved absolute time, e.g. `Reminder #12 set for Fri 2026-10-23 14:22 PDT (in 20m).` Refuses past times, empty text, text over 2000 characters, and more than `scheduler.max_reminders_per_channel` pending reminders. |
| `list_reminders` | — | Pending rows on the calling channel, earliest first, with ids and due times. Operator jobs targeting the channel are listed as read-only. |
| `cancel_reminder` | `id: int` | Cancels a tool-created reminder on the calling channel. Refuses operator jobs; another channel's row reads as not found. |

## Bundled plugins

The kernel's plugins follow. The cognition plugins (`MemoryPlugin`,
`MemoryToolsPlugin`, `FunnelPlugin`, `AppraisalPlugin`, `CritiquePlugin`,
`OutcomeLogPlugin`, `GatePlugin`, `HeartbeatPlugin`) ship in the `cognition` extra and are described in
[cognition.md](cognition.md), which has reference sections for the memory,
funnel and outcome-log plugins.

These plugins ship with Corvidae and are registered via entry points as part of the default daemon. They can be omitted if the application does not need their functionality; each degrades gracefully when absent.

### McpClientPlugin (`corvidae/mcp_client.py`)

Connects to external [MCP](https://modelcontextprotocol.io/) servers and
exposes their tools to the agent loop. Entry point name: `"mcp"`.

Implements three hooks:

- `on_start` — connects to all servers listed under `mcp.servers`
  in `agent.yaml`, fetches their tool lists, and builds cached `Tool` instances.
  Runs in the broadcast; completes before `ToolCollectionPlugin.on_start`
  (trylast) calls `register_tools`.
- `register_tools` — appends the cached tools to `tool_registry`.
- `on_stop` — signals each server's owner task to close its session and
  transport (enter and exit happen in the same task), all servers at once.

**Config:**
```yaml
mcp:
  servers:
    filesystem:
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
      env:                     # optional: extra env vars for the subprocess
        NODE_ENV: production
      tool_prefix: "fs"        # optional: prefix for tool names; defaults to server name
      timeout_seconds: 30      # optional: per-call timeout; default 30 (a value above tools.call_timeout is cut at call_timeout)
      connect_timeout_seconds: 30  # optional: startup handshake bound; default 30

    remote_api:
      transport: sse
      url: "http://localhost:8001/sse"
      timeout_seconds: 60

    wiki:
      transport: streamable_http
      url: "http://localhost:8090/mcp"
      headers:
        Authorization: "Bearer ${OTTERWIKI_MCP_TOKEN}"
```

`streamable_http` is MCP's current HTTP transport; `sse` is the older one.
Both take an optional `headers` map, sent on every request. `${VAR}` in a
header value is replaced from the daemon's environment at connect time, so
a token never has to sit in `agent.yaml`. A variable that is unset, empty, or holds a newline or NUL
skips that server with one WARNING naming the server, header and variable,
and the server is not queued for the retry below (the environment
can't change without a restart). The same holds for every config error:
unknown `transport`, missing `command`/`url`, non-positive
`connect_timeout_seconds`. Header values never reach the log. The
plugin builds the HTTP client itself (`create_mcp_http_client`) and closes
it with the server.

Tool names are prefixed to avoid collisions: a server named `filesystem`
produces tools like `filesystem__read_file`. Set `tool_prefix: ""` to
disable prefixing. If two servers produce the same tool name, the first
wins and the duplicate is skipped with a WARNING. A server that does not
complete its handshake within `connect_timeout_seconds` is skipped with a
WARNING and its subprocess terminated; startup continues without it. A
server that fails to connect at startup is kept and retried by a background
loop on the kernel clock (the plugin declares `depends_on = {"clock"}`): 10 s,
doubling to 300 s, reset on success, stopped in `on_stop`. An MCP tool call
also triggers a retry unless one is already running. A successful retry adds
its tools to the cached list and the shared registry; a failed retry logs a
WARNING naming the server, its endpoint, the exception type and the next delay, and leaves
the current tool call untouched. A server that connects at startup also
rebuilds the registry at the end of the plugin's `on_start`, because the
tools plugin may have collected before the connect finished (#405). A
reconnect rebuilds the registry too,
and the running Agent picks the tools up on its next turn (#109); a plugin
that calls `ToolCollectionPlugin.rebuild_registry` at runtime reaches the
Agent the same way.

**Without this plugin:** MCP servers cannot be used. No tools are affected
if the `mcp:` config key is absent — the plugin no-ops.

### ThinkingPlugin (`corvidae/thinking.py`)

Strips `<think>...</think>` blocks and `reasoning_content` from LLM
output. Entry point name: `"thinking"`.

Implements three hooks, driven by the resolved `reasoning_history`
(`drop` | `exchange` | `keep`):

- `before_agent_turn` — applies the policy to the whole in-memory window
  before each LLM call: `drop` strips every assistant message's
  `reasoning_content`, `exchange` only those before the last `role: user`
  message, `keep` none.
- `after_persist_assistant` — under `drop`, strips the just-appended
  message. The DB copy is already written; this only affects subsequent
  prompt builds.
- `transform_display_text` (`@hookimpl(wrapper=True)`) — receives the
  chain result and calls `strip_thinking` on it unconditionally. Returns
  the stripped string. Operates as a sync wrapper; does not take
  `channel`, `text`, or `result_message` parameters directly.

**Without this plugin:** `<think>` blocks pass through to the channel
verbatim, and `reasoning_content` remains in in-memory history
regardless of the `reasoning_history` config value.

### CompactionPlugin (`corvidae/compaction.py`)

Compacts conversation history when it approaches the channel's
`max_context_tokens` limit. Entry point name: `"compaction"`.

Implements one hook:

- `resolve_compaction` (`trylast=True`) — fires before each LLM call. Checks whether the token estimate exceeds `compaction_threshold * max_tokens`. If so, and if the conversation has more than `min_messages_to_compact` messages, summarizes older messages to fit within `compaction_retention * max_tokens`, moving the cut off any tool cycle that straddles it where that still leaves a useful compaction (design.md, CompactionPlugin step 3). With `agent.forks` on for `compaction`, the summary comes from a fork of the dropped prefix (`agent.compaction_fork_prompt`); otherwise, or if the fork fails, from the same render sent on the compaction stage's profile (`llm.stages.compaction`, else the default chat profile) with no tools (the view call); if that fails too, the over-budget fallback applies. Fires `before_compaction` before truncating and `on_compaction` after. Returns True to stop the `firstresult` chain. Runs after any third-party handlers at default priority.

Token counting uses tiktoken (cl100k_base encoding) via `count_tokens()` in `corvidae/context.py`. When tiktoken is unavailable, a character-based fallback is used (3.5 chars per token). `agent.chars_per_token` is deprecated and only takes effect in the fallback path.

**Config:**
```yaml
agent:
  compaction_threshold: 0.8      # compact when token estimate exceeds this fraction of max_context_tokens
  compaction_retention: 0.5      # retain this fraction of max_context_tokens after compaction
  min_messages_to_compact: 5     # skip compaction if conversation has this many messages or fewer
  # chars_per_token: 3.5         # deprecated; only used when tiktoken is unavailable
```

**Without this plugin:** conversations grow without bound. The LLM will
receive an error from the API when the context limit is exceeded.

### PersistencePlugin (`corvidae/persistence.py`)

Opens the SQLite database, runs schema migrations, and sets the journal mode.
Entry point name: `"persistence"`. Implements `on_start`,
`on_stop`, `resolve_conversation`, `persist_message`, and `on_compaction`.

**Config:**
```yaml
daemon:
  session_db: sessions.db       # path to SQLite database file (default "sessions.db")
  sqlite_journal_mode: wal      # SQLite journal mode (default "wal");
                                # allowed values: delete, truncate, persist, memory, wal, off
```

**Without this plugin:** `resolve_conversation` returns no history; conversation
history is not persisted across restarts.

### JsonlLogPlugin (`corvidae/jsonl_log.py`)

Writes an append-only JSONL log of conversation events alongside the SQLite
store. Each `persist_message` and `on_compaction` call produces one
JSON line in a per-channel file. Entry point name: `"jsonl_log"`.

Implements three hooks:

- `on_start` — reads `daemon.jsonl_log_dir` and creates the directory.
  If the key is absent the plugin is a no-op for the rest of its lifetime.
- `persist_message` — writes a record with fields `ts`, `channel`,
  `type` (message type string), and `message` (the untagged message dict).
- `on_compaction` — writes a record with `type: "summary"` and the untagged
  summary dict.

File names are derived from the channel ID with `/` and `:` replaced by `_`.
Files are opened in append mode and kept open until `on_stop`. Each is capped
by `logging.jsonl.max_bytes` and `backup_count` (default 10 MiB x 5, as for
`logging.file`): a write that would pass the cap first rotates the file to
`<name>.1` and shifts older ones up (`corvidae/jsonl_rotate.py`).

**Config:**
```yaml
daemon:
  jsonl_log_dir: logs/   # path relative to the config file; omit to disable
```

**Without this plugin:** no JSONL log is written. Conversation history is
still persisted in SQLite by `PersistencePlugin`.

### MetricsPlugin, UsageLogPlugin, MetricsJsonlPlugin (`corvidae/metrics.py`)

The observability sinks. All three are fail-soft: a metering failure is
logged with a traceback and never breaks an LLM call.

- **`MetricsPlugin`** (entry point `"metrics"`) — consumes `on_llm_response`
  and emits `on_metrics` events: `llm.tokens.prompt`, `llm.tokens.completion`,
  `llm.tokens.total` (each skipped if the usage dict lacks it),
  `llm.latency_ms` (always), and `llm.errors` (value 1.0, on terminal
  failure). Tags: `{"profile", "model", "stage", "channel"}` — stage and
  channel come from the call's attribution snapshot.
- **`UsageLogPlugin`** (entry point `"usage_log"`, depends on
  `"persistence"`) — consumes `on_llm_response` and writes one row per LLM
  call into the `usage_log` SQLite table (in the persistence plugin's
  database): timestamp, request id, profile, model, stage, channel id,
  trace id (NULL for calls outside any trace), trigger,
  mode (`fork`/`standalone` for stages that can fork, else NULL), token
  counts, cached tokens, llama-server's `cache_n` and `prompt_n` (prompt
  tokens reused from cache vs prefilled), latency (the final attempt),
  and error; the call's `started_at`, `ended_at`, `total_ms` (including
  retries) and `attempts`; the request's `message_count` and
  `tool_count`; the channel's `max_context_tokens` (found through the
  optional `"registry"` plugin; NULL when unknown); and llama-server's
  `prompt_ms` and `predicted_ms` (prefill and decode time of the final
  attempt, NULL when the backend reports none). Subtracting those two
  from `latency_ms` approximates the time a call queued in the server.
  The table is created lazily on first use (`CREATE TABLE IF NOT EXISTS`).
- **`MetricsJsonlPlugin`** (entry point `"metrics_jsonl"`) — consumes
  `on_metrics` and appends one JSON line per event
  (`{"ts", "name", "value", "tags"}`) to the file named by
  `daemon.metrics_jsonl`, rotated by the same `logging.jsonl.*` caps as the
  transcripts. A complete no-op when the key is unset.

**Config:**
```yaml
daemon:
  metrics_jsonl: metrics.jsonl   # path relative to the config file; omit to disable
```

**Without these plugins:** the `on_llm_request`/`on_llm_response` hooks
still fire from LLMPlugin's observer; nothing records them.

### LLMPlugin (`corvidae/llm_plugin.py`)

Owns the `LLMClient` instance lifecycle. Entry point name: `"llm"`. Other plugins retrieve clients via
`get_dependency(pm, "llm", LLMPlugin)`.

Implements these lifecycle hooks:

- `on_start` — broadcasts `register_llm_stages` to collect the model-calling
  stages, parses the `llm:` section (`parse_llm_config`: profiles, `extends`,
  `llm.defaults`, `llm.stages`), checks the stage map against the registered
  stages, creates an `LLMClient` for each profile, starts their aiohttp
  sessions, and logs where each stage resolves (`llm stages resolved: ...`).
- `on_stop` — closes all aiohttp sessions.
- `on_config_reload` — swaps no client. A reload that changes `llm:` logs a
  warning, and the change applies on restart; nothing under `llm:` is
  hot-reloaded, `llm.main` included.

Each created client gets a hook observer injected that fires
`on_llm_request`/`on_llm_response` (with the client's profile and the current
attribution snapshot) around every chat call. `corvidae/llm.py` itself
stays free of plugin-system imports — the observer seam is a plain object
with async `request`/`response` methods, defined in `llm_plugin.py`.

Each key under `llm:` is a **profile**. A plugin that calls a model owns one
or more **stages** and names them in `register_llm_stages`:

```python
@hookimpl
def register_llm_stages(self, stages: dict[str, str]) -> None:
    stages["summarize"] = "chat"       # or "embedding"
```

A stage resolves to the profile the operator maps it to under `llm.stages`,
else to its kind's default profile (`llm.defaults`; `chat` to `main` and
`embedding` to `embedding` when omitted). There are no fallback chains, and
a plugin never names a profile. Then:

```python
llm = get_dependency(self.pm, "llm", LLMPlugin)
client = llm.get_client("summarize")   # None if the stage resolves to no profile
main = llm.get_client()                # the main-turn client (default chat profile)
settings = llm.profile_config("summarize")  # the stage's resolved profile settings, read-only
```

`get_client(stage)` raises `ValueError` for a name no plugin registered, and
profile names count: `get_client("main")` raises, so use `get_client()`.
Resolve the client when you call it, as the stages in this repo do, not in your
own `on_start`: hooks run concurrently, so `LLMPlugin` may not have started
yet. The `main_client` and `background_client` properties and
`embedding_dimensions` are gone.

**Config:**
```yaml
llm:
  main:                        # required unless llm.defaults.chat names another profile
    base_url: https://api.openai.com/v1
    model: gpt-4o
    api_key: sk-...            # optional; can also use environment variable
    extra_body: {}             # optional: extra fields merged into request body
    max_retries: 3             # optional (default 3)
    retry_base_delay: 2.0      # optional (default 2.0)
    retry_max_delay: 60.0      # optional (default 60.0)
    timeout: 120               # optional: request timeout in seconds
  embedding:                   # optional — the embedding stage gets no client without it
    kind: embedding            # required for an embedding profile
    base_url: http://localhost:8082/v1
    model: nomic-embed-text
    dimensions: 768            # required
  # stages: {compaction: main}   # optional: map a stage to a defined profile
```

See [`llm`](configuration.md#llm--llm-backend) for `kind`, `extends`,
`defaults` and `stages`.

**Without this plugin:** `Agent` and `CompactionPlugin` cannot create LLM
clients and will raise `RuntimeError` during startup.

### TaskPlugin (`corvidae/task.py`)

Owns the `TaskQueue` and delivers task results via the `on_notify` hook.
Entry point name: `"task"`. If that broadcast raises for a tool task, it
logs the failure and asks the Agent (`deliver_tool_error`) to queue an
error tool result instead, unless the real result was already queued, so
the tool cycle still finishes. The Agent's own no-task-queue error path
falls back to `deliver_tool_error` the same way.

**Config:**
```yaml
daemon:
  max_task_workers: 4       # concurrent task workers (default 4)
  completed_task_buffer: 100  # number of completed task records to retain in memory
```

**Without this plugin:** tool calls that return asynchronously (e.g., `subagent`)
cannot complete. `Agent` logs an error when tool dispatch is attempted
without a `TaskQueue`.

### RuntimeSettingsPlugin (`corvidae/tools/settings.py`)

Registers the `set_settings` tool, which lets the agent update per-channel
LLM inference parameters and framework settings at runtime. Entry point name: `"runtime_settings"`.

Implements one hook:

- `register_tools` — appends the `set_settings` tool to `tool_registry`.

The `set_settings` tool accepts a `settings` dict. Supported keys include
LLM inference parameters (`temperature`, `top_p`, `top_k`,
`frequency_penalty`, `presence_penalty`, `max_tokens`, `seed`), framework
parameters (`max_turns`, `max_context_tokens`, `reasoning_history`),
and dotted plugin tunables (`critique.*`, `appraisal.*`; see
configuration.md). Pass `null` for a key to clear the override and revert to
the static config value. `system_prompt` is always blocked. Additional keys
can be blocked via `agent.immutable_settings` in `agent.yaml`. Any other
undotted key (`messages`, `model`, `tools`, …) rejects the whole call; only
the inference parameters reach the LLM request body. See
[threat-model.md](threat-model.md), finding 1.

**Config:**
```yaml
agent:
  immutable_settings: [temperature]   # keys the agent cannot change at runtime
```

**Without this plugin:** the `set_settings` tool is not available. Channel
settings can only be configured statically in `agent.yaml`.

### ToolCollectionPlugin (`corvidae/tool_collection.py`)

Collects tools from all registered plugins and owns the `ToolRegistry`.
Entry point name: `"tools"`.

Implements one hook:

- `on_start` (trylast=True) — calls the sync `register_tools` broadcast after
  all other `on_start` hooks have completed. Builds a `ToolRegistry` from the
  collected items, then drops any tool named in `tools.disabled` — the
  single chokepoint every registered tool passes through, so there are no
  per-plugin opt-outs. Reads `tools.max_result_chars` (or the deprecated
  `agent.max_tool_result_chars`): the length past which a tool result is cut
  and spilled.

`Agent` retrieves the registry via
`get_dependency(pm, "tools", ToolCollectionPlugin)`. The `trylast=True`
marker guarantees all tool providers have fully initialized before collection
runs.

Other plugins can access the registry:
```python
from corvidae.hooks import get_dependency
from corvidae.tool_collection import ToolCollectionPlugin

tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
registry = tools_plugin.get_registry()
tools_dict, schemas = tools_plugin.get_tools()
```

**Without this plugin:** `Agent` cannot retrieve tools and will raise
`RuntimeError` during startup.

### SchedulerPlugin (`corvidae/scheduler.py`)

Clock-driven producer of reminders and operator jobs (the heartbeat is cognition's `HeartbeatPlugin` now; see [cognition.md](cognition.md#heartbeatplugin)). Entry point name: `"scheduler"`; `depends_on = {"registry", "persistence", "clock"}`. Owns the `schedule` table in `sessions.db` (created on first use) and one timer task, which sleeps until the earliest due row and is woken early whenever a row changes. A due row fires through `on_notify` with `source` and `annotations["origin"]` set to `reminder`, so persistence, critique eligibility, attribution, and the outbound send work unchanged. Jobs under `scheduler.jobs` sync by name on start and on every reload. Full behaviour: [design.md](design.md#schedulerplugin); keys: [configuration.md](configuration.md#scheduler--reminders-and-periodic-jobs).

It follows the two patterns a plugin owning persistent state and a background task needs. Schema creation is lazy on first use (`PersistencePlugin.db` may not be open during its `on_start`, as with `MemoryPlugin`). The timer is an `asyncio.Task` cancelled in `on_stop` (as with `ConfigWatcherPlugin`). It sleeps on the kernel clock, so its tests use a fake clock and never sleep.

**Without this plugin:** no reminders and no scheduled jobs; the three tools are absent.

### ConfigWatcherPlugin (`corvidae/config_watcher.py`)

Polls `agent.yaml` for mtime changes and dispatches `on_config_reload` to all registered plugins when the file changes. Entry point name: `"config_watcher"`.

Implements three hooks:

- `on_init` — reads `_config_path`, `_cli_overrides`, and `daemon.config_poll_interval` from the config dict.
- `on_start` — records the file's current mtime (to suppress a reload on the first poll) and starts the background polling task.
- `on_stop` — cancels the polling task and waits for it to finish.

On each poll cycle, when the mtime has changed:
1. The file is parsed as YAML.
2. CLI overrides are deep-merged on top of the new YAML.
3. The `llm` section is validated (`parse_llm_config`: a malformed profile set, `extends` problem, `defaults` or `stages` entry, or no default chat profile). On failure the reload is skipped.
4. `ChannelRegistry.agent_defaults` is updated from the new `agent` section.
5. `load_channel_config` is called: it registers new exact entries, updates `tools.disabled` on channels already registered, and replaces transport-wide entries (`signal:*`), re-matching them on every live channel. A malformed section changes no channel.
6. `on_config_reload` is dispatched to each plugin individually. A plugin that raises logs an error but does not block the other plugins.

**Config:**
```yaml
daemon:
  config_poll_interval: 2.0   # seconds between mtime checks (default 2.0)
```

See [Hot-reload](configuration.md#hot-reload) in the configuration reference for which settings update on reload and which require restart.

**Without this plugin:** `agent.yaml` changes are not detected at runtime. Reload requires a daemon restart.
