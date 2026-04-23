# Sherman — Agent Daemon

Asyncio agent daemon connecting to chat platforms (IRC first, Signal/BlueSky later),
routing messages through a local LLM via a hand-rolled agent loop, with a pluggy/apluggy
plugin system for hot-loadable components.

Hardware: i7-8700K / RTX 3060 Ti / 128GB DDR4-2666.
LLM: Qwen3.6-35B-A3B served by llama-server on host via OpenAI-compatible API.

## File Map

```
sherman/
├── hooks.py               # AgentSpec hookspecs (7 lifecycle hooks)
├── plugin_manager.py      # create_plugin_manager()
├── channel.py             # Channel, ChannelConfig, ChannelRegistry, load_channel_config
├── llm.py                 # LLMClient — async aiohttp wrapper for Chat Completions API
├── agent_loop.py          # run_agent_loop(), tool_to_schema(), strip_thinking(), strip_reasoning_content()
├── conversation.py        # ConversationLog — SQLite-backed append-only log + compaction
├── prompt.py              # resolve_system_prompt() — string passthrough or file list concatenation
├── agent_loop_plugin.py   # AgentLoopPlugin — wires LLMClient/run_agent_loop/ConversationLog into hooks
├── main.py                # Daemon entry point — YAML config, plugin registration, SIGINT/SIGTERM
tests/
├── conftest.py            # Shared fixtures (in-memory SQLite DB)
├── test_hooks.py
├── test_agent_loop.py
├── test_conversation.py
├── test_channel.py
├── test_main.py
├── test_agent_loop_plugin.py
└── test_prompt.py
prompts/
├── SOUL.md                # Sample identity/personality prompt
└── IRC.md                 # Sample IRC channel-specific prompt
plans/
├── design.md              # Primary design document — authoritative source
└── phase2-design.md       # Phase 2 detailed spec (complete)
```

## Phase Status

- **Phase 0** ✓ — Smoke test. Validated aiohttp ↔ llama-server, tool calling, thinking tokens.
  See `plans/smoke-test.md`.
- **Phase 1** ✓ — Core framework. 39 tests. hooks, plugin manager, LLMClient, agent loop,
  ConversationLog, main.py.
- **Phase 1.5** ✓ — Channel abstraction. 68 tests. Channel/ChannelConfig/ChannelRegistry,
  Channel passed through all hooks (was raw channel_id string), `pm.registry` on plugin manager.
- **Phase 2** ✓ — Agent loop plugin. 87 tests. AgentLoopPlugin wires LLMClient, run_agent_loop,
  and ConversationLog into the hook system.
- **Phase 2.5** ✓ — Composable system prompts. 108 tests. `system_prompt` accepts list of file
  paths; resolved in `_ensure_conversation` via `resolve_system_prompt()` in `prompt.py`.
  `base_dir` injected via `config["_base_dir"]`; defaults to `Path(".")` when absent.

## Key Architectural Decisions

**Thinking token 3-layer strategy** (confirmed by Phase 0 smoke test):
- llama-server puts thinking tokens in `reasoning_content`, not `content`.
- Display: use `content` directly; `strip_thinking()` is a defensive fallback only.
- Persistent log: full message dict preserved including `reasoning_content`.
- Active history: `reasoning_content` stripped from newly appended assistant messages
  after `run_agent_loop()` returns (when `keep_thinking_in_history=false`, which is the default).
  Only newly appended messages are stripped — prior turns were already stripped.

**Lazy ConversationLog init on channel** (Phase 2):
- `channel.conversation` is None until first message arrives on that channel.
- `AgentLoopPlugin._ensure_conversation()` initializes it on demand.
- No plugin-level `self.conversations` dict — the Channel object owns the log.

**register_tools sync call ordering** (Phase 2):
- `register_tools` is synchronous, called via `pm.hook.register_tools()` (not `pm.ahook`).
- AgentLoopPlugin calls it during `on_start` to collect tools from already-registered plugins.
- Tool-providing plugins (e.g., CoreToolsPlugin in Phase 3) MUST be registered before
  AgentLoopPlugin, or their tools will be missed.

**messages_before off-by-one** (Phase 2):
- `conv.build_prompt()` returns `[system_msg, *conv.messages]` — one longer than `conv.messages`.
- `messages_before = len(messages)` must use the `build_prompt()` result, NOT `len(conv.messages)`.
  Using `len(conv.messages)` would include the last user message in `new_messages` and double-persist it.

**System prompt resolution** (Phase 2.5):
- `resolve_system_prompt(value, base_dir)` in `prompt.py` handles both string and list values.
- String passes through unchanged. List: each entry read as a file, stripped, joined with `\n\n`.
- Relative paths resolved against `base_dir` (`Path(config_path).parent`), so config is portable.
- Called in `_ensure_conversation`, not `ChannelConfig.resolve()` — raw list preserved in config.
- `_base_dir` injected into config dict by `main.py`; no hookspec change required.
- `plugin.base_dir` defaults to `Path(".")` (never `None`).
- Re-reads files each time a conversation is initialized, so prompt edits take effect on next
  new conversation without a restart (sets up Phase 5 hot-reload).

**Channel ID format:** `transport:scope` — e.g., `irc:#lex`, `signal:+15551234567`.

**Registry injection:** `pm.registry` is set as a plain attribute on the plugin manager in `main.py`.
Plugins access it via `self.pm.registry`.
