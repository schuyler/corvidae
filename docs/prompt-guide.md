# Prompt Construction in Corvidae

This document explains how the system prompt is assembled before each LLM call. Understanding this pipeline helps you configure prompts correctly, write plugins that inject context, and reason about what the LLM actually sees.

---

## Overview

The prompt sent to the LLM is a list of message dicts assembled by `ContextWindow.build_prompt()`:

```
[system message, conversation message, conversation message, ..., injected context, ...]
```

The **system message** is fixed for the lifetime of a channel's conversation. **Conversation messages** accumulate turn by turn and may be compacted when they approach the token budget. **Injected context** entries (`MessageType.CONTEXT`) are appended into `conv.messages` by plugins before each LLM call and interleaved with other messages — they are not a separate trailing section.

---

## The system prompt

### Where it comes from

The system prompt is set in `agent.yaml`. There are two forms:

**Inline string:**
```yaml
agent:
  system_prompt: "You are a helpful assistant."
```

**File list (composable):**
```yaml
agent:
  system_prompt:
    - prompts/SOUL.md
    - prompts/IRC.md
```

When given a list, Corvidae reads each file and joins them with double newlines (`\n\n`) into a single string. Relative paths are resolved against the directory containing `agent.yaml` (the `_base_dir`). Absolute paths are used as-is. If `_base_dir` is unavailable, paths resolve against the current working directory.

This resolution happens in `corvidae/channel.py:resolve_system_prompt()`.

### Per-channel overrides

Any channel can override the agent-level default:

```yaml
channels:
  irc:#general:
    system_prompt: "You are the #general channel bot."
  cli:local:
    system_prompt:
      - prompts/SOUL.md
      - prompts/cli-addendum.txt
```

The lookup priority is:

1. Per-channel `system_prompt` (if set)
2. Agent-level `system_prompt` (fallback)
3. Built-in default: `"You are a helpful assistant."`

Runtime overrides (via the `set_settings` tool) can change many channel settings, but the system prompt is **always blocked** — it cannot be changed mid-session. See `agent.immutable_settings` in [configuration.md](configuration.md) for related controls.

### When it is resolved

The system prompt is resolved **on the first message to a channel**. It is stored in `channel.conversation.system_prompt` (a `ContextWindow` attribute) and does not change for the lifetime of that conversation. Restarting the daemon re-reads the config and creates a fresh `ContextWindow`.

---

## Conversation history

After the system message, `build_prompt()` appends all messages from `ContextWindow.messages`:

```python
# corvidae/context.py:94
def build_prompt(self) -> list[dict]:
    cleaned = []
    for msg in self.messages:
        if "_message_type" in msg:
            msg = {k: v for k, v in msg.items() if k != "_message_type"}
        cleaned.append(msg)
    return [{"role": "system", "content": self.system_prompt}] + cleaned
```

Each message is a standard dict with `"role"` and `"content"` keys. The internal `_message_type` tag is stripped before the list is returned — the LLM never sees it.

### Message types

Every entry in `ContextWindow.messages` carries a `_message_type` that controls persistence and compaction behavior:

| Type | Meaning |
|------|---------|
| `MESSAGE` | Ordinary user or assistant turn |
| `SUMMARY` | Compaction summary replacing older messages |
| `CONTEXT` | Plugin-injected context; survives compaction |

### Compaction

When `token_estimate() >= compaction_threshold × max_context_tokens` (default: 80% of budget), `CompactionPlugin` summarizes older `MESSAGE` entries via the LLM and replaces them with a single `SUMMARY` entry. `CONTEXT` and `SUMMARY` entries are not themselves compacted.

Token estimation is character-based: `int(total_chars / chars_per_token)`. The system prompt length is included. The default ratio is 3.5 characters per token; configure via `agent.chars_per_token`.

---

## Plugin-injected context

Before every LLM call, Corvidae fires the `before_agent_turn` hook. Plugins use this hook to inject contextual information — memory, retrieved documents, background notes — by appending to `channel.conversation`:

```python
# corvidae/hooks.py (hookspec)
async def before_agent_turn(self, channel) -> None: ...
```

Example plugin:

```python
from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, hookimpl

class MemoryPlugin(CorvidaePlugin):
    @hookimpl
    async def before_agent_turn(self, channel) -> None:
        notes = await self.fetch_relevant_notes(channel.id)
        if notes:
            channel.conversation.append(
                {"role": "user", "content": f"[Context]\n{notes}"},
                message_type=MessageType.CONTEXT,
            )
```

`CONTEXT` entries survive compaction — `CompactionPlugin` only summarizes `MESSAGE` entries.

Each message injected during `before_agent_turn` is individually persisted by `Agent` via `on_conversation_event`, in a loop that runs after the hook completes (agent.py:516–524).

### Avoiding duplicate injections

`ContextWindow` provides `remove_by_type(MessageType.CONTEXT)` to clear all in-memory `CONTEXT` entries before re-injecting fresh ones. This prevents accumulation across turns. The method raises `ValueError` if called with `MESSAGE` or `SUMMARY` — those are managed by the agent and compaction system, not plugins.

---

## Post-LLM steps that affect subsequent prompts

After the LLM responds, two steps affect what `build_prompt()` returns on the **next** turn:

**Step 8 — Append assistant message.** The assistant message dict (including any `reasoning_content` from a reasoning model) is appended to `conv.messages` and persisted.

**Step 9 — `after_persist_assistant` hook.** Plugins may mutate the in-memory message dict after the DB record is written. `ThinkingPlugin` uses this hook to strip `reasoning_content` from the in-memory dict when `keep_thinking_in_history` is `false`. The DB copy is already written and is not affected. Because `build_prompt()` reads `conv.messages` directly, subsequent turns will not include reasoning content in prompt history when this setting is off.

---

## Full pipeline

The full sequence for processing a queue item (`agent.py:_process_queue_item`):

1. **Lazy-init conversation** — on first message, resolve config, call `resolve_system_prompt()`, restore persisted history via the `load_conversation` hook (the return value is assigned directly to `conv.messages`).
2. **Append user message** — add to `conv.messages` as `MessageType.MESSAGE`.
3. **Persist** — fire `on_conversation_event` hook for the user message.
4. **Compact if needed** — fire `compact_conversation` hook. `CompactionPlugin` summarizes old messages if the token budget is near full (runs `trylast=True`, returns `True` to stop the chain).
5. **Inject context** — fire `before_agent_turn` hook; plugins append `CONTEXT` entries. Each injected entry is then individually persisted via `on_conversation_event`.
6. **Build prompt** — call `conv.build_prompt()` to produce the final message list.
7. **Call LLM** — send `[system, ...history, ...injected context]` plus tool schemas.
8. **Append assistant response** — add to `conv.messages`, fire `on_conversation_event`.
9. **Post-process in-memory message** — fire `after_persist_assistant`; `ThinkingPlugin` may strip `reasoning_content` from the in-memory dict.
10. **Dispatch** — handle tool calls (re-enters the pipeline) or send text response to the channel.

---

## What the LLM sees

A typical prompt for a CLI session with a composable system prompt:

```
[system]    You are Sherman, a helpful and knowledgeable agent. ...
            (SOUL.md content joined with IRC.md content if using a file list)

[user]      Hello, what can you help me with?
[assistant] I can help with ...

            ... older turns, or a SUMMARY entry from compaction ...

[user]      [Context]                        ← injected by before_agent_turn
            ...retrieved notes...

[user]      What was that command again?     ← current user message
```

---

## Note on ContextCompactPlugin

`ContextCompactPlugin` provides an alternative compaction strategy that generates persistent background blocks from older conversation segments and injects them before each agent turn. However, it is **not currently active** — it is absent from the `[project.entry-points.corvidae]` section of `pyproject.toml` and is therefore not loaded by the daemon.

If enabled, it would run on the `compact_conversation` hook with `tryfirst=True` (returning `None` to let `CompactionPlugin` continue), and on `before_agent_turn` to inject the most recent background block as a `CONTEXT` entry. Its configuration keys are documented in [configuration.md](configuration.md).

---

## Key source locations

| Component | File | What it does |
|-----------|------|--------------|
| `resolve_system_prompt()` | `corvidae/channel.py:210` | Reads file list or returns literal string |
| `ChannelConfig.resolve()` | `corvidae/channel.py:40` | Merges agent defaults, channel overrides, runtime overrides |
| `ContextWindow` | `corvidae/context.py:40` | Holds messages and system prompt in memory |
| `ContextWindow.build_prompt()` | `corvidae/context.py:94` | Assembles final message list for LLM |
| `ContextWindow.token_estimate()` | `corvidae/context.py:107` | Character-based token count |
| `ContextWindow.remove_by_type()` | `corvidae/context.py:122` | Clears in-memory entries of a given type |
| Lazy initialization | `corvidae/agent.py:442` | First-message conversation setup |
| `before_agent_turn` hook | `corvidae/agent.py:506` | Context injection point |
| `compact_conversation` hook | `corvidae/agent.py:495` | Compaction trigger |
| `after_persist_assistant` hook | `corvidae/agent.py:555` | Post-LLM in-memory message post-processing |
| `ThinkingPlugin` | `corvidae/thinking.py` | Strips `reasoning_content` from in-memory history |

For full configuration details, see [configuration.md](configuration.md).
