# Prompt Construction in Corvidae

This document explains how the system prompt is assembled before each LLM call. Understanding this pipeline helps you configure prompts correctly, write plugins that inject context, and reason about what the LLM actually sees.

---

## Overview

The final prompt sent to the LLM is a list of message dicts:

```
[system message] + [conversation history] + [plugin-injected context]
```

The system message is fixed for the lifetime of a channel conversation. Conversation history accumulates turn by turn, is compacted when it approaches the token budget, and may be restored from the database on startup. Plugin-injected context is added fresh before every LLM call.

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

When given a list, Corvidae reads each file and joins them with double newlines into a single string. Relative paths are resolved against the directory containing `agent.yaml` (the `_base_dir`). Absolute paths are used as-is.

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

### When it is resolved

The system prompt is resolved **lazily**, on the first message to a channel. It is stored in the channel's `ContextWindow.system_prompt` and does not change for the lifetime of that conversation. Restarting the daemon re-reads the config and creates a fresh `ContextWindow`.

The system prompt is **immutable at runtime**. The `set_settings` tool blocks it unconditionally; it cannot be changed mid-session.

---

## Conversation history

After the system message, `build_prompt()` appends all messages from `ContextWindow.messages` in order:

```python
# corvidae/context.py
def build_prompt(self) -> list[dict]:
    cleaned = [strip_metadata(msg) for msg in self.messages]
    return [{"role": "system", "content": self.system_prompt}] + cleaned
```

Each message is a standard dict with `"role"` and `"content"` keys. An internal `_message_type` tag is stripped before the message reaches the LLM.

### Message types

Every entry in `ContextWindow.messages` carries an internal `_message_type` that controls persistence and compaction behavior, but is invisible to the LLM:

| Type | Meaning |
|------|---------|
| `MESSAGE` | Ordinary user or assistant turn |
| `SUMMARY` | Compaction summary replacing older messages |
| `CONTEXT` | Plugin-injected context; survives compaction |

### Compaction

When `token_estimate() >= compaction_threshold × max_context_tokens` (default: 80% of the budget), `CompactionPlugin` summarizes older `MESSAGE` entries via the LLM and replaces them with a single `SUMMARY` entry. `CONTEXT` entries are not compacted.

Token estimation is character-based: `total_chars / chars_per_token` (default ratio: 3.5). Configure `agent.chars_per_token` in `agent.yaml`.

---

## Plugin-injected context

Before every LLM call, Corvidae fires the `before_agent_turn` hook. Plugins use this hook to inject contextual information—memory, retrieved documents, background notes—by appending to `channel.conversation`:

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
                {"role": "user", "content": f"[Memory]\n{notes}"},
                message_type=MessageType.CONTEXT,
            )
```

Injected `CONTEXT` entries appear in the prompt for that turn and are persisted, but they are excluded from compaction. The built-in `ContextCompactPlugin` uses this mechanism to inject background summary blocks.

---

## Full pipeline

The sequence from user message to LLM call (in `agent.py:_process_queue_item`):

1. **Lazy-init conversation** — on first message, resolve config, call `resolve_system_prompt()`, load persisted history via `load_conversation` hook.
2. **Append user message** — add to `conv.messages` as `MessageType.MESSAGE`.
3. **Persist** — fire `on_conversation_event` hook.
4. **Compact if needed** — fire `compact_conversation` hook; `CompactionPlugin` summarizes old messages if the token budget is close to full.
5. **Inject context** — fire `before_agent_turn` hook; plugins append `CONTEXT` entries.
6. **Build prompt** — call `conv.build_prompt()` to produce the final message list.
7. **Call LLM** — send `[system, ...history, ...context]` plus tool schemas.

---

## What the LLM sees

A typical prompt looks like this:

```
[system]    You are Sherman, a helpful and knowledgeable agent. ...
            (+ IRC.md content if using a file list)

[user]      Hello, what can you help me with?
[assistant] I can help with ...

...older turns or a compaction summary...

[user]      [Memory]                     ← injected by before_agent_turn
            ...retrieved notes...

[user]      What was that command again?  ← current user message
```

---

## Configuration quick reference

```yaml
agent:
  system_prompt: "You are a helpful assistant."  # or list of file paths
  max_context_tokens: 24000
  compaction_threshold: 0.8    # compact at 80% of token budget
  compaction_retention: 0.5    # retain 50% after compaction
  min_messages_to_compact: 5   # don't compact very short conversations
  chars_per_token: 3.5         # character-to-token ratio for estimation

channels:
  irc:#general:
    system_prompt: "You are the channel bot."
  cli:local:
    system_prompt:
      - prompts/SOUL.md
      - prompts/cli-addendum.txt
```

---

## Key source locations

| Component | File | What it does |
|-----------|------|--------------|
| `resolve_system_prompt()` | `corvidae/channel.py:210` | Reads file list or returns literal string |
| `ChannelConfig.resolve()` | `corvidae/channel.py:40` | Merges agent defaults, channel overrides, runtime overrides |
| `ContextWindow` | `corvidae/context.py:40` | Holds messages and system prompt in memory |
| `ContextWindow.build_prompt()` | `corvidae/context.py:94` | Assembles final message list for LLM |
| `ContextWindow.token_estimate()` | `corvidae/context.py:107` | Character-based token count |
| Lazy initialization | `corvidae/agent.py:442` | First-message conversation setup |
| `before_agent_turn` hook | `corvidae/agent.py:506` | Context injection point |
| `compact_conversation` hook | `corvidae/agent.py:495` | Compaction trigger |
