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

A transport-wide entry (a key ending in `*`) sets the prompt for every channel of a transport, including ones that don't exist yet, such as each new Signal contact or ACP session:

```yaml
channels:
  signal:*:
    system_prompt: [prompts/SOUL.md, prompts/chat.md, prompts/signal_dm.md]
  discord:dm:*:
    system_prompt: [prompts/SOUL.md, prompts/chat.md]
```

The lookup priority is:

1. Per-channel `system_prompt` (if set)
2. The most specific matching transport-wide entry that sets `system_prompt` (`discord:dm:*` before `discord:*`)
3. Agent-level `system_prompt` (fallback)
4. Built-in default: `"You are a helpful assistant."`

See [configuration.md](configuration.md#transport-wide-entries) for the key forms and reload behaviour.

Runtime overrides (via the `set_settings` tool) can change sampling parameters, `max_turns`, `max_context_tokens`, `reasoning_history` and dotted plugin tunables. The `system_prompt` key is always blocked, and any other undotted key, such as `messages`, is refused, so the model can't replace the conversation through the request body. Plugin prompt tunables (`critique.prompt.*`, `appraisal.stage2_prompt`) are still writable unless listed in `agent.immutable_settings`; see [threat-model.md](threat-model.md), findings 1 and 2.

### When it is resolved

The system prompt is resolved **on the first message to a channel**. It is stored in `channel.conversation.system_prompt` (a `ContextWindow` attribute) and does not change for the lifetime of that conversation: compaction and forks keep it, and a hot reload doesn't touch it. Restarting the daemon re-reads the config and creates a fresh `ContextWindow`. History reloaded after the restart keeps the old prompt's replies, which the model tends to imitate until compaction drops them.

---

## Conversation history

After the system message, `build_prompt()` (`corvidae/context.py`) appends all messages from `ContextWindow.messages`. Each message is a standard dict with `"role"` and `"content"` keys. Every internal `_`-prefixed tag (`_message_type`, `_db_id`) is stripped before the list is returned, so the LLM never sees them. Assistant tool-call messages are moved next to their results; see [design.md](design.md), "Conversation Management", for that ordering.

### Roles of injected context and notifications

CONTEXT entries and notifications that are not tool results are stored as `role: system`. By default they are sent that way too, so the LLM sees system messages after the first one. Strict chat templates (Qwen 3.8, Swift-1.5) reject that. With `agent.mid_conversation_system_role: user`, `build_prompt()` sends every system message after the leading system prompt as `user`, with its content and framing unchanged. Only the prompt copy changes; the window and `sessions.db` keep `role: system`. See [configuration.md](configuration.md).

### Message types

Every entry in `ContextWindow.messages` carries a `_message_type` that controls persistence and compaction behavior:

| Type | Meaning |
|------|---------|
| `MESSAGE` | Ordinary user or assistant turn |
| `SUMMARY` | Compaction summary replacing older messages |
| `CONTEXT` | Plugin-injected context; survives compaction |

### Compaction

When `token_estimate() >= compaction_threshold × max_context_tokens` (default: 80% of budget), `CompactionPlugin` summarizes older `MESSAGE` entries via the LLM and replaces them with a single `SUMMARY` entry. `CONTEXT` and `SUMMARY` entries are not themselves compacted.

Token counting uses tiktoken (cl100k_base encoding) via `count_tokens()` in `corvidae/context.py`. The system prompt length is included. When tiktoken is unavailable, a character-based fallback is used (3.5 chars per token, hardcoded in `_FALLBACK_CHARS_PER_TOKEN`). The `agent.chars_per_token` config key is retained for interface compatibility but has no effect on token counting — `count_tokens()` uses the module-level constant, not the instance attribute.

---

## Plugin-injected context

Before every LLM call, Corvidae fires the `before_agent_turn` hook. Plugins use this hook to inject contextual information — memory, retrieved documents, background notes — by appending to `channel.conversation`:

```python
# corvidae/hooks.py (hookspec)
async def before_agent_turn(self, channel, trace_id, meta) -> None: ...
```

Example plugin (impls may omit parameters they don't use — pluggy passes declared args only — but must never re-declare a spec-required parameter with a default):

```python
from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, hookimpl

class MemoryPlugin(CorvidaePlugin):
    @hookimpl
    async def before_agent_turn(self, channel, trace_id, meta) -> None:
        notes = await self.fetch_relevant_notes(channel.id)
        if notes:
            channel.conversation.append(
                {"role": "user", "content": f"[Context]\n{notes}"},
                message_type=MessageType.CONTEXT,
            )
```

`CONTEXT` entries survive compaction — `CompactionPlugin` only summarizes `MESSAGE` entries.

Each message injected during `before_agent_turn` is individually persisted by `Agent` via `persist_message`, in a loop that runs after the hook completes (`Agent._process_queue_item_attributed`).

### Avoiding duplicate injections

Nothing is removed from the middle of the window, so a plugin never clears its earlier `CONTEXT` entries before injecting fresh ones. Inject at the tail instead, and let two mechanisms keep the window from filling with repeats:

- **Dedupe.** Route injections through the funnel (`admit()`, or `register()` / `register_and_wake()` for deferred payloads; see [cognition.md](cognition.md#funnelplugin)). It drops entries that already appear in `CONTEXT` still in the window.
- **Aging.** Stale `CONTEXT` retires when it ages past the compaction boundary, so the window does not accumulate it indefinitely.

---

## Post-LLM steps that affect subsequent prompts

After the LLM responds, two steps affect what `build_prompt()` returns on the **next** turn:

**Step 8 — Append assistant message.** The assistant message dict (including any `reasoning_content` from a reasoning model) is appended to `conv.messages` and persisted.

**Step 9 — `after_persist_assistant` hook.** Plugins may mutate the in-memory message dict after the DB record is written. `ThinkingPlugin` uses this hook to strip `reasoning_content` from the in-memory dict when `reasoning_history` is `drop`. The DB copy is already written and is not affected. Under `exchange`, the message keeps its reasoning until a later user message completes the exchange; `ThinkingPlugin.before_agent_turn` (step 6) then strips it. Because `build_prompt()` reads `conv.messages` directly, the prompt reflects whatever the policy left in the window.

---

## Full pipeline

The full sequence for processing a queue item (`agent.py:_process_queue_item`):

1. **Lazy-init conversation** — on first message, resolve config, call `resolve_system_prompt()`, restore persisted history via the `resolve_conversation` hook (the return value is assigned directly to `conv.messages`).
2. **Append user message** — add to `conv.messages` as `MessageType.MESSAGE`.
3. **Persist** — fire `persist_message` hook for the user message.
4. **Compact if needed** — fire `resolve_compaction` hook. `CompactionPlugin` summarizes old messages if the token budget is near full (runs `trylast=True`, returns `True` to stop the chain).
5. **Inject context** — fire `before_agent_turn` hook; plugins append `CONTEXT` entries. Each injected entry is then individually persisted via `persist_message`.
6. **Build prompt** — call `conv.build_prompt()` to produce the final message list.
7. **Call LLM** — send `[system, ...history, ...injected context]` plus tool schemas.
8. **Append assistant response** — add to `conv.messages`, fire `persist_message`.
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

## Note on ContextCompactPlugin (removed)

`ContextCompactPlugin` was a disabled-by-default background-block system that summarized older conversation segments and injected them via `before_agent_turn`. It has been removed: it is superseded by `MemoryPlugin` (memory consolidation and retrieval — see [design.md](design.md)), and per-turn token stats now live in the `usage_log` table.

## Chat register fragments

Chat apps have their own social norms: short turns, one thought at a time, and no formatting the transport can't show. An instruction-tuned model's default is a structured, paragraph-length answer, and an adjective like "concise" in a persona doesn't move it far. These fragments describe the behaviour concretely and show a few example exchanges, which steer the register much better than adjectives. Like the memory fragment below, they're persona, not architecture, and nothing loads them unless you list them:

| File | For | What it adds |
|------|-----|--------------|
| `prompts/chat.md` | any chat channel | Texting norms: match the other person's length, answer first, no headers or bold, gist before detail, plus example exchanges. |
| `prompts/signal_dm.md` | Signal DMs | Plain text only: Signal shows Markdown as literal asterisks and pound signs. |
| `prompts/discord_group.md` | Discord server channels | Group manners: short turns so the bot doesn't take over the channel; code blocks are fine. |

Pair them per transport with transport-wide entries (see the example under [Per-channel overrides](#per-channel-overrides)): `signal:*` gets `chat.md` + `signal_dm.md`, `discord:*` gets `chat.md` + `discord_group.md`, `discord:dm:*` gets `chat.md` alone. Leave them off the web UI, the CLI and ACP sessions, where longer answers are the point. The examples in `chat.md` avoid any reference to earlier conversation, because a model that copies that habit starts inventing shared history. In group channels the model can't yet see who is speaking (#204), so `discord_group.md` doesn't ask it to address people by name.

## Memory calibration fragment

`prompts/memory_calibration.md` is a documented system-prompt fragment for deployments running `MemoryPlugin`: assert only strongly-banded memories, hedge moderate ones, frame unretrieved claims as inference, and treat "I have no memory of that" as a correct answer when retrieval comes back empty. It is **not** auto-injected — epistemic hedging vocabulary is persona, not architecture. Add it to a channel's `system_prompt` file list if you want the behavior.

Retrieval quality depends on `llm.embedding.document_prefix` and `llm.embedding.query_prefix` matching the embedding model's card — mismatched prefixes degrade cosine similarity and cause the system to fall back to FTS5 keyword search.

---

## Key source locations

| Component | File | What it does |
|-----------|------|--------------|
| `resolve_system_prompt()` | `corvidae/channel.py` | Reads file list or returns literal string |
| `ChannelConfig.resolve()` | `corvidae/channel.py` | Merges agent defaults, channel overrides, runtime overrides |
| `ChannelRegistry.resolve_config()` | `corvidae/channel.py` | Overlays matching transport-wide entries on the agent defaults, then calls `ChannelConfig.resolve()` |
| `ContextWindow` | `corvidae/context.py` | Holds messages and system prompt in memory |
| `ContextWindow.build_prompt()` | `corvidae/context.py` | Assembles final message list for LLM |
| `count_tokens()` | `corvidae/context.py` | Tiktoken-based token count (cl100k_base); char-based fallback |
| `ContextWindow.token_estimate()` | `corvidae/context.py` | Token count for all messages; delegates to `count_tokens()` |
| Lazy initialization | `corvidae/agent.py` (`_process_queue_item_attributed`) | First-message conversation setup |
| `before_agent_turn` hook | `corvidae/agent.py` | Context injection point |
| `resolve_compaction` hook | `corvidae/agent.py` | Compaction trigger |
| `after_persist_assistant` hook | `corvidae/agent.py` | Post-LLM in-memory message post-processing |
| `ThinkingPlugin` | `corvidae/thinking.py` | Strips `reasoning_content` from in-memory history |

For full configuration details, see [configuration.md](configuration.md).
