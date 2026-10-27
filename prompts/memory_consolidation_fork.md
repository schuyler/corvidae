# Memory consolidation fork instruction

The documented copy of `DEFAULT_CONSOLIDATION_FORK_PROMPT` in
`packages/corvidae-cognition/corvidae_cognition/memory.py`. With `agent.forks.enabled` (and `consolidation` in
`agent.forks.stages`), the memory record isn't written by a separate
background-model call. It's written by a fork of the live window: the main
turn's render (up to the compaction cut, or the whole window for the idle
trigger), followed by this text as a `user` turn (GitHub issue #51). The
agent's own system prompt is above it, so "I" needs no explaining.

`MemoryPlugin` appends one sentence naming where the range starts:

- "Record the whole conversation above." when nothing precedes it;
- "Record the conversation above that follows the summary at its top."
  when only the compaction summary does;
- otherwise "Record the conversation above from the message that begins
  «…» onward. Earlier messages are already in your memory; do not restate
  them.", quoting the first 120 characters of the first new message.

The standalone prompt (`memory.consolidation_prompt`) is still used when
forks are off, the range has left the window, or the fork fails. Override
this one with a literal string via `memory.consolidation_fork_prompt`. It's
static config, not a runtime tunable. The reply is parsed exactly like the
standalone one.

---

[Memory consolidation] Pause the conversation here and write a record of
part of it for your long-term autobiographical memory, in the first person,
as yourself.

Keep what you were told apart from what you inferred or speculated
("Schuyler told me...", "I speculated that...", "I suggested..."). Keep
concrete details that would matter later: names, decisions, commitments,
corrections, and outcomes. Omit filler. Do not call tools.

Respond with a single JSON object, nothing else:

```json
{
  "summary": "<first-person summary, a few sentences>",
  "topic_tags": ["<short topic tag>", ...],
  "participants": ["<sender name>", ...]
}
```
