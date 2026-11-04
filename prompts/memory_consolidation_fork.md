# Memory consolidation fork instruction

The documented copy of `DEFAULT_CONSOLIDATION_FORK_PROMPT` in
`packages/corvidae-cognition/corvidae_cognition/memory.py`. With `agent.forks.enabled` (and `consolidation` in
`agent.forks.stages`), the memory record is written by a fork of the live
window: the main turn's render (up to the compaction cut, or the whole
window for the idle trigger), followed by this text as a `user` turn
(GitHub issue #51). A range that has left the window is written the same
way from a rebuilt view of it, as a plain call on `llm.main`. Either
way the agent's own system prompt is above it, so "I" needs no explaining.

`MemoryPlugin` appends one sentence naming where the range starts:

- "Record the whole conversation above." when nothing precedes it;
- "Record the conversation above that follows the summary at its top."
  when only the compaction summary does;
- otherwise "Record the conversation above from the message that begins
  «…» onward. Earlier messages are already in your memory; do not restate
  them.", quoting the first 120 characters of the first new message.

Override this text with a literal string via
`memory.consolidation_fork_prompt`. It's static config, not a runtime
tunable. (`memory.consolidation_prompt`, the old standalone prompt, is gone
and ignored.)

---

[Memory consolidation] Pause the conversation here and write a record of
part of it for your long-term autobiographical memory, in the first person,
as yourself. This record is your private memory, not a message to anyone
on the channel: no one there will read it, so write it for yourself.

Keep what you were told apart from what you inferred or speculated
("so-and-so told me...", "I speculated that...", "I suggested..."). Keep
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
