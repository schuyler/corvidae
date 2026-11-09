# Memory consolidation fork instruction

The documented copy of `DEFAULT_CONSOLIDATION_FORK_PROMPT` in
`packages/corvidae-cognition/corvidae_cognition/memory.py`. With `agent.forks.enabled` (and `consolidation` in
`agent.forks.stages`), the memory entries are written by a fork of the live
window: the main turn's render (up to the compaction cut, or the whole
window for the idle trigger), followed by this text as a `user` turn
(GitHub issue #51). A range that has left the window is written the same
way from a rebuilt view of it, as a plain call on the `consolidation` stage's profile. Either
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

Narrate this stretch of the conversation from your own point of view, step
by step, in your own words: what was said, what you did, what you learned,
and what was decided or left open. Don't quote anyone verbatim, but keep
facts and opinions exactly as they were expressed, and who expressed each
(names, numbers, dates, identifiers, preferences, commitments,
corrections), so you can recall them accurately later. Keep what you were
told apart from what you inferred or speculated ("so-and-so told me...",
"I guessed that..."). Say what you did with tools and what it told you,
not their raw output. Bracketed CONTEXT blocks are things you recalled or
were shown, not new events; don't record them as if they happened now. Skip
pleasantries and chatter.

Write it as separate entries, one for each distinct stretch or topic, in
order. Each entry is a paragraph at most and must make sense read on its
own, months from now, with nothing else: name the people, projects and
things involved rather than "he" or "it". Don't attribute a fact from one
entry's topic to another's.

Then rate how important this stretch is to remember long-term, from 0
(routine) to 1 (essential). Do not call tools.

Respond with a single JSON object, nothing else:

```json
{
  "entries": [
    {"narrative": "<first-person narrative of one stretch or topic>",
     "topic_tags": ["<short topic tag>", "..."]}
  ],
  "participants": ["<sender name>", "..."],
  "importance": "<number from 0 to 1>"
}
```
