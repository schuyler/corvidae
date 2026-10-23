# Critique fork instruction: provenance

The documented copy of `DEFAULT_FORK_PROMPTS["provenance"]` in
`corvidae/critique.py`. With `agent.forks.enabled` (and `critique` in
`agent.forks.stages`), the lens runs as a fork of the live window instead
of a separate critic call: the main turn's render of the whole window,
followed by this text as a `user` turn (GitHub issue #51). It runs on
`llm.main` with the objections `response_format` and `tool_choice: "none"`.

Fired by the mechanical provenance gate, independent of appraisal.
`CritiquePlugin` appends the CONTEXT snapshot recorded for the exchange
after the quote, under "Retrieved context snapshot (all CONTEXT in your
window when you wrote the response):".

`CritiquePlugin` appends the exchange's opening message and the response
the user was sent, each quoted to its first 120 characters ("The exchange
began with the message «…», and the response you sent began «…»."). The
standalone prompt (`critique.prompt.provenance`,
`prompts/critique_provenance.md`) is still used when forks are off, the
exchange's originating row has left the window, or this lens's fork fails.
Override this one with a literal string via
`critique.fork_prompt.provenance`. Unlike `critique.prompt.provenance`,
it's static config, not a runtime tunable, so `set_settings` can't soften
the critic.

---

[Self-critique: provenance] Pause the conversation here. Audit the response
you gave in the exchange that began with the message quoted below for
confabulated memory. You asserted something about past events, statements,
or commitments. The memory system reports that retrieval for this exchange
was weak and a search of the raw conversation log found nothing matching.
Below is a snapshot of ALL the retrieved context that was in your window
when you wrote the response.

For each claim about the past in your response:

- Is the claim supported by the conversation above or by the context
  snapshot?
- If not, you asserted a recollection with no record behind it: object,
  quoting the unsupported claim exactly.
- Claims explicitly framed as uncertainty or inference ('I don't recall',
  'I would guess...') are correctly calibrated: do not object to them.

Confident phrasing is not evidence. A claim is supported by records, not by
how sure you sounded. An empty objections list is a good outcome.

Do not call tools. Respond with a single JSON object, nothing else:

```json
{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}
```
