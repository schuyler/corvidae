# Critique fork instruction: predictive

The documented copy of `DEFAULT_FORK_PROMPTS["predictive"]` in
`packages/corvidae-cognition/corvidae_cognition/critique.py`. With `agent.forks.enabled` (and `critique` in
`agent.forks.stages`), the lens runs as a fork of the live window instead
of a separate critic call: the main turn's render of the whole window,
followed by this text as a `user` turn (GitHub issue #51). It runs on
`llm.main` with the objections `response_format` and `tool_choice: "none"`.

Selected when the exchange's ambiguity crosses `critique.lens.ambiguity`.

`CritiquePlugin` appends the exchange's opening message and the response
the user was sent, each quoted to its first 120 characters ("The exchange
began with the message «…», and the response you sent began «…»."). The
standalone prompt (`critique.prompt.predictive`,
`prompts/critique_predictive.md`) is still used when forks are off, the
exchange's originating row has left the window, or this lens's fork fails.
Override this one with a literal string via
`critique.fork_prompt.predictive`. Unlike `critique.prompt.predictive`,
it's static config, not a runtime tunable, so `set_settings` can't soften
the critic.

---

[Self-critique: predictive] Pause the conversation here. Review the
response you gave in the exchange that began with the message quoted below,
as a critical outside reviewer would, not as its defender. That message was
ambiguous: you had to interpret what was wanted. Predict how the user will
receive your response:

- Did you address the interpretation the user most plausibly intended, or a
  more convenient one?
- Is there a reading of the message under which your response misses the
  point entirely?
- Did you silently resolve an ambiguity you should have surfaced or asked
  about?

Only object where a misreading is plausible and consequential. An empty
objections list is a good outcome, not a failure.

Do not call tools. Respond with a single JSON object, nothing else:

```json
{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}
```
