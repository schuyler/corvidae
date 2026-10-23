# Critique fork instruction: constrained

The documented copy of `DEFAULT_FORK_PROMPTS["constrained"]` in
`corvidae/critique.py`. With `agent.forks.enabled` (and `critique` in
`agent.forks.stages`), the lens runs as a fork of the live window instead
of a separate critic call: the main turn's render of the whole window,
followed by this text as a `user` turn (GitHub issue #51). It runs on
`llm.main` with the objections `response_format` and `tool_choice: "none"`.

Selected when the exchange's commitment density crosses
`critique.lens.commitment`.

`CritiquePlugin` appends the exchange's opening message and the response
the user was sent, each quoted to its first 120 characters ("The exchange
began with the message «…», and the response you sent began «…»."). The
standalone prompt (`critique.prompt.constrained`,
`prompts/critique_constrained.md`) is still used when forks are off, the
exchange's originating row has left the window, or this lens's fork fails.
Override this one with a literal string via
`critique.fork_prompt.constrained`. Unlike `critique.prompt.constrained`,
it's static config, not a runtime tunable, so `set_settings` can't soften
the critic.

---

[Self-critique: constrained] Pause the conversation here. Review the
response you gave in the exchange that began with the message quoted below,
as a critical outside reviewer would, not as its defender. The exchange
contains concrete commitments, facts, numbers, dates, or constraints. Check
your response against them mechanically:

- Does every number, date, and name in your response agree with the ones
  given in the conversation?
- Did you make a commitment you cannot keep, or contradict a commitment
  already on record?
- Did you drop a stated constraint (a budget, a deadline, an exclusion)
  when forming your answer?

Only object to concrete violations you can point at. An empty objections
list is a good outcome, not a failure.

Do not call tools. Respond with a single JSON object, nothing else:

```json
{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}
```
