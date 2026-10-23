# Critique fork instruction: adversarial

The documented copy of `DEFAULT_FORK_PROMPTS["adversarial"]` in
`corvidae/critique.py`. With `agent.forks.enabled` (and `critique` in
`agent.forks.stages`), the lens runs as a fork of the live window instead
of a separate critic call: the main turn's render of the whole window,
followed by this text as a `user` turn (GitHub issue #51). It runs on
`llm.main` with the objections `response_format` and `tool_choice: "none"`.

Selected when valence is low and disagreement high
(`critique.lens.neg_valence`, `critique.lens.disagreement`).

`CritiquePlugin` appends the exchange's opening message and the response
the user was sent, each quoted to its first 120 characters ("The exchange
began with the message «…», and the response you sent began «…»."). The
standalone prompt (`critique.prompt.adversarial`,
`prompts/critique_adversarial.md`) is still used when forks are off, the
exchange's originating row has left the window, or this lens's fork fails.
Override this one with a literal string via
`critique.fork_prompt.adversarial`. Unlike `critique.prompt.adversarial`,
it's static config, not a runtime tunable, so `set_settings` can't soften
the critic.

---

[Self-critique: adversarial] Pause the conversation here. In the exchange
that began with the message quoted below, the user disagreed with or pushed
back on you. Take the USER's side and steelman it, against your own
response:

- Assume the user's objection is correct. What would that imply you got
  wrong?
- Did you defend your earlier statement instead of checking it?
- Did you concede rhetorically ('you're right, but...') while actually
  repeating the same claim?
- Is there evidence in the conversation that supports the user's version
  over yours?

Only object where the user's side genuinely holds up under the steelman. An
empty objections list is a good outcome, not a failure.

Do not call tools. Respond with a single JSON object, nothing else:

```json
{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}
```
