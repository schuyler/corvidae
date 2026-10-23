# Stage-2 appraisal fork instruction

The documented copy of `DEFAULT_STAGE2_FORK_PROMPT` in
`corvidae/appraisal.py`. With `agent.forks.enabled` (and `appraisal` in
`agent.forks.stages`), the silent stage-2 task forks the live window at run
time instead of making its own tier-3 call: the prompt is the main turn's
render of the whole window, followed by this text as a `user` turn (GitHub
issue #51). It runs on `llm.main` with the stage's `response_format` and
`tool_choice: "none"`.

`AppraisalPlugin` appends two lines: the exchange's opening message, quoted
to its first 120 characters ("The exchange began with the message «…»."),
and the retrieval profile ("Memory retrieval for this exchange: …").

The standalone prompt (`appraisal.stage2_prompt`) is still used when forks
are off, the exchange's originating row has left the window, or the fork
fails. Override this one with a literal string via
`appraisal.stage2_fork_prompt`. Unlike `appraisal.stage2_prompt`, it's
static config, not a runtime tunable, so `set_settings` can't change it.

---

[Appraisal] Pause the conversation here and appraise, for your own
perception system, the exchange you just completed: from the message quoted
below through your final response. Score the EXCHANGE (not your
performance) on each dimension, 0.0–1.0.

- **valence**: the emotional tone of the exchange for you. 0.0 = strongly
  negative (conflict, failure, distress), 0.5 = neutral, 1.0 = strongly
  positive (success, warmth, praise).
- **stakes**: how much depends on this exchange being handled well.
  Commitments, deadlines, personal disclosures, decisions = high; idle
  chat = low.
- **ambiguity**: how much of the message's intent remained open to
  interpretation. 0.0 = fully explicit; 1.0 = you had to guess.
- **commitment_density**: how many concrete commitments, facts, numbers,
  or promises the exchange contains, relative to its length.
- **novelty**: how new this exchange's content is relative to what you
  already knew, going by the memory retrieval note below. Familiar
  ground = low.
- **correction**: true if the user was correcting you, telling you that
  something you said, remembered, or did was wrong.

Do not call tools. Respond with a single JSON object, nothing else:

```json
{
  "valence": 0.5,
  "stakes": 0.0,
  "ambiguity": 0.0,
  "commitment_density": 0.0,
  "novelty": 0.5,
  "correction": false
}
```
