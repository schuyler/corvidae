# Before-send appraisal fork instruction

The documented copy of `DEFAULT_BEFORE_SEND_FORK_PROMPT` in
`packages/corvidae-cognition/corvidae_cognition/appraisal.py`. With
`agent.forks.enabled` (and `appraisal` in `agent.forks.stages`), appraisal's
`should_send_response` forks the live window once on each final draft that is
not a decline, before it is sent, and asks the agent to score the lenses that
peers offer for `before_send` (#1169). The prompt is the main turn's render of
the whole window, followed by this text as a `user` turn. It runs on the
`appraisal` stage's profile with the stage's `response_format` and
`tool_choice: "none"`.

`AppraisalPlugin` appends the exchange's opening message, the draft quoted
whole (whitespace flattened to one line, not truncated: the draft is the thing
being scored and the window doesn't hold it yet) and the same
`[Critique selection]` block as the idle fork, one `- <name>: <question>` line
per offered lens. The reply schema holds the `lenses` object only: this fork
lists no claims, and the provenance check runs only after the exchange.

Before send, critique offers the built-in lenses except retention, and a
plugin lens only when it declares `before_send=True`. With no lenses on offer
the fork does not run. A miss (forks off, the exchange left the window, the
fork fails or its reply is unusable) logs a warning and tells the peers there
is no vector. The draft is never held back for any of this.

Override this one with a literal string via `appraisal.before_send_fork_prompt`.
That key is resolved at start-up and again on a config reload, and the result
is cached, so an `agent.yaml` change applies from the next reload;
`set_settings` accepts a write to it but nothing reads the override.

---

[Appraisal] Pause the conversation here. You have drafted the reply quoted below but not sent it yet. For your own perception system, score the self-checks listed below against that draft: which one, if any, is most worth running on the reply before it goes out.

Do not call tools. Respond with a single JSON object, nothing else, holding only the "lenses" object described below.
