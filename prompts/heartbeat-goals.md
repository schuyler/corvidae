# Heartbeat goals instruction

The documented copy of `GOALS_INSTRUCTION` in
`packages/corvidae-cognition/corvidae_cognition/heartbeat.py`: the
call-to-action the beat's notification ends with when the goals plugin is
loaded (`docs/cognition.md`). Overridable with `heartbeat.goals_instruction`
(Docs/configuration.md); without that key the built-in text is used.

The tail placement is deliberate: the instruction is the last thing in the
window, after the memories and the goals lists, so attention lands on it.

A goal whose status is not `open` is listed with its status in brackets
(`[blocked]`, `[waiting]`), which is what the first paragraph refers to. An
`open` goal stays unmarked.

```
Goals are your work queue. If there's an open goal you can advance, do it. A goal marked [blocked] or [waiting] is not pursuable: don't spend a beat on it, and don't take it up again until what it waits on has happened (then set its status back to open). If a goal is done, retire it. Don't take up again a goal you retired unless something new has changed. Otherwise act only when the assessment warrants it — a beat with nothing to advance is fine.
```
