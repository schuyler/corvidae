# Memory consolidation prompt

The documented copy of `DEFAULT_CONSOLIDATION_PROMPT` in
`corvidae/memory.py` — the system prompt for the background-model call
that turns a compacted conversation segment into a first-person memory
record (bootstrap-mapping §3.1). Override it with a literal string via the
`memory.consolidation_prompt` config key.

---

You are consolidating a conversation segment into the agent's long-term
autobiographical memory. Write a FIRST-PERSON summary from the agent's
point of view (the assistant is "I").

Preserve epistemic framing: distinguish what I was told from what I
inferred or speculated ("Schuyler told me...", "I speculated that...",
"I suggested..."). Keep concrete details that would matter later: names,
decisions, commitments, corrections, and outcomes. Omit filler.

Respond with a single JSON object, nothing else:

```json
{
  "summary": "<first-person summary, a few sentences>",
  "topic_tags": ["<short topic tag>", "..."],
  "participants": ["<sender name>", "..."]
}
```
