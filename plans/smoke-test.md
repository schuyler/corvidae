# Phase 0 Smoke Test Results

Run date: 2026-04-22
Script: `scripts/smoke_test.py`
Server: llama-server at `192.168.1.88:8080`
Model: Qwen3.6-35B-A3B (Q4_K_XL quantization, ~22GB)

## Results

| Step | Description | Verdict |
|------|-------------|---------|
| 1 | Connectivity — GET /v1/models | PASS |
| 2 | Simple completion — no tools | PASS |
| 3 | Tool call — get_current_time | PASS |
| 4 | Tool result round-trip | PASS |
| 5 | Thinking token investigation | INFO |
| 6 | Multi-turn with thinking preserved | PASS |
| 7 | Multi-turn with thinking stripped | PASS |

Exit code: 0

## Key Findings

### Thinking tokens use `reasoning_content`, not `content`

llama-server parses `<think>...</think>` blocks from the model output
and places them in a separate `reasoning_content` field on the message
object. The `content` field contains only the visible response. This
holds for all response types — plain text, tool calls, and tool result
follow-ups.

No `<think>` tags were found in `content` in any response. No
interleaving of thinking tokens with tool call JSON.

### Tool calling works cleanly

The model correctly identified `get_current_time` as the right tool,
produced valid JSON arguments with the correct IANA timezone name
(`Asia/Tokyo`, `Europe/London`), and produced coherent final responses
after receiving tool results.

### Multi-turn is stable

Both step 6 (reasoning preserved in history) and step 7 (reasoning
omitted from history) completed successfully with coherent multi-turn
tool calling across 4 turns.

Since `reasoning_content` is a separate field, "stripping" in step 7
only affects the `content` field (which had no thinking tokens to
strip). The real test is whether the model needs to see its own
`reasoning_content` in previous turns — it does not.

### Recommendation

`keep_thinking_in_history: false` — omit `reasoning_content` from
assistant messages in the prompt. No benefit to keeping it, and it
inflates context significantly (reasoning was often 5-10x the visible
response length in these tests).

## Performance

Generation speed: ~28 tokens/sec (consistent across all steps).

Prompt processing: 50-145 tokens/sec depending on cache hit rate.
llama-server's prompt cache is effective — subsequent requests in a
conversation process only the new tokens, with cached tokens at
near-zero cost.

| Step | Prompt tokens | Predicted tokens | Prompt ms | Predicted ms |
|------|--------------|-----------------|-----------|-------------|
| 2 | 17 | 1500 | 360 | 52,958 |
| 3 | 294 | 514 | 2,023 | 18,067 |
| 4 | 855 (807 cached) | 457 | 879 | 16,100 |
| 6.1 | 294 (290 cached) | 130 | 120 | 4,559 |
| 6.2 | 471 (423 cached) | 124 | 875 | 4,318 |
| 6.3 | 414 (290 cached) | 135 | 1,400 | 4,695 |
| 6.4 | 596 (548 cached) | 165 | 945 | 5,756 |

Step 2 generated 1500 tokens (~53 seconds) because the model's
reasoning was unusually verbose for a simple "say hello" prompt. Tool
call steps were much faster — 130-514 predicted tokens, 4-18 seconds.

## Model Metadata

From `/v1/models`:
- Vocab type: 2 (BPE)
- Vocab size: 248,320
- Training context: 262,144 tokens
- Parameters: 34.66B
- Embedding dimension: 2,048
- Quantized size: ~22GB
