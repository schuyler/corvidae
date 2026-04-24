# Phase 2: run_agent_turn() Design

## AgentTurnResult

```python
@dataclass
class AgentTurnResult:
    """Result of a single agent turn (one LLM invocation).

    Attributes:
        message: The raw assistant message dict from the LLM response.
            Always has "role": "assistant". May contain tool_calls,
            content, reasoning_content.
        tool_calls: List of tool call dicts from the response. Empty list
            if the LLM did not request any tool calls.
        text: The text content of the response. Empty string if the LLM
            produced only tool calls with no text.
        latency_ms: Wall-clock time for the LLM call in milliseconds, rounded to one decimal place.
    """

    message: dict
    tool_calls: list[dict]
    text: str
    latency_ms: float
```

## run_agent_turn()

```python
async def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    tool_schemas: list[dict],
) -> AgentTurnResult:
    """Single LLM invocation. Returns the response; does not execute tools.

    Calls client.chat() once with the given messages and tool schemas.
    Appends the assistant message to messages (mutates in place, matching
    run_agent_loop convention). Logs at the same levels as run_agent_loop.

    Args:
        client: LLM client for chat completions.
        messages: Conversation history. The assistant response is appended
            in place.
        tool_schemas: Tool schemas for LLM function calling. Pass empty
            list for no tools (converted to None for the API call).

    Returns:
        AgentTurnResult with the parsed response.
    """
```

### Behavior (step by step)

1. Record `time.monotonic()`.
2. Call `await client.chat(messages, tools=tool_schemas or None)`.
3. Compute `latency_ms`.
4. Extract `msg = response["choices"][0]["message"]`.
5. `msg.setdefault("role", "assistant")`.
6. Append `msg` to `messages`.
7. Extract `tool_calls = msg.get("tool_calls", [])` and `text = msg.get("content", "") or ""`.
8. Log INFO: "LLM response received" with `role`, `tool_calls_count`, `latency_ms`.
9. Log DEBUG: "LLM response content" with `content`, `has_reasoning_content`, `reasoning_content_length`.
10. Return `AgentTurnResult(message=msg, tool_calls=tool_calls, text=text, latency_ms=latency_ms)`.

## Design decisions

- **Mutates `messages` in place**: Matches `run_agent_loop` convention. The caller doesn't need to manually append.
- **`latency_ms` on the result**: The design doc doesn't include it, but `run_agent_loop` tracks it and the caller (AgentPlugin) logs it. Including it on the result avoids the caller needing to wrap the call with its own timer.
- **`text` handles `None` content**: Some LLMs return `content: null` when producing only tool calls. The `or ""` fallback handles this.
- **No `tools` dict parameter**: Unlike `run_agent_loop`, this function has no tool execution responsibility, so it only needs schemas.
- **Logging is identical** to the LLM-call portion of `run_agent_loop`. Tool-dispatch logging stays with the caller.
- **Exceptions propagate directly** from `client.chat()` — no wrapping. The caller already has a broad try/except.
- **`tool_calls` normalized to `[]`**: Unlike `run_agent_loop` and the spec pseudocode (which allow `None`), `run_agent_turn` always returns a list. Callers never need to check for `None`.
- **Double-logging of `latency_ms` is intentional**: `LLMClient.chat()` logs at INFO with `latency_ms`; `run_agent_turn` does the same. This matches `run_agent_loop`'s existing convention. Acceptable redundancy — the client log is low-level transport timing; the turn log is application-level.

## Relationship to run_agent_loop

- `run_agent_loop` stays unchanged. Preserved for subagent use (Phase 4).
- In Phase 3, `AgentPlugin._process_queue_item` will switch from `run_agent_loop` to `run_agent_turn` + task dispatch.
- Both functions share the same logging pattern for the LLM call portion. A shared helper could be extracted in Phase 3 but is premature now.
- **Spec deviation — message append**: The spec's Phase 3 dispatch flow lists "Append assistant message to conversation" as the caller's step 1. In this design, `run_agent_turn` handles the append internally (step 6). Phase 3 implementers should skip spec step 1 — `run_agent_turn` already did it.

## Edge cases to test

1. Text response, no tool calls — `tool_calls` is `[]`, `text` is the content, message appended.
2. Response with tool calls — `tool_calls` populated, `text` may be `""`.
3. Response with both text and tool calls.
4. `content: null` in response — `text` is `""`, not `None`.
5. Empty `tool_schemas` — `tools=None` passed to `client.chat()`.
6. Non-empty `tool_schemas` — `tools=tool_schemas` passed.
7. Message mutation — assistant message appended to input list.
8. `latency_ms` is positive float.
9. Logging — INFO "LLM response received" with latency_ms; DEBUG "LLM response content".
10. `reasoning_content` present — DEBUG log attributes.
11. Exception from `client.chat()` — `messages` list unchanged, exception propagates to caller.

## Design Report

`run_agent_turn()` is a focused, single-responsibility function: one LLM call, structured result, no side effects beyond message mutation and logging. It follows existing conventions from `run_agent_loop` (message mutation, logging patterns) while removing tool execution and looping. The `latency_ms` field is an addition to the original design doc spec, justified by existing usage patterns.

Recommendation: **yes**, proceed to review.

## Design Review

### Critical

**1. Message mutation location deviates from spec without explicit acknowledgment.**

`task-system-design.md` defines the caller's dispatch flow as: (1) Append assistant message to conversation, (2) dispatch tool calls as Tasks, (3) send text if no tool calls, (4) return. The design moves step 1 inside `run_agent_turn`. This is acknowledged in "Design decisions" but not flagged as a spec deviation.

The risk is concrete: a Phase 3 implementer reading the spec's dispatch steps could implement step 1 in the caller _and_ rely on `run_agent_turn` also appending — resulting in a duplicate message appended on every turn, corrupting the conversation history.

Fix: Add an explicit note in the design (or a Phase 3 callout) that the spec's step 1 is handled by `run_agent_turn` internally. The Phase 3 dispatch flow should begin at step 2.

### Important

**2. Missing test case: exception propagation leaves `messages` unmutated.**

`client.chat()` can raise `RuntimeError` (session not started) or `aiohttp.ClientResponseError` (HTTP error). Because the append (step 6) happens after the `await` (step 2), the current step order is correct — but there is no test verifying this. A caller that retries on a transient error needs the guarantee that `messages` is unchanged after a raise.

Fix: Add test case 11 — `client.chat()` raises an exception; verify `messages` is unchanged and the exception propagates unmodified.

**3. `latency_ms` is logged twice at INFO level.**

`LLMClient.chat()` already emits INFO "chat completion returned" with `latency_ms`. `run_agent_turn` emits a second INFO "LLM response received" with the same value. This matches `run_agent_loop`'s existing behavior (acknowledged in the design), but means every LLM call produces two INFO entries for the same timing datum.

This is not a blocker, but the design should explicitly state this redundancy is intentional (matching existing convention), or drop `latency_ms` from the "LLM response received" log since the client layer already captured it.

### Cosmetic

**4. `tool_calls` normalization (`[]` vs `None`) is an improvement over both the spec and `run_agent_loop` but is not called out.**

The spec's pseudocode and `run_agent_loop` both allow `tool_calls` to be `None`. The design normalizes it to `[]`. This simplifies callers. Worth a one-liner in "Design decisions" so the behavioral guarantee is explicit.

**5. `latency_ms` docstring could note precision.**

`run_agent_loop` uses `round(..., 1)`. The `AgentTurnResult` docstring says "Wall-clock time for the LLM call in milliseconds" but doesn't mention rounding. Minor, but consistent with the level of detail elsewhere.

### Recommendation

**Yes, proceed.** The design is correct and well-scoped. Two fixes before implementation: (1) document the spec deviation on message mutation to prevent a Phase 3 bug, and (2) add the exception-propagation test case. The double-logging issue can be resolved during implementation.

## Design Fix #1 Report

### Changes made

**Design decisions section** — two bullets added:
- `tool_calls` normalized to `[]`: documents that `run_agent_turn` always returns a list, unlike `run_agent_loop` and the spec pseudocode which allow `None`. Addresses review finding #4 (cosmetic).
- Double-logging of `latency_ms` is intentional: documents that the redundancy is deliberate and maps to distinct logging layers (transport vs. application). Addresses review finding #3 (important).

**Relationship to run_agent_loop section** — one bullet added:
- Spec deviation — message append: flags that Phase 3 implementers must skip spec step 1 because `run_agent_turn` performs the append internally. Addresses review finding #1 (critical).

**Edge cases to test section** — one case added:
- Case 11: Exception from `client.chat()` — verifies `messages` is unchanged and exception propagates unmodified. Addresses review finding #2 (important).

**`AgentTurnResult` docstring** — `latency_ms` description updated to note rounding to one decimal place. Addresses review finding #5 (cosmetic).

### Proceed?

**Yes.**

## Design Re-review #1

### Verification of fixes

All five original findings verified as fixed:

1. **(Critical) Spec deviation on message mutation** — FIXED. Lines 82-83 document the deviation with Phase 3 guidance.
2. **(Important) Missing exception propagation test** — FIXED. Edge case 11 added at line 96.
3. **(Important) Double-logging of latency_ms** — FIXED. Line 75 documents intentional redundancy with rationale.
4. **(Cosmetic) tool_calls normalization** — FIXED. Line 74 documents `[]` guarantee.
5. **(Cosmetic) latency_ms precision** — FIXED. Line 18 notes rounding.

### New issues identified

None.

### Recommendation

**Yes, proceed to implementation.** All findings addressed, no new issues introduced.

## Red TDD Review

### Summary

8 test functions covering all 11 edge cases. Skip guard correct. One test has a wrong assertion that will cause a false failure on a correct implementation.

### Critical

**C1. Wrong `assert_awaited_once_with` in `test_run_agent_turn_tool_calls_only`.**

The assertion checks `messages[:1] + [messages[-1]]` after mutation, which evaluates to `[user_msg, assistant_msg]`. But `client.chat` is called at step 2 (before append at step 6) — a correct implementation calls it with `[user_msg]` only. Fix: snapshot before the call.

### Important

**I1. `result.message` never asserted.** No test checks the `message` field of `AgentTurnResult`. Add `assert result.message is messages[-1]` and `assert result.message.get("role") == "assistant"`.

**I2. Case 5 accepts both `tools=None` and missing `tools` kwarg.** Known permissiveness.

**I3. `tool_calls_count`/`role` on INFO log not checked.** Consistent with existing tests.

### Cosmetic

**M1.** `latency_ms` precision not verified. **M2.** Redundant `is not None` assertion.

### Recommendation

**No.** Fix C1 and I1 before proceeding to green phase.

## Red TDD Re-review #1

### Verification

**C1 (Critical):** FIXED. `test_run_agent_turn_tool_calls_only` now snapshots `messages` before the call (`expected_call_arg = list(messages)`) and uses the snapshot in `assert_awaited_once_with`.

**I1 (Important):** FIXED. `test_run_agent_turn_text_response` now asserts `result.message is messages[-1]` and `result.message.get("role") == "assistant"`.

### New issues

None.

### Recommendation

**Yes, proceed to green phase.**

## Green TDD Report

### Implementation

Added to `sherman/agent_loop.py` before `run_agent_loop`:

- `from dataclasses import dataclass` import added.
- `AgentTurnResult` dataclass with fields `message`, `tool_calls`, `text`, `latency_ms`.
- `run_agent_turn()` async function following the 10-step behavior spec.

### Deviation from spec

`client.chat()` is called with `list(messages)` (a shallow copy) rather than `messages` directly. This is required to make `test_run_agent_turn_tool_calls_only` pass.

The test snapshots messages before mutation with `expected_call_arg = list(messages)` and then calls `client.chat.assert_awaited_once_with(expected_call_arg, tools=tool_schemas)`. Python's `AsyncMock` stores argument references, not copies — so if `messages` is passed directly, the mock records the live list reference, which by assertion time contains the appended assistant message. Passing a copy means the mock records a reference to a list that is never mutated, so the assertion sees the pre-mutation state.

The shallow copy is safe: `client.chat()` only reads the list and its contained dicts (it doesn't mutate them), and the messages list is still mutated in place after the call (step 6).

### Test results

34 passed, 0 skipped, 0 failed.

### Proceed?

**Yes.**
