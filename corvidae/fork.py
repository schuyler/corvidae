"""Forks — background processes run as branches of the main window (#51).

A fork is one LLM call whose prompt is a prefix of the channel's rendered
window plus one appended ``user`` instruction. Because that prompt is a
strict prefix of something llama-server already holds in its slot, the fork
reuses the cached KV prefix instead of prefilling a separate prompt, and it
sees the agent's own system prompt, so first-person output is written by
the agent it describes (plans/forks.md Part 1).

This module holds the pure pieces: the prompt builder, the tool-block cut
snap, the ``agent.forks`` config parser, and the result/error types. The
call itself is ``Agent.fork`` (agent.py), which owns the five inputs the
prefix depends on (client, tool schemas, build_prompt arguments, runtime
override filter, reasoning effort). Plugins reach it through
:func:`fork_agent`.

Config:
    agent:
      forks:
        enabled: false                # opt-in until measured on Lex (#73)
        reasoning_budget_tokens: 2048 # per fork call; null = server default
        stages: null                  # null = every stage; or a list of names

Stage names are opaque here: core never names the cognition plugins
(tests/test_correlation_core.py), so each plugin passes its own stage name
to :func:`fork_agent` and a name nobody uses simply matches nothing.
"""

from __future__ import annotations

import json

from dataclasses import dataclass

from corvidae.context import ContextWindow

#: Default per-fork reasoning budget. A fork keeps the main reasoning effort
#: (it is rendered into the system block, so changing it breaks the cached
#: prefix) and bounds its think with this per-request budget instead.
DEFAULT_FORK_REASONING_BUDGET = 2048

#: Characters of a message quoted in a fork instruction to name where an
#: exchange or a consolidation range begins.
QUOTE_CHARS = 120

_FORKS_KEYS = frozenset({"enabled", "reasoning_budget_tokens", "stages"})


class ForkRefused(Exception):
    """The model answered a fork with tool calls instead of text.

    Forks never dispatch tools (``tool_choice: "none"`` plus this check),
    so the caller treats it as a failed fork and falls back.
    """


class ForkUnavailable(Exception):
    """A fork could not be built: no live window or no main client."""


@dataclass(frozen=True)
class ForkResult:
    """What a fork returns.

    Attributes:
        text: The reply's content ("" when the model produced none).
        message: The raw assistant message dict from the response.
        usage: The response's usage envelope, or None.
    """

    text: str
    message: dict
    usage: dict | None = None


@dataclass(frozen=True)
class ForksConfig:
    """Parsed ``agent.forks`` section. Static operator flags, never tunables."""

    enabled: bool = False
    reasoning_budget_tokens: int | None = DEFAULT_FORK_REASONING_BUDGET
    #: None means every stage; otherwise only the named stages fork.
    stages: frozenset[str] | None = None

    def stage_enabled(self, stage: str) -> bool:
        """Whether ``stage`` runs as a fork under this config."""
        return self.enabled and (self.stages is None or stage in self.stages)


def parse_forks_config(agent_config: dict) -> ForksConfig:
    """Parse and validate ``agent.forks`` from the agent config section.

    Raises:
        ValueError: on an unknown key, a non-boolean ``enabled``, a budget
            that is not a non-negative integer or null, or ``stages`` that
            is neither null nor a list of strings.
    """
    raw = agent_config.get("forks")
    if raw is None:
        return ForksConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"agent.forks must be a mapping; got {raw!r}")

    # Unknown keys are rejected so a typo doesn't silently leave forks off.
    unknown = set(raw) - _FORKS_KEYS
    if unknown:
        raise ValueError(f"agent.forks: unknown key(s) {sorted(unknown)}")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"agent.forks.enabled must be true or false; got {enabled!r}")

    # bool is an int subclass; reject it explicitly so `true` isn't a budget of 1.
    budget = raw.get("reasoning_budget_tokens", DEFAULT_FORK_REASONING_BUDGET)
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
    ):
        raise ValueError(
            "agent.forks.reasoning_budget_tokens must be a non-negative integer "
            f"or null; got {budget!r}"
        )

    stages_raw = raw.get("stages")
    if stages_raw is not None and (
        not isinstance(stages_raw, list) or not all(isinstance(s, str) for s in stages_raw)
    ):
        raise ValueError(
            f"agent.forks.stages must be a list of stage names or null; got {stages_raw!r}"
        )

    return ForksConfig(
        enabled=enabled,
        reasoning_budget_tokens=budget,
        stages=frozenset(stages_raw) if stages_raw is not None else None,
    )


def build_fork_prompt(
    conv: ContextWindow,
    cut: int | None,
    instruction: str,
    pending_ids: frozenset,
    mid_role: str,
) -> list[dict]:
    """``build_prompt(window[:cut])`` plus one appended user instruction.

    Pure: renders a throwaway window holding the first ``cut`` messages, so
    the channel's window is never touched. ``cut`` None means the whole
    window. The render is a prefix of the main turn's render only when no
    tool cycle straddles ``cut`` (see :func:`snap_cut_to_tool_boundary`).

    Raises:
        ValueError: if ``cut`` is outside ``0..len(conv.messages)``.
    """
    n = len(conv.messages)
    if cut is None:
        cut = n
    if not 0 <= cut <= n:
        raise ValueError(f"fork cut {cut} outside 0..{n}")

    # Same system prompt and the same build_prompt arguments as step 7 of
    # the agent loop; only the message list is truncated.
    prefix = ContextWindow(conv.channel_id, chars_per_token=conv.chars_per_token)
    prefix.system_prompt = conv.system_prompt
    prefix.messages = conv.messages[:cut]
    prompt = prefix.build_prompt(pending_ids, mid_conversation_system_role=mid_role)
    prompt.append({"role": "user", "content": instruction})
    return prompt


def _first_result_index(messages: list[dict]) -> dict[str, int]:
    """Index of the first result for each tool_call_id (build_prompt's rule)."""
    result_at: dict[str, int] = {}
    for i, m in enumerate(messages):
        call_id = m.get("tool_call_id")
        if m.get("role") == "tool" and call_id and call_id not in result_at:
            result_at[call_id] = i
    return result_at


def snap_cut_forward(messages: list[dict], cut: int) -> int:
    """Move ``cut`` forward past the last result of every straddling cycle.

    The other way to clear a straddle (see
    :func:`snap_cut_to_tool_boundary`): drop the whole cycle instead of
    keeping it. Moving forward can straddle a later call whose results sit
    further on, so this repeats until stable. May return ``len(messages)``.
    """
    result_at = _first_result_index(messages)
    while True:
        new_cut = cut
        for i in range(cut):
            for c in messages[i].get("tool_calls") or []:
                j = result_at.get(c["id"], -1)
                if j >= new_cut:
                    new_cut = j + 1
        if new_cut == cut:
            return cut
        cut = new_cut


def snap_cut_to_tool_boundary(messages: list[dict], cut: int) -> int:
    """Move ``cut`` back until no tool cycle straddles it.

    build_prompt emits an assistant ``tool_calls`` message at the position
    of its last result. If a call sits before ``cut`` and any of its results
    at or after it, the prefix renders the call in place with placeholder
    results while the full window renders it later with the real ones, so
    the two prompts diverge there (and compaction would drop the call while
    keeping its results). Backing up to the call can land inside an earlier
    cycle, so this repeats until stable. A call with no result anywhere
    (in flight or abandoned) renders in place either way and never
    straddles.
    """
    result_at = _first_result_index(messages)

    # Back up to the earliest straddling call; repeat to a fixpoint.
    while True:
        new_cut = cut
        for i in range(cut):
            calls = messages[i].get("tool_calls") or []
            if any(result_at.get(c["id"], -1) >= cut for c in calls):
                new_cut = i
                break
        if new_cut == cut:
            return cut
        cut = new_cut


def looks_like_tool_call(text: str) -> bool:
    """Whether a fork's text reply is a tool call written out as text.

    ``tool_choice: "none"`` stops the server emitting structured
    ``tool_calls``, but the tools are still in the prompt and a model can
    write call markup into its content instead. That must not be taken as
    a summary, record or verdict: a bare ``{"name": ..., "arguments": ...}``
    object would even parse as a reply with nothing to report.
    """
    stripped = text.strip()
    if stripped.startswith(("<tool_call>", "<function")):
        return True
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            return False
        return isinstance(data, dict) and "name" in data and "arguments" in data
    return False


def window_db_ids(messages: list[dict]) -> set[int]:
    """The ``message_log`` rowids (``_db_id`` tags) present in a window's
    messages (or a slice of them)."""
    return {m["_db_id"] for m in messages if isinstance(m.get("_db_id"), int)}


def quote(text: str | None, limit: int = QUOTE_CHARS) -> str:
    """The first ``limit`` characters of ``text`` on one line, for naming a
    message in a fork instruction."""
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def fork_agent(pm, stage: str):
    """Return the Agent plugin when ``stage`` runs as a fork, else None.

    Duck-typed (``pm.get_plugin("agent")``), the same pattern the
    cognition plugins use for each other, so plugins don't import
    Agent and a disabled agent plugin degrades to the standalone path.
    """
    if pm is None:
        return None
    agent = pm.get_plugin("agent")
    enabled = getattr(agent, "fork_enabled", None)
    if not callable(enabled) or not enabled(stage):
        return None
    return agent
