"""Runtime-tunable settings resolution — the two-surface seam.

Any plugin parameter must be adjustable at runtime without a daemon
restart, through BOTH tuning surfaces (operator directive 2, 2026-07-06):

  1. the agent-facing ``set_settings`` tool (``RuntimeSettingsPlugin`` →
     ``channel.runtime_overrides``) — per-channel;
  2. operator hot config reload (``ConfigWatcherPlugin`` →
     ``on_config_reload``) — global.

Plugins call :func:`resolve_tunable` at decision time — never caching the
resolved value at init — so both surfaces take effect immediately (trap #8).
Dotted keys (``"memory.retrieval_k"``) are the namespace convention for
plugin tunables; the step-7 ``extra_body`` filter in ``agent.py`` excludes
any dotted key from the LLM request body.
"""

from typing import Any


def resolve_tunable(channel, config: dict, key: str, default: Any) -> Any:
    """Per-decision setting resolution (operator directive 2, 2026-07-06).

    Order (first hit returns):
      1. ``channel.runtime_overrides[key]``   — set_settings, per-channel
      2. ``config`` walked by dotted path     — agent.yaml, hot-reloadable
      3. ``default``                          — best-guess constant

    ``channel`` may be any duck-typed object exposing ``runtime_overrides``.
    The dotted-path walk tolerates missing intermediate keys and non-dict
    nodes by falling through to ``default``.

    Intentional divergence: this does NOT reuse ``ChannelConfig.resolve``
    (channel.py) — that method merges a fixed set of typed framework fields
    last-wins; this is a per-decision first-hit-wins lookup for arbitrary
    dotted plugin keys. Do not unify them: doing so would reintroduce the
    FRAMEWORK_KEYS-only semantics this seam exists to escape.
    """
    # Surface 1: per-channel runtime overrides set via the set_settings tool.
    overrides = getattr(channel, "runtime_overrides", None)
    if overrides is not None and key in overrides:
        return overrides[key]

    # Surface 2: walk the (hot-reloadable) config dict by dotted path. Any
    # missing or non-dict intermediate node means the key is not configured.
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
