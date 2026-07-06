"""Deterministic eval-harness foundations (metric functions, fixture format).

Everything in this package that runs in CI is red/green with no network.
Live LLM-judge evaluation lives in scripts/ and behind the `eval` pytest
marker (deselected by default, run with --run-eval).
"""
