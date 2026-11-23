"""Shared fixtures for the cognition tests."""

import pytest

from corvidae.llm_plugin import LLMPlugin
from helpers import KERNEL_LLM_STAGES

#: Stages the memory plugin registers through register_llm_stages.
COGNITION_LLM_STAGES = {
    "consolidation": "chat",
    "rubric": "chat",
    "embedding": "embedding",
}


@pytest.fixture(autouse=True)
def llm_stages_registered(monkeypatch):
    """Fixtures here build an LLMPlugin and inject clients without running
    on_start, so register_llm_stages never collects its owners' stages. Seed
    them, as on_start would; a test that runs on_start replaces the table
    with what its pm collects."""
    original = LLMPlugin.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._stages = {**KERNEL_LLM_STAGES, **COGNITION_LLM_STAGES}

    monkeypatch.setattr(LLMPlugin, "__init__", init)
