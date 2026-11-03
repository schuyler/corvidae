"""Behavioural eval for issue #215 (S2, S3): does the model follow the
silence instruction? Live LLM, deselected by default (``--run-eval``),
modelled on tests/test_compaction_quality.py TestLiveCompactionEval. Skips
without an agent.yaml carrying llm.main.base_url and model.

  (a) chat register (shipped chat register + REGISTER_FRAGMENT): closers get
      exactly NO_REPLY, a substantive question gets a real reply (S2, S3).
  (b) task register (no register, as ACP): an addressed request gets a real
      reply that is not NO_REPLY (S3, PRD 3.12.7). A closer is not asserted:
      a task register may stay silent where nothing was asked.
"""

from pathlib import Path

import pytest

from corvidae.channels._registers import shipped_register

IDENTITY = "You are a helpful assistant."
CLOSERS = ["thanks, night!", "lol ok"]
QUESTION = "Can you explain how a B-tree keeps itself balanced as it grows?"
REQUEST = "Please list, in order, the first three things you would check when a service will not start."


def _llm_config() -> tuple[str, str, str | None]:
    """(base_url, model, api_key) from agent.yaml; skips when absent."""
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "agent.yaml"
    if not config_path.exists():
        pytest.skip("No agent.yaml found")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    llm = config.get("llm", {}).get("main", {})
    if not llm.get("base_url") or not llm.get("model"):
        pytest.skip("LLM config not found in agent.yaml")
    return llm["base_url"], llm["model"], llm.get("api_key")


async def _reply(system: str, history: list[dict], user: str) -> str:
    """One live completion for ``user`` after ``history`` under ``system``."""
    import aiohttp

    from scripts.eval_compaction import call_llm

    base_url, model, api_key = _llm_config()
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]
    async with aiohttp.ClientSession() as session:
        # A reasoning model may return null content; treat it as empty.
        return ((await call_llm(session, base_url, model, api_key, messages)) or "").strip()


def _chat_system() -> str:
    """Identity plus the chat register plus the silence fragment."""
    # Skip on missing LLM config before importing the plugin, so the skip
    # wins over an import error.
    _llm_config()
    from corvidae_cognition.silence import REGISTER_FRAGMENT

    return IDENTITY + "\n\n" + shipped_register("chat") + "\n\n" + REGISTER_FRAGMENT


# A short, finished exchange for a closer to follow.
WOUND_DOWN = [
    {"role": "user", "content": "what's the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]


@pytest.mark.eval
@pytest.mark.timeout(180)
class TestLiveSilenceEval:
    """Run with: pytest --run-eval tests/cognition/test_silence_eval.py"""

    @pytest.mark.parametrize("closer", CLOSERS)
    async def test_chat_closer_gets_silence(self, closer):
        reply = await _reply(_chat_system(), WOUND_DOWN, closer)
        assert reply == "NO_REPLY", f"closer {closer!r} drew {reply!r}"

    async def test_chat_substantive_question_gets_a_reply(self):
        reply = await _reply(_chat_system(), [], QUESTION)
        assert reply and reply != "NO_REPLY"

    async def test_task_register_request_gets_a_reply(self):
        reply = await _reply(IDENTITY, [], REQUEST)
        assert reply and reply != "NO_REPLY"
