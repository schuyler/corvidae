"""Grice lens plugin tests (#914).

Verifies that the GriceLensPlugin:
- contributes one sample-rate lens named ``grice`` through on_critique_init
- names all four maxims in its fork prompt
- uses critique's objection format (no custom reply shape)
- has critique as an optional dependency only
"""

from unittest.mock import Mock

import pytest

from corvidae_grice import GriceLensPlugin


async def _contributed(plugin):
    """Run on_critique_init and return the lenses it added."""
    collected = []
    await plugin.on_critique_init({}, collected.append)
    return collected


class TestGriceLens:
    """The plugin contributes one valid lens."""

    async def test_contributes_one_sample_rate_lens(self):
        [lens] = await _contributed(GriceLensPlugin())
        assert lens.name == "grice"
        assert lens.thresholds == ()
        assert lens.reply_patterns == ()
        assert lens.sample_rate == 0.1

    def test_critique_is_an_optional_dependency(self):
        """A config that disables critique must still boot (docs/cognition.md)."""
        assert GriceLensPlugin.optional_depends_on == frozenset({"critique"})
        assert not GriceLensPlugin.depends_on

    async def test_brief_names_all_four_maxims(self):
        [lens] = await _contributed(GriceLensPlugin())
        prompt = lens.fork_prompt
        for maxim in ("Quantity", "Quality", "Relation", "Manner"):
            assert maxim in prompt, f"maxim {maxim} missing from fork prompt"

    async def test_brief_has_conservative_default(self):
        """The prompt must instruct: when in doubt, no objection."""
        [lens] = await _contributed(GriceLensPlugin())
        prompt = lens.fork_prompt
        assert "clear and material" in prompt
        assert "empty objections list" in prompt

    async def test_brief_uses_critiques_objection_format(self):
        """Critique appends its objections tail and parses only objections,
        so the brief must put the maxim and violation in the claim and must
        not ask for a reply shape of its own."""
        [lens] = await _contributed(GriceLensPlugin())
        prompt = lens.fork_prompt
        assert "claim" in prompt
        assert "objection" in prompt
        # No custom reply format (no JSON schema, no specific keys beyond
        # claim/objection)
        assert "json" not in prompt.lower()
        assert "schema" not in prompt.lower()

    async def test_brief_caps_at_four_objections(self):
        [lens] = await _contributed(GriceLensPlugin())
        assert "max 4" in lens.fork_prompt

    async def test_brief_one_objection_per_maxim(self):
        """The prompt caps at one objection per violated maxim."""
        [lens] = await _contributed(GriceLensPlugin())
        assert "One objection per violated maxim" in lens.fork_prompt


class TestEntryPoint:
    """The entry point is registered correctly for plugin discovery."""

    def test_entry_point_exists(self):
        from importlib.metadata import entry_points
        eps = entry_points(group="corvidae")
        names = [ep.name for ep in eps]
        assert "corvidae_grice" in names, (
            f"corvidae_grice entry point missing; found: {names}"
        )
        ep = next(ep for ep in eps if ep.name == "corvidae_grice")
        assert ep.value == "corvidae_grice:GriceLensPlugin"
