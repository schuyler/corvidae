"""Grice lens plugin tests (#914).

Verifies that the AlignmentLensPlugin:
- contributes one sample-rate lens named ``alignment`` through on_critique_init
- names all four maxims in its fork prompt
- uses critique's objection format (no custom reply shape)
- has critique as an optional dependency only
"""



from corvidae_alignment import AlignmentLensPlugin


async def _contributed(plugin):
    """Run on_critique_init and return the lenses it added."""
    collected = []
    await plugin.on_critique_init({}, collected.append)
    return collected


class TestGriceLens:
    """The plugin contributes one valid lens."""

    async def test_contributes_one_lens(self):
        [lens] = await _contributed(AlignmentLensPlugin())
        assert lens.name == "alignment"

    def test_critique_is_an_optional_dependency(self):
        """A config that disables critique must still boot (docs/cognition.md)."""
        assert AlignmentLensPlugin.optional_depends_on == frozenset({"critique"})
        assert not AlignmentLensPlugin.depends_on

    async def test_brief_names_all_four_maxims(self):
        [lens] = await _contributed(AlignmentLensPlugin())
        prompt = lens.fork_prompt
        for maxim in ("Quantity", "Quality", "Relation", "Manner"):
            assert maxim in prompt, f"maxim {maxim} missing from fork prompt"

    async def test_brief_has_conservative_default(self):
        """The prompt must instruct: when in doubt, no objection."""
        [lens] = await _contributed(AlignmentLensPlugin())
        prompt = lens.fork_prompt
        assert "clear and material" in prompt
        assert "empty objections list" in prompt

    async def test_brief_uses_critiques_objection_format(self):
        """Critique appends its objections tail and parses only objections,
        so the brief must put the maxim and a quote of the offending passage
        in the claim (#914) and must not ask for a reply shape of its own."""
        [lens] = await _contributed(AlignmentLensPlugin())
        prompt = lens.fork_prompt
        assert (
            'The "claim" is the maxim name and a quote of the offending\n'
            "  passage from the reply."
            in prompt
        )
        assert '- The "objection" is a one-sentence rationale.' in prompt
        # No custom reply format (no JSON schema, no specific keys beyond
        # claim/objection)
        assert "json" not in prompt.lower()
        assert "schema" not in prompt.lower()

    async def test_brief_caps_at_four_objections(self):
        [lens] = await _contributed(AlignmentLensPlugin())
        assert "max 4" in lens.fork_prompt

    async def test_brief_one_objection_per_maxim(self):
        """The prompt caps at one objection per violated maxim."""
        [lens] = await _contributed(AlignmentLensPlugin())
        assert "One objection per violated maxim" in lens.fork_prompt


class TestAppraisalQuestion:
    """#1078 T10: the lens declares the question the appraisal fork answers."""

    async def test_lens_declares_a_one_line_question(self):
        [lens] = await _contributed(AlignmentLensPlugin())
        assert lens.appraisal_question.strip()
        assert "\n" not in lens.appraisal_question


class TestEntryPoint:
    """The entry point is registered under the name `plugins.disabled`
    matches, the one docs/design.md tells operators to list."""

    def test_entry_point_exists(self):
        from importlib.metadata import entry_points
        eps = entry_points(group="corvidae")
        names = [ep.name for ep in eps]
        assert "alignment" in names, (
            f"alignment entry point missing; found: {names}"
        )
        ep = next(ep for ep in eps if ep.name == "alignment")
        assert ep.value == "corvidae_alignment:AlignmentLensPlugin"
