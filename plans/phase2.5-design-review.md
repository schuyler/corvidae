# Phase 2.5 Design Review: Composable System Prompts

Reviewer: Chico
Date: 2026-04-23
Document reviewed: `plans/phase2.5-design.md`
Requirements source: `plans/design.md` § Phase 2.5

---

## Summary

The design is correct, well-grounded in the codebase, and covers all
requirements. Two important concerns need resolution before implementation.
No critical issues.

---

## Important (2)

### 4.1 `self.base_dir` type mismatch

The design proposes `self.base_dir: Path | None = None` on `AgentLoopPlugin`
but `resolve_system_prompt(base_dir: Path)` — a type mismatch. If
`_ensure_conversation` is called before `on_start`, `None` flows into a
`Path`-typed parameter.

**Fix:** Use `self.base_dir: Path = Path(".")` in `__init__`. Drop `| None`.

### 4.2 Test setup for `test_ensure_conversation_resolves_file_list` underspecified

The test plan lists this test but doesn't describe how `base_dir` gets set up
in a test that bypasses `on_start`. The existing `_build_plugin_and_channel`
helper doesn't call `on_start`.

**Fix:** Specify: create temp files with `tmp_path`, set
`plugin.base_dir = tmp_path`, then call `on_message`.

---

## Cosmetic (4)

1. Absolute path behavior works via Python `Path.__truediv__` semantics but
   should be noted in the `resolve_system_prompt()` docstring.
2. `_base_dir` config dict injection vs. hookspec parameter left as open
   question — should be decided before implementation.
3. Empty list returning `""` is consistent with existing override semantics
   but should be documented in the docstring.
4. Conversation-level vs. turn-level prompt freezing should be stated
   explicitly for Phase 5 hot-reload clarity.

---

## Overall Assessment

**Needs minor revision.** Neither important concern requires redesign — both
are resolved with a one-line clarification in the design document.
