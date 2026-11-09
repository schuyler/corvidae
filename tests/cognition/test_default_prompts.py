"""Shipped default prompts (#380, #388): R4 (the consolidation instruction
tells the agent its record is private memory) and R9 (no shipped default
names an operator)."""

import ast
from pathlib import Path

from corvidae_cognition.memory import DEFAULT_CONSOLIDATION_FORK_PROMPT

REPO = Path(__file__).resolve().parents[2]

#: The R4 sentence, added to the default consolidation instruction.
PRIVATE_MEMORY_SENTENCE = (
    "This record is your private memory, not a message to anyone on the "
    "channel: no one there will read it, so write it for yourself."
)


def _flat(text: str) -> str:
    """``text`` with every whitespace run collapsed to one space, so a
    markdown copy wrapped across lines still contains the sentence."""
    return " ".join(text.split())


def test_consolidation_instruction_says_private_memory():
    """R4: the code default and its documented copy in prompts/ both tell
    the agent the record is its private memory, not a message to anyone."""
    copy = (REPO / "prompts" / "memory_consolidation_fork.md").read_text()

    assert PRIVATE_MEMORY_SENTENCE in _flat(DEFAULT_CONSOLIDATION_FORK_PROMPT)
    assert PRIVATE_MEMORY_SENTENCE in _flat(copy)


def _prompt_strings(path: Path) -> list[tuple[str, str]]:
    """(name, text) of every string in an assignment, anywhere in the module
    (class attributes included), whose target name contains ``PROMPT``, read
    from source (no import, so optional SDKs a module needs do not matter)."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        for name in names:
            if "PROMPT" not in name:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    found.append((name, sub.value))
    return found


def test_no_shipped_default_names_an_operator():
    """R9: no ``*PROMPT*`` string assignment (module or class level) in ``corvidae`` or
    ``corvidae_cognition``, and no ``prompts/*.md`` copy, contains an
    operator's name. An example in a default prompt must be a placeholder."""
    sources = [
        *sorted((REPO / "corvidae").rglob("*.py")),
        *sorted((REPO / "packages" / "corvidae-cognition" / "corvidae_cognition").rglob("*.py")),
    ]
    offenders = [
        f"{path.relative_to(REPO)}:{name}"
        for path in sources
        for name, text in _prompt_strings(path)
        if "Schuyler" in text
    ]
    offenders += [
        str(path.relative_to(REPO))
        for path in sorted((REPO / "prompts").glob("*.md"))
        if "Schuyler" in path.read_text()
    ]

    assert not offenders, f"shipped defaults name an operator: {offenders}"


#: The #635 consolidation instruction: paragraphs separated by a blank line,
#: the JSON template's lines separated by newlines.
_OPENING = (
    "[Memory consolidation] Pause the conversation here and write a record of "
    "part of it for your long-term autobiographical memory, in the first "
    "person, as yourself. " + PRIVATE_MEMORY_SENTENCE
)
_NARRATE = (
    "Narrate this stretch of the conversation from your own point of view, "
    "step by step, in your own words: what was said, what you did, what you "
    "learned, and what was decided or left open. Don't quote anyone verbatim, "
    "but keep facts and opinions exactly as they were expressed, and who "
    "expressed each (names, numbers, dates, identifiers, preferences, "
    "commitments, corrections), so you can recall them accurately later. Keep "
    "what you were told apart from what you inferred or speculated "
    '("so-and-so told me...", "I guessed that..."). Say what you did with '
    "tools and what it told you, not their raw output. Bracketed CONTEXT "
    "blocks are things you recalled or were shown, not new events; don't "
    "record them as if they happened now. Skip pleasantries and chatter."
)
_ENTRIES = (
    "Write it as separate entries, one for each distinct stretch or topic, in "
    "order. Each entry is a paragraph at most and must make sense read on its "
    "own, months from now, with nothing else: name the people, projects and "
    "things involved rather than \"he\" or \"it\". Don't attribute a fact from "
    "one entry's topic to another's."
)
_RATE = (
    "Then rate how important this stretch is to remember long-term, from 0 "
    "(routine) to 1 (essential). Do not call tools."
)
_TEMPLATE = "\n".join([
    "Respond with a single JSON object, nothing else:",
    "{",
    '  "entries": [',
    '    {"narrative": "<first-person narrative of one stretch or topic>",',
    '     "topic_tags": ["<short topic tag>", "..."]}',
    "  ],",
    '  "participants": ["<sender name>", "..."],',
    '  "importance": "<number from 0 to 1>"',
    "}",
])


def test_consolidation_instruction_asks_for_entries():
    """C5: the code default is exactly the entries-shape instruction, and the
    prompts/ copy carries the same body paragraphs and template keys."""
    expected = "\n\n".join([_OPENING, _NARRATE, _ENTRIES, _RATE, _TEMPLATE])
    copy = _flat((REPO / "prompts" / "memory_consolidation_fork.md").read_text())

    assert DEFAULT_CONSOLIDATION_FORK_PROMPT == expected
    for paragraph in (_NARRATE, _ENTRIES, _RATE):
        assert _flat(paragraph) in copy
    for key in ('"entries"', '"narrative"', '"topic_tags"', '"participants"', '"importance"'):
        assert key in copy
