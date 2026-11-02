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
