"""Invariants of the container image (PRD §3.11.12, #37).

These read the Dockerfile and .dockerignore as text. They can't prove the image
builds (the manual `docker build` in docs/configuration.md does), but they pin
the properties that fail silently if someone edits the files carelessly: a
port exposed by default, the daemon running as root, an unlocked dependency
install, or a build context that leaks `agent.yaml`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dockerfile() -> list[str]:
    """Dockerfile instructions, one per logical line (continuations joined)."""
    path = ROOT / "Dockerfile"
    assert path.exists(), f"{path} is missing; PRD §3.11.12 describes the image it holds"
    text = path.read_text()
    # Join backslash continuations so a multi-line RUN is one instruction.
    text = re.sub(r"\\\n\s*", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop blanks and comments; what's left is instructions.
    return [ln for ln in lines if ln and not ln.startswith("#")]


@pytest.fixture(scope="module")
def runtime(dockerfile: list[str]) -> list[str]:
    """Instructions of the final stage only, which is what the shipped image inherits.

    Settings in the builder stage (ENV, WORKDIR, USER) don't carry over, so a test about
    the image has to read this, not the whole file.
    """
    last_from = max(i for i, ln in enumerate(dockerfile) if ln.upper().startswith("FROM "))
    return dockerfile[last_from:]


@pytest.fixture(scope="module")
def ignored() -> list[str]:
    """Non-comment patterns in .dockerignore."""
    path = ROOT / ".dockerignore"
    assert path.exists(), f"{path} is missing; the build context would include agent.yaml"
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _instructions(lines: list[str], name: str) -> list[str]:
    """Arguments of every `name` instruction, in file order."""
    prefix = name + " "
    return [ln[len(prefix) :] for ln in lines if ln.upper().startswith(prefix)]


def test_entrypoint_is_corvidae_and_default_command_serves(runtime):
    """Bare `corvidae` prints help, so the image has to default to `serve`."""
    assert _instructions(runtime, "ENTRYPOINT") == ['["corvidae"]']
    assert _instructions(runtime, "CMD") == ['["serve"]']


def test_working_directory_is_data(runtime):
    """`agent.yaml`, sessions.db and logs resolve against the working directory."""
    assert _instructions(runtime, "WORKDIR")[-1] == "/data"
    # Exactly /data, not a subdirectory of it: a substring match would accept /data/x.
    assert _instructions(runtime, "VOLUME") == ["/data"]


def test_no_port_is_exposed(runtime):
    """PRD §3.11.12: no ports unless `web:` is configured, and then the operator publishes."""
    assert _instructions(runtime, "EXPOSE") == []


def test_final_user_is_not_root(runtime):
    """Tools run unsandboxed as the daemon user (threat model, finding 4)."""
    users = _instructions(runtime, "USER")
    assert users, "no USER instruction: the daemon would run as root"
    assert users[-1].split(":")[0] not in {"root", "0"}


def test_dependencies_install_from_the_lock(dockerfile):
    """PRD §3.11.7: uv.lock pins everything, and the build must fail if it is stale."""
    syncs = [ln for ln in _instructions(dockerfile, "RUN") if "uv sync" in ln]
    assert syncs, "no `uv sync` in the Dockerfile"
    for sync in syncs:
        assert "--locked" in sync, f"unlocked install: RUN {sync}"
        assert "--no-dev" in sync, f"dev dependencies in the image: RUN {sync}"


def test_workspace_members_reach_the_builder(dockerfile):
    """`uv sync --locked` needs every workspace member on disk to validate the lock."""
    copies = " ".join(_instructions(dockerfile, "COPY"))
    assert "packages" in copies, "packages/ is not copied; --locked cannot resolve the workspace"
    assert "uv.lock" in copies, "uv.lock is not copied into the build"


def test_apluggy_git_dependency_has_a_git_binary(dockerfile):
    """apluggy is pinned to a git commit, so the builder stage needs git to fetch it."""
    runs = " ".join(_instructions(dockerfile, "RUN"))
    assert re.search(r"apt-get install[^&]*\bgit\b", runs)


def test_extras_are_a_build_arg_defaulting_to_the_kernel(dockerfile):
    """The default image is the kernel alone; extras are opt-in at build time."""
    args = [a for a in _instructions(dockerfile, "ARG") if a.split("=", 1)[0] == "EXTRAS"]
    assert args, "no EXTRAS build arg"
    # An empty default means the kernel alone; a non-empty one would ship an extra unasked.
    for arg in args:
        default = arg.partition("=")[2].strip("\"'")
        assert default == "", f"EXTRAS defaults to {default!r}, not the kernel alone"
    assert any("--extra" in ln for ln in _instructions(dockerfile, "RUN"))


@pytest.mark.parametrize("pattern", [".git", ".venv", "agent.yaml", "state", "*.db"])
def test_dockerignore_keeps_local_state_and_secrets_out(ignored, pattern):
    """`agent.yaml` holds API keys; the others are local state or bulk."""
    assert any(p.rstrip("/") == pattern for p in ignored), f"{pattern!r} is not in .dockerignore"


@pytest.mark.parametrize("needed", ["uv.lock", "pyproject.toml", "corvidae", "packages"])
def test_dockerignore_does_not_hide_build_inputs(ignored, needed):
    """Ignoring one of these breaks `uv sync --locked`, with an error far from the cause."""
    assert not any(p.rstrip("/") == needed for p in ignored), f"{needed!r} is needed to build"


def test_tiktoken_encoding_is_baked_into_the_image(dockerfile, runtime):
    """tiktoken fetches cl100k_base on first import; a container would redo that on every
    recreate, and a host with no outbound network would silently fall back to a character
    estimate. The build fetches it once, where the network is, and fails loudly if it can't."""
    # The builder's ENV doesn't reach the shipped image, so read the final stage.
    env = " ".join(_instructions(runtime, "ENV"))
    assert "TIKTOKEN_CACHE_DIR=" in env, "the runtime stage does not point tiktoken at a cache"
    runs = " ".join(_instructions(dockerfile, "RUN"))
    assert "get_encoding" in runs and "cl100k_base" in runs, "the build never warms the cache"
    copies = " ".join(_instructions(runtime, "COPY"))
    assert "--from=builder" in copies and "tiktoken" in copies, "the warmed cache is not copied"
