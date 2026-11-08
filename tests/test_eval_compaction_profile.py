"""scripts/eval_compaction.py resolves the stage profile the daemon uses (#470).

The eval measures compaction quality, so it must send to the model the
compaction stage actually resolves to -- through llm.stages, llm.defaults and
`extends` -- not to llm.main read raw.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_compaction.py"


@pytest.fixture(scope="module")
def eval_script():
    """Import the script as a module, since its name is not a valid import."""
    spec = importlib.util.spec_from_file_location("eval_compaction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_compaction"] = module
    spec.loader.exec_module(module)
    return module


BASE = {"base_url": "http://base:8080", "model": "base-model"}


def test_stage_profile_wins(eval_script):
    """R470.1: llm.stages.compaction names the profile to use."""
    llm = {
        "main": {"base_url": "http://main:1", "model": "main-model"},
        "foo": {"base_url": "http://foo:2", "model": "foo-model", "api_key": "k"},
        "stages": {"compaction": "foo"},
    }
    resolved = eval_script.resolve_llm_defaults(llm)
    assert resolved["base_url"] == "http://foo:2"
    assert resolved["model"] == "foo-model"
    assert resolved["api_key"] == "k"


def test_defaults_chat_without_main(eval_script):
    """R470.2: no llm.main, but llm.defaults.chat names one."""
    llm = {
        "foo": {"base_url": "http://foo:2", "model": "foo-model"},
        "defaults": {"chat": "foo"},
    }
    resolved = eval_script.resolve_llm_defaults(llm)
    assert resolved["base_url"] == "http://foo:2"
    assert resolved["model"] == "foo-model"


def test_extends_inherits_keys(eval_script):
    """R470.3: a profile built with extends resolves its inherited keys."""
    llm = {
        "base": dict(BASE),
        "child": {"extends": "base", "model": "child-model"},
        "defaults": {"chat": "child"},
    }
    resolved = eval_script.resolve_llm_defaults(llm)
    assert resolved["base_url"] == "http://base:8080"
    assert resolved["model"] == "child-model"


def test_malformed_llm_raises_a_config_error(eval_script):
    """R470.4: a malformed section is a ValueError naming the problem."""
    with pytest.raises(ValueError) as exc:
        eval_script.resolve_llm_defaults({"main": {"model": "no-base-url"}})

    assert "base_url" in str(exc.value)


def test_no_default_chat_profile_raises(eval_script):
    """§3.11.1: a config with no chat default is invalid, and the error says
    what to define. `parse_llm_config` enforces this, so the script inherits it."""
    with pytest.raises(ValueError) as exc:
        eval_script.resolve_llm_defaults({})

    message = str(exc.value)
    assert "default chat profile" in message or "llm.defaults.chat" in message


def test_script_reports_a_missing_corvidae_import_clearly(tmp_path):
    """Run outside the project environment, it says how to run it (#470 review).

    Before the module-scope guard, a plain interpreter died with a bare
    ModuleNotFoundError from inside a helper, which reads as a broken script
    rather than the wrong environment.
    """
    import subprocess
    import sys as _sys

    # No corvidae on the path, and none of the venv's site-packages.
    result = subprocess.run(
        [_sys.executable, "-S", str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=20,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""},
    )
    output = result.stdout + result.stderr
    assert "uv run" in output or result.returncode == 0
