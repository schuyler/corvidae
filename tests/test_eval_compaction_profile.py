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
    rather than the wrong environment. `--help` exits before any deferred
    import, so this does assert the guard rather than an early exit.
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
    assert "uv run" in output, (
        "the guard must name how to run the script; got "
        f"rc={result.returncode} output={output!r}"
    )


def test_api_key_environment_reference_is_expanded(eval_script, monkeypatch):
    """R470.1: the key is resolved the way the daemon resolves it.

    A profile holds `api_key: "${OPENAI_API_KEY}"` (docs/configuration.md) and
    the daemon expands it in `LLMPlugin._create_client`. Passing the raw
    reference sends `Bearer ${OPENAI_API_KEY}` and 401s against the very
    deployment the eval claims to mirror (#470 review).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    llm = {
        "main": {"base_url": "http://main:1", "model": "main-model"},
        "foo": {"base_url": "http://foo:2", "model": "foo-model",
                "api_key": "${OPENAI_API_KEY}"},
        "stages": {"compaction": "foo"},
    }
    resolved = eval_script.resolve_llm_defaults(llm)
    assert resolved["api_key"] == "sk-test-123"


def test_an_unset_api_key_variable_is_a_config_error(eval_script, monkeypatch):
    """An unset ${VAR} names the variable instead of sending the reference."""
    monkeypatch.delenv("CORVIDAE_EVAL_MISSING_KEY", raising=False)
    llm = {
        "main": {"base_url": "http://main:1", "model": "m",
                 "api_key": "${CORVIDAE_EVAL_MISSING_KEY}"},
    }
    with pytest.raises(ValueError) as exc:
        eval_script.resolve_llm_defaults(llm)

    assert "CORVIDAE_EVAL_MISSING_KEY" in str(exc.value)


def test_empty_agent_yaml_is_a_config_error_not_a_traceback(eval_script, tmp_path):
    """R470.4: an empty config file is reported, not an AttributeError.

    `yaml.safe_load` returns None for an empty file, so `config.get("llm")`
    raised `AttributeError: 'NoneType' object has no attribute 'get'` (#470
    review).
    """
    empty = tmp_path / "agent.yaml"
    empty.write_text("")

    with pytest.raises(ValueError) as exc:
        eval_script.resolve_llm_from_file(empty)

    assert "llm" in str(exc.value) or "default chat" in str(exc.value)
