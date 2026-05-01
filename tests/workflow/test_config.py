"""Tests for workflow.config — load_agent_config, create_llm_client."""

import pytest
from workflow.config import load_agent_config, create_llm_client


def test_load_agent_config_reads_yaml(tmp_path):
    """load_agent_config returns parsed YAML dict."""
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost:8080\n    model: test\n"
    )
    config = load_agent_config(config_file)
    assert config["llm"]["main"]["base_url"] == "http://localhost:8080"
    assert config["llm"]["main"]["model"] == "test"


def test_load_agent_config_missing_file_raises():
    """load_agent_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_agent_config("/nonexistent/agent.yaml")


def test_create_llm_client_returns_client(tmp_path):
    """create_llm_client creates an LLMClient from config."""
    config = {
        "llm": {
            "main": {
                "base_url": "http://localhost:8080",
                "model": "test-model",
                "api_key": "sk-test",
            }
        }
    }
    client = create_llm_client(config)
    assert client.base_url == "http://localhost:8080"
    assert client.model == "test-model"
    assert client.api_key == "sk-test"


def test_create_llm_client_missing_main_raises():
    """create_llm_client raises KeyError when llm.main is missing."""
    with pytest.raises(KeyError):
        create_llm_client({"llm": {}})


def test_create_llm_client_optional_fields():
    """create_llm_client handles missing optional fields."""
    config = {"llm": {"main": {"base_url": "http://x", "model": "m"}}}
    client = create_llm_client(config)
    assert client.api_key is None
    assert client.extra_body is None
