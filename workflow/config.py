"""Config loading — reads agent.yaml and creates an LLMClient.

Reuses the same agent.yaml that the Corvidae daemon uses, so there's
a single config file for both systems.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from corvidae.llm import LLMClient


def load_agent_config(config_path: str | Path = "agent.yaml") -> dict:
    """Load and return the agent.yaml config dict.

    Args:
        config_path: Path to agent.yaml. Defaults to agent.yaml in CWD.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def create_llm_client(config: dict) -> LLMClient:
    """Create an LLMClient from agent.yaml config.

    Reads llm.main config section and constructs an LLMClient
    with the same parameters the Corvidae daemon uses.

    Raises:
        KeyError: If llm.main config section is missing.
    """
    llm_config = config["llm"]["main"]
    return LLMClient(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key=llm_config.get("api_key"),
        extra_body=llm_config.get("extra_body"),
        max_retries=llm_config.get("max_retries", 3),
        retry_base_delay=llm_config.get("retry_base_delay", 2.0),
        retry_max_delay=llm_config.get("retry_max_delay", 60.0),
        timeout=llm_config.get("timeout"),
    )
