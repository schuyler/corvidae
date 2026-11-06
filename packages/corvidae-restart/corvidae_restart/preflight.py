"""Pre-flight check for the restart plugin.

Validates the config file using the daemon's own validation functions,
so a config that passes the pre-flight can actually boot. Run as a
subprocess by the restart tool before the daemon is killed.

Usage:
    python -P -m corvidae_restart.preflight --config <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight check for restart")
    parser.add_argument(
        "--config", required=True, help="Path to the agent config file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        print(f"config file could not be parsed: {exc}", file=sys.stderr)
        return 1

    if not isinstance(config, dict):
        print("config file is not a YAML mapping", file=sys.stderr)
        return 1

    # Run the daemon's own validation, same as `corvidae serve`.
    try:
        from corvidae.channel import validate_agent_config
        from corvidae.idle import validate_daemon_config

        if "agent" in config:
            validate_agent_config(config["agent"])
        if "daemon" in config:
            validate_daemon_config(config["daemon"])
    except Exception as exc:
        print(f"config validation failed: {exc}", file=sys.stderr)
        return 1

    print("pre-flight OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
