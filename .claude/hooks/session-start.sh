#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Web sessions get a fresh container with the repo cloned but no virtualenv and
# no dependencies, which makes the red/green TDD workflow in AGENTS.md
# impossible until something installs them. This does that.
#
# Local CLI sessions are skipped — they already have direnv and .envrc.
set -euo pipefail

# Only run in the remote (web) environment; local sessions manage their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "session-start: not a remote session, skipping dependency install"
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Install runtime deps plus the dev extra (pytest, pytest-asyncio) and the dev
# dependency group (pytest-timeout, required by the global timeout in
# tool.pytest.ini_options). uv sync is idempotent and reuses its cache.
echo "session-start: installing dependencies with uv"
uv sync --extra dev

echo "session-start: dependencies ready"
