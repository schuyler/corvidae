"""CLI entry point for the workflow engine.

Usage:
    python -m workflow implement --issue <url> [--repo-path <path>]

The workflow engine reads agent.yaml for LLM configuration,
fetches the issue body via gh CLI, and runs the issue
implementation workflow.

Logging setup:
    configure_workflow_logging() installs dual handlers:
    - File handler: DEBUG-level structured logs (full detail)
    - STDOUT handler: only shows workflow-level events
      (stage start/complete via the display module)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from corvidae.logging import StructuredFormatter
from workflow.config import load_agent_config, create_llm_client
from workflow.context import WorkflowContext
from workflow.engine import WorkflowEngine
from workflow.workflows.issue import build_issue_workflow, parse_issue_url

logger = logging.getLogger(__name__)


def configure_workflow_logging(
    *,
    log_file: str = "workflow.log",
) -> None:
    """Set up dual-handler logging for the workflow.

    Two handlers are installed on the 'workflow' logger:

    1. File handler (DEBUG level): Writes full structured logs to
       the given file path. Uses StructuredFormatter for key=value
       extra fields. Creates parent directories if needed.

    2. STDOUT handler (WARNING level): Only lets through
       WARNING+ messages to STDOUT. Stage progress is shown via
       the Display module (plain print), not via the logging system.
       This prevents INFO/DEBUG noise from flooding the terminal.

    The root logger is set to WARNING to suppress third-party noise.
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Get the workflow logger
    workflow_logger = logging.getLogger("workflow")
    workflow_logger.setLevel(logging.DEBUG)
    workflow_logger.propagate = False

    # Remove any existing handlers to avoid duplicates on re-configure
    workflow_logger.handlers.clear()

    # File handler: full structured debug logs
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    workflow_logger.addHandler(file_handler)

    # STDOUT handler: only warnings and above
    # Stage progress is shown via the Display module, not through logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    workflow_logger.addHandler(stdout_handler)

    # Quiet the root logger to suppress third-party noise
    logging.getLogger().setLevel(logging.WARNING)


async def fetch_issue_body(repo: str, issue_number: int) -> str:
    """Fetch issue body via gh CLI."""
    from corvidae.tools.shell import shell
    cmd = (
        f"gh issue view {issue_number} "
        f"--repo {repo} --json body -q .body"
    )
    return await shell(cmd, timeout=30)


async def run_workflow(issue_url: str, repo_path: str) -> None:
    """Parse issue, build workflow, and run it."""
    # Parse the issue URL
    repo, issue_number = parse_issue_url(issue_url)
    logger.info("Running workflow for %s/issues/%d", repo, issue_number)

    # Fetch issue body
    issue_body = await fetch_issue_body(repo, issue_number)
    if issue_body.startswith("Error:"):
        logger.error("Failed to fetch issue: %s", issue_body)
        sys.exit(1)

    # Load LLM config from agent.yaml
    config = load_agent_config()
    client = create_llm_client(config)
    await client.start()

    # Read agent identity config for co-authored-by and signatures
    workflow_config = config.get("workflow", {})
    agent_name = workflow_config.get("agent_name", "corvidae-workflow")
    agent_email = workflow_config.get("agent_email", "agent@corvidae")
    gh_user = workflow_config.get("gh_user")

    # If gh_user not configured, try to read it from gh CLI
    if not gh_user:
        try:
            from corvidae.tools.shell import shell
            gh_user = (await shell("gh api user -q .login", timeout=10)).strip()
        except Exception:
            logger.debug("Could not determine gh user, using 'user'")
            gh_user = "user"

    try:
        # Build and run the workflow
        stages = build_issue_workflow(
            repo=repo,
            issue_number=issue_number,
            repo_path=repo_path,
            issue_body=issue_body,
            agent_name=agent_name,
            agent_email=agent_email,
            gh_user=gh_user,
        )

        context = WorkflowContext()
        context.facts["issue_url"] = issue_url
        context.facts["repo"] = repo
        context.facts["issue_number"] = issue_number

        engine = WorkflowEngine(client)
        await engine.run(stages, context)

        logger.info("Workflow complete")
    finally:
        await client.stop()


def main() -> None:
    """Parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(
        description="Corvidae Workflow Engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    impl = subparsers.add_parser(
        "implement",
        help="Implement a GitHub issue",
    )
    impl.add_argument(
        "--issue", required=True,
        help="GitHub issue URL",
    )
    impl.add_argument(
        "--repo-path", default=".",
        help="Local repo path (default: current directory)",
    )

    args = parser.parse_args()

    # Configure dual-handler logging: file gets DEBUG detail,
    # STDOUT only shows human-readable stage progress.
    configure_workflow_logging()

    if args.command == "implement":
        asyncio.run(run_workflow(args.issue, args.repo_path))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
