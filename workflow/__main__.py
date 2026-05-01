"""CLI entry point for the workflow engine.

Usage:
    python -m workflow implement --issue <url> [--repo-path <path>]

The workflow engine reads agent.yaml for LLM configuration,
fetches the issue body via gh CLI, and runs the issue
implementation workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from workflow.config import load_agent_config, create_llm_client
from workflow.context import WorkflowContext
from workflow.engine import WorkflowEngine
from workflow.workflows.issue import build_issue_workflow, parse_issue_url

logger = logging.getLogger(__name__)


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

    try:
        # Build and run the workflow
        stages = build_issue_workflow(
            repo=repo,
            issue_number=issue_number,
            repo_path=repo_path,
            issue_body=issue_body,
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

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.command == "implement":
        asyncio.run(run_workflow(args.issue, args.repo_path))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
