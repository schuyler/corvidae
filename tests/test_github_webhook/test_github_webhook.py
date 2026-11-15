"""Tests for the github_webhook plugin (#792).

A real plugin manager with the real WebhookPlugin (not started), the plugin's
real subscribe hook, and a recording on_notify, so a line's header renders as
the kernel would render it. Payloads are the fields GitHub actually sends,
cut to what each line reads.
"""

from __future__ import annotations

import logging

import pytest
from fake_clock import install_kernel_clock
from github_webhook import GithubWebhookPlugin

from corvidae.channel import ChannelRegistry
from corvidae.channels.webhook import WebhookDelivery, WebhookPlugin
from corvidae.hooks import create_plugin_manager, hookimpl, install_plugin_hookspecs
from corvidae.spill import NotificationSpill

ENDPOINTS = {"github": ["irc:#a", "irc:#b"]}
# One channel, for tests that read a single call rather than the fan-out.
ONE = {"github": ["irc:#a"]}


class Recorder:
    """Records every on_notify call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations) -> None:
        self.calls.append(
            dict(channel=channel.id, source=source, text=text, annotations=annotations)
        )


async def build(base, block=None, extra=()):
    """The plugin on a real pm, initialised and subscribed."""
    pm, plugin, rec = create_plugin_manager(), GithubWebhookPlugin(), Recorder()
    install_kernel_clock(pm, started=False)
    for name, p in [
        ("webhook", WebhookPlugin()),
        ("registry", ChannelRegistry()),
        ("spill", NotificationSpill()),
        ("github_webhook", plugin),
        ("rec", rec),
        *extra,
    ]:
        pm.register(p, name=name)
    install_plugin_hookspecs(pm)
    config = {"_base_dir": base}
    if block is not None:
        config["github_webhook"] = block
    await pm.ahook.on_init(pm=pm, config=config)
    subs: dict = {}
    await pm.ahook.on_webhook_subscribe(subscriptions=subs)
    return plugin, rec, subs


def github(
    event: str, payload: dict, endpoint: str = "github", delivery_id: str = "d1"
) -> WebhookDelivery:
    return WebhookDelivery(
        event_type=event, endpoint=endpoint, payload=payload, delivery_id=delivery_id
    )


#: Marks "no conclusion given": the fixture then picks what GitHub sends.
_DEFAULT = object()


def workflow_run(
    action="completed",
    number=6,
    conclusion=_DEFAULT,
    branch="main",
    workflow_id=42,
    repo="o/r",
    name="CI",
) -> dict:
    # GitHub sends "conclusion": null until a run completes (#847 review), so a
    # non-terminal fixture carries None unless a test says otherwise.
    if conclusion is _DEFAULT:
        conclusion = "success" if action == "completed" else None
    return {
        "action": action,
        "workflow_run": {
            "conclusion": conclusion,
            "head_branch": branch,
            "run_number": number,
            "html_url": f"https://x/{number}",
            "name": name,
        },
        "workflow": {"id": workflow_id},
        "repository": {"full_name": repo},
    }


def pull_request(action="opened", number=7, title="Fix the thing", merged=False) -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": number,
            "title": title,
            "merged": merged,
            "html_url": f"https://x/pull/{number}",
        },
    }


def push(branch="main", commits=2, head="abcdef1234567890") -> dict:
    return {
        "ref": f"refs/heads/{branch}",
        "after": head,
        "commits": [{"id": str(n)} for n in range(commits)],
        "compare": "https://x/compare",
    }


async def test_subscribes_one_handler_per_endpoint(tmp_path):
    plugin, _, subs = await build(tmp_path, {"endpoints": ENDPOINTS})
    assert subs == {"github": [plugin._deliver]}


async def test_a_completed_run_is_one_line_to_every_channel(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"endpoints": ENDPOINTS})
    await plugin._deliver(github("workflow_run", workflow_run()))
    assert [(c["channel"], c["text"]) for c in rec.calls] == [
        ("irc:#a", "CI success: CI on main (run #6) https://x/6"),
        ("irc:#b", "CI success: CI on main (run #6) https://x/6"),
    ]
    assert {c["source"] for c in rec.calls} == {"webhook:github"}
    assert all(c["annotations"] == {"origin": "webhook:github"} for c in rec.calls)


async def test_job_status_churn_is_dropped_and_counted(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    plugin, rec, _ = await build(tmp_path, {"endpoints": ENDPOINTS})
    await plugin._deliver(github("workflow_job", {"action": "in_progress"}))
    await plugin._deliver(github("workflow_job", {"action": "completed"}))
    assert rec.calls == []
    dropped = [
        r.getMessage() for r in caplog.records if "dropped event workflow_job" in r.getMessage()
    ]
    assert dropped and "(2 so far)" in dropped[-1]


async def test_a_superseded_cancellation_is_dropped(tmp_path, caplog):
    """Run 5 requested, 6 requested, 5 cancelled: 5 is noise, 6 is reported."""
    caplog.set_level(logging.DEBUG)
    plugin, rec, _ = await build(tmp_path, {"endpoints": ONE})
    await plugin._deliver(github("workflow_run", workflow_run("requested", number=5)))
    await plugin._deliver(github("workflow_run", workflow_run("requested", number=6)))
    await plugin._deliver(github("workflow_run", workflow_run(conclusion="cancelled", number=5)))
    assert rec.calls == [], "the cancelled run was replaced by run 6"
    await plugin._deliver(github("workflow_run", workflow_run(number=6)))
    [call] = rec.calls
    assert call["text"] == "CI success: CI on main (run #6) https://x/6"


async def test_a_lone_cancellation_is_reported(tmp_path):
    """Nothing replaced it, so the cancellation may matter."""
    plugin, rec, _ = await build(tmp_path, {"endpoints": ONE})
    await plugin._deliver(github("workflow_run", workflow_run(conclusion="cancelled", number=3)))
    [call] = rec.calls
    assert call["text"] == "CI cancelled: CI on main (run #3) https://x/3"


async def test_supersession_is_per_repo_workflow_and_branch(tmp_path):
    """A high run number elsewhere must not silence this branch's run."""
    plugin, rec, _ = await build(tmp_path, {"endpoints": ONE})
    await plugin._deliver(
        github("workflow_run", workflow_run("requested", number=9, branch="other"))
    )
    await plugin._deliver(github("workflow_run", workflow_run(conclusion="cancelled", number=3)))
    [call] = rec.calls
    assert "cancelled" in call["text"]


@pytest.mark.parametrize(
    ("action", "merged", "expected"),
    [
        ("opened", False, "PR #7 opened: Fix the thing https://x/pull/7"),
        ("reopened", False, "PR #7 reopened: Fix the thing https://x/pull/7"),
        ("closed", True, "PR #7 merged: Fix the thing https://x/pull/7"),
        ("closed", False, "PR #7 closed: Fix the thing https://x/pull/7"),
    ],
)
async def test_pull_request_transitions(tmp_path, action, merged, expected):
    plugin, rec, _ = await build(tmp_path, {"endpoints": ONE})
    await plugin._deliver(github("pull_request", pull_request(action, merged=merged)))
    assert [c["text"] for c in rec.calls] == [expected]


async def test_other_pr_actions_are_dropped(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"endpoints": ENDPOINTS})
    await plugin._deliver(github("pull_request", pull_request("synchronize")))
    assert rec.calls == []


async def test_a_pr_title_is_cut_so_a_sender_cannot_pad_the_line(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"endpoints": ONE})
    await plugin._deliver(github("pull_request", pull_request(title="x" * 400)))
    [call] = rec.calls
    assert "…" in call["text"]
    assert len(call["text"].split(": ", 1)[1]) < 200
    # Sender text never reaches an annotation, which the kernel does not fence.
    assert call["annotations"] == {"origin": "webhook:github"}


async def test_a_push_is_reported_only_for_the_named_branches(tmp_path):
    block = {"endpoints": ENDPOINTS, "push_branches": ["main"]}
    plugin, rec, _ = await build(tmp_path, block)
    await plugin._deliver(github("push", push("main", commits=3)))
    assert [c["text"] for c in rec.calls] == [
        "Push to main: 3 commits, head abcdef1 https://x/compare"
    ] * 2
    rec.calls.clear()
    await plugin._deliver(github("push", push("feature")))
    assert rec.calls == [], "feature is not in push_branches"


async def test_one_commit_reads_singular(tmp_path):
    block = {"endpoints": {"github": {"channels": ["irc:#a"], "push_branches": ["main"]}}}
    plugin, rec, _ = await build(tmp_path, block)
    await plugin._deliver(github("push", push("main", commits=1)))
    assert rec.calls[0]["text"].startswith("Push to main: 1 commit, head")


async def test_a_tag_push_is_dropped(tmp_path):
    block = {"endpoints": {"github": {"channels": ["irc:#a"], "push_branches": ["main"]}}}
    plugin, rec, _ = await build(tmp_path, block)
    await plugin._deliver(
        github("push", {"ref": "refs/tags/v1", "after": "a" * 40, "commits": [], "compare": "x"})
    )
    assert rec.calls == []


async def test_a_branch_deletion_is_not_reported_as_a_push(tmp_path):
    """A push that deletes a listed branch carries no commits and an all-zero
    head; it is not a push the agent can act on."""
    block = {"endpoints": {"github": {"channels": ["irc:#a"], "push_branches": ["main"]}}}
    plugin, rec, _ = await build(tmp_path, block)
    # commits=1 so a fix that drops every empty push cannot pass this.
    payload = push("main", commits=1, head="0" * 40) | {"deleted": True}
    await plugin._deliver(github("push", payload))
    assert rec.calls == []


async def test_a_push_with_a_non_string_ref_is_skipped(tmp_path):
    """A malformed payload must not raise into the ingress worker."""
    block = {"endpoints": {"github": {"channels": ["irc:#a"], "push_branches": ["main"]}}}
    plugin, rec, _ = await build(tmp_path, block)
    await plugin._deliver(github("push", push("main") | {"ref": 123}))
    assert rec.calls == []


async def test_a_push_without_compare_does_not_print_none(tmp_path):
    block = {"endpoints": {"github": {"channels": ["irc:#a"], "push_branches": ["main"]}}}
    plugin, rec, _ = await build(tmp_path, block)
    payload = push("main", commits=2)
    del payload["compare"]
    await plugin._deliver(github("push", payload))
    [call] = rec.calls
    assert "None" not in call["text"], call["text"]
    assert call["text"].startswith("Push to main: 2 commits, head abcdef1"), call["text"]


async def test_ping_is_logged_and_not_notified(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    plugin, rec, _ = await build(tmp_path, {"endpoints": ENDPOINTS})
    await plugin._deliver(github("ping", {"hook_id": 9, "zen": "Keep it logically awesome."}))
    assert rec.calls == []
    [info] = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert "github" in info and "9" in info


async def test_a_delivery_without_an_event_header_is_skipped_loudly(tmp_path, caplog):
    plugin, rec, _ = await build(tmp_path, {"endpoints": ENDPOINTS})
    await plugin._deliver(github(None, {"hi": "there"}, delivery_id="d-77"))
    assert rec.calls == []
    [warn] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "github" in warn and "d-77" in warn and "X-GitHub-Event" in warn


@pytest.mark.parametrize(
    ("event", "payload", "missing"),
    [
        (
            "workflow_run",
            {"action": "completed", "workflow_run": {"run_number": 1, "head_branch": "main"}},
            "workflow_run.conclusion",
        ),
        ("pull_request", {"action": "opened", "pull_request": {"number": 1}}, "pull_request.title"),
        ("push", {"ref": "refs/heads/main", "after": "a" * 40}, "commits"),
    ],
)
async def test_a_missing_field_is_skipped_loudly_not_raised(
    tmp_path, caplog, event, payload, missing
):
    """A KeyError must never reach the ingress worker."""
    rec_block = {"endpoints": {"github": {"channels": ["irc:#a"], "push_branches": ["main"]}}}
    plugin, rec, _ = await build(tmp_path, rec_block)
    await plugin._deliver(github(event, payload, delivery_id="d-9"))
    assert rec.calls == []
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(missing in w and "d-9" in w for w in warns), warns


async def test_an_endpoint_shared_with_webhook_notify_warns_once(tmp_path, caplog):
    plugin = None
    pm_block = {"endpoints": ENDPOINTS}
    caplog.set_level(logging.INFO)
    # webhook_notify's own block is read out of the same config, so build the
    # plugin manager by hand to get both sections in front of on_init.
    pm, rec = create_plugin_manager(), Recorder()
    install_kernel_clock(pm, started=False)
    plugin = GithubWebhookPlugin()
    for name, p in [
        ("registry", ChannelRegistry()),
        ("spill", NotificationSpill()),
        ("github_webhook", plugin),
        ("rec", rec),
    ]:
        pm.register(p, name=name)
    install_plugin_hookspecs(pm)
    await plugin.on_init(
        pm,
        {
            "_base_dir": tmp_path,
            "github_webhook": pm_block,
            "webhook_notify": {"endpoints": ENDPOINTS},
        },
    )
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "github" in warns[0] and "twice" in warns[0]


async def test_an_unready_channel_is_skipped_and_the_other_still_gets_it(tmp_path, caplog):
    class Sig:
        transport = "sig"

        @hookimpl
        async def resolve_channel(self, channel_id: str):
            return None

    block = {"endpoints": {"github": ["sig:x", "irc:#a"]}}
    plugin, rec, _ = await build(tmp_path, block, extra=[("sig", Sig())])
    await plugin._deliver(github("pull_request", pull_request()))
    assert [c["channel"] for c in rec.calls] == ["irc:#a"]
    [warn] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "sig:x" in warn and "github" in warn


@pytest.mark.parametrize(
    ("block", "key", "bad", "remedy"),
    [
        ("oops", "github_webhook", "'oops'", "mapping"),
        ({"max_payload_chars": 0}, "github_webhook.max_payload_chars", "0", "positive integer"),
        ({"push_branches": "main"}, "github_webhook.push_branches", "'main'", "list"),
        ({"push_branches": [1]}, "github_webhook.push_branches", "1", "branch names"),
        ({"endpoints": ["x"]}, "github_webhook.endpoints", "['x']", "mapping"),
        ({"endpoints": {"github": []}}, "github_webhook.endpoints.github", "[]", "non-empty list"),
        (
            {"endpoints": {"github": {"push_branches": ["main"]}}},
            "github_webhook.endpoints.github.channels",
            "None",
            "required",
        ),
        (
            {"endpoints": {"github": {"channels": ["irc:#a"], "bogus": 1}}},
            "github_webhook.endpoints.github.bogus",
            "1",
            "channels or push_branches",
        ),
        (
            {"endpoints": {"github": ["irc"]}},
            "github_webhook.endpoints.github",
            "'irc'",
            "transport:scope",
        ),
    ],
)
async def test_invalid_config_raises(tmp_path, block, key, bad, remedy):
    with pytest.raises(ValueError) as exc:
        await build(tmp_path, block)
    for part in (key, bad, remedy, "check agent.yaml"):
        assert part in str(exc.value)


@pytest.mark.parametrize("block", [None, {}, {"endpoints": None}])
async def test_an_absent_block_leaves_it_inert(tmp_path, block, caplog):
    caplog.set_level(logging.DEBUG)
    plugin, _, subs = await build(tmp_path, block)
    assert subs == {}
    assert plugin._dropped == {}
