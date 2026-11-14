"""Corvidae plugin: report GitHub webhook events as terminal outcomes (#792).

GitHub emits a ``workflow_job`` event for every job status change and a
``workflow_run`` event for each run's *requested*, *in_progress* and
*completed* states, so one push through a multi-job workflow is a dozen
notifications, and a burst of merges multiplies that again (each new push
cancels the previous run). Routed to a human's channel, that is hundreds of
turns of churn (#790). This plugin subscribes to the webhook ingress itself
and reports only what the agent can act on.

Reported:

| Event | When | Line |
|---|---|---|
| ``workflow_run`` | ``completed`` unless superseded | ``CI <conclusion>: <workflow> on <branch> (run #<n>) <url>`` |
| ``pull_request`` | ``opened``, ``reopened``, ``closed`` | ``PR #<n> opened/reopened/merged/closed: <title> <url>`` |
| ``push`` | the branch is in ``push_branches``, not deleted | ``Push to <branch>: <n> commit(s), head <sha> [<compare>]`` |
| ``ping`` | always | not notified; one INFO naming the endpoint and the hook |

Everything else — ``workflow_job``, non-terminal ``workflow_run``, other
events — is dropped and counted at DEBUG. A delivery with no
``X-GitHub-Event`` header, or one missing a field its line needs, is skipped
with a ``WARNING`` naming the delivery: a ``KeyError`` must not reach the
ingress worker.

Supersession: a run GitHub cancelled because a newer run replaced it is noise,
and the payload never says so, so it is inferred. The highest ``run_number``
seen per ``(repository.full_name, workflow_id, head_branch)`` is tracked from
every ``workflow_run`` event; a completed ``cancelled`` run below that mark is
dropped. The map is in memory and lost on restart, where the worst case is one
extra cancelled line. It needs the hook to send ``workflow_run`` events,
``requested`` included.

Sender-controlled text — PR titles, branch names, workflow names — goes only
in the notification text, which the kernel fences, never in an annotation, and
titles are cut to 120 characters.

Origin ``webhook:<endpoint>`` in the cognition plugins, as for
``webhook_notify``:

- appraisal: runs stage 2 as for any non-critique exchange.
- outcome_log: records the origin verbatim in ``exchange_log.origin``.
- gates: non-``user``, so the decide gate may withhold a low-salience final
  when ``gate.send.enforce`` is set, as for heartbeat and task.
- critique: not eligible, as for heartbeat.
- funnel: keys stubs on its own origins only; unaffected.

Endpoints are owned, not shared: every subscriber to an endpoint receives
every delivery, so an endpoint listed in both this plugin and
``webhook_notify`` is delivered twice, and startup says so with a ``WARNING``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from corvidae.channel import (
    ChannelNotReady,
    ChannelRegistry,
    resolve_persisted_channel,
)
from corvidae.channels.webhook import WebhookDelivery, WebhookHandler
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.spill import NotificationSpill
from corvidae.tuning import config_value

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAYLOAD_CHARS = 2000
TITLE_CHARS = 120
# Keys the supersession map keeps before dropping the least recently updated.
SUPERSESSION_KEYS = 500

_REPORTED_PR_ACTIONS = ("opened", "reopened", "closed")


def _bad(key: str, value: object, want: str) -> ValueError:
    """The config error naming the key, the rejected value and the fix."""
    return ValueError(f"Config error: {key} is {value!r}; {want} — check agent.yaml")


def _cut(text: object, limit: int = TITLE_CHARS) -> str:
    """A sender-controlled string, cut to a fixed length for the line."""
    out = str(text).replace("\n", " ").strip()
    return out if len(out) <= limit else out[: limit - 1] + "…"


@dataclass(frozen=True)
class _Target:
    """One GitHub endpoint: its channels and which pushes it reports."""

    channels: tuple[str, ...]
    push_branches: frozenset[str]


class GithubWebhookPlugin(CorvidaePlugin):
    """Reports GitHub webhook events as terminal outcomes, one line each."""

    depends_on = frozenset({"registry", "spill"})
    optional_depends_on = frozenset({"webhook"})
    model_visible_annotations = frozenset({"origin"})

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._registry = get_dependency(pm, "registry", ChannelRegistry)
        self._spill = get_dependency(pm, "spill", NotificationSpill)
        self._max_chars = config_value(config, "github_webhook.max_payload_chars", None)
        if self._max_chars is None:
            self._max_chars = DEFAULT_MAX_PAYLOAD_CHARS
        if type(self._max_chars) is not int or self._max_chars < 1:
            raise _bad(
                "github_webhook.max_payload_chars", self._max_chars, "use a positive integer"
            )
        block = config.get("github_webhook")
        if block is not None and not isinstance(block, dict):
            raise _bad("github_webhook", block, "it must be a mapping")
        self._dropped: dict[str, int] = {}
        self._targets: dict[str, _Target] = {}
        self._highest_run: OrderedDict[tuple[str, str, str], int] = OrderedDict()
        push_branches = self._parse_branches((block or {}).get("push_branches"))
        endpoints = (block or {}).get("endpoints")
        if endpoints is not None and not isinstance(endpoints, dict):
            raise _bad(
                "github_webhook.endpoints",
                endpoints,
                "use a mapping of endpoint name to a list of channel ids",
            )
        for name, spec in (endpoints or {}).items():
            self._targets[name] = self._parse_endpoint(name, spec, push_branches)
        self._warn_shared_endpoints(config)

    # -- config ------------------------------------------------------------

    def _parse_branches(self, block: object) -> frozenset[str]:
        """`github_webhook.push_branches`: the branches a push is reported for."""
        if block is None:
            return frozenset()
        if not isinstance(block, list):
            raise _bad(
                "github_webhook.push_branches", block, "use a list of branch names, e.g. [main]"
            )
        for branch in block:
            if not isinstance(branch, str) or not branch.strip():
                raise _bad(
                    "github_webhook.push_branches", branch, "use branch names as strings, e.g. main"
                )
        return frozenset(block)

    def _parse_endpoint(self, name: str, spec: object, push_branches: frozenset[str]) -> _Target:
        """One endpoint: a bare channel list, or a mapping with `channels`."""
        key = f"github_webhook.endpoints.{name}"
        if isinstance(spec, list):
            channels, overrides = spec, {}
        elif isinstance(spec, dict):
            channels = spec.get("channels")
            overrides = {k: v for k, v in spec.items() if k != "channels"}
            if channels is None:
                raise _bad(
                    f"{key}.channels", None, "it is required, as a non-empty list of channel ids"
                )
        else:
            raise _bad(
                key,
                spec,
                "use a non-empty list of channel ids, or a mapping "
                "with channels: and push_branches:",
            )
        if not isinstance(channels, list) or not channels:
            raise _bad(key, channels, "use a non-empty list of channel ids such as irc:#general")
        for cid in channels:
            transport, _, scope = cid.partition(":") if isinstance(cid, str) else ("", "", "")
            if not transport or not scope:
                raise _bad(key, cid, "use transport:scope, e.g. irc:#general")
        branches = push_branches
        for field, value in overrides.items():
            if field != "push_branches":
                raise _bad(f"{key}.{field}", value, "use channels or push_branches")
            branches = self._parse_branches(value)
        # dict.fromkeys drops duplicate ids and keeps config order.
        return _Target(channels=tuple(dict.fromkeys(channels)), push_branches=branches)

    def _warn_shared_endpoints(self, config: dict) -> None:
        """An endpoint in both this plugin and webhook_notify is delivered twice."""
        other = config.get("webhook_notify")
        if not isinstance(other, dict):
            return
        listed = other.get("endpoints")
        if not isinstance(listed, dict):
            return
        for endpoint in self._targets:
            if endpoint in listed:
                logger.warning(
                    "github_webhook: endpoint %s is also in "
                    "webhook_notify.endpoints, so every delivery reaches both "
                    "plugins and is notified twice; remove it from one of them",
                    endpoint,
                )

    # -- delivery ----------------------------------------------------------

    @hookimpl
    async def on_webhook_subscribe(self, subscriptions: dict[str, list[WebhookHandler]]) -> None:
        # One bound method serves every endpoint; it reads delivery.endpoint.
        for endpoint in self._targets:
            subscriptions.setdefault(endpoint, []).append(self._deliver)

    async def _deliver(self, delivery: WebhookDelivery) -> None:
        """Turn one delivery into at most one line, or drop it."""
        endpoint = delivery.endpoint
        event = delivery.event_type
        if not event:
            logger.warning(
                "github_webhook: delivery %s on endpoint %s has no "
                "X-GitHub-Event header, so it is not a GitHub event; dropped",
                delivery.delivery_id,
                endpoint,
            )
            return
        if event == "ping":
            payload = delivery.payload
            hook_id = payload.get("hook_id") if isinstance(payload, dict) else None
            logger.info(
                "github_webhook: ping on endpoint %s (hook id %s); not notified",
                endpoint,
                hook_id,
            )
            return
        line = getattr(self, f"_line_{event}", None)
        if line is None:
            self._count(f"event {event}", endpoint)
            return
        text = line(delivery)
        if text is None:
            return
        await self._notify(endpoint, text)

    def _count(self, reason: str, endpoint: str) -> None:
        """Count one dropped delivery at DEBUG, once per reason per endpoint."""
        key = f"{endpoint}/{reason}"
        self._dropped[key] = self._dropped.get(key, 0) + 1
        logger.debug(
            "github_webhook: dropped %s on endpoint %s (%d so far)",
            reason,
            endpoint,
            self._dropped[key],
        )

    def _field(self, delivery: WebhookDelivery, *path: str) -> Any:
        """A payload field by path, or None with a WARNING naming it."""
        value: Any = delivery.payload
        for step in path:
            if not isinstance(value, dict) or step not in value:
                logger.warning(
                    "github_webhook: %s delivery %s on endpoint %s has no %s; "
                    "skipped, and the hook's payload is not what GitHub sends",
                    delivery.event_type,
                    delivery.delivery_id,
                    delivery.endpoint,
                    ".".join(path),
                )
                return None
            value = value[step]
        return value

    # -- one line per event ------------------------------------------------

    def _line_workflow_run(self, delivery: WebhookDelivery) -> str | None:
        """A completed run, unless a newer run on the same branch replaced it."""
        action = self._field(delivery, "action")
        if action is None:
            return None
        run = self._field(delivery, "workflow_run")
        if run is None:
            return None
        if not isinstance(run, dict):
            self._count("workflow_run without a run object", delivery.endpoint)
            return None
        # The supersession key and run number come first: every event carries
        # them, while conclusion is null until a run completes (#847 review).
        branch = self._field(delivery, "workflow_run", "head_branch")
        number = self._field(delivery, "workflow_run", "run_number")
        if branch is None or number is None:
            return None
        workflow_id = self._field(delivery, "workflow", "id")
        repo = self._field(delivery, "repository", "full_name")
        if not isinstance(number, int):
            self._count("workflow_run without a run_number", delivery.endpoint)
            return None
        # Track the highest run per workflow and branch from every event,
        # including the requested/in_progress ones otherwise dropped.
        key = (str(repo), str(workflow_id), str(branch))
        highest = self._highest_run.get(key, 0)
        if number > highest:
            self._highest_run[key] = number
            self._highest_run.move_to_end(key)
            while len(self._highest_run) > SUPERSESSION_KEYS:
                self._highest_run.popitem(last=False)
        if action != "completed":
            self._count(f"workflow_run {action}", delivery.endpoint)
            return None
        # Only a completed run needs the fields its line prints.
        fields = {}
        for name in ("conclusion", "html_url", "name"):
            fields[name] = self._field(delivery, "workflow_run", name)
        # _field already warned for each absent one; a line needs all of them.
        if any(value is None for value in fields.values()):
            return None
        if fields["conclusion"] == "cancelled" and number < self._highest_run[key]:
            self._count("superseded cancelled run", delivery.endpoint)
            return None
        return (
            f"CI {fields['conclusion']}: {_cut(fields['name'])} on "
            f"{_cut(branch)} (run #{number}) {fields['html_url']}"
        )

    def _line_pull_request(self, delivery: WebhookDelivery) -> str | None:
        """A PR opened, reopened, merged or closed — and nothing in between."""
        action = self._field(delivery, "action")
        if action is None:
            return None
        if action not in _REPORTED_PR_ACTIONS:
            self._count(f"pull_request {action}", delivery.endpoint)
            return None
        pr = self._field(delivery, "pull_request")
        if pr is None:
            return None
        if not isinstance(pr, dict):
            self._count("pull_request without a pull_request object", delivery.endpoint)
            return None
        for name in ("number", "title", "html_url", "merged"):
            if name not in pr:
                self._field(delivery, "pull_request", name)
                return None
        if action == "closed":
            what = "merged" if pr["merged"] is True else "closed"
        else:
            what = action
        return f"PR #{pr['number']} {what}: {_cut(pr['title'])} {pr['html_url']}"

    def _line_push(self, delivery: WebhookDelivery) -> str | None:
        """A push, but only to the branches the endpoint names."""
        ref = self._field(delivery, "ref")
        if ref is None:
            return None
        # A malformed payload must not raise into the ingress worker.
        if not isinstance(ref, str):
            logger.warning(
                "github_webhook: push delivery %s on endpoint %s has a ref of "
                "type %s, not a string; skipped",
                delivery.delivery_id,
                delivery.endpoint,
                type(ref).__name__,
            )
            return None
        branch = ref.removeprefix("refs/heads/")
        if not ref.startswith("refs/heads/"):
            self._count("push to a non-branch ref", delivery.endpoint)
            return None
        if branch not in self._targets[delivery.endpoint].push_branches:
            self._count(f"push to {branch}", delivery.endpoint)
            return None
        # A branch deletion is a push with an all-zero head: nothing to act on.
        if delivery.payload.get("deleted") is True:
            self._count(f"deletion of {branch}", delivery.endpoint)
            return None
        commits = self._field(delivery, "commits")
        head = self._field(delivery, "after")
        compare = delivery.payload.get("compare")
        if not isinstance(commits, list) or not isinstance(head, str):
            self._count("push without commits or after", delivery.endpoint)
            return None
        count = len(commits)
        line = f"Push to {_cut(branch)}: {count} commit{'s' if count != 1 else ''}, head {head[:7]}"
        # The compare URL is optional: a line without it still says what moved.
        return f"{line} {compare}" if compare else line

    # -- sending -----------------------------------------------------------

    async def _notify(self, endpoint: str, text: str) -> None:
        """Send one line to every channel of the endpoint."""
        origin = f"webhook:{endpoint}"
        channels = []
        for cid in self._targets[endpoint].channels:
            try:
                channel = await resolve_persisted_channel(self.pm, self._registry, cid)
            except ChannelNotReady:
                logger.warning(
                    "github_webhook: endpoint %s skipped channel %s, whose "
                    "transport cannot build it yet; this line does not reach it",
                    endpoint,
                    cid,
                )
                continue
            channels.append(channel)
        if not channels:
            return
        # The kernel fences the text and cuts it at the limit.
        await self._spill.notify(
            channels,
            source=origin,
            text=text,
            annotations={"origin": origin},
            limit=self._max_chars,
        )
