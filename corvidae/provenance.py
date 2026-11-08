"""Kernel provenance: what the kernel observed about a queue item, rendered
into the prompt as a code-built header, and external payloads fenced.

The kernel sets an ``ItemMetadata`` on every item it builds, stores it with
the row under ``PROVENANCE_KEY``, and ``frame`` renders it at prompt-build
time. Nothing a plugin supplies can change the header wording or the fence;
plugins only declare which annotation keys the model may see
(``CorvidaePlugin.model_visible_annotations``).

Logging:
    - WARNING: a stored provenance record that does not decode (row id, channel)
    - WARNING: the fence key file is unreadable or too short; a process-only key is used
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import logging
import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROVENANCE_KEY = "corvidae_provenance"
#: Longest sender or channel name rendered, in characters; ids are never cut.
NAME_MAX = 64
KEY_FILE = "provenance.key"
_NOT_HUMAN = "not from this channel's human"


@dataclass(frozen=True)
class ItemMetadata:
    """What the kernel itself observed about a queue item. Which fields apply
    follows the row's role: user / system (notification) / tool."""

    # role user: the transport's sender id and the names it supplied at admission
    sender: str | None = None
    sender_name: str | None = None
    channel_name: str | None = None
    addressed: bool | None = None
    # role system (notification): the attribution context of the code that called on_notify
    origin_channel: str | None = None
    origin_trace: str | None = None
    # role tool: the dispatched call's name, and the registered channel other
    # than this one that its arguments named
    tool: str | None = None
    target: str | None = None
    # role tool: True when the tool declared its output untrusted at dispatch
    # (True or absent)
    untrusted: bool | None = None
    # role system (notification): how the kernel cut the text, and where the
    # full text went (spill_failed is True or absent)
    cut_shown: int | None = None
    cut_total: int | None = None
    spill_path: str | None = None
    spill_chars: int | None = None
    spill_failed: bool | None = None


_FIELDS = frozenset(f.name for f in dataclasses.fields(ItemMetadata))
_BOOL_FIELDS = frozenset({"addressed", "spill_failed", "untrusted"})
_INT_FIELDS = frozenset({"cut_shown", "cut_total", "spill_chars"})


def _well_typed(name: str, value: object) -> bool:
    """Whether ``value`` has the type ItemMetadata field ``name`` stores."""
    if name in _BOOL_FIELDS:
        return isinstance(value, bool)
    if name in _INT_FIELDS:
        # type() is int, not isinstance: True is not a length.
        return type(value) is int
    return isinstance(value, str)


def encode(metadata: ItemMetadata, annotations: Mapping[str, str]) -> dict:
    """The row's stored record: set fields only, annotations when there are any."""
    record = {k: v for k, v in dataclasses.asdict(metadata).items() if v is not None}
    if annotations:
        record["annotations"] = dict(annotations)
    return record


def decode(value: object) -> tuple[ItemMetadata, dict[str, str]] | None:
    """The record ``encode`` wrote, or None when ``value`` is not one."""
    if not isinstance(value, dict):
        return None
    annotations = value.get("annotations", {})
    fields = {k: v for k, v in value.items() if k != "annotations"}
    if (
        not isinstance(annotations, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in annotations.items())
        or not fields.keys() <= _FIELDS
        or not all(_well_typed(k, v) for k, v in fields.items())
    ):
        return None
    return ItemMetadata(**fields), annotations


def load_fence_key(base_dir: Path) -> bytes:
    """The stable fence key in ``base_dir``, created (mode 0600) when absent.

    shortcut: an unreadable key file degrades to a process-only key, so nonces
    change every restart; upgrade if that path is ever routine.
    """
    path = Path(base_dir) / KEY_FILE
    try:
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(os.urandom(32))
        key = path.read_bytes()
        if len(key) >= 32:
            return key
        problem = f"holds {len(key)} bytes, fewer than 32"
    except OSError as exc:
        problem = f"cannot be read or created ({type(exc).__name__})"
    logger.warning(
        "fence key file %s %s; using a key that lasts until restart, so payload "
        "fences change on the next start — fix the file's permissions or delete it",
        path, problem,
    )
    return os.urandom(32)


def clean_value(s: str, max_len: int | None = None) -> str:
    """``s`` as one short header-safe line: no control, format or bracket characters."""
    s = unicodedata.normalize("NFKC", s).replace(";", " ")
    # Characters are dropped before whitespace collapses, so none leave doubled spaces.
    s = "".join(
        c for c in s
        if c not in "<>[]" and unicodedata.category(c) not in ("Cc", "Cf")
    )
    s = " ".join(s.split())
    s = s[:max_len].strip() if max_len is not None else s.strip()
    return s or "?"


def clean_payload(s: str) -> str:
    """``s`` without control or format characters, newline and tab excepted."""
    return "".join(
        c for c in s
        if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cf")
    )


def _nonce(msg: dict, content: str, fence_key: bytes | None) -> str:
    """The fence nonce: HMAC of the stored row id, or of the text for an unsaved row."""
    row_id = msg.get("_db_id")
    subject = f"message_log:{row_id}" if isinstance(row_id, int) else f"text:{content}"
    # surrogatepass: an unsaved row's text may hold a lone surrogate (JSON "\ud83d").
    return hmac.new(fence_key, subject.encode(errors="surrogatepass"), hashlib.sha256).hexdigest()[:16]


def _person_header(meta: ItemMetadata, channel_id: str) -> str | None:
    """The #513 wording behind a ``metadata:`` label, or None with no names
    and no addressing."""
    if not meta.sender_name and not meta.channel_name and meta.addressed is None:
        return None
    sender_id = clean_value(meta.sender or "")
    sender = (
        f"sender: {clean_value(meta.sender_name, NAME_MAX)} (id {sender_id})"
        if meta.sender_name else f"sender: id {sender_id}"
    )
    scope = channel_id.split(":", 1)[-1]
    if scope.startswith("dm:"):
        where = f"channel: direct message (id {clean_value(scope)})"
    elif meta.channel_name:
        where = f"channel: {clean_value(meta.channel_name, NAME_MAX)} (id {clean_value(channel_id)})"
    else:
        where = f"channel: id {clean_value(channel_id)}"
    # Fixed words: no transport text reaches the clause.
    addressed = "" if meta.addressed is None else f"; addressed: {'yes' if meta.addressed else 'no'}"
    return f"[metadata: {sender}; {where}{addressed}]"


def _cut_line(meta: ItemMetadata) -> str:
    """The ``metadata:`` line saying a notification was cut and where the rest is."""
    line = f"[metadata: cut to the first {meta.cut_shown} of {meta.cut_total} characters"
    if meta.spill_path:
        path = clean_value(meta.spill_path)
        if meta.spill_chars == meta.cut_total:
            line += f"; the full text is in {path}"
        else:
            line += f"; the first {meta.spill_chars} characters are in {path}"
    elif meta.spill_failed:
        line += "; the full text was not saved"
    return line + "]"


def frame(
    msg: dict, rendered: dict, channel_id: str, fence_key: bytes | None
) -> tuple[dict | None, dict]:
    """Render the stored row ``msg``'s provenance.

    ``rendered`` is the caller's rendering of ``msg``. Returns a header message
    to place before the row (person rows only) and the row as it goes to the
    model, with the stored provenance key removed.
    """
    rendered = {k: v for k, v in rendered.items() if k != PROVENANCE_KEY}
    if PROVENANCE_KEY not in msg:
        return None, rendered
    decoded = decode(msg[PROVENANCE_KEY])
    if decoded is None:
        row_id = msg.get("_db_id")
        logger.warning(
            "channel %s: the provenance record on %s does not decode; rendering "
            "the row without it (inspect the row)",
            channel_id,
            f"message_log id {row_id}" if isinstance(row_id, int) else "an unsaved row",
            extra={"channel": channel_id},
        )
        return None, rendered
    meta, annotations = decoded
    role = msg.get("role")
    if role == "user":
        header = _person_header(meta, channel_id)
        lines = [header] if header else []
    elif role == "system":
        if meta.origin_channel:
            trace = f" (trace {clean_value(meta.origin_trace)})" if meta.origin_trace else ""
            what = f"notification from your own activity on {clean_value(meta.origin_channel)}{trace}"
        else:
            what = "notification"
        lines = [f"[metadata: {what}; {_NOT_HUMAN}]"]
        if meta.cut_total is not None:
            lines.append(_cut_line(meta))
    elif role == "tool" and meta.untrusted:
        # A declared-untrusted result names the call (read from the stored
        # row, so a reload renders the same bytes) and labels its content.
        call_id = msg.get("tool_call_id")
        on = f" on {clean_value(meta.target)}" if meta.target else ""
        lines = [
            f"[metadata: result of your own {clean_value(meta.tool or '')} call "
            f"{clean_value(call_id if isinstance(call_id, str) else '')}{on}; "
            f"untrusted outside content, {_NOT_HUMAN}]"
        ]
    elif role == "tool" and meta.target:
        lines = [
            f"[metadata: result of your own {clean_value(meta.tool or '')} action on "
            f"{clean_value(meta.target)}; {_NOT_HUMAN}]"
        ]
    else:
        return None, rendered
    if lines and annotations:
        shown = ", ".join(
            f"{clean_value(k)}={clean_value(v)}" for k, v in sorted(annotations.items())
        )
        lines.append(f"[annotations: {shown}]")
    if role == "user":
        return ({"role": "system", "content": "\n".join(lines)} if lines else None), rendered
    # Notifications and cross-channel results carry a code-built header and a
    # fenced payload in one message, so a call and its result stay adjacent.
    content = clean_payload(str(rendered.get("content") or ""))
    nonce = _nonce(msg, content, fence_key)
    lines += [f"[payload {nonce} begins]", content, f"[payload {nonce} ends]"]
    return None, {**rendered, "content": "\n".join(lines)}
