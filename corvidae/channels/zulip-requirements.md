# Zulip transport requirements

What must be true of `corvidae/channels/zulip.py`. This is the Zulip entry that
`docs/prd.md` §3.2 would carry, kept here instead: by the owner's decision on
#640, Zulip is exempt from PRD updates, and `docs/prd.md` is not edited for it.
Everything in the PRD's §3.2 contract (items 1 to 10) still applies; this file
adds what is specific to Zulip and says where each requirement comes from.
`docs/design.md` ("Zulip") says how the code meets them, and `docs/zulip-ops.md`
is the operator's guide.

Provenance tags: **[PRD 3.2.n]** the §3.2 contract item; **[PRD p10]** a
principle; **[#640 Dn]** an owner decision on the ticket; **[Zulip]** a fact from
Zulip's API documentation, which is also where each of these stops being
verified (see the last section).

## Configuration

1. No `zulip:` block: the plugin loads inert. No session, no task, nothing
   logged above DEBUG. [PRD 3.2.1]
2. A present block that is malformed raises in `on_init`, naming the field as
   `zulip.<key>`, the rejected value, and what is accepted. Validation completes
   before anything is stored, so a failed init leaves the plugin inert. [PRD 3.2.1]
3. `site` is required: a bare `http(s)://host[:port]`, with no path, query,
   fragment or credentials. Plain `http://` to a host other than loopback logs a
   WARNING naming the site and saying to use `https://`. [#640]
4. `email` and `api_key` are required and may come from `ZULIP_EMAIL` and
   `ZULIP_API_KEY`; an empty config value falls through to the environment. The
   API key is never logged, at any level. [PRD p9]
5. `allow` is a list of Zulip **user ids** (ints or digit strings, compared as
   strings). An email, a name, a boolean or a non-list is rejected, naming the
   entry. Default-deny: absent or empty refuses everyone. [PRD 3.2.2, PRD p10]
6. `require_mention` is a boolean, default `true`. `channels` maps **stream ids**
   (digits, no leading zero, no prefix) to `{require_mention: bool}`; a name, a
   prefixed key, a duplicate or an unknown key is rejected rather than ignored.
7. `message_chunk_size` is an integer from 100 to 10000 (default 10000);
   `default_topic` is a string of 1 to 60 characters (default `corvidae`).

## Identity

8. A channel id is built from ids alone: `zulip:<stream_id>`; `zulip:dm:<user_id>`
   for a 1:1 DM, keyed on the other person; `zulip:dm:<id>,<id>` for a group DM,
   ascending, the bot's own id left out. Names, emails and topics are never part
   of an id. [PRD p10, #640 D1]
9. A topic is a label and reply routing, not a key: every topic of a channel
   shares one channel, one history, one window. [#640 D1; Zulip: topics are
   renamable and expose no id]
10. `channel.name` reads `#<channel> > <topic>` for a channel message (the empty
    topic shows as the realm's own display label) and `group DM with <names>` for
    a group DM; it is left unset for a 1:1 DM. The sender passed to `on_message`
    is the user id, with the display name in `channel.sender_names`. Both names
    are strings others chose and are data, never instructions. [PRD 3.3.11, PRD p9]
11. `resolve_channel` answers only for `zulip:` ids whose scope is a stream id or
    `dm:` with strictly ascending user ids, and returns None before `on_start`. A
    reply needs no live connection to be addressed. [PRD 3.2.7]

## Connecting

12. Needs Zulip 9.0 (feature level 248) or newer. An older server stops the
    transport with one ERROR naming the site, the level and the remedy, and is
    not retried. [#640, Zulip: `channel` and `direct` message types]
13. Registration asks for message events only, raw Markdown (`apply_markdown`
    false), `fetch_event_types=["realm"]` (without it the limits are not
    returned), and the `empty_topic_name` client capability. [Zulip]
14. A rejected login (401, or a deactivated user or organisation) stops the
    transport with one ERROR naming the site, the bot email and the config keys
    to check, and is not retried. [PRD p5]
15. A lost event queue (`BAD_EVENT_QUEUE_ID`) is re-registered at once, with a
    WARNING naming the site and the bot and saying that messages sent meanwhile
    are not replayed. They are not fetched. [#640 D4]
16. A transient failure (network, timeout, 5xx, a reply that is not Zulip's JSON)
    retries the same queue on the shared ladder: 10 s, doubling, capped at 300 s,
    reset on success; each retry names the site and the bot identity. [PRD 3.2.8]
17. A 429 waits the server's `retry-after`, capped at 60 s, then continues. [Zulip]
18. Events are handled in id order and acknowledged by the next poll. A malformed
    message event is skipped with a WARNING naming the event and message ids; it
    never stops the loop. Edits and deletes are never delivered (not
    requested): the log is append-only. [PRD p5, PRD p4]

## Inbound

19. The bot's own messages are dropped by sender id, whatever the allowlist or
    `require_mention` say.
20. The gate is default-deny on the sender's user id. A refusal is one INFO line
    naming the id; nothing is sent back, no typing is shown, and no channel is
    registered for the sender. [PRD 3.2.2, 3.2.5, 3.2.6]
21. In a channel or group DM a message is admitted only when the server's
    `mentioned` flag is set (a direct mention, or a mention through a user group
    the bot belongs to), unless `require_mention` (global, or the channel's
    entry) is false. Wildcard mentions (`@**all**`, `@**topic**`) never count. A
    1:1 DM needs no mention. A dropped unmentioned message is one DEBUG line
    naming the sender and channel. [PRD 3.2.2, PRD p9]
22. `channel.addressed` is set before every `on_message`: the mention verdict in
    channels and group DMs, `None` in a 1:1 DM.
23. The bot's own mention token is stripped from the text (`@**Name**`,
    `@**Name|id**`, silent `@_**Name**`, any name carrying the bot's id); other
    people's mentions stay (so a user-group token that admitted the message stays
    in the text). A message with nothing left is dropped.
24. The channel label, the sender-name cache and the reply topic are set only by
    an allowed sender, so a stranger cannot relabel a channel or move where the
    agent answers. [PRD p9]
25. The transport mints the trace id and fires `on_message` for every message that
    survives the filters above; the gate is the one place that vetoes. Every
    inbound hookimpl opens with `if not self.owns(channel): return`. [PRD 3.1.4, 3.2.2]

## Outbound

26. A reply goes to the topic of the newest allowed message that reached the agent on that channel, held
    in memory only, whatever started the turn (a message, a reminder, the
    heartbeat, a cross-channel send); only when none is remembered for the channel
    (the first turn after a restart, or a channel no allowed message has reached)
    does it go to `default_topic`. Two allowed senders in different topics of one
    channel can therefore have an answer land in the other's topic. The empty topic
    round-trips as the empty topic. A DM goes to its user ids. [#640 D1, PRD 3.2.10]
27. Reasoning and tool status are not displayed. [PRD 3.2.3]
28. Text is split, never truncated, at the smaller of `message_chunk_size` and the
    server's `max_message_length`, counted in UTF-8 bytes (never fewer than the
    code points Zulip limits). Chunks concatenate back to the text; whitespace-only
    chunks are not sent. Known gap: the split is not fence-aware, so a Markdown
    code fence can be cut across two messages. [PRD 3.2.4]
29. Notifying mentions in generated text (`@` directly before `*`: `@**all**`,
    `@**Name**`, `@*group*`) are defused before splitting, so a reply cannot page
    anyone. Silent mentions (`@_**Name**`) and ordinary `@` signs are untouched.
    [PRD p9]
30. Replies that cannot be sent because the server is unreachable are queued in
    order, bounded at 50, flushed after the next successful poll; past the bound
    the oldest is dropped and the channel and drop count are logged. [PRD 3.2.8]
31. A refused post (a 4xx) drops the rest of that reply and logs the channel, the
    topic and the server's own explanation. A 429 on a send waits `retry-after` and
    retries up to twice before the chunk is held in the queue. [PRD p5]

## Liveness

32. Typing is shown only to allowed senders: sent on the first outstanding message,
    refreshed every 8 s while a reply is owed, and stopped with an explicit `stop`
    when a final is sent, when a final is withheld, or when a message the gate never
    admitted gives its slot back. It is never left on. A failed typing call is a
    WARNING and never costs the reply. [PRD 3.2.5]
33. One final releases every outstanding slot (it answers a coalesced burst). On
    the withheld-final path (`on_agent_response`) the ownership check,
    `final_was_unsent` and the release run before the first `await`; a sent final
    releases after it has delivered, as Discord and Signal do, so a message that
    arrives mid-delivery has its slot released with the rest. Progress text is
    delivered without releasing a slot.

## Testing

34. Behaviour is pinned against a fake Zulip server that speaks the real HTTP
    protocol, with scripted failures. There is no local Zulip server in
    `harness/`, so there is no end-to-end harness test. [PRD 3.2.9]
35. The errors that stop the transport, and every config error, are tested for the
    identifier they name (site, bot, `zulip.<key>`), not only for the raise.
    Remedies are asserted for the startup-stopping errors only. [AGENTS.md]

## Taken from the documentation, not observed

No real server has been run against this transport. Until one is, treat these as
assumptions:

- `mentioned` is set for a direct mention and for a user-group mention, and not
  for a wildcard (the user-group part is documented, the rest is not observed);
- the exact `@**Name**` token forms;
- the server expires a typing notice after roughly 15 s, which the 8 s refresh
  assumes;
- a bot receives channel messages only for channels it is subscribed to;
- `max_message_length` and the empty-topic label are returned only with
  `fetch_event_types=["realm"]`, and `event_types=["message"]` excludes edits and
  deletes;
- the error codes `BAD_EVENT_QUEUE_ID`, `RATE_LIMIT_HIT`, `UNAUTHORIZED`,
  `USER_DEACTIVATED` and `REALM_DEACTIVATED`, and `retry-after` in the body;
- the server counts message length in code points;
- the empty topic round-trips with the `empty_topic_name` capability on every
  server version from feature level 248;
- the web-app steps in `docs/zulip-ops.md` (Personal settings → Bots; a profile
  showing the user ID).

Known parity gap: PRD §3.2 item 2 admits a sender the agent has messaged first
(§3.5.9). This transport's gate is the allowlist only, as Discord's is; nothing
in the tree implements that admission for either.
