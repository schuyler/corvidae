# Zulip operations

How to give corvidae a Zulip bot: make the bot, subscribe it to a channel, and point corvidae's `zulip:` config block at it.

```yaml
zulip:
  site: https://chat.example.org
  email: corvidae-bot@chat.example.org   # or set ZULIP_EMAIL
  api_key: "..."                          # or set ZULIP_API_KEY
  allow: [123, 456]                       # Zulip user ids; default-deny
```

There is nothing to install. The transport speaks Zulip's REST API over aiohttp, which corvidae already needs. It does need a server on Zulip 9.0 or newer; an older one is refused at startup with an ERROR that names the server and its feature level.

There is no local Zulip server in `harness/` (one needs PostgreSQL, Redis, RabbitMQ and memcached), so this transport has no end-to-end test. The unit tests run the real HTTP client against a fake server that speaks the slice of the API corvidae uses. Expect to find the first rough edge yourself, and file it.

## Create the bot

1. In the web or desktop app, click the gear icon, choose **Personal settings**, then **Bots**, then **Add a new bot**.
2. Pick **Generic bot**. It acts like a normal Zulip user that can only use the API, which is what an agent that reads and writes messages needs. (Incoming and outgoing webhook bots can't do that.)
3. Copy the bot's email and API key into `zulip.email` and `zulip.api_key`, or into `ZULIP_EMAIL` and `ZULIP_API_KEY`. The key is the bot: anyone holding it can act as it, and `read_file` and `shell` can read both `agent.yaml` and the daemon's environment. It can be regenerated from the same Bots page.
4. Subscribe the bot to the channels it should answer in, the way you'd subscribe a person. A bot only sees messages in channels it is subscribed to, plus direct messages sent to it.

## Find ids

The allowlist and `channels` take numbers, not names or emails, because names and emails change and ids don't.

- **User id** (for `allow`): open the person's profile in the app; it lists their user ID. Or ask the API: `curl -u BOT_EMAIL:API_KEY https://chat.example.org/api/v1/users` (`-u` puts the key in your shell history and process list; for more than a one-off, use a `~/.netrc` entry and `curl -n`) returns every `user_id` with its `full_name`.
- **Channel id** (for `channels`): Zulip's API has a "Get channel ID" call (`GET /api/v1/get_stream_id`, see its page in Zulip's API docs for the parameter). Easier: set `logging.level: DEBUG`, post a message in the channel without mentioning the bot, and read the line saying it does not mention the bot: it names the channel as `zulip:7`, and the key is the `7`.

## How it behaves

- **1:1 DMs**: every message from an allowed user reaches the agent (which may still decline to reply).
- **Channels**: only a mention of the bot does (`@**Corvidae Bot**`, or a user group it belongs to, since Zulip sets its `mentioned` flag for both; the group's token stays in the text), unless `zulip.require_mention` is `false` (everywhere) or `zulip.channels.<id>.require_mention` is (one channel). `@**all**`, `@**channel**` and `@**topic**` do not count.
- **Group DMs** follow `zulip.require_mention` like a channel does.
- **Everyone else** is refused silently: one INFO line naming their user id, nothing sent back. (A stranger's message that doesn't mention the bot in a mention-only channel is dropped earlier, with a DEBUG line instead.)

### Topics

Every channel message in Zulip lives in a topic, and corvidae answers in the topic of the latest message from an allowed user that reached the agent in that channel (an unmentioned message in a mention-only channel is dropped first, so it does not move the reply). The topic is not part of the channel's identity: history, memory and `channels.zulip:*` entries are keyed on the channel id (`zulip:7`), so everything said in any topic of that channel shares one conversation. The model sees the channel as `#general > plans`, so it knows where it is. Topics can be renamed by anyone, and an id that depends on a name would orphan its history when they do (PRD principle 10). If you want a separate conversation per topic, that is a decision to make in the PRD, not a setting.

The topic to answer in is remembered in memory only. Only when nothing is remembered for the channel (the first turn after a restart, or a channel no allowed message has reached yet) does the reply go to `zulip.default_topic` (default `corvidae`). Otherwise a reminder, a heartbeat post or a cross-channel `send_message` to `zulip:7` lands in that last remembered topic. If two allowed people write in different topics of one channel close together, the answer to the first goes to the second's topic.

### Mentions in replies

Zulip has no switch like Discord's allowed-mentions, so corvidae breaks the sigil itself: `@**all**`, `@**Alice**` and `@*group*` in a reply get a zero-width space after the `@` and notify no one. Silent mentions (`@_**Alice**`) pass through, since they don't notify. The agent can't ping people, and its register says so.

## When something goes wrong

| What you see | What it means |
|---|---|
| `refused the login for <email>` (ERROR, once) | Wrong or deactivated bot. Check `zulip.email` and `zulip.api_key`. Not retried. |
| `older than 248 (Zulip 9.0)` (ERROR, once) | The server is too old to post to. Upgrade it, or point `zulip.site` at a newer one. Not retried. |
| `event queue ... was garbage-collected` (WARNING) | The server dropped the bot's queue, usually after a long outage. Corvidae registers a new one. **Messages sent while it was gone are not replayed.** |
| `connection error to <site> as <email>, retrying in Ns` | Network or 5xx. Retries at 10 s, doubling to 300 s. |
| `could not post to zulip:7 (topic 'x') ... dropped` | The server refused the message; its own explanation is in the line. Usually the bot isn't subscribed or may not post there. |
| `message queued until the server answers` | The server is unreachable; replies wait in a 50-chunk queue (a long reply takes several slots) and are lost if corvidae stops first. |

## Limits

- Edits and deletes aren't followed: only new messages are read, and the log is append-only.
- Attachments aren't downloaded; an uploaded file shows up as the link Zulip put in the message text.
- A reply split across messages is cut at the configured size without regard to Markdown code fences (Discord's splitter keeps fences balanced; this one doesn't yet).
- One `zulip:` block is one organisation.

## Content is stored in plaintext

Zulip messages aren't end-to-end encrypted: your Zulip server has them. On this side they land in `sessions.db`, the JSONL log (`zulip_<stream_id>.jsonl`, or `zulip_dm_<user ids>.jsonl`, when enabled) and memory, as ordinary text. Use `corvidae redact` as described in the [Signal section of configuration.md](configuration.md#signal--signal-transport), with `channel_id = 'zulip:<stream_id>'` or `'zulip:dm:<user ids>'`.
