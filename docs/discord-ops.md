# Discord operations

How to give corvidae a Discord bot: create an application in the Developer
Portal, get its token, invite it to a server, and point corvidae's
`discord:` config block at it.

The SDK is an optional extra: add `--extra discord` to your `uv sync` line,
for example `uv sync --extra discord --extra cognition` for the full daemon.
`uv sync` removes any extra you don't name, so list all of them. Without the
SDK, a `discord:` block fails at startup and says what to install.

Unlike Signal, there is no daemon to run. corvidae logs in to Discord's
gateway itself using discord.py, retries on the same backoff ladder IRC
uses if the connection drops, and reconnects when Discord is back. Everything
on this page is about the Discord side of that login.

There is no local Discord server to test against, so this transport has no
harness end-to-end test; the unit tests use fake clients. Expect to find
the first rough edge yourself, and file it.

## Create the application and bot

1. Open <https://discord.com/developers/applications> and choose **New
   Application**. The name is what people see as the bot's name.
2. Open the **Bot** page. A bot user already exists; there is nothing to
   add.
3. Under **Privileged Gateway Intents**, leave everything off. corvidae does
   not request the Message Content intent: Discord delivers message content
   for DMs and for messages that @-mention the bot without it, and those are
   the only messages corvidae processes. Turning it on does nothing useful.
4. Decide whether the bot should be installable by anyone. **Public Bot**
   is on by default; if you do not want strangers inviting it elsewhere,
   switch it off. The invite link below keeps working for you.

The Developer Portal moves its buttons around. If a label above does not
match, look for the same thing under a nearby name.

## Get the token

On the **Bot** page, choose **Reset Token** and copy it. Discord shows it
once. Treat it like a password: whoever has it is the bot, and can read and
send everything the bot can. If it leaks, reset it again; the old one stops
working immediately.

Give it to corvidae either way:

```yaml
discord:
  token: "MTIz...your.bot.token"
```

or, preferably if `agent.yaml` is shared or checked in anywhere:

```sh
export DISCORD_TOKEN='MTIz...your.bot.token'
```

If `discord.token` is absent, null, or empty, `DISCORD_TOKEN` is used. If
neither is set and the `discord:` block exists, corvidae refuses to start.

## Invite the bot

Open **OAuth2** and build an invite URL (the portal's URL generator will do
it for you):

- Scope: `bot`. You do not need `applications.commands`; corvidae registers
  no slash commands.
- Permissions: **View Channels**, **Send Messages**, and **Send Messages in
  Threads**.

You do not need **Read Message History**, **Manage Messages**,
**Mention Everyone**, **Embed Links**, or **Attach Files**. corvidae only
reads messages the gateway pushes to it, sends plain text, and shows the
typing indicator, which needs only Send Messages. Every outbound message
also suppresses mentions on its own, so Mention Everyone would do nothing.
Give it less than you think it needs, and add permissions only if Discord
reports a failure.

The resulting URL looks like this, with your application id and the
permissions integer the generator computes:

```
https://discord.com/oauth2/authorize?client_id=<application id>&scope=bot&permissions=<integer>
```

Open it, pick the server, and authorize. DMs need no invite; a user can DM
the bot if Discord's own rules let them (typically, they share a server
with it).

## Find ids

corvidae identifies everything by numeric id, not by name, because names
change and ids are what history is stored under.

In Discord, turn on **User Settings, Advanced, Developer Mode**. Then
right-click (or long-press) a user, channel, or thread and choose **Copy
User ID** or **Copy Channel ID**.

- Your user id goes in `discord.allow`.
- A server channel's id gives the channel `discord:<channel_id>`.
  A thread has its own id and its own history; it does not share its
  parent channel's.
- A DM with user 123 is `discord:dm:123`, whichever DM channel Discord
  happens to use.

You need channel ids only if you want per-channel overrides under
`channels:` (see [configuration.md](configuration.md#channels--per-channel-overrides))
or to find a conversation in `sessions.db`.

## Configure corvidae

```yaml
discord:
  allow:
    - 123456789012345678       # your user id
```

with `DISCORD_TOKEN` set in the environment, or `token:` added. Restart
corvidae; `discord:` is read at startup. On success the log shows
`discord: connected as user id <id>`.

`allow` is default-deny. An empty or missing list means nobody can use the
bot, and it will not tell them so.

## How the bot behaves

- **DMs:** every message from an allowlisted user is answered.
- **Servers and threads:** the bot answers only messages that @-mention it
  directly. `@everyone`, `@here`, and role pings do not count. The mention
  is stripped before the model sees the text. Messages that do not mention
  it are not even delivered with content, and are ignored.
- **Other bots, and itself:** ignored.
- **Typing indicator:** shown while a reply is owed, and only for allowed
  users. There are no read receipts.
- **Long replies:** split at 2000 bytes by default, keeping code blocks
  intact, and sent as several messages.
- **Mentions in replies** never ping anyone, whatever the model writes.
- **Attachments** are discarded; text sent with them is still read.
  Outbound is text only.
- **Reasoning and tool-status** messages are not sent to Discord.

## Troubleshooting

The bot is online but never answers. There are three likely causes, and corvidae
cannot tell you which from Discord's side. Check the log (INFO or lower):

- `discord: rejected message from unauthorized sender <id>` means the user
  is not in `discord.allow`. Add the id from the log line, as a number, and
  restart.
- Nothing logged at all in a server channel means the message did not
  mention the bot. Mention it with the autocomplete (typing `@name` as
  plain text does not make a mention), or DM it.
- `mention in channel <id> arrived with empty content; dropping` at WARNING
  means Discord sent a mention without text. Discord delivers the content
  of messages that mention the bot even without the Message Content
  intent, so this should not happen; if you see it, Discord's behavior has
  changed and the transport needs a code change (it would have to request
  that intent). Enabling the intent in the portal alone does nothing.

`discord: login failed; check that discord.token (or DISCORD_TOKEN) is a
valid bot token` at ERROR means Discord rejected the token. It is logged
once and not retried; fix the token (look for stray quotes or whitespace,
or reset it) and restart corvidae. The rest of the daemon keeps running.

`discord: connection error, retrying` at WARNING is anything else: network
down, Discord outage, rate limiting. It retries after 10s, doubling up to
300s, and resets after a good connection.

`discord: cannot reach discord:<id>` at WARNING on a send means the bot
cannot see that channel or user: it was removed from the server, the
channel was deleted, or View Channels / Send Messages was revoked. A
reply to it is dropped, not queued.

A reply arrives in several pieces. That is the 2000-byte limit, and you can
lower it with `message_chunk_size` (100 to 2000). See the splitter's known
limits in [design.md](design.md#discord-channelsdiscordpy): ~~~ fences and
four-backtick runs are not tracked.

## Privacy

Discord messages are not end-to-end encrypted. Discord can read everything
sent to and from the bot. On the corvidae side the same text is written in
plaintext to `sessions.db`, the JSONL logs, and memory. See the plaintext
note in [configuration.md](configuration.md#discord--discord-transport) and
[threat-model.md](threat-model.md) for what an allowlisted user can do from
a channel other people can read. The short version is: use DMs, or channels
where everyone present is someone you would put in `allow`.
