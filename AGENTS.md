# Coding conventions

* Package management in this project is done with uv.
* Use red/green test-driven development on this project.
* This project relies on asyncio, so all pytest runs need to employ timeouts to prevent hangs.
* Always log exceptions, pass through, or re-raise -- never swallow exceptions with bare `except Exception: pass`.
* Use idiomatic Python with proper type hinting.
* Comment every block of code with a description of its purpose so that humans can tell what's going on.

# Error handling and messages

A failure has two audiences, and each gets different text from the same exception: a human reading a terminal or a chat reply, and whoever reads the log afterwards.

* Human-facing text says what failed and what to do next, in plain language: `Config error: <detail> — check /etc/corvidae/agent.yaml`. Exception text and tracebacks belong in the log line, which names the exception type, the identifier, and carries the traceback.
* Name the identifier in every message: the path, URL, tool, MCP server, task, channel, record id, config key or field that failed. `Error: search failed` is unfinished; `Search failed for "foo" (QueryError); try a simpler query` is finished.
* State the remedy. When the caller can fix it, say how; when only a retry helps, say so.
* Exhaust the cheap rescue before failing: retry a transient, fall back to a simpler path, skip the corrupt record and return the rest, degrade to a partial result. Then report what was reduced.
* Log every rescue at warning level with the reason it was needed, so a fallback never hides a break. Where a fallback has a known edge, mark it `shortcut: <limit>, <when to upgrade>`.
* On rejected input, name the key, the rejected value, and the accepted values.
* Test the message, not just the raise. Assert that the identifier and the remedy appear; `pytest.raises` alone proves nothing about what the user reads.

# Issues and pull requests

A PR is the unit of work and the issue is where its requirements live, so both are read by someone deciding whether to spend attention on them. An agent-written body is accurate and complete, and completeness is what makes it expensive to read: a reader who sees six paragraphs before the first fact stops reading. Write both in two layers.

**The opener is your judgement, in a few sentences.** Before any detail, one short paragraph -- two to four sentences -- saying what was wrong or missing and what the change does about it, in plain language, with no headings and no bullet list. If the change is visible, a screenshot after the opener says more than a paragraph of description. This paragraph is a promise that the rest is worth reading; it is the only part most readers will read, so it must be written, not generated: if you cannot compress the change into four sentences, you do not yet know what the change is.

**Everything else goes inside `<details><summary>`.** The requirements, the PRD comparison, the implementation plan, what you ran, the failure output, the alternatives you rejected, the follow-ups you left: all of it below the opener, collapsed. The summary names what wrote it, model and effort level (`<summary>Claude Opus 5.5 Medium</summary>`), so a reader can tell agent prose from the author's own summary, and so the same rule reads sensibly on a human-written PR. `<details><summary>` is GitHub Markdown; it renders in issues, PR bodies, comments and reviews, and the order is details-summary -- `<summary>` is the line that summarises, and `<details>` wraps everything else.

```markdown
The poster row drew more avatars than the width it measured could hold, so the
last one was drawn under the forward arrow. It now budgets the count label from
the string it will print.

<details><summary>Claude Opus 5.5 Medium</summary>

### What was measured
...
</details>
```

**Same shape for review feedback.** A review comment or verdict opens with the verdict and its one-sentence reason -- `REQUEST_CHANGES -- the budget is real now, but the label is wider than it reserves` -- and puts the probes, the numbers, the file and line references and the suggested patch inside a `<details>`. A blocking finding always gets its reason in the opener, never only in the collapsed part: a reader who expands nothing must still know what is being asked of them.

**The two-layer rule does not license vagueness.** The opener is short because it is selected, not because the detail is thin. Everything a reviewer needs to check the work is still written down, in the detail block, and a body whose detail block is empty has failed this section rather than complied with it.

# Process instructions

* Read `docs/prd.md` before implementing anything. The PRD is the reference for what must be true; issues, plans, design docs and the existing code derive from it.
* Compare what an issue (or plan, or request) asks for against the PRD before implementing. The PRD's opening paragraph "What this document governs" says where it binds and where it leaves behaviour to a plugin. Sort each difference into one of two kinds:
  * **Flag it to the owner and wait** when the issue contradicts or rules out a PRD requirement, omits one, or weakens a principle; or when the PRD is silent about something that is not one plugin's own business: the kernel (anything under `corvidae/`), a hook or other contract that other plugins rely on, or a new transport (who may reach the agent). Don't silently follow either side, and don't treat an issue's older wording as a requirement the PRD has since dropped.
  * **Proceed** when the PRD is silent about behaviour inside one plugin: its tools, config keys, prompts, storage, tuning. List these in the issue comment as "plugin-defined" with the issue as their provenance, and record them in the plugin's docs (`docs/plugin-checklist.md` step 6). If untrusted text can now reach somewhere new or a tool can do something new, the change still needs a `docs/threat-model.md` entry (PRD principle 9).
* If you are solving a GitHub issue, post the requirements, the PRD comparison above, and the implementation plan as a comment on the issue before implementing -- opened with the short summary the section above asks for, with the lists inside one `<details>` block.
* Check @docs/ before your work is complete and make sure the documentation is updated.
* If you are adding or changing a plugin, read `docs/plugin-checklist.md` first. It covers placement (a plugin, not the kernel), the package layout and CI wiring, and the traps that fail silently. `docs/plugin-guide.md` has the depth.
