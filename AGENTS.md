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

**Everything else goes inside `<details><summary>`.** The requirements, the PRD comparison, the implementation plan, what you ran, the failure output, the alternatives you rejected, the follow-ups you left: all of it below the opener, collapsed. The summary names the agent that wrote it, in the form `<summary>Agent details (Alfred)</summary>`, so a reader can tell agent prose from the author's own summary and can go back to that agent with a question. Name the agent rather than the model behind it: a fixed format does not go stale when the model changes under it, and the agent is the party a reader can ask. A human-written PR can use the same shape with their own name, or leave the block out entirely. `<details><summary>` is GitHub Markdown; it renders in issues, PR bodies, comments and reviews, and the order is details-summary -- `<summary>` is the line that summarises, and `<details>` wraps everything else.

```markdown
The poster row drew more avatars than the width it measured could hold, so the
last one was drawn under the forward arrow. It now budgets the count label from
the string it will print.

<details><summary>Agent details (Alfred)</summary>

### What was measured
...
</details>
```

**Same shape for review feedback.** A review comment or verdict opens with the verdict and its one-sentence reason -- `REQUEST_CHANGES -- the budget is real now, but the label is wider than it reserves` -- and puts the probes, the numbers, the file and line references and the suggested patch inside a `<details>`. A blocking finding always gets its reason in the opener, never only in the collapsed part: a reader who expands nothing must still know what is being asked of them.

**The two-layer rule does not license vagueness.** The opener is short because it is selected, not because the detail is thin. Everything a reviewer needs to check the work is still written down, in the detail block, and a body whose detail block is empty has failed this section rather than complied with it.

# Review loops

* **Every non-trivial change gets a full review loop before it is merged.** The criterion is whether behaviour, or something a reader or another plugin acts on, could change, not the file type: a config key, a prompt, a documented contract, a docs page readers act on, a one-line code fix and a refactor all count. What can change neither does not (a typo, a broken link, a comment that was wrong about unchanged code), and the PR body says which case it is.
* **The review is blind.** The reviewer gets the diff and the requirement as it was written -- the issue or the owner's words, never the author's paraphrase -- and nothing else: no reasoning, no commit-message justification, no test names offered as proof the tests are right. It reads the connected code, not only the diff. Use a subagent where the tooling allows. If no issue or owner's words state the requirement, write it on an issue before the review and give the reviewer that issue; never write it into the reviewer's prompt.
* **Critical blocks the merge:** the change is wrong -- logic, safety, data loss, or a path nothing tests.
* **Important blocks the merge:** the change deviates from scope in either direction. Too much implementation is as much a deviation as too little (an option nobody asked for, an abstraction with one caller, a config key with no user), and so is a test suite past its use-by: tests that re-prove behaviour another test already proves, or any test that cannot fail. Past 3:1 (test lines against the non-test lines changed in the same PR), the reviewer checks for exactly that. **A regression test for the bug being fixed is always in scope.**
* **Everything else is the author's judgement** -- naming, structure, an alternative weighed, a nit -- and does not buy another round.
* **Re-review is cold, never a conversation with the first reviewer.** Give the new reviewer the earlier findings as the reviewers wrote them, each with its outcome -- fixed, declined as non-blocking, or ruled on by the owner (quote the ruling) -- and no argument for any of them. The reviewer does not reopen what is listed; a finding about code changed since is new.
* **A round closes on the reviewer's verdict, not the author's reading of it.** The author may argue a non-blocking finding down inside the round. A critical or important one is either fixed and re-reviewed cold, or, if the author thinks it is wrong, taken to the owner: quote the finding verbatim and give the author's case in one sentence; the owner's ruling joins the earlier findings for the next round. The author never closes the round over the reviewer.
* **Loop until a round closes with nothing critical or important open.**
* **Three rounds with a blocking finding still open is a stop, not a fourth round.** Ask the owner whether the scope or the approach is wrong.
* **The reviewer does not expand scope.** It may name a larger problem it noticed as a one-line follow-up with its own issue; it may not fold that problem in, and the author does not adopt one to be agreeable. Scope grows by a decision recorded on an issue.
* **No agent self-certifies and merges.** A non-blocking verdict makes a change admissible, not accepted: merging needs the owner's permission, either for that change or as a standing instruction the owner wrote, and the message says which. A review written under the author's own account is not a second party and never counts as permission. Because agents act under the owner's account, the author field proves nothing: permission is the owner's words in a live session, or a comment the owner points to from one.
* **An emergency merge without the loop** needs the owner's agreement recorded before it happens; it says so at the top of the PR body and is reviewed immediately afterwards, before anything else starts.
* **Say what was reviewed in the PR body:** the rounds it took.

# The review loop

A change is reviewed blind: by someone who reads the requirement and the diff, not the author's reasoning. The loop below is what both agents run, so that a review is a gate rather than an opinion.

* **Three verdicts, not two.** `SOUND` means no blocking finding. `APPROVE WITH FOLLOW-UPS` means nits and non-blocking notes only: they are listed, the author has them in hand, and the PR may merge. `NEEDS-CHANGES` means a CRITICAL or IMPORTANT finding: a route that still fails, a test that cannot fail, a doc claim the code contradicts, scope beyond the issue. A nit never blocks a merge; it is written down and tracked, not re-reviewed.
* **The author's evidence comes before round one.** Every new behavioural test is shown to fail with the fix reverted, and the body carries the pairs: this test, this mutation, nothing else breaks. An assertion that also passes on the parent commit is not a test of this change. An assertion that compares two values the test itself never set proves nothing at all.
* **Code first, words last.** Rounds one and two read the diff and its tests. `docs/`, the PR body and the review comments are reviewed once the code is final. Prose restated in every round drifts in every round, and a doc finding costs the same round as a bug.
* **The body describes the shipped design.** Not the design as it was drafted, and not the one an earlier round rejected. A body narrating a superseded approach misleads whoever reads the commit next, and costs a round to correct.
* **A rejected third round triggers a meta-analysis.** Before a fourth review: are the findings shrinking in scope and severity, or is one mechanism producing them? If it is one mechanism — an invariant checked at call sites instead of at a single choke point, a race window one `await` narrower each time — stop fixing and say so in the channel. Change the design, or hand it to the owner with the finding. A loop finding instances of one pattern is not converging, and a fourth round will find the next instance.
* **Three rounds with a blocking finding is a stop.** Write the open finding on the PR, hand it to the owner, and say plainly that the loop has not converged. Whether to keep going is the owner's decision.
* **Reviewing is the reviewer's to do; merging is the owner's.** A review never waits for permission. A merge always does.

# Process instructions

* Read `docs/prd.md` before implementing anything. The PRD is the reference for what must be true; issues, plans, design docs and the existing code derive from it.
* Compare what an issue (or plan, or request) asks for against the PRD before implementing. The PRD's opening paragraph "What this document governs" says where it binds and where it leaves behaviour to a plugin. Sort each difference into one of two kinds:
  * **Flag it to the owner and wait** when the issue contradicts or rules out a PRD requirement, omits one, or weakens a principle; or when the PRD is silent about something that is not one plugin's own business: the kernel (anything under `corvidae/`), a hook or other contract that other plugins rely on, or a new transport (who may reach the agent). Don't silently follow either side, and don't treat an issue's older wording as a requirement the PRD has since dropped.
  * **Proceed** when the PRD is silent about behaviour inside one plugin: its tools, config keys, prompts, storage, tuning. List these in the issue comment as "plugin-defined" with the issue as their provenance, and record them in the plugin's docs (`docs/plugin-checklist.md` step 6). If untrusted text can now reach somewhere new or a tool can do something new, the change still needs a `docs/threat-model.md` entry (PRD principle 9).
* If you are solving a GitHub issue, post the requirements, the PRD comparison above, and the implementation plan as a comment on the issue before implementing -- opened with the short summary the section above asks for, with the lists inside one `<details>` block.
* Check @docs/ before your work is complete and make sure the documentation is updated.
* If you are adding or changing a plugin, read `docs/plugin-checklist.md` first. It covers placement (a plugin, not the kernel), the package layout and CI wiring, and the traps that fail silently. `docs/plugin-guide.md` has the depth.
* **The plan comment is the claim.** Before starting on an issue, read its comments and the open PRs that mention it; if someone has claimed it, stop and ask. Otherwise post the plan comment above before writing code -- that comment is what the next agent checks for.
* A claim about code, config or runtime state names the command that produced it and the tree it ran against. A worktree, the live checkout and the installed package are three different answers to the same import.
* A new test has to fail on the parent commit before it counts as a test. Run it there, not only in your head.
* Comment every block with its purpose, as the coding conventions say. `docs/plugin-checklist.md` asks for the shorter "why it is there"; that is a compression of this rule, not a different one, and this file wins.
