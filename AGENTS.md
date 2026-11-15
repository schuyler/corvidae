# Coding conventions

* Package management in this project is done with uv.
* Use red/green test-driven development on this project. A guard test goes red under a mutation of the change rather than on the parent commit (see "Process instructions").
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

**Same shape for review feedback.** A review comment or verdict opens with the verdict and its one-sentence reason -- `NEEDS-CHANGES -- the budget is real now, but the label is wider than it reserves` -- and puts the probes, the numbers, the file and line references and the suggested patch inside a `<details>`. A blocking finding always gets its reason in the opener, never only in the collapsed part: a reader who expands nothing must still know what is being asked of them.

**The two-layer rule does not license vagueness.** The opener is short because it is selected, not because the detail is thin. Everything a reviewer needs to check the work is still written down, in the detail block, and a body whose detail block is empty has failed this section rather than complied with it.

# Review loops

* **Every non-trivial change gets a review loop before it is merged.** The criterion is whether behaviour, or something a reader or another plugin acts on, could change, not the file type: a config key, a prompt, a documented contract, a docs page readers act on, a one-line code fix and a refactor all count. What can change neither does not (a typo, a broken link, a comment that was wrong about unchanged code), and the PR body says which case it is.
* **The review is blind.** The reviewer gets the diff; the requirement as written -- the issue or the owner's words, never the author's paraphrase; the PR body, as claims to check; the severity rules below, verbatim; any owner rulings, as settled (disagreement goes to the owner and does not change the verdict); and on a re-review, the earlier findings with their outcomes. Nothing else: no reasoning beyond what the PR body states, no argument for or against a finding. It reads the connected code, not only the diff. Use a subagent where the tooling allows. If no issue or owner's words state the requirement, write it on an issue first; never write it into the reviewer's prompt.

## Severity

Findings are rated by consequence, not reachability (PRD principle 12).

* **Critical:** on an input the reviewer names, the change causes a security hole, data loss, a hang or crash, or a wrong result the user sees.
* **Important:**
  * the change deviates from scope in either direction. Too much implementation is as much a deviation as too little (an option nobody asked for, an abstraction with one caller, a config key with no user), and so is a test suite past its use-by: tests that re-prove behaviour another test already proves. Past 3:1 (test lines against the non-test lines changed in the same PR), the reviewer checks for exactly that. **A regression test for the bug being fixed is always in scope.**
  * a test the change adds that no mutation of the change makes fail, or whose claim rests on values neither the test nor the code under test set.
  * a path the change adds that no test reaches.
  * a claim the change makes, or leaves standing, that the code or another doc contradicts.
* **Everything else is non-blocking.**
* **Verdicts:** `SOUND` (no findings), `APPROVE WITH FOLLOW-UPS` (non-blocking findings only), `NEEDS-CHANGES` (a critical or important finding). The verdict opens the review, as "Issues and pull requests" asks.

## The author's side

* **Sweep before you hand it over.** Apply the review's checks to the whole change yourself first. Review is for what you could not see, not a substitute for looking.
* **Show your tests can fail.** The PR body gives, for each new behavioural test, a mutation of the change that makes it fail on the assertion carrying its claim.
* **Drive the real path.** Acceptance tests run the hook order the agent runs, on data shaped the way production writes it. A test that does not says why it is equivalent.
* **Run the full suite before answering a round**, docs-only changes included -- tests read docs too -- and give the count.
* **Fix the class, not the example.** Name the finding's class and fix every instance the change covers; the reviewer's instance is a lower bound.
* **Triage honestly and disclose what you skip.** Fix the findings that land against the requirement. List the ones you skip in the PR body, file a follow-up issue for any that is not cosmetic, and name a skipped critical in the opener. Skip non-blocking findings unless the fix is a line or two: every addition is new surface for the next round.
* **The PR body describes the design that ships**, not the draft and not one an earlier round rejected.

## Rounds

* **Severity sets the verdict; the author's triage ends the loop.** The loop ends when every critical and important finding is fixed or disclosed as skipped, and nothing has changed since the last round.
* **Re-review is cold, never a conversation with the first reviewer.** The new reviewer checks that the fixes landed, marks a finding that comes back as a recurrence, and does not reopen the rest.
* **Meta-review when findings recur.** When a finding recurs or a second round ends `NEEDS-CHANGES`, stop fixing and sweep the whole change for the class behind the findings. Post the diagnosis on the PR -- a flawed design, a flawed remedy, reviewer noise, or scope creep -- and take it to the owner unless the loop is converging.
* **From the third round on, a round ending `NEEDS-CHANGES` stops the loop** until the owner says to continue.
* **The reviewer does not expand scope.** It may name a larger problem it noticed as a one-line follow-up with its own issue; it may not fold that problem in, and the author does not adopt one to be agreeable. Scope grows by a decision recorded on an issue.

## Merging

* **No agent self-certifies and merges.** A finished loop makes a change admissible, not accepted: merging needs the owner's permission, either for that change or as a standing instruction the owner wrote, and the message says which. A review written under the author's own account is not a second party and never counts as permission. Because agents act under the owner's account, the author field proves nothing: permission is the owner's words in a live session, or a comment the owner points to from one.
* **An emergency merge without the loop** needs the owner's agreement recorded before it happens; it says so at the top of the PR body and is reviewed immediately afterwards, before anything else starts.
* **Say what was reviewed in the PR body:** the rounds it took, any meta-review, and the findings skipped.

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
* A new test has to fail before it counts as a test: a test of new or fixed behaviour fails on the parent commit; a guard test, which pins behaviour the change must not break, fails under a mutation of the change. Run it, not only in your head.
* Comment every block with its purpose, as the coding conventions say. `docs/plugin-checklist.md` asks for the shorter "why it is there"; that is a compression of this rule, not a different one, and this file wins.
