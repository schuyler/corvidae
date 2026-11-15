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

A review is a gate, not an opinion: every agent runs the same loop against the same text, so a verdict means the same thing whoever wrote it.

## When, and who

* **Every non-trivial change gets a full review loop before it is merged.** The criterion is whether behaviour, or something a reader or another plugin acts on, could change, not the file type: a config key, a prompt, a documented contract, a docs page readers act on, a one-line code fix and a refactor all count. What can change neither does not (a typo, a broken link, a comment that was wrong about unchanged code), and the PR body says which case it is.
* **The review is blind.** A reviewer gets what the next rule lists and nothing else: no reasoning, no commit-message justification, no argument for or against any finding. Nothing the author offers as proof, test names and mutations included, counts until the reviewer has checked it. It reads the connected code, not only the diff. Use a subagent where the tooling allows. If no issue or owner's words state the requirement, write it on an issue before the review and give the reviewer that issue; never write it into the reviewer's prompt.
* **What a reviewer gets.** This is the one list of a reviewer's inputs; other rules point here.
  * Every review: the diff; the requirement as it was written -- the issue or the owner's words, never the author's paraphrase; the severity definitions and verdicts below, quoted verbatim so no reviewer gates on a paraphrase; and what is settled -- an owner's ruling, a design the owner approved -- listed as settled.
  * Every review, as claims to check against the code and not as argument: the PR body, including its description of the design and its mutation evidence.
  * A re-review, in addition: every earlier finding as its reviewer wrote it, with its outcome but without the reason for a skip.
  * A reviewer who thinks a settled decision is wrong reports it as an *upstream concern*, listed under the verdict: the author forwards it to the owner, and it does not change the verdict or fail the round.
* **Finding outcomes.** Every finding leaves its round as `fixed` or `skipped`, and either may carry an owner's ruling, quoted. A skipped finding is listed in the PR body, with its follow-up issue if it has one; the reason is given in the round's answer on the PR, never in the PR body. An upstream concern is not a finding.

## Severity and verdicts

Findings gate on consequence, not reachability (PRD principle 12).

* **Critical blocks the merge:** on an input the reviewer names, the change causes a security hole, data loss, a hang or crash, or a wrong result the user sees. The finding states the input and the consequence.
* **Important blocks the merge:**
  * the change deviates from scope in either direction. Too much implementation is as much a deviation as too little (an option nobody asked for, an abstraction with one caller, a config key with no user), and so is a test suite past its use-by: tests that re-prove behaviour another test already proves. Past 3:1 (test lines against the non-test lines changed in the same PR), the reviewer checks for exactly that. **A regression test for the bug being fixed is always in scope.**
  * a test the change adds that cannot fail: one that no mutation of the change makes fail, or whose claim rests on values neither the test nor the code under test set, such as a default compared with a default.
  * a path the change adds that no test reaches.
  * a claim the change makes, or leaves standing about what it changes, in the docs or the PR body, that the code, another rule or another doc contradicts.
* **Everything else is non-blocking** and does not buy another round: naming, structure, an alternative weighed, a nit, and a failure the reviewer can reach whose consequence is none of the critical ones and that is none of the important kinds above. The reviewer reports that last kind with its input, its consequence and a rough fix size.
* **Three verdicts.** `SOUND`: no findings. `APPROVE WITH FOLLOW-UPS`: non-blocking findings only, listed. `NEEDS-CHANGES`: at least one critical or important finding. The verdict opens the review, as "Issues and pull requests" asks.

## The author's side

* **Sweep before you hand it over.** Before round one, the author applies the review's own checks to the whole change -- each severity kind above, against concrete cases -- and fixes what it finds. Review is for what the author could not see, not a substitute for looking.
* **Evidence comes with the change.** The PR body gives, before round one and kept current each round, for every new behavioural test, a mutation of the change that makes it fail. The test must fail on the assertion that carries its claim; a test that fails for another reason while that assertion would still pass is not evidence for it.
* **Acceptance tests drive the real path:** the hook order the agent actually runs, data shaped the way production writes it. A test that calls one function directly or builds its data by hand says so, and says why that is equivalent.
* **Run the full suite before answering a round**, not only the files touched -- tests read docs too -- and give the count in the answer.
* **Fix the class, not the example.** An answer to a blocking finding names the finding's class -- for example "checked, then an `await`, then acted on", or "an invariant kept at each exit point" -- and fixes every instance on the paths the change covers. The reviewer's instance is a lower bound. Instances outside the change's scope go on an issue.
* **The author triages every finding.** The author decides, honestly, whether each finding lands against the stated requirement or is trivial. One that lands is fixed, as a class (above). One the author skips is disclosed as "Finding outcomes" says, and a non-cosmetic one gets a follow-up issue. A non-blocking finding is skipped unless the fix is a line or two with no new mechanism: every addition is new surface for the next round. A skipped finding with a critical consequence is named in the PR body's opener, so the owner sees it before giving merge permission (PRD principle 12). The author may take any finding to the owner instead: quote it verbatim and give the author's case in one sentence.
* **The PR body describes the design that ships**, not the draft and not one an earlier round rejected. Rewrite it when a round changes the design.

## Rounds and stopping

* **Re-review is cold, never a conversation with the first reviewer.** The new reviewer gets what "What a reviewer gets" lists for a re-review. It checks that each `fixed` finding is fixed -- one that is not is a recurrence -- and does not reopen any other finding listed there; a finding about code changed since is new. It marks each finding new or a recurrence of an earlier one (`RECURRENCE of round 2, #1`).
* **Loop until a round leaves every critical or important finding fixed or disclosed as skipped, and nothing in the diff has changed since it.** A follow-up taken after the last round is reviewed in another round, unless the owner waives it.
* **Meta-review a loop that is not converging.** The second round to end `NEEDS-CHANGES`, a blocking recurrence, or a fix that would add mechanism or tests the requirement does not ask for, triggers one. After a meta-review finds a loop converging, the next trigger starts another. The author runs it before the next round, and stops fixing to do so: the reviewers caught what the author overlooked, so the author now sweeps the whole change for the class behind the findings -- every instance, not the next one. It names the cause -- a flawed design, a flawed remedy, reviewer noise, or scope creep -- and gives the fix for the whole class or the single place the invariant could be enforced, and it is posted on the PR. A loop it finds converging continues up to the stop; any other diagnosis goes to the owner now, not after another round.
* **From the third round on, a round ending `NEEDS-CHANGES` is a stop, not another round:** the next round needs the owner's authorization. Ask the owner whether the scope or the approach is wrong, with the meta-review.
* **The reviewer does not expand scope.** It may name a larger problem it noticed as a one-line follow-up with its own issue; it may not fold that problem in, and the author does not adopt one to be agreeable. Scope grows by a decision recorded on an issue.

## Merging

* **No agent self-certifies and merges.** A loop that has ended (see "Loop until") makes a change admissible, not accepted: merging needs the owner's permission, either for that change or as a standing instruction the owner wrote, and the message says which. A review written under the author's own account is not a second party and never counts as permission. Because agents act under the owner's account, the author field proves nothing: permission is the owner's words in a live session, or a comment the owner points to from one.
* **An emergency merge without the loop** needs the owner's agreement recorded before it happens; it says so at the top of the PR body and is reviewed immediately afterwards, before anything else starts.
* **Say what was reviewed in the PR body:** the rounds it took, and any meta-review.

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
