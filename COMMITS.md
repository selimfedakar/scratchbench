# Commit queue — `selimfedakar/scratchbench`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else.

Every block starts with `cd ~/scratchbench` so a block is safe to paste on its
own, into any shell, without checking which directory you are in.

The loop variable is `file` and never `path`. In zsh `$path` is the array form
of `$PATH`, so `read path` empties it and every command after the first
iteration becomes "command not found", with `set -e` unable to fire because the
shell cannot find the binary that was supposed to fail. `docs/LESSONS.md` L24.

---

## Pending

### Session 16 — laptop candidate 2, written and refused

```bash
bash ~/scratchbench-session16-commits.sh
```

43 commits: the task in the order it was written, thirty draws one file each,
then the metadata and the record the draws license. **One push at the end**, no
sleeps, nothing squashed.

The order is a hard constraint in exactly one place, and it is the same one as
last session: all thirty draws land before
`tasks/chunked_batchnorm_backward/meta.yaml`, because `tools/check_calibration.py`
runs in CI and re-derives every block from the files in `calibration/` and
`leaderboard/`. A block that arrives before its evidence is a red build telling
the truth. The four task files land before the draws for a different reason —
`meta.yaml` is what makes the directory a task, so until it arrives the loader
does not see the other four and CI cannot care what order they came in.

| Group | Files | What |
|---|---|---|
| 1 | `tasks/chunked_batchnorm_backward/{reference,starter,hidden_tests,prompt.md}` ×4 | the task, in the order `CLAUDE.md` mandates |
| 2 | `tools/mutate_v2_tasks.py` | sixteen mutants, fourteen caught and two expected to survive |
| 3 | `calibration/*.json` ×30 | ten draws each for Opus, Sonnet and Haiku, laptop tier |
| 4 | `tasks/chunked_batchnorm_backward/meta.yaml` | `frozen_set: warmup` and the calibration block the rule produced |
| 5 | `README.md` | the new row, the fifth-task section, and three stale counts |
| 6 | `docs/V2_DESIGN.md` | §2.0 scoped to the accelerated tier |
| 7 | `docs/ROADMAP.md` | item 1 closed, item 7 opened, §9.3 written |
| 8 | `docs/LESSONS.md` | L44 |
| 9 | `docs/sessions/16-…md` | the journal |
| 10 | `CLAUDE.md` | the state block |
| 11 | `COMMITS.md` | this block |

After it: `tools/check_calibration.py` reports **32 entries from 212 draws**,
`tools/check_cost.py` **212 files**, `validate --tier all` **15 validated, 1 not
checked here**, and the harness suite **110 passed**.

### Session 15 — the stall was never the task's — ✅ all three batches ran



```bash
bash ~/scratchbench-session15-commits.sh
```

Session 15 went out in **two batches**, because the diagnosis was pushed while
the fix it argued for was still being written.

**Batch one, already committed and pushed** — six commits, `15eba19` through
`a52b826`, carrying session 14's thirteen with them: the probe, L42, the debt
ledger rewrite, journal 15, the state block, and the queue entry. Twenty-two
commits, `main` clean against `origin/main`.

**Batch three, pending** — thirty calibration draws and everything they settle:

```bash
bash ~/scratchbench-session15-calibration.sh
```

38 commits: one per draw file, the way every calibration in this repository has
been committed, then the metadata the draws license, then the record. One push
at the end.

The order is a hard constraint, not tidiness. All thirty draws must land before
either `meta.yaml`, because `tools/check_calibration.py` runs in CI and
re-derives every block from the files in `calibration/` and `leaderboard/` — a
block that arrives before its evidence is a red build that is telling the truth.

| Group | Files | What |
|---|---|---|
| 1 | `calibration/*.json` ×30 | ten draws each for Opus, Sonnet and Haiku, `--tier accelerated`, taken under the stall guard |
| 2 | `tasks/metal_softmax_backward_kernel/meta.yaml` | `frozen_set: v2` and its first calibration block |
| 3 | `tasks/metal_cross_entropy_kernel/meta.yaml` | the block rebuilt over twenty draws, and the `timeout` kept as a failure |
| 4 | `README.md` | the new task row, twenty-nine entries, and the paragraph about the re-draw |
| 5 | `docs/LESSONS.md` | L43 |
| 6 | `docs/ROADMAP.md` | items 5 and 6 closed, §9.1 decided, §5's table updated |
| 7 | `docs/sessions/15-…md` | the draws and the defect that was already published |
| 8 | `CLAUDE.md` | the state block |
| 9 | `COMMITS.md` | this block |

After it: `python tools/check_calibration.py` reports **29 entries from 182
draws**, `tools/check_cost.py` **182 files**, the harness suite **110 passed**.

**Batch two, already committed and pushed** — the guard Selim chose in §9.2, and
the four record files updated to say it landed:

```bash
bash ~/scratchbench-session15-commits.sh
```

8 commits, **one push at the end**, no sleeps, nothing squashed.

The order is one constraint and then tidiness. `runner/mps_stall.py` holds the
threshold that `tools/check_mps_stall.py` imports and that `runner/sandbox.py`
acts on, so it lands before either of them; the record files land last, because
every number in them is produced by the code above.

| # | File | What |
|---:|---|---|
| 1 | `runner/mps_stall.py` | the probe, and the pytest hook that refuses to grade in a stalled process |
| 2 | `runner/sandbox.py` | the `mps_stalled` status, the guard copy, the relaunch loop |
| 3 | `tools/check_mps_stall.py` | rewritten to import the threshold instead of holding a second copy of it |
| 4 | `tests/test_runner.py` | five tests: arming, retrying, and the status that must not be a `timeout` |
| 5 | `docs/ROADMAP.md` | item 3 closed, §9.2 decided, §5's precondition table |
| 6 | `docs/sessions/15-the-stall-was-never-the-tasks.md` | the guard, and what it does not do |
| 7 | `CLAUDE.md` | the state block and the map |
| 8 | `COMMITS.md` | this block |

`docs/LESSONS.md` is not in the list: L42 went out in batch one and nothing
about the guard changed what it says.

No task files change and no draws are added, so `check_cost.py` and
`check_calibration.py` have nothing new to re-derive; both were run after the
guard landed and reported 152 files and 26 entries, unchanged. The harness suite
is **110 passed**, up from 105.

### Session 14 — the second Metal task, and a gate that refused everything

```bash
bash ~/scratchbench-session14-commits.sh
```

13 commits, **one push at the end**, no sleeps between commits. Nothing is
squashed.

**No calibration draws this time, so there is no first push of results files.**
The task is finished and deliberately uncalibrated: about half of its failing
runs enter a state where every MPS operation in the process is a hundred times
slower, which turns a wrong solution into a `timeout` rather than a `failed`.
The pass rate would survive that and the failure shape would not. `meta.yaml`
carries `frozen_set: unvalidated` and says why, so the loader is not being
argued with — it is being told the truth.

| # | File | What |
|---:|---|---|
| 1 | `tasks/…/reference/softmax_backward_kernel.py` | three reductions, one scratch array |
| 2 | `tasks/…/hidden_tests/test_softmax_backward_kernel.py` | 81 tests, tolerance from a measurement |
| 3 | `tasks/…/starter/softmax_backward_kernel.py` | the signature and nothing else |
| 4 | `tasks/…/prompt.md` | written last, on purpose |
| 5 | `tasks/…/meta.yaml` | unvalidated, with the reason in the file |
| 6 | `tools/mutate_metal_task.py` | both Metal tasks; runs bounded by the task's own limit |
| 7 | `README.md` | task and test counts |
| 8 | `docs/V2_DESIGN.md` | §5, the third accelerated task |
| 9 | `docs/LESSONS.md` | L38, L39, L40, L41 |
| 10 | `docs/ROADMAP.md` | §1.3 gate verdicts, §2.2, §9 debt ledger, §9.1 |
| 11 | `docs/sessions/14-…md` | journal 14 |
| 12 | `CLAUDE.md` | state |
| 13 | `COMMITS.md` | this |

One ordering note that still holds from session 13: `meta.yaml` without
`hidden_tests/` beside it is a task the loader refuses, so any tip inside
commits 1 through 5 is red on a commit that is not wrong. The single push at the
end makes it moot.

### Session 13 — a laptop task, refused, and what it taught

Two runs, because the push cadence changed in the middle of the session.

**Part one, already pushed:** the thirty calibration draws behind the new task's
block, 30 commits and one push. They land first because `meta.yaml` carries a
calibration block and `tools/check_calibration.py` re-derives it from those
files in CI.

**Part two, the rest:**

```bash
bash ~/scratchbench-session13-rest.sh
```

14 commits, **one push at the end**, no sleeps. That is the standing cadence
now: one file per commit, every commit landing separately so each one is its own
contribution, and a single push because a CI run per commit was never what the
gaps were buying. Nothing here is squashed.

| # | File | What |
|---:|---|---|
| 1 | `tasks/…/reference/scaled_swish.py` | the reference backward, differentiable in turn |
| 2 | `tasks/…/hidden_tests/test_scaled_swish.py` | 116 tests, the second-order ones are the point |
| 3 | `tasks/…/starter/scaled_swish.py` | the skeleton |
| 4 | `tasks/…/prompt.md` | written last, on purpose |
| 5 | `tasks/…/meta.yaml` | refused from v2, warmup with its block |
| 6 | `tools/mutate_v2_tasks.py` | twelve mutants, three expected to survive |
| 7 | `tools/check_cost.py` | it walked one directory of the two costs live in |
| 8 | `README.md` | the fourth task and the sharpened question |
| 9 | `docs/V2_DESIGN.md` | §4b the headline decision, §2.0 sharpened |
| 10 | `docs/LESSONS.md` | L36, L37 |
| 11 | `docs/ROADMAP.md` | candidate one spent, three left in the bank |
| 12 | `docs/sessions/13-…md` | journal 13 |
| 13 | `CLAUDE.md` | state |
| 14 | `COMMITS.md` | this |

The task's five files used to need a push of their own for a different reason,
and it is worth keeping in mind for any future queue that pushes more than once:
`meta.yaml` without `hidden_tests/` beside it is a task the loader refuses, so a
tip inside that group is red on a commit that is not wrong. A single push at the
end makes the question moot.

### Session 12 — the admission rule reads the top two

Sixteen commits, fifteen pushes, sixty seconds between them. The runnable form
is a script outside the repository, because it is long enough that pasting it
into a terminal in two halves is how a queue gets half-run:

```bash
bash ~/scratchbench-session12-commits.sh
```

**Push 1 carries two commits on purpose, and this is the part not to "fix".**
`runner/tasks.py` and `tests/test_runner.py` are a **coupled pair**: with one
file per push, whichever lands first leaves a tip where the tests describe a
rule the loader does not implement, and CI goes red on a commit that is not
wrong. One file per commit is untouched — both are separate commits — but they
travel in one push. That is `docs/LESSONS.md` L17 applied on purpose rather than
after the fact.

Every push retries three times: on 2026-08-17 GitHub answered a valid credential
with `Invalid username or token` while returning 503 on the API. A push failure
is not an authentication failure.

The order, and the tip after each one is green:

| # | Files | Commit message |
|---:|---|---|
| 1 | `runner/tasks.py` **+** `tests/test_runner.py` | the rule, then its tests (two commits, one push) |
| 2 | `tasks/flash_attention_backward/meta.yaml` | flash_attention_backward earns v2 on Sonnet's two lost draws |
| 3 | `tasks/fused_rmsnorm_kernel/meta.yaml` | fused_rmsnorm_kernel carries the draws that qualified it for v2 |
| 4 | `tasks/speculative_decoding_verify/meta.yaml` | Say which two models cleared speculative_decoding_verify |
| 5 | `tasks/activation_checkpointing_rng/meta.yaml` | Say which two models cleared activation_checkpointing_rng |
| 6 | `TASK_FORMAT.md` | The contract states the two-entry admission rule |
| 7 | `README.md` | The README says which task the rule was rewritten for |
| 8 | `CONTRIBUTING.md` | Calibrate against two models, because one is not a field |
| 9 | `leaderboard/README.md` | The CUDA rows say which set their task moved to |
| 10 | `docs/V2_DESIGN.md` | V2_DESIGN records the revision and what it rejected |
| 11 | `docs/LESSONS.md` | L35: I changed the admission rule after it refused a task I liked |
| 12 | `docs/sessions/12-the-rule-that-read-one-model.md` | Journal 12: the rule that read one model |
| 13 | `docs/ROADMAP.md` | A roadmap: every remaining session from here to launch |
| 14 | `CLAUDE.md` | Update state after the admission rule widened |
| 15 | `COMMITS.md` | Update commit queue |

## Committed

### 2026-08-17 — session 11 tail, two commits

The GitHub description and topics were set through the API, which leaves no file
in the tree; the badge row in `README.md` is the part that does. Both commits
pushed a minute apart, both runs green, `main` at 274 commits.

### 2026-08-17 — session 11, forty-nine commits

The Metal task and its calibration, the three v1 tasks that moved to `warmup`,
and the documents around both. Forty-nine commits in four groups, one push per
commit, every run green; `main` at 272 commits.

Two things went differently from the plan and neither cost anything. Group 1 ran
on 2026-08-14 and the rest on 2026-08-17, so the session spans three days in the
history. And GitHub returned `Invalid username or token` to a push in the middle
of the last group while answering 503 to the same credential on the API — a
platform incident wearing an authentication error's clothes. `git ls-remote`
with that credential succeeded a minute later. One commit sat unpushed until
the retry; the loop in the last script now tries three times before it stops.

### 2026-08-13 — session 10, a hundred and one commits

The three v2 candidates written end to end, the calibration machinery that
refused all three, and the 65 draws that refusal was computed from. Written on
2026-08-09 and queued; the queue ran on 2026-08-13 in three groups of about
thirty-four, one push per commit, every run green. `main` at 223 commits.

The group boundaries were the two ordering constraints, not tidiness: all 65
calibration draws had to land before `tests/test_runner.py`, which re-derives
every block from them, and the `check_calibration.py` step in `ci.yml` had to
land after both the tool and the draws.

### 2026-08-09 — session 09, twenty-two commits

The accelerated tier's first model runs, the leaderboard rewritten around both
tiers with the failure shape beside the rate, journal 09 and L27. Twenty-two
commits, twenty-two pushes a minute apart, all green. `main` at 122 commits.

### 2026-08-09 — session 08, ten commits

The five Opus draws on the laptop tier, the leaderboard page rewritten around
them, journal 08, and L26. Ten commits, ten pushes a minute apart, all green.

### 2026-08-09 — the replay, 45 commits, and CI green

`main` was reset to `fa24428` and everything after it replayed, ten commits on
2026-08-03 and thirty-five the next day, one push per commit. Ninety commits,
working tree clean, every workflow run green.

The fifteen red crosses are gone from the history because the commits they
belonged to are: a check run belongs to a SHA, and none of those SHAs is
reachable from `main` any more. All fifteen had the same cause and all fifteen
were correct — `adapters/__init__.py` was committed importing
`adapters.anthropic_api`, which was never committed, so `pytest` exited 2 at
collection from the first CI commit onwards. `docs/LESSONS.md` L17 and L24.

Per-commit pushing was safe there because of where the workflow file sat in the
queue: `ci.yml` was commit 34 of 45, so the thirty-three pushes before it started
no run at all, and the tree at 34 had been assembled and run against CI's four
commands beforehand. That ordering is not a general property. `ci.yml` in its
original position lands before `tools/check_cost.py` has anything in
`leaderboard/` to check and goes red on arrival.

The Actions tab keeps run history independently of commits, so the fifteen
failures stay listed there until deleted:

```bash
gh run list --limit 20 --json databaseId,conclusion \
  --jq '.[] | select(.conclusion=="failure") | .databaseId' \
  | while read -r id; do gh run delete "$id"; done
```

### Session 16, second half — candidate 3, and a sweep the credit cut short

```bash
bash ~/scratchbench-session16b-commits.sh
```

Twenty draws plus ten files. **One push at the end**, nothing squashed. Every
commit in it is guarded — `git add` then commit only if something was staged —
so it is safe to run whether or not the first-half script has already gone
through, and a file that is already committed and unchanged is skipped rather
than aborting under `set -e`.

Same hard constraint as always: the twenty draws land before
`tasks/chunked_batchnorm_reduction/meta.yaml`, because CI re-derives every
calibration block from `calibration/` and `leaderboard/`.

⚠ The block it commits is **incomplete on purpose**. `claude-opus-5` is done at
ten draws, `claude-sonnet-5` has one graded draw and nine `adapter_error`s from
the credit balance running out, and Haiku was never asked. The task is
`frozen_set: unvalidated` and says so in the file. Resuming needs credit, not
code: nine more Sonnet draws and ten Haiku draws, about $2.

### Session 17 — the sweep finished, and one loss in thirty was the task

```bash
bash ~/scratchbench-session17-commits.sh
```

Nineteen draws plus six files, twenty-five commits. **One push at the end**,
nothing squashed, every commit guarded so a re-run skips what is already in.

The same hard constraint: the nineteen draws land before
`tasks/chunked_batchnorm_reduction/meta.yaml`, because CI re-derives every
calibration block from `calibration/` and `leaderboard/`.

What the block now says is complete: `claude-opus-5` 9/10, `claude-sonnet-5`
10/10, `claude-haiku-4-5` 0/10. The task is `frozen_set: warmup`, and it is
worth knowing that the admission rule would have accepted `v2` — it refuses only
a task the top two measured entries both clear. It is `warmup` because ten of
the eleven lost draws also fail `chunk_sizes0`, the single-chunk split where the
chunked mistake is correct by construction. See `docs/LESSONS.md` L45 and
`docs/ROADMAP.md` 9.4.

### Session 17, second half — the finish plan, section 4, and section 3

```bash
bash ~/scratchbench-session17b-commits.sh
```

Sixteen files, sixteen commits, one push. Sections **4 and 3 both land here**,
which is two roadmap sessions in one. `ROADMAP.md` gains §10: `v2` ships now with
option C's reporting shape, the OpenAI adapter is in launch scope, a CUDA box is
rented for one session, ~$40 over four sessions. It also carries **all of §4** —
`failed_tests` at results version 3, `cost_per_solved`, and
`tools/render_leaderboard.py` with its CI gate.

⚠ `leaderboard/README.md` changes in this batch. The rates and counts are
unchanged; what changes is that the four tables are now generated, gain a
`$ per solved` column, and `.github/workflows/ci.yml` fails the build if the page
and the files behind it ever disagree.

⚠ `adapters/model_api.py` stops being a skeleton module and becomes the shared
contract: the prompt, the schema and the writer move there out of
`anthropic_api.py`, unchanged, so both vendors ask the same question. The
Anthropic adapter imports them back under the same names and
`AnthropicResponseError` becomes a subclass of the shared `ResponseError`, so
nothing that caught it before stops working. `tools/check_cost.py` routes a
model to its own vendor's price table through `adapters.resolve`, rather than
holding a second prefix table that could disagree with the runner about who
serves what.

⛔ No results files are committed. The OpenAI draws are one per model, and a
calibration block needs ten.

No draws and no `meta.yaml`, so there is no calibration ordering constraint. The
order that is there is the usual one: the three code files land before the three
records, because every number in the records is produced by the code above them.
The harness suite is **120 passed** after it, up from 110, and
`tools/check_calibration.py` and `tools/check_cost.py` both still read all 251
published files at the older schema.

---

## Session 18, first half — the red badge, the claims audit, and a count that is evidence

Ran before the `v2` sweep's draws are in, so that CI validates the schema change
on its own rather than inside a batch that also moves numbers.

The session opened on `gh run list`, which is the command the session 17 restart
file told me to run and I had not: **CI had been red on every push since
2026-09-08**, three pushes, both Python versions, one test. The test asserted
against the machine it ran on rather than the mechanism it was written to check,
so it passed here and failed on every GPU-less runner. `docs/LESSONS.md` **L47**,
and the fix is already pushed.

What follows it is §6.3's claims audit and what the audit found underneath the
eight wrong numbers: the README was holding derived values by hand while a tool
existed to derive them. So the counts and the task table are now generated the
way `leaderboard/README.md`'s tables already are, and CI fails the build when the
page and the directories disagree.

⚠ **`meta.yaml` gains a required key, `hidden_tests`.** The first version of the
renderer counted the tests by collecting them, which worked here and crashed on
CI: `fused_rmsnorm_kernel` imports triton, which has no macOS wheel and is not
installed on a CPU-only runner. A count that can only be derived on some machines
is the same defect as a test that only passes on some machines — L47 reproduced
by the tool written to prevent its cousin. So the number is checked in as
evidence and `scratchbench validate` re-derives it wherever the task can actually
run, which is the rule the accelerated tier already follows.

The order below is the usual one: the contract before the files that satisfy it,
the tool before the page it renders, the gate after the page is correct, and the
records last because every number in them is produced by the code above them.
All seventeen `meta.yaml` files carry a count derived this session, totalling
**805**, and `validate` reports `ok` on all fifteen tasks it could run here.

⚠ **`runner/sandbox.py` changes too, and it is a separate finding.** The `v2`
sweep produced two `collection_error` draws on a task `validate` says is intact.
That status does not mean the model failed — it means the **task** is broken and
every number measured against it is void, and it prints as `BROKEN`. The cause
is a bool with three answers: `starter_collects_cleanly` returns `False` both
when the baseline ran and could not collect, and when it never ran at all
because every attempt stalled. So the machine's state was published as a verdict
on the task. It now returns `bool | None` and `grade` maps `None` to
`mps_stalled`. Both statuses are `NO_EVIDENCE`, so **no published rate moves**;
what moves is the meaning. `docs/LESSONS.md` **L48**, which also records that my
first attempt to confirm the diagnosis used a number that could not have refuted
it.

⛔ **The historical draws are not relabelled.** Whether those two were stalls is
unestablished, and rewriting recorded evidence to match a diagnosis is L43 —
which does not become acceptable because the rewrite would flatter my own.

---

## Session 18, second half — the first v2 sweep, and the task it removed

⛔ **Run the first-half script before this one.** These `meta.yaml` files are
touched in both batches, and the `hidden_tests` schema has to land first.

⛔ **The sixty draws are committed BEFORE any `meta.yaml`.** CI derives every
calibration block from the draw files, so a block committed ahead of its evidence
fails the build at that tip for a reason unrelated to the change.

Six models, ten draws each, both vendors, `--set v2 --tier all` on the M1 Pro.
**$8.4807** against §10.4's ~$30. The sweep was scheduled to produce a `v2` row
and produced a decision instead.

`flash_attention_backward` entered `v2` in August as the one laptop task
separating two frontier Claude models. Sonnet's two losses did not repeat and two
OpenAI models cleared it outright, so cumulatively the top two measured entries
are both at the ceiling and `_check_admission` refuses it. It is `warmup` now, and
**`v2` ships as three accelerated tasks with no laptop member at all** — stated on
the page in those words rather than buried. `docs/ROADMAP.md` §11 carries the
decision and the three alternatives it rejects; the one worth naming is that
omitting the OpenAI models from the blocks would have kept the task by choosing
which evidence counts after seeing which way it went (L21, L35, L43).

⭐ The better half of the news is the accelerated tier: both Metal tasks now carry
**six models at six distinct rates**, and every failing test instance clusters on
threadgroup sizes 7, 33 and 96 — with `claude-sonnet-5` failing only there and
never at 1, 32 or 1024. That is the hazard both tasks were written around, read
out of the failure shapes rather than out of the author's description.

⚠ `tools/render_leaderboard.py` also stops printing `0/0` for a task no draw ever
graded. `fused_rmsnorm_kernel` wants CUDA and the sweep ran on a Mac, and `0/0`
is arithmetically true while reading as "failed everything".

---

## Session 18, third half — the invariant that was a fact about one vendor

CI went red on the sweep's own evidence.
`test_nothing_published_was_billed_at_a_cache_rate` asserted every published
draw read zero cached tokens, on the reasoning that each task is sent once as
its own prompt so there is no prefix to reuse. That reasoning was correct and it
was a fact about **Anthropic**, where caching is opt-in and this adapter never
sets `cache_control`. OpenAI caches long prompt prefixes automatically, and nine
of every ten OpenAI draws read about 90% of their prompt from a cache nobody
requested. All 86 Claude draws still read zero.

⚠ **The pricing was never wrong** — `price_from_counts` subtracts the cached
count from the input count because the API reports it as a subset, and all 311
published costs re-derive. What is wrong is publishing a warm-cache price on a
page whose argument is that you can re-run this: the OpenAI rows are 5.4% to
6.1% cheaper than the same draws cost cold. `leaderboard/README.md` now carries
the cold-cache column beside them.

The test splits into the three invariants that are actually true across both
vendors: nobody is billed for a cache **write**; a cached read is always a
**subset** of the input it came from, which is the assertion protecting the
subtraction from silently under-billing every row; and **Anthropic** draws read
zero, which is what makes two Claude rows comparable to each other.

`docs/LESSONS.md` **L49**. The rule worth keeping: an invariant that holds by
construction is only as portable as the construction.

---

## Session 19 — an exit code is not a verdict (F01/F03/F06)

```bash
bash ~/scratchbench-session19-commits.sh
```

Eight commits, one file each, one push at the end, nothing squashed. Order does
not matter for CI here — no draws, no `meta.yaml`, no calibration — but the files
go fix first, then the tests that pin it, then the prose:

1. `runner/mps_stall.py` — the guard writes a sentinel when it fires.
2. `runner/sandbox.py` — a pass requires the suite to have reached the declared
   count; a stall requires the sentinel; a bare 125 is the solution's failure.
3. `runner/cli.py` — `validate` keeps an unrun task off the validated total and
   grows a `--strict` flag.
4. `.github/workflows/ci.yml` — CI validates with `--strict`.
5. `tests/test_runner.py` — six regression tests (one per door plus the bare-125
   retry case), and the three stall tests updated to model a real stall by
   writing the sentinel rather than returning the exit code alone.
6. `docs/LESSONS.md` — L50.
7. `docs/sessions/19-an-exit-code-is-not-a-verdict.md` — the journal.
8. `README.md` — the LESSONS entry count, forty-nine to fifty.

The review that produced these (`scratchbench-review-2026-10-16/`) is not in the
repository and does not go in it. F02 — the run environment is not a security
sandbox — is the next work package, not this one.

---

## Review follow-up, sessions 19–23 (2026-10-16/17) — ONE script, supersedes the session-19 pointer above

```bash
bash ~/scratchbench-review-fixes-commits.sh
```

The 2026-10-16 technical review's eight findings, closed. Delivered as a single
script and a single push because the work was prepared in one working tree: the
README's LESSONS count and the task counts climb across the packages, so an
intermediate push would show a count ahead of the file that justifies it. One
commit per file, nothing squashed, no AI attribution, push once at the end.

Twenty-nine files. The order groups them by the finding they close, fix before
the tests that pin it before the prose:

- **F01/F03/F06** — `runner/mps_stall.py`, `runner/sandbox.py`, `runner/cli.py`,
  `.github/workflows/ci.yml`.
- **F04/F05** — both batchnorm references, their hidden tests, the reduction
  prompt, both `meta.yaml` (version 2).
- **F02** — `runner/sandbox.py` (credential strip, shared commit with F01),
  README trust note.
- **F07 / §5.1** — `runner/report.py` (version-keyed grouping, commit
  provenance), `leaderboard/SET_MANIFESTS.yaml`.
- **§5.2 / §6 / F08** — `tools/render_leaderboard.py`, `leaderboard/README.md`,
  `README.md`, `TASK_FORMAT.md`, `CLAUDE.md`, `adapters/model_api.py`,
  `runner/tasks.py`.
- **The record** — `tests/test_runner.py`, `docs/LESSONS.md` (L50–L54), five
  session journals, `docs/ROADMAP.md`, this file.

Verified before delivery: full suite **155 passed** on the no-GPU plugin,
`validate` 14/14, `check_cost` 311, `check_calibration` 44, both renderers in
`--check`. The review itself (`scratchbench-review-2026-10-16/`) is not in the
repository and does not go in it.

---

## Session 24 — the CUDA sweep (2026-10-28)

The review follow-up above was never committed, and session 24 then edited some
of the same files, so both ship together in one push. The queue is the script
`~/scratchbench-session24-commits.sh` (git is Selim's to run):

- 120 new calibration draws, one commit each (house style)
- 69 superseded v1 batchnorm draws removed, one commit each
- the two batchnorm tasks (reference, tests, prompt, meta) and the rmsnorm meta
- the grader/harness review fixes (sandbox, mps_stall, cli, report, tasks, model_api)
- tools (render_leaderboard, new render_chart), tests, CI
- docs (ROADMAP §13, LESSONS L55/L56, SET_MANIFESTS, README, CLAUDE, sessions 19-24, pass-rates.svg)
- a single push at the end, three attempts

Verified before delivery: full suite **157 passed** on the no-GPU plugin,
`validate` 14/14, `check_calibration` 48/362, `check_cost` 362, all three
renderers in `--check`.
