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
