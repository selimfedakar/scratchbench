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
