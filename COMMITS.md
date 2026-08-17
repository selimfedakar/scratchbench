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

### Session 11 — the Metal task, its calibration, and three tasks moved to warmup

Forty-nine commits in three groups. Run them in order; a group is safe to stop
after, and the boundaries are the two ordering constraints below rather than
tidiness.

**Why this order.** `discover_tasks` only loads a directory that has a
`meta.yaml`, so the new task is invisible to `validate`, to the control run and
to the whole suite until its metadata lands. It therefore goes in as reference,
hidden tests, starter, prompt — four pushes CI cannot see — and its `meta.yaml`
opens group 3, by which time both the machinery and the evidence its calibration
block claims are already in the repository.

Two constraints are load-bearing and neither is obvious:

- `tools/check_calibration.py` has to land **before** the three `warmup`
  metadata files. Their blocks are re-derived from `leaderboard/`, and the
  version on `main` reads `calibration/` only, so the order the other way round
  turns CI red on a claim whose evidence the checker cannot see.
- `tests/test_runner.py` has to land **before** the Metal task's `meta.yaml`.
  The old validate-count test subtracts every accelerated task from the number
  of tasks it expects to be checked, which was right when the only accelerated
  task needed hardware this machine does not have, and is wrong the moment one
  of them runs here.

The thirty results files in group 2 are read by `tools/check_cost.py` on every
push, and every one of them re-derives from its own tokens; nothing else reacts
to them until the `meta.yaml` at the end.

No force. `origin/main` and `main` are the same commit, so every push is a
fast-forward.

#### Group 1 — the task, the machinery, and the three tasks that moved (10 commits, ~10 minutes)

```bash
cd ~/scratchbench
set -e

n=0
total=10
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n/$total pushed: $message"
done <<'QUEUE'
tasks/metal_cross_entropy_kernel/reference/cross_entropy_kernel.py|metal_cross_entropy_kernel: the reduction, at a threadgroup size it does not choose
tasks/metal_cross_entropy_kernel/hidden_tests/test_cross_entropy_kernel.py|metal_cross_entropy_kernel: hidden tests against a float64 loss on the CPU
tasks/metal_cross_entropy_kernel/starter/cross_entropy_kernel.py|metal_cross_entropy_kernel: starter
tasks/metal_cross_entropy_kernel/prompt.md|metal_cross_entropy_kernel: the prompt states the launch, not the kernel
tools/mutate_metal_task.py|tools: the Metal task's mutation pass, with the verdict each mutant is expected to get
tools/check_calibration.py|tools: a draw is evidence wherever it is published, so read leaderboard/ too
tests/test_runner.py|tests: calibration from both directories, one accelerator missing is not all of them, and a draw with no call has no cost
tasks/attention_causal_mask/meta.yaml|attention_causal_mask: warmup, on the draws already published
tasks/quantization_error_bounds/meta.yaml|quantization_error_bounds: warmup, on the draws already published
tasks/sharded_dataloader/meta.yaml|sharded_dataloader: warmup, on the draws already published
QUEUE
```

#### Group 2 — the thirty calibration draws (30 commits, ~30 minutes, unattended)

Results files and nothing else. No workflow can fail on any of these trees, so
this group is safe to start and walk away from. `sleep 20` instead of `sleep 60`
is fine here and cuts it to ten minutes.

```bash
cd ~/scratchbench
set -e

n=0
total=30
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n/$total pushed: $message"
done <<'QUEUE'
leaderboard/accelerated-metal-claude-opus-5-20260813-draw1.json|leaderboard: claude-opus-5 on the Metal kernel, draw 1 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw2.json|leaderboard: claude-opus-5 on the Metal kernel, draw 2 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw3.json|leaderboard: claude-opus-5 on the Metal kernel, draw 3 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw4.json|leaderboard: claude-opus-5 on the Metal kernel, draw 4 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw5.json|leaderboard: claude-opus-5 on the Metal kernel, draw 5 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw6.json|leaderboard: claude-opus-5 on the Metal kernel, draw 6 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw7.json|leaderboard: claude-opus-5 on the Metal kernel, draw 7 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw8.json|leaderboard: claude-opus-5 on the Metal kernel, draw 8 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw9.json|leaderboard: claude-opus-5 on the Metal kernel, draw 9 of 10
leaderboard/accelerated-metal-claude-opus-5-20260813-draw10.json|leaderboard: claude-opus-5 on the Metal kernel, draw 10 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw1.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 1 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw2.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 2 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw3.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 3 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw4.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 4 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw5.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 5 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw6.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 6 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw7.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 7 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw8.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 8 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw9.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 9 of 10
leaderboard/accelerated-metal-claude-sonnet-5-20260813-draw10.json|leaderboard: claude-sonnet-5 on the Metal kernel, draw 10 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw1.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 1 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw2.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 2 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw3.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 3 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw4.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 4 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw5.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 5 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw6.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 6 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw7.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 7 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw8.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 8 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw9.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 9 of 10
leaderboard/accelerated-metal-claude-haiku-4-5-20260813-draw10.json|leaderboard: claude-haiku-4-5 on the Metal kernel, draw 10 of 10
QUEUE
```

#### Group 3 — the metadata that turns it on, and the documents (9 commits, ~9 minutes)

The first commit here is the one CI reacts to: the task becomes visible, its
calibration block is checked against the draws from group 2, and `validate`
reports it as `UNCHECKED` on the Linux runner, which has no Metal device.

```bash
cd ~/scratchbench
set -e

n=0
total=9
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n/$total pushed: $message"
done <<'QUEUE'
tasks/metal_cross_entropy_kernel/meta.yaml|metal_cross_entropy_kernel: metadata, and the first calibration block a numbered set accepted
leaderboard/README.md|leaderboard: the Metal row, the failures by name, and what v1 means after today
README.md|README: the kernels row stops promising Metal and starts reporting it
CONTRIBUTING.md|CONTRIBUTING: the Metal half of the accelerated evidence runs on a laptop
docs/V2_DESIGN.md|V2_DESIGN: the accelerated tier got its second task, and the three under review moved
docs/LESSONS.md|Lessons: a mutant that was a correct program, a race the hardware hides, a kernel that skips the kernel, and one reserved word
docs/sessions/11-a-metal-kernel-on-the-machine-i-already-have.md|Journal 11: a Metal kernel on the machine I already have
CLAUDE.md|Update state after the first task a frontier model fails
COMMITS.md|Update commit queue
QUEUE
```

#### After the last push

```bash
cd ~/scratchbench

git status --porcelain              # expect: empty
git log --oneline | wc -l           # expect 272
python -m pytest -q                 # expect 103 passed
python -m runner.cli validate --tier all   # expect 12 validated, 1 not checked here
python tools/check_cost.py          # expect 57 file(s) checked
python tools/check_calibration.py   # expect 21 calibration entries
python tools/mutate_metal_task.py   # expect all 13 mutants as expected
gh run list --limit 3               # expect success
```

## Committed

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
