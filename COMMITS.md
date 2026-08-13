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

### Session 10 — three v2 candidates, the calibration rule, and its evidence

A hundred and one commits in three groups of about thirty-four. Run them in order; a
group is safe to stop after, and the group boundaries are where they are because
of the two ordering constraints below, not for tidiness.

**Why this order.** `discover_tasks` only loads a directory that has a
`meta.yaml`, so a task is invisible to `validate`, to the control run and to the
whole suite until its metadata lands. Each task therefore goes in as reference,
hidden tests, starter, prompt — four pushes CI cannot see — and then `meta.yaml`,
at which point all four parts exist and it validates on arrival.

Two ordering constraints are load-bearing and neither is obvious:

- `tests/test_runner.py` re-derives every calibration block from
  `calibration/`, so **all 65 draw files must land before it** or the suite goes
  red on a claim whose evidence has not arrived yet.
- `.github/workflows/ci.yml` gains a `check_calibration.py` step, so it lands
  **after** both the tool and the draws. Before that commit nothing runs the
  checker, which is what makes block B silent.

The three runner files go in before `tests/test_runner.py` because the new tests
import behaviour that does not exist until they land. The old tests pass against
the new runner files, which is what makes those commits green rather than merely
quiet.

No force. `origin/main` and `main` are the same commit, so every push is a
fast-forward.

#### Grup 1 — the tasks, the tools, the runner, and the first draws (32 commits, ~32 minutes)

The first twenty are the ones CI reacts to. Each task goes in as reference,
hidden tests, starter, prompt — four pushes CI cannot see, because
`discover_tasks` only loads a directory that has a `meta.yaml` — and then its
`meta.yaml`, at which point all four parts exist and the task validates on
arrival. The three runner files land after them and before any test that needs
them. The last twelve are calibration draws, which nothing reads yet.

```bash
cd ~/scratchbench
set -e

n=0
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n/32 pushed: $message"
done <<'QUEUE'
tasks/speculative_decoding_verify/reference/speculative_decoding.py|speculative_decoding_verify: the accept-and-correct step, derived from its guarantee
tasks/speculative_decoding_verify/hidden_tests/test_speculative_decoding.py|speculative_decoding_verify: hidden tests, graded on the output distribution
tasks/speculative_decoding_verify/starter/speculative_decoding.py|speculative_decoding_verify: starter
tasks/speculative_decoding_verify/prompt.md|speculative_decoding_verify: the prompt states the guarantee, not the formula
tasks/speculative_decoding_verify/meta.yaml|speculative_decoding_verify: metadata and calibration
tasks/flash_attention_backward/reference/flash_backward.py|flash_attention_backward: the backward pass, one block of keys at a time
tasks/flash_attention_backward/hidden_tests/test_flash_backward.py|flash_attention_backward: hidden tests against autograd, and one against algebra
tasks/flash_attention_backward/starter/flash_backward.py|flash_attention_backward: starter
tasks/flash_attention_backward/prompt.md|flash_attention_backward: the prompt claims only what the interface enforces
tasks/flash_attention_backward/meta.yaml|flash_attention_backward: metadata and calibration
tasks/activation_checkpointing_rng/reference/checkpointed_mlp.py|activation_checkpointing_rng: recomputation that puts the generator back
tasks/activation_checkpointing_rng/hidden_tests/test_checkpointed_mlp.py|activation_checkpointing_rng: hidden tests, including the generator state itself
tasks/activation_checkpointing_rng/starter/checkpointed_mlp.py|activation_checkpointing_rng: starter
tasks/activation_checkpointing_rng/prompt.md|activation_checkpointing_rng: the prompt asks for the bracket without describing it
tasks/activation_checkpointing_rng/meta.yaml|activation_checkpointing_rng: metadata and calibration
tools/mutate_v2_tasks.py|tools: the mutation pass for the laptop-tier v2 candidates
tools/check_calibration.py|tools: every calibration block, re-derived from the draws behind it
runner/tasks.py|runner: a calibration block, the admission rule, and a frozen-set filter
runner/report.py|runner: a sweep that died halfway is not a draw of the set rate
runner/cli.py|runner: per-draw workdirs, the calibration column, and --set
calibration/claude-haiku-4-5-20260809T160658+0000-draw1.json|calibration: claude-haiku-4-5, draw 1 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T160730+0000-draw2.json|calibration: claude-haiku-4-5, draw 2 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T160807+0000-draw3.json|calibration: claude-haiku-4-5, draw 3 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T160836+0000-draw4.json|calibration: claude-haiku-4-5, draw 4 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T160901+0000-draw5.json|calibration: claude-haiku-4-5, draw 5 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T161145+0000-draw1.json|calibration: claude-haiku-4-5, draw 1 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T161238+0000-draw2.json|calibration: claude-haiku-4-5, draw 2 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T161309+0000-draw3.json|calibration: claude-haiku-4-5, draw 3 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T161340+0000-draw4.json|calibration: claude-haiku-4-5, draw 4 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T161405+0000-draw5.json|calibration: claude-haiku-4-5, draw 5 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T162905+0000-draw1.json|calibration: claude-haiku-4-5, draw 1 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T162941+0000-draw2.json|calibration: claude-haiku-4-5, draw 2 of a v2-candidate sweep
QUEUE
```

#### Grup 2 — calibration draws (32 commits, ~32 minutes, unattended)

Thirty-two results files and nothing else. No workflow can fail on any of these
trees, so this group is safe to start and walk away from. `sleep 20` instead of
`sleep 60` is fine here and cuts it to eleven minutes.

```bash
cd ~/scratchbench
set -e

n=0
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n/32 pushed: $message"
done <<'QUEUE'
calibration/claude-haiku-4-5-20260809T163008+0000-draw3.json|calibration: claude-haiku-4-5, draw 3 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163043+0000-draw4.json|calibration: claude-haiku-4-5, draw 4 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163114+0000-draw5.json|calibration: claude-haiku-4-5, draw 5 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163152+0000-draw6.json|calibration: claude-haiku-4-5, draw 6 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163226+0000-draw7.json|calibration: claude-haiku-4-5, draw 7 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163251+0000-draw8.json|calibration: claude-haiku-4-5, draw 8 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163318+0000-draw9.json|calibration: claude-haiku-4-5, draw 9 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163342+0000-draw10.json|calibration: claude-haiku-4-5, draw 10 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163903+0000-draw1.json|calibration: claude-haiku-4-5, draw 1 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163915+0000-draw2.json|calibration: claude-haiku-4-5, draw 2 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163926+0000-draw3.json|calibration: claude-haiku-4-5, draw 3 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163936+0000-draw4.json|calibration: claude-haiku-4-5, draw 4 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163947+0000-draw5.json|calibration: claude-haiku-4-5, draw 5 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T163957+0000-draw6.json|calibration: claude-haiku-4-5, draw 6 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T164006+0000-draw7.json|calibration: claude-haiku-4-5, draw 7 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T164025+0000-draw8.json|calibration: claude-haiku-4-5, draw 8 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T164035+0000-draw9.json|calibration: claude-haiku-4-5, draw 9 of a v2-candidate sweep
calibration/claude-haiku-4-5-20260809T164047+0000-draw10.json|calibration: claude-haiku-4-5, draw 10 of a v2-candidate sweep
calibration/claude-opus-5-20260809T160108+0000-draw1.json|calibration: claude-opus-5, draw 1 of a v2-candidate sweep
calibration/claude-opus-5-20260809T160217+0000-draw2.json|calibration: claude-opus-5, draw 2 of a v2-candidate sweep
calibration/claude-opus-5-20260809T160340+0000-draw3.json|calibration: claude-opus-5, draw 3 of a v2-candidate sweep
calibration/claude-opus-5-20260809T160447+0000-draw4.json|calibration: claude-opus-5, draw 4 of a v2-candidate sweep
calibration/claude-opus-5-20260809T160607+0000-draw5.json|calibration: claude-opus-5, draw 5 of a v2-candidate sweep
calibration/claude-opus-5-20260809T161732+0000-draw1.json|calibration: claude-opus-5, draw 1 of a v2-candidate sweep
calibration/claude-opus-5-20260809T161843+0000-draw2.json|calibration: claude-opus-5, draw 2 of a v2-candidate sweep
calibration/claude-opus-5-20260809T161955+0000-draw3.json|calibration: claude-opus-5, draw 3 of a v2-candidate sweep
calibration/claude-opus-5-20260809T162110+0000-draw4.json|calibration: claude-opus-5, draw 4 of a v2-candidate sweep
calibration/claude-opus-5-20260809T162223+0000-draw5.json|calibration: claude-opus-5, draw 5 of a v2-candidate sweep
calibration/claude-opus-5-20260809T162343+0000-draw6.json|calibration: claude-opus-5, draw 6 of a v2-candidate sweep
calibration/claude-opus-5-20260809T162443+0000-draw7.json|calibration: claude-opus-5, draw 7 of a v2-candidate sweep
calibration/claude-opus-5-20260809T162544+0000-draw8.json|calibration: claude-opus-5, draw 8 of a v2-candidate sweep
calibration/claude-opus-5-20260809T162653+0000-draw9.json|calibration: claude-opus-5, draw 9 of a v2-candidate sweep
QUEUE
```
-- burdan 
#### Grup 3 — the last draws, then the tests, CI and the documents (37 commits, ~37 minutes)

The remaining twenty-one draws first, and only then `tests/test_runner.py`,
because it re-derives every calibration block from `calibration/` and would go
red against evidence that has not finished arriving. `.github/workflows/ci.yml`
comes after both the tool and the draws: before that commit nothing runs
`check_calibration.py`, which is what makes groups 1 and 2 silent.

```bash
cd ~/scratchbench
set -e

n=0
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n/37 pushed: $message"
done <<'QUEUE'
calibration/claude-opus-5-20260809T162804+0000-draw10.json|calibration: claude-opus-5, draw 10 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163416+0000-draw1.json|calibration: claude-opus-5, draw 1 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163457+0000-draw2.json|calibration: claude-opus-5, draw 2 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163524+0000-draw3.json|calibration: claude-opus-5, draw 3 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163556+0000-draw4.json|calibration: claude-opus-5, draw 4 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163618+0000-draw5.json|calibration: claude-opus-5, draw 5 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163648+0000-draw6.json|calibration: claude-opus-5, draw 6 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163715+0000-draw7.json|calibration: claude-opus-5, draw 7 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163744+0000-draw8.json|calibration: claude-opus-5, draw 8 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163816+0000-draw9.json|calibration: claude-opus-5, draw 9 of a v2-candidate sweep
calibration/claude-opus-5-20260809T163850+0000-draw10.json|calibration: claude-opus-5, draw 10 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T182928+0000-draw1.json|calibration: claude-sonnet-5, draw 1 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T183125+0000-draw2.json|calibration: claude-sonnet-5, draw 2 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T183310+0000-draw3.json|calibration: claude-sonnet-5, draw 3 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T183440+0000-draw4.json|calibration: claude-sonnet-5, draw 4 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T183557+0000-draw5.json|calibration: claude-sonnet-5, draw 5 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T183821+0000-draw6.json|calibration: claude-sonnet-5, draw 6 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T184254+0000-draw7.json|calibration: claude-sonnet-5, draw 7 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T184431+0000-draw8.json|calibration: claude-sonnet-5, draw 8 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T184614+0000-draw9.json|calibration: claude-sonnet-5, draw 9 of a v2-candidate sweep
calibration/claude-sonnet-5-20260809T184800+0000-draw10.json|calibration: claude-sonnet-5, draw 10 of a v2-candidate sweep
tests/test_runner.py|tests: calibration, the warm-up set, partial draws, per-draw workdirs and the set filter
.github/workflows/ci.yml|ci: calibration blocks reproduce from their own draws
leaderboard/claude-sonnet-5-20260809-draw1.json|leaderboard: claude-sonnet-5 on the laptop tier, draw 1 of 5
leaderboard/claude-sonnet-5-20260809-draw2.json|leaderboard: claude-sonnet-5 on the laptop tier, draw 2 of 5
leaderboard/claude-sonnet-5-20260809-draw3.json|leaderboard: claude-sonnet-5 on the laptop tier, draw 3 of 5
leaderboard/claude-sonnet-5-20260809-draw4.json|leaderboard: claude-sonnet-5 on the laptop tier, draw 4 of 5
leaderboard/claude-sonnet-5-20260809-draw5.json|leaderboard: claude-sonnet-5 on the laptop tier, draw 5 of 5
leaderboard/README.md|leaderboard: a third model on the v1 laptop tier, and a draw that measured seven of eight
TASK_FORMAT.md|Task format: frozen sets, calibration, and where the draws live
CONTRIBUTING.md|Contributing: how to calibrate a task, and what a refusal means
docs/V2_DESIGN.md|Design: the property the criterion was missing, and what v2 contains
docs/LESSONS.md|Lessons: a design claim no test could keep, a half-empty error bar, and the wrong property
docs/sessions/10-calibrating-against-models.md|Journal 10: calibrating against models instead of against me
README.md|README: the next set has to earn its place, and the first three did not
CLAUDE.md|Update state after the first calibrated tasks
COMMITS.md|Update commit queue
QUEUE
```

Then confirm:

```bash
cd ~/scratchbench

git status --porcelain              # expect: empty
git log --oneline | wc -l           # expect 223
python -m pytest -q                 # expect 103 passed
python -m runner.cli validate       # expect 11 task(s) validated
python tools/check_cost.py          # expect 27 file(s) checked
python tools/check_calibration.py   # expect 9 calibration entries re-derived
gh run list --limit 3               # expect success
```

## Committed

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
