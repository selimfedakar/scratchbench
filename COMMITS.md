# Commit queue — `selimfedakar/scratchbench`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else.

Every block starts with `cd ~/scratchbench` so a block is safe to paste on its
own, into any shell, without checking which directory you are in.

## Why this queue replays history instead of continuing it

Fifteen commits on `main` carry a red cross and **every one of them is correct**:
the tree really is broken at each of them, and it has been broken since the
commit that introduced CI. The cause is one missing file.

`adapters/__init__.py` was committed at `0711605` already importing
`.anthropic_api`, and `adapters/anthropic_api.py` was never committed. Anything
that imports `adapters.reference` therefore pulls in `adapters/__init__.py`,
which fails:

```
tests/test_runner.py:17: in <module>
    from adapters.reference import solve as reference_solve
adapters/__init__.py:23: in <module>
    from .anthropic_api import AnthropicAdapter
E   ModuleNotFoundError: No module named 'adapters.anthropic_api'
```

`pytest` exits 2 at collection, before `validate` is ever reached. The first CI
commit `a970b0a` was already broken on arrival, and so was every commit after
it. Pushing one commit at a time is what turned that into fifteen separate red
runs: **GitHub starts a workflow run per push, against the tip of what was
pushed.** Commits that are never a push tip get no run and no cross.

The crosses cannot be removed without replacing the commits, because a check run
belongs to a commit SHA. So this queue resets to `fa24428` — the last commit
before CI existed, and therefore the last one with no run attached — and replays
everything from there. Nothing is lost: `--mixed` moves the branch pointer and
leaves every file on disk exactly as it is now.

**A push per commit is safe here, and that is a property of the order rather
than a general rule.** The workflow file is commit 34 of 45. Every tree before
it has no `.github/workflows/` at all, so those thirty-three pushes start no run
whatsoever. From commit 34 the tree is green, verified by running CI's four
commands against it before this queue was written, and the eleven commits after
it are documents no CI step reads. The result is zero red runs and twelve green
ones, however the pushes are spaced.

Get the order wrong and this stops being true: `ci.yml` in its original position
lands before `tools/check_cost.py` has anything in `leaderboard/` to check, and
goes red on arrival for a third distinct reason.

`docs/LESSONS.md` L17 says not to rewrite history over a red repository, and
that still stands: it is about reaching for the biggest tool before reading a
log line. The log line has been read, the cause is known and fixed in this
queue, and what is left is a cosmetic cleanup of a public portfolio repository
that its owner asked for. Different decision, made with the diagnosis in hand.

---

## Step 1 — move the branch back, keep every file

```bash
cd ~/scratchbench

git fetch origin
git log --oneline | wc -l             # expect 60
git log --oneline -1 fa24428          # expect: fa24428 Journal 02: the runner

git reset --mixed fa24428

git log --oneline | wc -l             # expect 45
git log --oneline -1                  # expect: fa24428 Journal 02: the runner
```

`--mixed`, not `--hard`. The working tree is untouched; only the branch pointer
and the index move. If `git log` does not say `fa24428` afterwards, stop and do
not continue.

## Step 2 — today: the first ten commits, one push each

Split over two days on purpose, and the split point is not arbitrary. **None of
these ten trees contains `.github/workflows/ci.yml`**, because that file is
commit 34. GitHub runs the workflow that exists *in the pushed tree*, so a tree
with no workflow file starts no run at all. These ten pushes produce **zero
workflow runs and therefore zero crosses**, no matter how they are spaced.

That is also why the tenth commit stops where it does. `quantization_error_bounds`
ends the day without its `meta.yaml`, which is deliberate: the loader skips any
task directory that has no `meta.yaml`, so the task is invisible to the harness
rather than half-present. Its metadata is tomorrow's first commit.

The first push is forced and the rest are not. `origin/main` currently holds the
fifteen commits Step 1 discarded, so the branch has diverged and only the first
push has to replace it; every push after that is a fast-forward.

The loop variable is `file` and not `path` on purpose. In zsh — which is the
shell this runs in — `$path` is the array form of `$PATH`, so `read path`
empties it and every command after the first iteration becomes "command not
found", halfway through a replay, with `set -e` unable to fire because the shell
cannot find the binary that was supposed to fail. Found by running this loop
against a throwaway repository before it ever touched this one.

The `sleep 60` is a pacing choice rather than a technical requirement. GitHub
does not need time between pushes and there is no indexing window to wait out; a
push is visible the moment it is accepted. It is here because one push per
minute is the rhythm this repository is worked at, and a burst of ten pushes in
five seconds reads as a script rather than as work.

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
  if [ "$n" -eq 1 ]; then
    git push --force-with-lease origin main
  else
    git push origin main
  fi
  echo "  $n pushed: $message"
done <<'QUEUE'
adapters/anthropic_api.py|adapters: Anthropic, one attempt per task, no silent model substitution
tasks/sharded_dataloader/reference/sharded_sampler.py|sharded_dataloader: reference solution
tasks/sharded_dataloader/hidden_tests/test_sharded_sampler.py|sharded_dataloader: hidden tests
tasks/sharded_dataloader/starter/sharded_sampler.py|sharded_dataloader: starter
tasks/sharded_dataloader/prompt.md|sharded_dataloader: prompt
tasks/sharded_dataloader/meta.yaml|sharded_dataloader: metadata
tasks/quantization_error_bounds/reference/quantize.py|quantization_error_bounds: reference solution
tasks/quantization_error_bounds/hidden_tests/test_quantize.py|quantization_error_bounds: hidden tests
tasks/quantization_error_bounds/starter/quantize.py|quantization_error_bounds: starter
tasks/quantization_error_bounds/prompt.md|quantization_error_bounds: prompt
QUEUE
```

Nine minutes of `sleep`, ten commits, ten pushes. Then check that the Actions
tab gained nothing:

```bash
git log --oneline | wc -l        # expect 55
git status --porcelain | wc -l   # expect 25 lines, which is 35 files: an
                                 # untracked directory counts as one line
gh run list --limit 5            # expect no new run: these trees have no
                                 # workflow. The fifteen failures still listed
                                 # are the old ones, and their commits are no
                                 # longer on the branch
```

## Step 3 — tomorrow: the remaining thirty-five commits

Start by confirming yesterday finished. If either of these disagrees, stop and
bring the output back rather than pushing on top of a half-replayed branch.

```bash
cd ~/scratchbench
git fetch origin
git log --oneline | wc -l               # expect 55
git log --oneline origin/main..HEAD     # expect: empty, everything from day one is pushed
```

Commits 1 to 23 of this batch still have no workflow file, so they still start
no runs. **Commit 24 is `.github/workflows/ci.yml`, and that tree is green:** it
was assembled and run before this queue was written, with the four commands CI
runs.

```
=== CI ADIM 1: pytest ===        77 passed in 18.84s
=== CI ADIM 2: validate ===      8 task(s) validated: reference passes, starter fails cleanly.
=== CI ADIM 3: reference run === laptop       100% (8/8)  ·  headline
=== CI ADIM 4: check_cost ===    2 file(s) checked: every cost reproduces from its own tokens.
```

That ordering is the reason `ci.yml` sits where it does rather than in its
original place. It has to land after the tasks, the tools, the runner, the tests
**and the two leaderboard files**, because `tools/check_cost.py` exits non-zero
when there is nothing in `leaderboard/` to check. Committed in its old position
it would have gone red on arrival for a third reason.

The twelve commits from `ci.yml` onwards are documents and metadata that no CI
step reads, so each of them gets its own run and each of those runs is green.
Twelve green ticks, no crosses.

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
  echo "  $n pushed: $message"
done <<'QUEUE'
tasks/quantization_error_bounds/meta.yaml|quantization_error_bounds: metadata
tasks/online_softmax_attention/reference/tiled_attention.py|online_softmax_attention: reference solution
tasks/online_softmax_attention/hidden_tests/test_tiled_attention.py|online_softmax_attention: hidden tests
tasks/online_softmax_attention/starter/tiled_attention.py|online_softmax_attention: starter
tasks/online_softmax_attention/prompt.md|online_softmax_attention: prompt
tasks/online_softmax_attention/meta.yaml|online_softmax_attention: metadata
tasks/fused_rmsnorm_kernel/reference/rmsnorm_kernel.py|fused_rmsnorm_kernel: reference solution
tasks/fused_rmsnorm_kernel/hidden_tests/test_rmsnorm_kernel.py|fused_rmsnorm_kernel: hidden tests
tasks/fused_rmsnorm_kernel/starter/rmsnorm_kernel.py|fused_rmsnorm_kernel: starter
tasks/fused_rmsnorm_kernel/prompt.md|fused_rmsnorm_kernel: prompt
tasks/fused_rmsnorm_kernel/meta.yaml|fused_rmsnorm_kernel: metadata
tools/mutate_rmsnorm.py|tools: mutate the RMSNorm reference and check the tests catch it
tools/verify_accelerated.sh|tools: the accelerated tier's evidence, on a machine that has the hardware
tools/check_cost.py|tools: re-derive every published cost from the tokens beside it
runner/sandbox.py|runner: one classification of what counts as evidence, and a broken solution is not exempt
runner/report.py|runner: keep no-evidence outcomes out of every rate, compute the headline over one tier, record the tokens behind the cost
runner/cli.py|runner: token usage off the adapter, and repeated independent draws
tests/test_runner.py|Test the no-evidence rules, the headline tier, a broken solution, repeated draws and the published costs
CONTRIBUTING.md|Add contributing guide
TASK_FORMAT.md|Task format: which tier the pass rate is over, and what a solution that will not import scores
leaderboard/claude-opus-5-20260803.json|leaderboard: claude-opus-5 on task set v1
leaderboard/claude-haiku-4-5-20260803.json|leaderboard: claude-haiku-4-5 on task set v1
leaderboard/README.md|leaderboard: the first two rows, what a row is not, and 308 tests rather than 332
.github/workflows/ci.yml|CI: the harness suite, every task validated, a reference run and a cost check on every push
docs/LESSONS.md|Lessons from the adapter, the first kernel, a red repository and a rate that blended two tiers
docs/V2_DESIGN.md|Design the task set that replaces v1 as the headline
docs/sessions/03-ci-and-the-broken-task-alarm.md|Journal 03: CI
docs/sessions/04-coverage-tiers-and-three-tasks-at-once.md|Journal 04: coverage, tiers, parallel authoring
docs/sessions/05-the-first-model-adapter-and-the-first-kernel.md|Journal 05: the first model adapter and the first kernel
docs/sessions/06-the-hardware-run-the-live-api-and-a-red-repository.md|Journal 06: hardware, the live API, and why CI was red
docs/sessions/07-an-audit-before-the-gpu-run.md|Journal 07: the audit before the GPU run
README.md|README: the numbers exist now, so the README stops promising them
.gitignore|Keep the author's notes out of the repository
CLAUDE.md|Update state after the hardware run, the live API and the audit
COMMITS.md|Update commit queue
QUEUE
```

## Step 4 — confirm

```bash
cd ~/scratchbench

git status --porcelain          # expect: empty
git log --oneline | wc -l       # expect 90  (45 before the reset, 45 replayed)

python -m pytest -q                                       # expect 77 passed
python -m runner.cli validate --verbose                   # expect 8 task(s) validated
python -m runner.cli run --model reference --tasks all --no-write   # expect laptop 100% (8/8)
python tools/check_cost.py                                # expect 2 file(s) checked

gh run list --limit 15          # expect twelve runs, all success, no failure
```

## The Actions tab keeps its own history

A workflow run outlives the commit it ran on. The fifteen failures stay in
`gh run list` and in the Actions tab after the replay, pointing at SHAs that are
no longer reachable from `main` — which is why no commit shows a cross any more
while the runs are still listed.

They can go too, if the tab is meant to read as cleanly as the history:

```bash
cd ~/scratchbench

# look first: every one of these should be a failure from before the replay
gh run list --limit 20 --json databaseId,headSha,conclusion,displayTitle \
  --template '{{range .}}{{slice .headSha 0 7}}  {{.conclusion}}  {{.displayTitle}}{{"\n"}}{{end}}'

# then delete them
gh run list --limit 20 --json databaseId,conclusion \
  --jq '.[] | select(.conclusion=="failure") | .databaseId' \
  | while read -r id; do gh run delete "$id"; done
```

Deleting a run removes its logs and cannot be undone. It touches no commit and
no code, and the reason each of these failed is written down in
`docs/LESSONS.md` L17 and L24, which is the part worth keeping.

## If it still goes red

Read the log before touching the history again:

```bash
gh run view --log-failed
```

The failure has to be something Step 3 could not see: a Linux or Python 3.12
difference, or a missing dependency in CI. Rewriting the history a second time
fixes none of those.

---

## Committed

### 2026-08-03 — replaced by the replay above

Blocks 0 through 6 and part of Block 7 had run, one push at a time, which is
what produced fifteen red runs. Step 1 above resets past all of them and Step 2
replays the same content in one sequence. The old queue is not kept here because
it no longer describes any commit that exists.
