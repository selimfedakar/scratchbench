# 06 — The hardware run, the live API, and a red repository (2026-08-02)

## What this step was

Session 05 ended with a list of things that were written and not run. This one
exists to close that list item by item, on the machines that can actually answer
them: a rented CUDA box for the kernel task and the harness on Linux, the live
Anthropic API for the adapter, and GitHub for a repository whose last seven
commits were all carrying a red cross.

Nothing here is new surface. Every item below was already claimed somewhere in
session 05, and the work was finding out which of the claims were true.

## The kernel task, on an A4500

`fused_rmsnorm_kernel` had never executed. Three things needed hardware to
settle, and hardware answered two of them the way I hoped and one of them badly.

```
root@667b8f0ec86f:~/scratchbench# python -c "import torch, triton; print(torch.cuda.get_device_name(0), '| torch', torch.__version__, '| triton', triton.__version__)"
NVIDIA RTX A4500 | torch 2.8.0+cu128 | triton 3.4.0

root@667b8f0ec86f:~/scratchbench# python -m runner.cli validate --tasks fused_rmsnorm_kernel --verbose
task                  reference  untouched starter    time   verdict
--------------------  ---------  -------------------  -----  -------
fused_rmsnorm_kernel  25 passed  23 failed, 2 passed  6.76s  ok

1 task(s) validated: reference passes, starter fails cleanly.
```

The reference passes. That is the one I was least sure of — Triton's type
promotion, the `out_ptr.dtype.element_ty` cast on the store, and a `range` over
a runtime bound were all written from the documented semantics and never
executed. And the untouched starter fails with test failures rather than a
collection error, which was the other open question: `NotImplementedError` in a
`@triton.jit` body is raised at compile time, inside a test, where pytest counts
it as a failure.

`23 failed, 2 passed` is the part that is not fine, and the mutation run made it
worse before it made it better:

```
root@667b8f0ec86f:~/scratchbench# python ~/mutate_rmsnorm.py ~/scratchbench
=== reference ===
25 passed in 2.25s

=== untouched starter ===
23 failed, 2 passed in 2.21s

=== mutants ===
SURVIVED  float16 accumulator (sums the squares in the input dtype instead of float32)
          25 passed in 4.56s
CAUGHT    off-by-one mean (divides the sum of squares by n_cols - 1)
          12 failed, 13 passed in 4.84s
CAUGHT    epsilon outside the root (adds eps to the root-mean-square instead of to the mean square)
          2 failed, 23 passed in 4.62s
CAUGHT    one stride for both tensors (walks the input with the output's row stride)
          3 failed, 22 passed in 4.54s
CAUGHT    weight indexed by row (scales the whole row by a single weight instead of one per column)
          16 failed, 9 passed in 4.72s

1 mutant(s) survived: the task has a hole in it

root@667b8f0ec86f:~/scratchbench# python -m pytest -q
............................................s...........
55 passed, 1 skipped in 10.82s
```

Four of five mutants caught, and the survivor is the one the task is named
after. Both problems are written up in `docs/LESSONS.md` — L14 for the surviving
mutant, L15 for the two tests an untouched starter passes — and both are fixed
here rather than argued away.

**The hole.** The prompt asks for float32 accumulation because a float16
accumulator cannot survive a long row, and one test was written to enforce
exactly that: 16384 values at a standard deviation of two, so the sum of squares
lands near the 65504 that float16 stops at. That reasoning describes an
implementation that holds the whole row in one accumulator. The reference does
not: it walks the row in blocks and keeps a `BLOCK_SIZE`-wide vector of partial
sums, so with a block of 1024 no lane ever accumulates more than sixteen
squares. The only place the large number exists is inside `tl.sum`, which
reduces in float32 whatever it is handed. The test was arithmetic about code I
was not running, and it is the reason mutation testing is in `CONTRIBUTING.md`
rather than in a footnote.

The replacement does not depend on the block size at all. float16 holds values
up to 65504, so anything past 256 squares to something it cannot represent:
rows at a scale of 300 are ordinary float16 numbers whose squares are around
9e4. Accumulated in float32 that is unremarkable. Squared in the input dtype it
is an infinity on the first element, the mean square is infinite, its reciprocal
root is zero, and the row comes back as zeros. The old long-row test stays,
renamed and re-commented to describe what it actually covers — sixteen passes
through both loops — because a test whose comment is wrong is a test the next
reader will trust for the wrong reason. The licensing sentence in `prompt.md`
changed with it.

**The two passing tests.** These needed two different fixes, which is the
interesting part. The `@triton.jit` type assertion is redundant: the hidden
tests launch the kernel with a grid subscript, and only a `JITFunction` supports
that, so a plain Python function already fails all nineteen direct-launch tests.
It is deleted, with a comment where it stood. The `torch.cuda.is_available()`
assertion is not redundant and not a test — it is a statement about the machine
rather than about the solution — so it moved to module scope. Missing hardware
is now a loud collection error instead of a green tick, which is what L12 asked
for, and it is no longer counted among the tests an untouched starter passes.
Twenty-five tests became twenty-four.

A sixth mutant went in at the same time: a mean taken over `BLOCK_SIZE` instead
of over `n_cols`, which is only correct by coincidence and which pins the masked
tail from a second direction.

**The mutation script now lives in the repository**, at
`tools/mutate_rmsnorm.py`. It was outside it before, and the first time it was
needed it was on a rented box that did not have it — twice, in the transcript
above. A check nobody can re-run is a claim, not a check. `tools/verify_accelerated.sh`
joins it: the five commands whose output an accelerated task needs before it can
enter the frozen set, in order, starting with the hardware probe that makes the
other four mean anything.

**The re-run, on the same box, through the script:**

```
root@180600ebaa2f:~/scratchbench# bash tools/verify_accelerated.sh
=== 1. hardware ===
NVIDIA RTX A4500 | torch 2.8.0+cu128 | triton 3.4.0

=== 2. harness suite ===
............................................s..............              [100%]
58 passed, 1 skipped in 10.84s

=== 3. validate (reference passes, untouched starter fails) ===
task                  reference  untouched starter  time   verdict
--------------------  ---------  -----------------  -----  -------
fused_rmsnorm_kernel  24 passed  24 failed          6.71s  ok

1 task(s) validated: reference passes, starter fails cleanly.

=== 4. control run through the full harness ===
  fused_rmsnorm_kernel ... pass  (24 passed)

reference  ·  2026-08-03T05:43:09+00:00  ·  task set unvalidated

task                  category  tier         diff  result  time   tests
--------------------  --------  -----------  ----  ------  -----  ---------
fused_rmsnorm_kernel  kernels   accelerated  4     pass    3.03s  24 passed

accelerated  100% (1/1)
4.6s wall clock  ·  1 attempt(s) per task
kernels 100%

=== 5. mutants ===
=== reference ===
24 passed in 2.22s

=== untouched starter ===
24 failed in 2.12s

=== mutants ===
CAUGHT    float16 accumulator (sums the squares in the input dtype instead of float32)
          2 failed, 22 passed in 4.56s
CAUGHT    off-by-one mean (divides the sum of squares by n_cols - 1)
          12 failed, 12 passed in 4.66s
CAUGHT    mean over the block rather than the row (divides by BLOCK_SIZE, which is only the row length by coincidence)
          11 failed, 13 passed in 4.61s
CAUGHT    epsilon outside the root (adds eps to the root-mean-square instead of to the mean square)
          2 failed, 22 passed in 4.65s
CAUGHT    one stride for both tensors (walks the input with the output's row stride)
          3 failed, 21 passed in 4.72s
CAUGHT    weight indexed by row (scales the whole row by a single weight instead of one per column)
          17 failed, 7 passed in 4.61s

all 6 mutants caught
```

Twenty-four passed against the reference, twenty-four failed against the
untouched starter, and six mutants out of six caught — including the one this
task exists to catch, which now fails two tests rather than none. `task set
unvalidated` in that transcript is the state the run was made in; `meta.yaml`
moved to `frozen_set: v1` afterwards, on the strength of it. The harness suite
also runs on Linux against a CUDA build of torch: 58 passed, 1 skipped, and the
skip is the Anthropic SDK test on a box where the extra is not installed.

`fused_rmsnorm_kernel` is the first task in the accelerated tier of the frozen
set. It is still reported beside the laptop rate and never folded into it.

## The adapter, against the live API

Session 05 shipped the Anthropic adapter with eight offline tests and no network
request. The first live request failed, and so did the second, and both failures
were mine.

```
$ python -m runner.cli run --model claude-haiku-4-5 --tasks softmax_stability
  softmax_stability ... ADAPTER  (BadRequestError: Error code: 400 - {'type': 'error', 'err…)

  softmax_stability: BadRequestError: Error code: 400 - {'type': 'error', 'error':
  {'type': 'invalid_request_error', 'message': 'adaptive thinking is not supported
  on this model'}, 'request_id': 'req_011CdfENeQhzEM892CXz7FM3'}
```

```
$ python -m runner.cli run --model claude-haiku-4-5 --tasks softmax_stability
  softmax_stability ... ADAPTER  (AnthropicResponseError: asked claude-haiku-4-5 and was an…)

  softmax_stability: AnthropicResponseError: asked claude-haiku-4-5 and was
  answered by claude-haiku-4-5-20251001
```

**The first failure is a design question wearing a bug's clothes.** The adapter
was asking for adaptive thinking, and Haiku 4.5 does not have it. The obvious
repair is a per-model table of which knobs each model accepts. I did not write
one, and the reason is the thing worth recording: a table like that goes stale
silently, and worse, it makes two columns of the same leaderboard mean different
things. A score measured at `effort: high` and a score measured at whatever a
cheaper model happens to support are not comparable, and nothing in the results
file would say so. So the adapter now sets no sampling, thinking or effort knobs
at all. Every model is asked at its own defaults, one code path, and what this
benchmark reports is the model as the API ships it. The only thing left in
`output_config` is the JSON schema, which constrains the shape of the answer
rather than the model's behaviour.

**The second failure is L16.** An alias resolves to a dated snapshot, and the
identity check — the thing standing between this repository and publishing one
model's score under another's name — read that as a substitution. The fix is one
rule applied in two places: an answer is the model that was asked for if it is
that string, or that string plus a trailing date. Matched as a date and not as a
prefix, because `claude-opus-4-5` starts with `claude-opus-4` and is a different
model. The price table needed the same rule for a quieter reason: keyed by
alias, it misses on a snapshot and reports `usd_cost: null`, which does not read
as "the lookup missed" — it reads as "this model has no published price".

Three offline tests cover it, including the prefix case a looser check would
have accepted:

```
$ python -m pytest -q
...........................................................              [100%]
59 passed in 15.96s
```

That is 56 from session 05 plus these three. Then the third live request, which
is the first one this repository has ever graded:

```
$ python -m runner.cli run --model claude-haiku-4-5 --tasks softmax_stability --tier laptop
  softmax_stability ... pass  (29 passed)

claude-haiku-4-5  ·  2026-08-03T05:43:14+00:00  ·  task set v1

task               category  tier    diff  result  time   tests
-----------------  --------  ------  ----  ------  -----  ---------
softmax_stability  numerics  laptop  1     pass    0.34s  29 passed

laptop       100% (1/1)
10.3s wall clock  ·  1 attempt(s) per task  ·  $0.01
numerics 100%
```

A model was asked to implement a numerically stable softmax from the
specification alone, wrote the file, and twenty-nine hidden tests it never saw
agreed. Ten seconds and eight tenths of a cent.

## The cost was a number nobody could check

Reading that run's results file field by field is what this session was for, and
it is where the last hole turned up. `attempts: 1`, `model: claude-haiku-4-5`,
`usd_cost: 0.008337` — all populated, all plausible. And that is the problem:
plausible is as far as anyone can get with it. `usd_cost` was the only trace of
the usage the run produced, so a price table that had drifted and a run that used
more tokens than expected produce the same figure, with nothing to separate them
afterwards. I asked whether the arithmetic held and found I had not left myself
the means to answer.

`stop_reason` never reached the file either, though that one is a smaller gap:
the interesting values already raise, and `adapter_error` carries the message.

The fix is the four counts the price is computed from:

```json
"usd_cost": 0.008337,
"tokens": {"input": ..., "output": ..., "cache_read": ..., "cache_write": ...}
```

With those and `PRICES`, anyone reading a leaderboard row can reproduce the cost
to the cent, or catch me if it does not. The reference solver has no calls and
therefore reports `tokens: null` rather than four zeros pretending to be a
measurement. Two tests cover both directions.

## The cost, checked

The second live request was supposed to be bookkeeping: run the same task again
with the `tokens` field in place and confirm the arithmetic. It held exactly.

```json
"usd_cost": 0.006197,
"tokens": {"input": 1182, "output": 1003, "cache_read": 0, "cache_write": 0}
```

Haiku 4.5 is $1 per million input and $5 per million output, so
`(1182 × 1 + 1003 × 5) / 1e6 = 0.006197`. To the last digit, from the file
alone, with no appeal to memory or to my own price table being right.

## And the same task failed

The run that produced those numbers also failed the task. Twenty-six passed,
three failed. The first run of the same model against the same task, four hours
earlier, had passed all twenty-nine.

Same model, same prompt, same adapter, one attempt each. Everything in this
repository is pinned to be deterministic and none of that pinning reaches the
model, which is sampled. Two sessions spent making the harness reproducible had
quietly become an assumption that a *result* was reproducible.

This is not an argument for retries. A second attempt *with the test results as
feedback* measures the scaffolding, which is why `max_attempts` is 1 and why
that number is in every results file. Two independent runs are a different
thing: the same experiment, sampled twice. The repository now distinguishes
them, because it did not have the words before (`docs/LESSONS.md` L19).

The consequence for the leaderboard is concrete: a sweep is one draw. The first
published row will say so, and until there are several, two models one task
apart are not distinguishable.

## The sweep that never reached the model, and what it exposed

The first attempt at a full eight-task sweep ran with no API key in the
environment. Every task came back `adapter_error`, and the results file said:

```json
"pass_rate": 0.0,
"measured": 8,
"not_measured": 0,
"pass_rate_by_category": {"attention": 0.0, "data": 0.0, "numerics": 0.0, ...}
```

That file says Claude Opus 5 scored zero on eight tasks. It never saw one of
them. A missing environment variable produced the exact failure this repository
exists to argue against.

It is L11 again, in a second place, and the reason is worth more than the fix:
the idea was written down twice. `report.py` already had
`HARNESS_FAILURES = {collection_error, adapter_error}` with the comment "the
number it produced is not a measurement of anything", and used it only for the
exit code; the rate arithmetic four functions above tested
`status != "needs_accelerator"` instead. Two correct sentences about the same
idea, neither aware of the other. L11 corrected one of them, which is precisely
why it did not correct this. `missing_deps` was leaking the same way and turned
up while writing the table.

So the fix is a mechanism rather than a value. `runner/sandbox.py` now holds one
classification, beside the code that produces the statuses:

```python
STATUSES: dict[str, StatusMeaning] = {
    "passed":            StatusMeaning("pass",    evidence=True,  harness_failure=False),
    "failed":            StatusMeaning("FAIL",    evidence=True,  harness_failure=False),
    "timeout":           StatusMeaning("TIMEOUT", evidence=True,  harness_failure=False),
    "collection_error":  StatusMeaning("BROKEN",  evidence=False, harness_failure=True),
    "adapter_error":     StatusMeaning("ADAPTER", evidence=False, harness_failure=True),
    "missing_deps":      StatusMeaning("SKIPPED", evidence=False, harness_failure=False),
    "needs_accelerator": StatusMeaning("NO GPU",  evidence=False, harness_failure=False),
}
```

`NO_EVIDENCE`, `HARNESS_FAILURES` and the printed labels are all derived from
it, and `Outcome.__post_init__` refuses to construct an outcome whose status is
not in it. A new status cannot reach the reporting layer unclassified. `timeout`
is the one non-obvious row and it is deliberate: a solution that ran and did not
finish inside the task's own limit is the solution failing, so it stays a
measurement.

The printed line was lying too, in a way that would have sent me looking in the
wrong place. The same sweep printed `laptop not run (8 task(s): no accelerator
on this machine)` — one message written when only one kind of absence existed.
Now:

```
$ python -m runner.cli run --model claude-opus-5 --tasks all --tier laptop
laptop       not run    (8 task(s): no result was produced; see the per-task detail)
0.7s wall clock  ·  1 attempt(s) per task
exit=1
```

and the accelerated tier still says what it means:

```
$ python -m runner.cli run --model reference --tasks all --tier all
accelerated  not run    (1 task(s): no accelerator on this machine)
laptop       100% (8/8)
```

Two new tests cover it: a run where every task is `adapter_error` reports
`pass_rate: null` with `measured: 0`, and an unclassified status raises rather
than being scored. Suite: 61 passed.

## The first sweep, and the first thing it says about the benchmark

With the arithmetic fixed and the rates honest, the set finally ran end to end
against two models.

```
claude-opus-5  ·  2026-08-03T10:29:32+00:00  ·  task set v1
laptop       100% (8/8)
213.1s wall clock  ·  1 attempt(s) per task  ·  $0.54

claude-haiku-4-5  ·  2026-08-03T10:32:01+00:00  ·  task set v1
laptop       38% (3/8)
116.6s wall clock  ·  1 attempt(s) per task  ·  $0.07
attention 33%  data 100%  numerics 50%  tokenization 0%  training 0%
```

Both costs reproduce from the tokens in their own results files:
`(15858 × 5 + 18453 × 25) / 1e6 = 0.540615` and
`(12103 × 1 + 11947 × 5) / 1e6 = 0.071838`. That is the first time a published
number in this repository could be checked without asking me.

Eight tasks, 332 hidden tests, and Claude Opus 5 passed every one of them on the
first attempt for fifty-four cents. That is a real result and it is also the
benchmark's own limit, stated plainly on the leaderboard page: **v1 does not
discriminate at the top.** It separates a small model from a large one cleanly
and it cannot tell two large models apart.

The failures are more informative than the rate. Haiku lost
`kv_cache_equivalence` on nineteen tests out of twenty-one and
`grad_accumulation` on seventeen out of twenty-seven — mechanisms that either
exist or do not. It lost `softmax_stability` and `bpe_merge_order` on exactly one
test each, which is a convention guessed differently, not a mechanism missing.
Those are two different kinds of difficulty and only one of them is worth
scaling up. `leaderboard/README.md` says so, because the next task set has to be
designed against that distinction rather than against my own sense of what feels
hard.

`leaderboard/` now holds both results files and a page that states what a row is
and what it is not: one draw, one attempt, no knobs, contamination assumed.

## Why the repository was red

Seven commits on `main`, seven red crosses, and the first suggestion in the room
was to delete the last ten commits and rebuild them. That would have been a
rewrite of a history that is fine.

```
$ gh run view 30782433453 --log-failed | grep -i error
ImportError while importing test module '/home/runner/work/scratchbench/scratchbench/tests/test_runner.py'.
E   ModuleNotFoundError: No module named 'adapters.anthropic_api'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
##[error]Process completed with exit code 2.
```

`tests/test_runner.py` was committed carrying an import of
`adapters/anthropic_api.py`, and that file is still in the commit queue. `pytest`
exits 2 at collection, before `validate` is ever reached. Every red cross is
correct: those trees really are broken.

The reason there are seven of them rather than one is the push cadence. GitHub
runs the workflow once per push, against the tip of what was pushed, and the
commits went up one at a time. One file per commit guarantees that most tips are
incomplete — a task with a reference and no hidden tests yet, a test module
whose import lands two commits later — so a push per commit is a red cross per
commit, by construction. `COMMITS.md` already said "push once at the end". I had
written that as a tidiness rule. It is not: it is the only thing that makes
"one file per commit" and "CI runs on every commit" compatible, and it now says
so at the top of the file, above the blocks.

No history is rewritten. `COMMITS.md` was re-derived from `git status` rather
than from what the old queue claimed was pending — it had drifted, which is its
own small lesson about a checklist nobody diffs against reality — and Blocks 7
through 12 are the remainder, ending in a single `git push`. The queue and
`git status` are now diffed against each other rather than trusted: nothing on
disk is missing from a block, and no block names a file that is already
committed.

## The accelerated CI job

Still **L0**, still never executed, and now with a reason to think it should not
be. The job targets a `[self-hosted, gpu]` runner, and this is a public
repository. GitHub's own guidance is not to attach self-hosted runners to public
repositories: a pull request from a fork can propose a workflow change, and a
returning contributor's pull request runs without approval, so the runner
executes code from the fork on the machine it is registered to. The job itself
is `workflow_dispatch`-only and could not be triggered that way — but the
*runner* is registered to the repository, not to the job, and `runs-on` is a
line in a file that a pull request can edit.

The rented box makes it worse rather than better: it is billed by the hour and
disappears, so the registration would have to be redone every session, and the
green tick would mean "a GPU existed at some point" rather than "this task was
verified on hardware I can name".

`tools/verify_accelerated.sh` is the alternative and it is already the thing
that produces the evidence: run it on the box, paste the output into the pull
request, and the claim carries the device name, the torch version and the triton
version with it. That is more than a green tick says.

**The job is deleted.** `ci.yml` keeps a comment where it stood, explaining the
hazard rather than leaving the next reader to wonder why a repository with an
accelerated tier has no accelerated job, and `CONTRIBUTING.md` now names the
script as the way that tier's evidence is collected. The undo is one revert; the
reason it should not be undone is that the runner is registered to the
repository and `runs-on` is a line in a file a pull request can edit.

## Verified this session

| claim | level | evidence |
|---|---|---|
| The kernel reference passes on a real GPU | L2 | `verify_accelerated.sh` step 3 → `24 passed`, pasted above |
| The untouched starter fails every test, with test failures | L2 | same line → `24 failed`, and step 5's starter run |
| All six mutants are caught | L2 | step 5, pasted above |
| The float32-accumulation claim now has a test behind it | L2 | `CAUGHT — float16 accumulator (2 failed, 22 passed)` |
| The kernel task through the full harness | L2 | step 4 → `accelerated 100% (1/1)`, 3.03s |
| The harness suite runs on Linux with a CUDA torch | L2 | step 2 → `58 passed, 1 skipped in 10.84s` |
| The first version of the task had two real holes | L2 | the first run: `23 failed, 2 passed` and `1 mutant(s) survived` |
| Harness suite here | L1 | `python -m pytest -q` → 59 passed in 15.79s |
| The eight laptop tasks still validate | L2 | `validate --tier all` → 8 ok, 1 not checked here |
| The control run is unaffected | L2 | `run --model reference --tasks all --tier all` → laptop 100% (8/8), accelerated not run, 4.7s |
| The live API rejects adaptive thinking on Haiku 4.5 | L2 | the 400, pasted above |
| An alias is answered by its dated snapshot | L2 | the `AnthropicResponseError`, pasted above |
| A model solved a task through the adapter, end to end | L2 | `claude-haiku-4-5 · laptop 100% (1/1) · 29 passed · $0.01` |
| `attempts`, `model` and `usd_cost` reach the results file | L2 | the results JSON, read field by field |
| `usd_cost` matches the usage it was computed from | L2 | `(1182 × 1 + 1003 × 5) / 1e6 = 0.006197`, and the file says `0.006197` |
| The same model, same task, gives different results | L2 | run one `29 passed`, run two `3 failed, 26 passed` |
| A sweep that reaches no model reports no rate | L2 | `pass_rate: null`, `measured: 0`, `not_measured: 8`, exit 1 |
| The reference control is unaffected by all of it | L2 | `laptop 100% (8/8)`, `accelerated not run`, 5.6s |
| Harness suite, after the taxonomy fix | L1 | 61 passed |
| The accelerated CI job | **removed** | see below; `tools/verify_accelerated.sh` replaces it |

## Next

One more live request, to read a results file that carries `tokens` and check
the cost against it — the run above predates the field. Then the first full
sweep across all eight laptop tasks, which is the first number this repository
will publish about somebody else's work, and therefore the first one worth being
slow about.
