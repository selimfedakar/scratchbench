# Task format

A task is a directory under `tasks/`. The name is a slug: lowercase, underscores, descriptive.

```
tasks/<slug>/
  meta.yaml          machine-readable description
  prompt.md          exactly what the model is shown
  starter/           files the model may edit
  hidden_tests/      pytest files the model never sees
  reference/         a correct solution (maintainers and CI only)
```

## meta.yaml

```yaml
slug: attention_causal_mask
version: 1                     # bump when the task changes in a scoring-relevant way
category: attention            # tokenization | attention | training | data | numerics | kernels
difficulty: 2                  # 1 warm-up .. 5 genuinely hard
probes: >                      # one sentence: what capability this isolates
  Whether the model gets masking direction and the -inf convention right,
  including the padded-batch case that most implementations quietly break on.
time_limit_s: 120              # wall clock for the hidden tests alone
tier: laptop                   # laptop (default) | accelerated
requires_gpu: false            # true is allowed only on the accelerated tier
deps: [numpy]                  # beyond the standard library and pytest
added: 2026-07-26
frozen_set: v1
```

## Frozen sets, and how a task gets into one

`frozen_set` says which published set a task belongs to, and from `v2` onward
getting into one is something a task has to earn:

| Value | Means |
|---|---|
| `unvalidated` | not finished. Its reference has never passed, usually because the hardware to run it on is somewhere else. Out of the leaderboard. |
| `v1` | the first set. Published as what it is: difficulty numbers assigned by how hard each task felt to write, before any model had been asked. |
| `warmup` | finished, verified, and solved by everything at the top. Useful for separating small models from large ones, and it measures nothing at the frontier, so it stays out of the headline. |
| `v2`, `v3`, … | calibrated. Carries the block below, and is refused without it. |

```yaml
calibration:
  - model: claude-opus-5
    draws: 15
    passed: 9
    date: 2026-08-09
  - model: claude-haiku-4-5
    draws: 20
    passed: 3
    date: 2026-08-09
```

Each entry is one model asked the same question several independent times —
draws, not retries. A numbered set from `v2` onward requires at least one entry,
at least one entry with five draws or more, and **no entry in which a model
passed every draw**. That last rule is the whole point: a task the best thing
you tried never fails is a task that has stopped measuring, it costs a sweep and
returns no information, and it belongs in `warmup`. The loader refuses the task
outright rather than warning, because a benchmark that admits tasks on the
author's estimate of difficulty is measuring the author (`docs/LESSONS.md` L21).

`difficulty` stays in the file and stops being an opinion once a calibration
block exists beside it: the number is then a summary of that block, not a
prediction.

**The draws go in the repository.** A task is refused from a frozen set on the
strength of these numbers, so the results files they were computed from are
checked in under `calibration/` and `tools/check_calibration.py` re-derives every
block from them in CI. Draws that produced no evidence — an adapter that never
answered — are in neither the numerator nor the denominator, on both sides of the
comparison. A number that decides admission is not allowed to be a number nobody
can reproduce.

## Tiers

Two tiers, and the separation is what lets this repository hold both of its
promises at once.

**`laptop`** is the default and the headline. No GPU, under five minutes,
reproducible by anyone who clones the repository. This is the tier the
leaderboard's pass rate is computed over, and `requires_gpu` must be false on
it. A benchmark whose number cannot be re-derived is a number you are trusting.

**`accelerated`** exists because writing a CUDA kernel is a real skill and
pretending otherwise would leave the most interesting question in machine
learning engineering out of the benchmark. These tasks may set
`requires_gpu: true` and declare which accelerator they need:

```yaml
tier: accelerated
requires_gpu: true
accelerator: cuda              # cuda | metal
```

The five-minute limit still applies — a GPU is not a licence to need an hour.

Accelerated tasks are reported **separately** and never folded into the laptop
pass rate. On a machine without the accelerator they come back as
`needs_accelerator`, which is neither a pass nor a failure: it is an absence of
evidence, and it is labelled as one rather than quietly rounded to zero.

Until an accelerated task's reference has been run on real hardware, it stays
out of the frozen set (`frozen_set: unvalidated`) and out of the leaderboard.
The rule that governs the whole repository does not bend for hardware: if the
reference has never passed, the task is not a task yet.

## prompt.md

What the model receives, and nothing else. It must:

- state the task in prose, the way a colleague would hand it over;
- name the exact files and function signatures to fill in;
- state every convention that the tests depend on — dtype, shape order, what
  happens on empty input, whether the operation is in place;
- **never** hint at the tests, name the test file, or list edge cases as a
  checklist. A specification that enumerates its own edge cases is measuring
  reading comprehension, not engineering.

If a hidden test can fail for a reason the prompt never mentioned, the task is
broken. That is the single rule task authors get wrong most often.

## starter/

Runnable from the first second. Correct signatures, imports in place, bodies
raising `NotImplementedError`. A model that fills in nothing must produce clean
test failures — never an import error, never a syntax error. Import errors and
real failures score the same and mean completely different things.

## hidden_tests/

Plain pytest. Rules:

- **Deterministic.** Fixed seeds. No network. No wall-clock assertions beyond
  the task's own time limit.
- **Test behaviour, not implementation.** Never assert on private helpers,
  variable names, or the number of loops. A different correct solution must
  pass.
- **Numerics get tolerances**, stated and justified in a comment.
- **Cover the specification exactly** — every claim in `prompt.md`, and nothing
  the prompt did not claim.
- **Under the time limit on a laptop.** If it needs a GPU, it does not belong
  in this repository.

## reference/

A correct solution, kept out of what the model sees. CI runs the hidden tests
against it on every commit: if the reference ever fails, the task is wrong, not
the model.

## Scoring

Binary. All hidden tests pass, or the task is failed.

A solution that will not import, or that renames what the hidden tests import,
is a failed task and not an absence of evidence. The starter imports cleanly by
contract, so a graded directory that cannot be collected was broken by whatever
wrote it. The harness confirms this rather than assuming it: on a collection
error it re-runs the hidden tests against the untouched starter, and only calls
the task broken if they fail there too.

The suite reports:

- `pass_rate` — fraction of solved tasks **on the headline tier**, named in the
  file as `headline_tier`. It is never an average across tiers; the others are
  reported in full beside it under `pass_rate_by_tier`
- `pass_rate_by_category` — where a model is strong and where it is not, within
  that same tier
- `attempts` — how many edit-and-run cycles the harness used
- `repeat` — which draw this is, when the sweep was repeated. Independent runs,
  not retries: no draw sees another's results
- `wall_clock_s`, `usd_cost` and the `tokens` the cost was computed from —
  because a model that solves everything in forty minutes and eleven dollars is
  a different tool than one that solves the same set in ninety seconds, and a
  cost nobody can re-derive is a cost you are trusting

## Contributing a task

1. Write `reference/` first, then `hidden_tests/`, then `prompt.md` last. In
   that order you discover the assumptions you were about to leave out.
2. Confirm the reference passes and the untouched starter fails cleanly.
3. Run at least two models against it. A task no model solves and a task every
   model solves are both worth having — a task nobody solves *for the wrong
   reason* is not, and running it is how you tell the difference.
4. Open a PR with the results in the description.
