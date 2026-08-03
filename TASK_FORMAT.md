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

The suite reports:

- `pass_rate` — fraction of tasks solved
- `pass_rate_by_category` — where a model is strong and where it is not
- `attempts` — how many edit-and-run cycles the harness used
- `wall_clock_s` and `usd_cost` — because a model that solves everything in
  forty minutes and eleven dollars is a different tool than one that solves the
  same set in ninety seconds

## Contributing a task

1. Write `reference/` first, then `hidden_tests/`, then `prompt.md` last. In
   that order you discover the assumptions you were about to leave out.
2. Confirm the reference passes and the untouched starter fails cleanly.
3. Run at least two models against it. A task no model solves and a task every
   model solves are both worth having — a task nobody solves *for the wrong
   reason* is not, and running it is how you tell the difference.
4. Open a PR with the results in the description.
