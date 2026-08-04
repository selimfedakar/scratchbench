# 03 — CI, and the alarm for a broken task (2026-07-26)

## What this step was

A GitHub Actions workflow that runs on every push and every pull request:

1. `python -m pytest -q` — the harness's own tests.
2. `scratchbench validate --verbose` — for every task, the reference passes the
   hidden tests and the untouched starter fails them cleanly.
3. `scratchbench run --model reference --tasks all --no-write` — the same
   solutions through the full harness, so the runner is exercised and not only
   the tasks.

Python 3.10 and 3.12, `fail-fast: false` so one version failing still reports
the other. Torch comes from the CPU wheel index: the premise of this repository
is that it runs without a GPU, and the CUDA build is a gigabyte of download
that would never be used.

## Why the second step is the one that matters

> If the reference ever fails, the task is wrong, not the model.

That sentence is already in `TASK_FORMAT.md`, and CI is what turns it from a
principle into an alarm. A benchmark decays quietly: a numpy release changes a
default, a tolerance that was always marginal drifts over the line, an edit to
a prompt stops matching the tests it licenses. Nothing announces any of it. The
first symptom is a model scoring badly on a task that is simply broken, and by
then the number is published.

So the reference solutions are not documentation. They are the canary, and they
get re-run on every commit.

`validate` also checks the half that is easier to forget: the untouched starter
must fail with real test failures. An import error and a wrong answer both
score zero and mean completely different things — one is the model getting it
wrong, the other is me shipping a task that was never runnable. `--verbose`
prints the pytest output for any task that fails either half, so a red CI run
says what broke rather than only that something did.

## Technologies

- **GitHub Actions**, `ubuntu-latest`, `actions/setup-python@v5` with pip
  caching.
- **A version matrix**, 3.10 and 3.12. 3.10 is what I develop on; 3.12 is what
  a contributor is most likely to have. Numeric tasks asserting agreement at
  1e-12 are exactly the kind of thing that a Python or numpy version can move.
- **The CPU torch index**, `download.pytorch.org/whl/cpu`.

## Also in this step

- `CONTRIBUTING.md`, which the README has been linking to since before it
  existed. It carries the authoring order, the both-halves check, and the two
  rules task authors break most often: a test that cannot fail is noise, and a
  task that cannot catch a plausible wrong answer is worse than no task.
- One correction to the README. It advertised
  `scratchbench run --model claude-opus-5`, which does not work, because the
  model adapters are deliberately skeletons. A public repository promising a
  command that raises is worse than one that says what it can do today, so the
  example is now the reference run, with a sentence explaining that the
  ordering is on purpose.

## Verified

Locally, all three CI steps:

```
python -m pytest -q               ->  35 passed in 6.86s
scratchbench validate             ->  5 task(s) validated
scratchbench run --model reference->  pass rate 100% (5/5), 6.4s wall clock
```

**Not verified:** the workflow has not run on GitHub yet — there is no remote
and no repository. The steps are the same commands that pass here, but the
runner image, the torch wheel index and the pip cache are untested until the
first push. Expect the first CI run to need a fix; that is normal and it is not
a reason to delay the push.

## Next

The model adapters, which is what turns this from a harness into a benchmark.
And more tasks: the current five leave `data` and `kernels` empty, and the
hardest task in the set is a 3.
