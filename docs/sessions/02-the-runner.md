# 02 — The runner (2026-07-26)

## What this step was

Until this step, five tasks existed and I had verified them by hand with `cp`
and `cd`. That is not a benchmark, it is a pile of directories. This step is
the harness: task loading, an isolated workdir, a deterministic environment, a
hard timeout, and a binary score.

```
runner/tasks.py     discovery and meta.yaml validation
runner/sandbox.py   workdir assembly, environment, pytest, the verdict
runner/report.py    the results JSON and the printed tables
runner/cli.py       list · validate · run · report
adapters/           reference (real) and model adapters (skeletons)
```

## The order inside `grade`, which is the only part with a security property

```
1. make a temp directory
2. copy starter/ into it
3. call the solver              <- it sees the starter and nothing else
4. copy hidden_tests/ in
5. run pytest, with a timeout
```

The hidden tests are not on disk while the solver is running. That is a cheaper
guarantee than any amount of instruction in a prompt, and there is a test in
`tests/test_runner.py` that passes a solver which lists the directory and
asserts it saw exactly one file.

## Choices worth writing down

**A flat workdir.** The solution files and the test files sit in the same
directory, so a hidden test says `from kv_cache import CachedAttention` and
nothing has to be done to `sys.path`. The cost is that every task has exactly
one module file, named identically in `starter/` and `reference/` — which the
loader enforces rather than trusts.

**A subprocess, not an import.** Graded code runs under `sys.executable` in its
own process, with `capture_output` and a `timeout`. An in-process import would
be faster and would let a solution hang the harness, poison the module cache
for the next task, or crash the run with a segfault. None of that is worth the
tenth of a second.

**A pinned environment.** `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `PYTHONPATH` removed. The thread
counts are the interesting ones: numpy and torch change their floating-point
reduction order with the number of threads, so a task asserting agreement at
1e-12 could pass on my laptop and fail in CI for reasons that have nothing to
do with the solution.

**Statuses, not a boolean.** A task comes back as `passed`, `failed`,
`timeout`, `collection_error`, `missing_deps`, or `adapter_error`. Only the
first counts as solved, but the distinctions decide what to do next: `failed`
is a result about the model, `collection_error` and `adapter_error` are results
about me. A run containing either exits non-zero, because the number it
produced is not a measurement of anything.

**`meta.yaml` is validated, not read.** `requires_gpu: true` is refused.
A `time_limit_s` over 300 is refused. An unknown category, a slug that does not
match its directory, a `reference/` whose file names differ from `starter/` —
all refused, with the path in the message. The founding constraint of this
repository is a laptop and five minutes, and a constraint that lives only in
prose is a constraint that erodes.

## One thing I got wrong

The first version of `grade` called the solver without a `try`. Pointing the
CLI at `--model claude-opus-5`, whose adapter is deliberately a skeleton, dumped
a Python traceback and killed the whole sweep.

The tempting fix was to make the skeleton adapters return quietly. That fixes
the symptom and leaves the mechanism: any solver can raise, and the real cases
are a rate limit or a dropped connection twenty tasks into a paid run. So the
fix went into `grade`, which is the thing that owns per-task isolation: a
solver that raises fails that task with status `adapter_error`, carrying the
exception message, and the sweep continues.

## Technologies, and what each is doing

- **subprocess + pytest** — the grader. Exit code 0 is a pass, 1 with no errors
  is a genuine failure, anything else means the tests never really ran.
- **PyYAML** — `meta.yaml`. A machine-readable task header is what lets `list`,
  `validate` and the results file agree about difficulty, category and limits.
- **argparse** — four subcommands, no dependency.
- **dataclasses** — `Task` and `Outcome`. Frozen where it should be frozen.
- **tempfile + shutil** — the isolated workdir, removed afterwards unless
  `--keep` is passed, which copies it out for inspection.
- No framework, no plugin system, no config file. The whole harness is five
  modules and it is meant to stay small enough that anyone re-running my
  leaderboard can read it before trusting it.

## Verified

The harness has its own test suite, because the harness is the thing that turns
a model into a number:

```
python -m pytest -q
35 passed in 6.86s
```

It covers the loader's refusals, the solver-never-sees-the-tests guarantee, the
timeout path (a solution that sleeps for a minute against a two second limit),
the syntax-error path, the missing-dependency path, the adapter-error path, the
results round trip, and three CLI commands.

Then the real thing, end to end:

```
scratchbench validate

task                   reference  untouched starter  time   verdict
---------------------  ---------  -----------------  -----  -------
attention_causal_mask  26 passed  26 failed          1.66s  ok
bpe_merge_order        30 passed  30 failed          0.26s  ok
grad_accumulation      27 passed  27 failed          4.84s  ok
kv_cache_equivalence   21 passed  21 failed          0.99s  ok
softmax_stability      29 passed  29 failed          1.01s  ok

5 task(s) validated: reference passes, starter fails cleanly.
```

```
scratchbench run --model reference --tasks all

task                   category      diff  result  time   tests
---------------------  ------------  ----  ------  -----  ---------
attention_causal_mask  attention     2     pass    0.52s  26 passed
bpe_merge_order        tokenization  2     pass    0.30s  30 passed
grad_accumulation      training      3     pass    4.86s  27 passed
kv_cache_equivalence   attention     3     pass    0.28s  21 passed
softmax_stability      numerics      1     pass    0.29s  29 passed

pass rate 100% (5/5)  ·  6.4s wall clock
attention 100%  numerics 100%  tokenization 100%  training 100%
```

Six and a half seconds for the whole task set. The promise on the tin is five
minutes per task; the current set is two orders of magnitude inside it, which
is the room I want for the harder tasks that come next.

## Not done, on purpose

The model adapters are skeletons that raise with a message pointing at their
contract. Writing them before the harness had been proven against a known
correct solution would mean the first model run could not distinguish "the
model failed" from "I failed", and that is exactly the mistake that makes a
benchmark worthless.

## Next

CI, so the reference solutions are re-checked on every commit rather than
whenever I remember to.
