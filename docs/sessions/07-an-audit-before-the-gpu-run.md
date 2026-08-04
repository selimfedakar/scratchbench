# 07 — An audit before the GPU run (2026-08-03)

## What this step was

Session 06 ended with a working benchmark, two published rows, and a list of
four things that had been claimed and not measured. The temptation was to start
on the hardest of them. Instead this session audited what already exists,
because two of the four open items involve pointing this harness at a machine
with a GPU, and the audit found that the harness produces a wrong number the
first time both tiers are measured in one run.

Nothing new was added to the task set. What changed is the arithmetic that turns
a run into a published number, and one status that was on the wrong side of the
evidence line.

## The starting state, verified rather than remembered

```
$ python -m pytest -q
61 passed in 17.03s

$ python -m runner.cli validate --tier all
task                       reference  untouched starter  time   verdict
-------------------------  ---------  -----------------  -----  -----------------------------------------
attention_causal_mask      26 passed  26 failed          0.35s  ok
bpe_merge_order            30 passed  30 failed          0.20s  ok
fused_rmsnorm_kernel       -          -                  -      UNCHECKED  no cuda device on this machine
grad_accumulation          27 passed  27 failed          1.57s  ok
kv_cache_equivalence       21 passed  21 failed          0.30s  ok
online_softmax_attention   57 passed  57 failed          0.31s  ok
quantization_error_bounds  65 passed  65 failed          0.31s  ok
sharded_dataloader         53 passed  53 failed          0.23s  ok
softmax_stability          29 passed  29 failed          0.27s  ok

8 task(s) validated: reference passes, starter fails cleanly.
1 task(s) not checked here: fused_rmsnorm_kernel
```

That matches what session 06 claimed, which is the point of running it first.

## The first finding: the headline rate was an average over tiers

The next thing on the list is to ask a model the accelerated task, on the rented
box. So the question worth asking before renting anything is what this harness
prints when both tiers are measured — a state no machine here has ever been in.

The probe builds the payload directly, with the laptop tasks passing and the
accelerated one failing:

```
=== PROBE A: does the headline pass_rate mix tiers? ===
headline pass_rate : 0.8888888888888888  measured: 9
by tier            : {'accelerated': 0.0, 'laptop': 1.0}
by category        : {'attention': 1.0, 'data': 1.0, 'kernels': 0.0, 'numerics': 1.0, 'tokenization': 1.0, 'training': 1.0}
```

`pass_rate` is the field `format_leaderboard` prints in the column called *pass
rate*. Eighty-nine percent is not a number about anything: it is a laptop tier
anyone can reproduce averaged with a GPU tier almost nobody can, and the
repository's central claim is that those are never mixed. `kernels` in the same
dictionary as `numerics` is the same fault one level down.

It survived because it cannot fire on a machine without a GPU. An accelerated
task here returns `needs_accelerator`, drops out on the no-evidence rule, and
never reaches the arithmetic. The existing test for the neighbouring case passes
for exactly that reason. `docs/LESSONS.md` L23 has the rest.

The fix is a constant and a filter:

```python
HEADLINE_TIER = "laptop"

headline = [o for o in measured if by_slug[o.slug].tier == HEADLINE_TIER]
```

plus `headline_tier` in the payload so a reader never has to infer it, per-tier
category rates so nothing is lost by narrowing the headline, and a
`headline_of()` that leaderboard rows read instead of counting every task in the
file. After:

```
headline_tier : laptop
pass_rate     : 1.0
by tier       : {'accelerated': 0.0, 'laptop': 1.0}
headline cats : {'attention': 1.0, 'data': 1.0, 'numerics': 1.0, 'tokenization': 1.0, 'training': 1.0}
accel cats    : {'kernels': 0.0}
leaderboard row: (1.0, 8, 8)
```

Results files written before the split carry no `headline_tier`. Every one of
them was laptop-only, so they are read as laptop rows and mean exactly what they
said. `RESULTS_VERSION` is 2.

## The second finding: a syntax error was scored better than a wrong answer

The second probe hands the grader a solver that writes an unparseable file:

```
=== PROBE B: what status does a model that emits invalid Python get? ===
status   : collection_error
passed   : False
evidence?: False
breaks run?: True
=> published pass_rate: 1.0 over measured: 1 of 2 tasks
```

One task solved, one task turned into garbage, and the file says 100%. The
no-evidence rule — the thing L11 and L20 exist to enforce — was taking a real
failure out of the denominator.

It is the wrong side of the line because the task contract already pins the
other half down: `validate` requires the untouched starter to import cleanly and
fail every hidden test with a real assertion error. The baseline collects by
construction, so a collection error in a graded directory came from whatever
wrote the file.

My first fix was an import check on the solution modules, and it is worth
recording because it is wrong in an instructive way: it catches `def answer(:`
and completely misses the commoner failure, a model that renames the function
the hidden tests import. That file imports perfectly and contains nothing the
task asked for.

What shipped asks the question about the baseline instead. On a collection
error, the hidden tests are re-run against the untouched starter:

```python
def starter_collects_cleanly(task: Task) -> bool:
    ...
    return returncode == PYTEST_TESTS_FAILED and not parse_summary(output).get("error")
```

If they collect and fail properly there, the task is intact and the solution
broke it: `solution_error`, which is evidence and is a failure. If the starter
cannot collect either, the task itself is broken: `collection_error`, still an
absence of evidence and still fatal to the run. The extra pytest run happens
only in the broken case. After:

```
status: solution_error  detail: SyntaxError: expected ':'
pass_rate: 0.5  measured: 2  not_measured: 0
```

Both new tests are in the suite, including the renamed-function case that the
first fix would have passed while measuring nothing.

## Repeated draws

L19 recorded the same model passing `softmax_stability` on one run and failing
it on the next, and the leaderboard says every row is one draw. That was a
caveat with no command behind it. Now:

```
$ python -m runner.cli run --model reference --tasks softmax_stability,bpe_merge_order --repeat 3
...
reference  ·  task set v1  ·  3 draw(s)  ·  laptop tier

task               passed  spread
-----------------  ------  ------
softmax_stability  3/3     always
bpe_merge_order    3/3     always

set pass rate: 100% in every draw
```

Each draw is a complete, independently valid run with its own results file, and
no draw sees another's results — these are repeated samples, not retries, and
`max_attempts` is still 1. `report --variance` does the same aggregation over
whatever is in `results/`.

Two things went wrong while writing it, both caught by tests written first. The
adapter accumulates spend across the tasks it solves, so reusing one across
draws wrote a running total into every file and made the last draw look three
times as expensive as the first: adapters are now built per sweep. And the
results filename is the model plus a one-second timestamp, while a reference
sweep finishes inside a second — three draws wrote to one filename and two of
them vanished. A variance measurement silently reduced to a single sample is a
funny way to fail, and the draw index is in the filename now.

## The published costs are a command, not a paragraph

`leaderboard/README.md` checked its own arithmetic by hand, once, in prose. That
check stops running the day it is written: a drifting price table, an edited
results file or a typo in a new model's rate all pass it forever.

```
$ python tools/check_cost.py
ok           claude-haiku-4-5-20260803.json  12103 in + 11947 out = $0.071838
ok           claude-opus-5-20260803.json  15858 in + 18453 out = $0.540615

2 file(s) checked: every cost reproduces from its own tokens.
```

It shares its arithmetic with the adapter rather than reimplementing it —
`price_from_counts` is now the one function both call, because a second copy
would make the check pass by construction. It rejects a tampered figure:

```
MISMATCH     wrong.json  file says $0.990000, the tokens say $0.540615
```

and a cost with no tokens behind it is reported `UNCHECKABLE` rather than
passed, which is the state L18 decided was not good enough. CI runs it on every
push.

This also settles what could be said about prompt caching. The adapter sends
each task once, as its own prompt, so there is no prefix to reuse and both cache
counts are zero by construction. That was an assertion; it is a test now, over
the published files. The cache multipliers remain exercised offline only, and
that sentence is now precise rather than a worry.

## What the audit did not change

Two open items from session 06 are unchanged because neither can be closed from
this machine and neither is blocked by anything in the code:

- **No model has been asked an accelerated task.** The task is verified, the
  adapter works, and the command is one line on a box with a GPU. It was worth
  fixing the tier arithmetic first, since the first `--tier all` sweep there is
  what would have written the blended number.
- **v1 does not discriminate at the top.** That is a task-set problem, not a
  harness problem. `docs/V2_DESIGN.md` is the design: what discriminates, eight
  candidates scored against it, the two to write first, and the process change
  that matters more than either — a task does not enter a v2 frozen set until it
  has been calibrated against models rather than against my sense of hard. The
  calibration machinery is deliberately not built yet, for the reason in L11:
  infrastructure written before its first user has tests that share the user's
  blind spot.

## The fifteen red crosses, and replaying instead of continuing

The last thing this session did was diagnose the redness properly rather than
work around it. Fifteen runs on `main`, all failures, and `gh run list` shows
them attached to fifteen consecutive commits starting with the one that
introduced CI. The log is one line:

```
adapters/__init__.py:23: in <module>
    from .anthropic_api import AnthropicAdapter
E   ModuleNotFoundError: No module named 'adapters.anthropic_api'
```

`adapters/__init__.py` was committed already importing a module that was never
committed, so the very first CI commit was broken on arrival and so was every
commit after it. Pushing one commit at a time turned one broken tree into
fifteen separate red runs, because GitHub starts a run per push against the tip
of what was pushed. Commits that are never a push tip get no run at all, which
is why the crosses stop where they do.

A check run belongs to a commit SHA, so the crosses cannot be removed without
replacing the commits. `COMMITS.md` now resets to the last commit CI never ran
on and replays all forty-five files, split into ten and thirty-five over two
days, one push per commit.

Per-commit pushing is what caused the problem, and it is safe here for a reason
that is worth stating precisely: `ci.yml` is commit 34 of 45, and GitHub runs
the workflow that exists in the pushed tree, so the thirty-three pushes before
it start no run at all. The tree at commit 34 was assembled and run against CI's
four commands before the queue was written:

```
=== CI ADIM 1: pytest ===        77 passed in 18.84s
=== CI ADIM 2: validate ===      8 task(s) validated: reference passes, starter fails cleanly.
=== CI ADIM 3: reference run === laptop       100% (8/8)  ·  headline
=== CI ADIM 4: check_cost ===    2 file(s) checked: every cost reproduces from its own tokens.
```

That check is what moved `ci.yml` from position 29 to 34. At 29 the leaderboard
files have not landed, `tools/check_cost.py` finds nothing to check and exits
non-zero, and the queue written to stop the repository going red would have made
it go red again.

Writing that replay produced L24, which is the entry I would least have
predicted. The loop read each line into `path`, and in zsh `$path` is the array
form of `$PATH`. The first iteration emptied it and every command afterwards was
`command not found`, `set -e` included, since the shell could not find the
binary that was supposed to fail. It surfaced because the loop was run against a
throwaway repository first:

```
--- loop, file degiskeniyle ---
commit sayisi     : 45
dosya/commit (tek?): 1
commit'lenmemis   : 0
```

Forty-five commits, exactly one file each, nothing left over. The queue is also
checked against the repository rather than trusted: forty-five files needed,
forty-five queued, no duplicates, no missing, no extras.

## State at the end

```
$ python -m pytest -q
77 passed in 18.26s
```

Sixty-one before, seventy-seven after. Task set unchanged: nine tasks, all nine
L2 verified, all nine `frozen_set: v1`.
