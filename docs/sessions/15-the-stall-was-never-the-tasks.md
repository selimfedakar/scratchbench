# 15 — The stall was never the task's

Session 14 ended with a finished task that could not be published. About half of
its runs were a hundred times slower, seven suspects had been killed by
measurement, and the entry recording that (`LESSONS.md` L41) closed with an
honest "no cause". The roadmap's instruction for this session was explicit:
stop theorising, bisect the suite until the smallest reproducer appears.

The bisection worked. It just did not cut where the roadmap expected, and what
it found is that both of last session's controls were coin flips.

## The cut that mattered was not a half of the suite

Every experiment in L41 varied something inside the task — its shapes, its
sentinel, its expected values, its kernel, the size of the tensors it compares —
and every one of them held the same thing fixed: *the untouched starter fails
all 81 tests*. So "the kernel writes nothing" and "81 assertions raise" had
never been separated. That is a factor, and it costs nothing to cross it.

Four workdirs, assembled exactly the way `runner/sandbox.py` assembles a graded
one, with the same pinned environment:

| Cell | Kernel | Tolerance | Assertions | Outcome |
|---|---|---|---|---|
| A | starter | as written | as written | 81 fail |
| B | reference | as written | as written | 81 pass |
| C | reference | zero | as written | 73 fail, 8 pass |
| D | starter | as written | removed | 81 pass |

Three interleaved rounds:

```
round  cell     wall     user     sys  summary
    1  A        2.62     1.28    0.49  81 failed in 2.08s
    1  B      100.11     1.02    0.35  KILLED
    1  C        2.05     1.24    0.32  73 failed, 8 passed in 1.45s
    1  D      100.11     3.24    0.57  81 passed in 97.58s
    2  A        2.10     1.21    0.37  81 failed in 1.57s
    2  B      120.10     1.03    0.37  KILLED
    2  C        2.17     1.24    0.38  73 failed, 8 passed in 1.57s
    2  D      120.08     2.91    0.48  KILLED
```

Two things there are worth more than the split. First, `user` is one second
against a hundred of `wall`, so a stalled process is not slow, it is blocked —
L40's card, and it points at what to do next rather than at what to think next.
Second, cell B is the *reference*, and L41's control says the reference is fast
every time. That contradiction was on the screen from the first round and I read
past it, because the split had a story: the cells that failed were fast and the
cells that passed were slow.

## The stack said blocked on the GPU; the GPU said it was free

`sample` on a stalled process, 2599 samples, every one of them under
`-[_MTLCommandBuffer waitUntilCompleted]` into `__psynch_cvwait`: 1337 beneath
the test's own `torch.mps.synchronize()`, 957 beneath `.to("mps")`, 305 beneath
`.cpu()`. Uniform, and spread across every kind of round trip.

So the process was waiting on the device. The obvious follow-up is whether the
device was busy. While one suite sat stalled — it ran 75 s and was killed — a
second, unrelated process was launched into that window and did 50 MPS softmaxes
in **0.687 s**. The same script with nothing else running took 2.398 s. The GPU
was not merely available during the stall, it was quicker than usual.

That is the measurement that should have ended the story about assertions and
buffers, and it did not, because I already had a mechanism.

## The mechanism was real, confirmed in both directions, and an artefact

Retention explained the split. A failing test dies at its assertion, pytest
holds the exception, the exception holds the frame, the frame holds the MPS
tensors — so a failing run keeps buffers out of the allocator and a passing run
returns them. Both directions were checked:

- the passing suite, with every MPS tensor deliberately kept alive: **2.46 s,
  1.72 s**, 81 passed
- the failing suite, with `gc.collect()` and `torch.mps.empty_cache()` after
  every test: **killed at 90 s**, twice

Clean, falsifiable, both directions. Then the minimal reproducer — one launch in
a loop, no pytest, no assertions, no expected values — reversed the sign: with
tensors held it was slow, with tensors released it was fast. The opposite result
from the same claim.

Both were the same artefact. The check is embarrassing and takes ninety
seconds. Run the *identical* command six times:

```
release #1  0.00s   release #2  43s   release #3  0.00s
release #4  46s     release #5  0.00s release #6  37s
```

The flag never mattered. In every pairing I had built, the slow cell was
whichever one ran second.

## What is actually true

Every other process on this machine that touches `torch.mps` is stalled for the
whole of its life. Fourteen consecutive launches of one script, each printing a
fixed start-up probe and then a workload:

```
launch  probe_ms  workload_s
     1      0.56       0.086
     2    776.01      10.201
     3      0.50       0.071
     4    420.51       8.500
     5      0.72       0.089
     6    462.61       9.524
     7      0.59       0.070
     8    300.48       8.055
```

Healthy round trips are 0.47–0.72 ms; stalled ones are 232–776 ms. The two
populations do not overlap and the workload behind them does not either:
0.070–0.089 s against 8.055–10.899 s. `tools/check_mps_stall.py` re-derives the
table, and `probe()` and `is_stalled()` are exported from it so a graded run can
ask the question before it grades anything.

Killed by measurement, one per line:

- **the task** — `metal_cross_entropy_kernel`, published since session 13,
  stalls identically: 1.89 s, 48.76 s, 1.50 s on three consecutive reference runs
- **the custom shader** — a loop of plain `torch.softmax` with no
  `compile_shader` anywhere stalls too
- **pass versus fail, and tensor retention** — both reproduced by running one
  unchanged command twice
- **a busy GPU** — the bystander above
- **a driver reset** — `AGXAccelerator` `recoveryCount` is 0 throughout, and its
  utilisation counters read 100 on a completely idle machine, so they are not
  evidence of anything
- **the previous process still tearing down** — a ten second gap between
  launches changes nothing
- **the last Metal client exiting** — a keeper process holding an MPS context
  open for the whole run changes nothing except which launches land on which
  side of the alternation

## What this costs the repository

`metal_softmax_backward_kernel` is not broken and never was. Neither is
`metal_cross_entropy_kernel` — but its calibration draws were taken on this
machine before any of this was known, so a `timeout` in them may be a stalled
process rather than a model that could not answer. The rate survives that
substitution and the failure shape does not, which is the exact argument that
kept the backward task unpublished, now pointing at something already public.
That is debt item 6.

What is left is not a diagnosis, it is a decision: `ROADMAP.md` §9.2 sets out
three responses — a start-up guard in `runner/sandbox.py` that refuses to grade
a stalled process and relaunches, a larger `time_limit_s`, or publishing with
the exposure documented — and recommends the guard, because it is the only one
of the three that makes a recorded `timeout` mean what the leaderboard says it
means.

## The guard, chosen and built the same day

Selim picked option A, so the session did not end at the diagnosis.

The constraint that shaped the design is that the stall is a property of a
*process*, decided before it does anything. So a probe run from the harness says
nothing about the child the harness is about to grade in — it has to be the
child that asks. `runner/mps_stall.py` therefore holds both halves: `probe()`,
which is the measurement, and `pytest_configure`, which is a pytest plugin hook.
`runner/sandbox.py` copies that module into a Metal task's workdir at the same
moment it copies the hidden tests, which is after the solver has finished
writing, and runs pytest with `-p _scratchbench_mps_stall`. A file that is not
on disk while the solver works cannot be subverted by it; that guarantee was
already paid for, and this reuses it rather than inventing a second one.

Three details are the ones worth defending:

- **The gate exits, it does not fail.** A stalled process has measured nothing
  about the solution. Failing its tests would turn an absence of evidence into a
  verdict, which is the one thing this repository cannot do. It exits `125` —
  outside pytest's own `0`-`5`, and distinct from the `124` that
  `tools/mutate_metal_task.py` already uses for a timeout, so a caller never has
  to guess which of the two it is holding.
- **Retry, then a status of its own.** `run_pytest_until_healthy` relaunches up
  to four times. Launches alternate almost perfectly, so two would usually do;
  four leaves room for the day that stops being true and still bounds the cost
  at a few seconds, because the gate fires before the first test. A run where
  every attempt stalls becomes **`mps_stalled`**, classified in `STATUSES` as no
  evidence and a harness failure. That status exists so that the alternative
  does not: without it a stalled run is a `timeout`, which is evidence *and* a
  failure, and the failure shape this repository publishes would be describing
  the machine.
- **It is armed only for Metal tasks.** The stall is a measured property of this
  Mac. Nothing has been measured on the rented CUDA box, and arming the guard
  there would be an assumption wearing a measurement's clothes.

Evidence, all of it from this session:

```
validate --tier all                                x3, clean
validate --tasks metal_softmax_backward_kernel     x6, clean
  81 passed  81 failed   7.60s · 3.27s · 11.90s · 9.37s · 9.05s · 14.35s
python -m pytest -q                                110 passed in 87.76s
tools/check_cost.py                                152 file(s) checked
tools/check_calibration.py                         26 entries, 152 draw(s)
```

That is the same task that returned `BROKEN` intermittently last session and
once took 2920 s. Six consecutive runs, longest 14.35 s. The harness suite gained
five tests and lost more than half its wall time, because it grades the Metal
tasks too and had been paying the stall all along without anyone noticing.

## Thirty draws, and the defect that was already published

With the guard in, the two debts the stall had been holding up became one sweep:
`--tier accelerated --repeat 10` for Opus, Sonnet and Haiku collects
`metal_softmax_backward_kernel`'s first calibration and
`metal_cross_entropy_kernel`'s re-draw at the same time. Thirty draws, sixty
gradings, $3.87.

The number that matters most is a zero. **No `timeout`, no `mps_stalled`, no
`collection_error` in any of the sixty.** That is the guard's field evidence, and
it is a different claim from the six clean `validate` runs: those were the
reference and the starter, these are twenty-one solutions that did not import,
seven that failed on their merits, and thirty-two that passed.

The admission rule decided the set, not me:

| Task | Opus 5 | Sonnet 5 | Haiku 4.5 | Set |
|---|---:|---:|---:|---|
| `metal_softmax_backward_kernel` | 8/10 | 6/10 | 2/10 | **v2** |

The top two rates are 80% and 60%, so the frontier does not clear it. What the
failure shape says is more interesting than the rate: every one of Opus's two
losses and Sonnet's four is a Metal compile error, not a wrong answer. The three
reductions — the thing the task was written to probe — are apparently not the
hard part. The type and address-space rules around a `threadgroup` array shared
by all three of them are. That is the second time a Metal task in this
repository has turned out to measure MSL's declaration rules rather than the
algorithm (`metal_cross_entropy_kernel` and `half`), and it is worth saying out
loud before a third one is written.

Then the re-draw found something I had written down as a risk and not checked.
Before spending the two dollars I counted what was already in `calibration/` and
`leaderboard/`:

```
metal_cross_entropy_kernel  claude-haiku-4-5
  {'solution_error': 6, 'adapter_error': 1, 'passed': 1, 'timeout': 1, 'failed': 1}
```

One `timeout`, in published evidence, in the denominator of a rate this
repository prints, since 2026-08-13. Debt item 6 had described that as something
that *might* have happened. It had already happened, and it was countable in
thirty seconds by anything that could read a dictionary. `LESSONS.md` L43 is
that, and the lesson is not about the stall: **a risk stated about data you
already hold is not a risk, it is an unread measurement.**

The draw stays counted as a failure. Removing it would improve Haiku's rate on
the strength of something learned after seeing which way the draw went, and a
benchmark that edits its own evidence in the direction it just learned about has
stopped measuring. It is also the conservative direction — a `timeout` scored as
a failure can only understate a model — and both `meta.yaml` and `README.md` now
say so in the file rather than in my head. The block is twenty draws per model:
Opus 14/20, Sonnet 12/20, Haiku 3/19.

`unvalidated` is now empty. `v1` five, `v2` four, `warmup` six.

## What this does not do

It does not explain the stall. Nothing here says why every other process on this
machine is blocked; the guard detects the state and refuses to grade in it, and
`LESSONS.md` L42 says as much in the entry rather than in a commit message.

It also does not clear the backlog the diagnosis created.
`metal_softmax_backward_kernel` still has no calibration block, so it is still
`unvalidated` and still belongs to no set — but its trigger is now met and the
thirty draws are the next thing to spend money on. `metal_cross_entropy_kernel`
is published on draws taken before the guard existed. And §5's `v2` sweep, which
was blocked on the stall, turns out to be blocked on four other things as well;
§5 now carries a table saying which, because "one session away" was never true
and saying so in the roadmap is cheaper than discovering it mid-sweep.

## What I would tell myself at the start of session 14

L40's rule was that a cause explaining one observation is not the cause. This
session's is its neighbour and it is about the other end of the experiment:
**a control that has not been repeated is not a control.** Both of L41's — the
reference is fast, the other Metal task is fast — were single-digit samples of
something that alternates, and every experiment for two sessions inherited the
frame they licensed.

The specific defence is cheap enough to be a habit. Interleaving cells protects
against drift and not against alternation. Before believing any difference
between two configurations, **run the same configuration twice in a row.**
