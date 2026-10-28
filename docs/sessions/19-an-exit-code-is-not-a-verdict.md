# 19 — An exit code is not a verdict

The plan for this session was still §10.3 — the rented CUDA box — but an outside
technical review landed first, and its first finding is the kind you stop
everything for. Handed one file to grade,

```
solution.py:
    import os
    os._exit(0)
```

the harness recorded **passed**. Not a near-miss in some corner of the
reporting layer: the headline grader, calling a no-op a success. This session is
the three findings in that review that touch what a results file is allowed to
claim — F01, F03, F06 — and nothing else, because none of the rest matters while
the grader will certify a solution that ran no tests.

## The common shape

All three read a number the process produced about itself as if it were the
thing the number stands for.

- **F01.** `os._exit(0)` during collection kills the process with status zero
  having judged nothing. `grade` read exit 0 as "every test passed". `0 passed`
  is not a pass.
- **F06.** 125 is the stall guard's exit code. A laptop task has no guard, so a
  solution that returned 125 on its own was recorded `mps_stalled` — an absence
  of evidence, dropped from the rate — when it was the solution failing.
- **F03.** `validate` counted a task that declared a missing dependency, and so
  never ran, inside its "14 task(s) validated" total.

I reproduced each against the current `main` before touching anything, with the
reviewer's own probe script:

```
{"probe": "exit_zero",           "status": "passed",      "passed": true}
{"probe": "spoof_stall_on_cpu",  "status": "mps_stalled", "passed": false}
{"probe": "missing_dependency_validate", "exit": 0, "stdout": "... 1 task(s) validated ..."}
```

## What changed

**A pass now requires the run to have reached the end.** The number of judged
tests must equal `hidden_tests` — the count `validate` already re-derives on
hardware that can run the task — so exit 0 with a short count falls through to
the same baseline disambiguation an unparseable solution does, and comes out
`solution_error`. The starter imports cleanly by contract, so the blame is the
solution's.

**A stall must prove itself.** `runner/mps_stall.py`'s gate writes a sentinel
file beside the hidden tests the instant it fires, and `runner/sandbox.py` reads
it back: a `mps_stalled` verdict, and the relaunch loop that spends up to four
processes looking for a healthy one, now both require the sentinel. A bare 125
from the solution is its own answer, retried zero times and classified as a
failure. This is the F06 acceptance's second half — a real guarded stall and a
solution's abnormal exit stay separate — not just the CPU case.

**`validate` stops counting what it did not run.** A missing dependency or an
absent accelerator is kept off the validated total and reported on its own line;
`--strict` turns "could not check everything asked for" into a non-zero exit,
and CI now passes it, because in an installed CI environment a missing
dependency is a broken task rather than an absent GPU.

## Verified

```
# the three probes, re-run after the fix
{"probe": "exit_zero",          "status": "solution_error", "passed": false}
{"probe": "spoof_stall_on_cpu", "status": "solution_error", "passed": false}
{"probe": "missing_dependency_validate", "exit": 0,
 "stdout": "... 0 task(s) validated ... 1 task(s) skipped (missing dependency) ..."}

$ python -m runner.cli validate
14 task(s) validated: reference passes, starter fails cleanly.
$ python -m runner.cli validate --strict ; echo $?
0
```

Six regression tests, one per door plus the bare-125 retry case, and the three
stall tests that encoded the old exit-code-only contract were updated to model a
real stall by writing the sentinel — a change to the test's fidelity, not a
loosening of it. Full harness suite green on the no-GPU plugin that reproduces
CI locally.

The lesson is `docs/LESSONS.md` L50. What this did not touch, and the review is
right that it must before launch: F02, the run environment is not a security
sandbox, and the solution can still do anything in-process a count cannot stop —
spoof the summary line included. F01 closes the structural false pass; the trust
boundary is its own work package.
