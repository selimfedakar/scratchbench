# 14 — The second Metal task, and a gate that refused everything it was pointed at

`ROADMAP.md` §2 asked for two things: the backward kernel that completes the
accelerated half of `v2`, and the second laptop candidate that §1.2's reopening
condition needs. The first exists. The second does not, and the reason it does
not is the more useful half of the session.

## The task: `metal_softmax_backward_kernel`

Softmax backward per row, fused, one threadgroup per row, with the threadgroup
size chosen by the caller exactly as in the forward task. Given the logits the
forward softmax was computed from and the gradient arriving from whatever
consumed it:

```
y[r, j]  = exp(logits[r, j]) / sum_k exp(logits[r, k])
dx[r, j] = y[r, j] * (dy[r, j] - sum_k y[r, k] * dy[r, k])
```

What makes it a different task rather than the forward one again is that it
reduces three times where the forward reduces twice, and it writes the row back
instead of a single number. The row maximum, the softmax denominator under that
shift, and the contraction of the upstream gradient against a softmax that is
never materialised — one scratch array serves all three, so every reuse needs a
barrier on both sides of it, and the last pass rebuilds the softmax from the
same shift and the same denominator every lane agreed on.

Journal 11's four measured API facts still carry it and were not re-derived.
Nothing the new kernel depends on was outside that list, and the probe that
checked said so: `exp` on this path agrees with a float64 CPU softmax to 1.6e-9
on a single lane, so there is no fast-math surprise to write up.

### The tolerance is a measurement

Over every configuration the hidden tests ship, the reference's worst absolute
deviation from a float64 CPU autograd gradient is **9.9e-8** and its worst
absolute row sum is **2.0e-7**. `ATOL` is 2e-6: twenty times the measured worst,
which leaves a different Apple GPU room, and four orders of magnitude under the
smallest gradient any test looks at.

The two shift-invariance tests get a looser 1e-5, and that is arithmetic rather
than slack — they compare two launches whose intermediate values differ by 137,
so the cancellation is real. Worst measured difference there: 7.5e-7.

### Both halves of L2

```
task                           reference  untouched starter  time   verdict
metal_softmax_backward_kernel  81 passed  81 failed          23.04s  ok

1 task(s) validated: reference passes, starter fails cleanly.
```

### The licensing walk found one real defect

Walking every assertion back to the sentence that licenses it turned up a
contradiction in my own prompt. It promised `dy` was "an ordinary gradient, of
order one", and `test_the_gradient_does_not_move_when_the_upstream_is_shifted`
feeds it `dy + 137`. That is the landmine inverted: not an assertion the prompt
fails to license, but a test that violates an input contract the prompt states.
The sentence now says `dy` is not extreme in the way the logits are, and that
nothing about the answer may depend on its scale or on where its mean sits —
which licenses the test instead of contradicting it.

## The mutants, and both of them were news

Nineteen mutants, and the two-way expectation earned itself twice in one run.
Each was given the verdict its sibling gets in the forward task, and two of those
guesses were wrong in opposite directions.

**The power-of-two fold survived, and it was a real hole.** It is caught by the
forward task and a real model wrote it unprompted in one of Sonnet's draws, so I
expected it caught. It passed all 76 tests, because the row shift is
algebraically neutral: a fold that loses some lanes' partial maxima still
produces *a* value from the row, and on ordinary logits any such value keeps
every exponential in range. One test closes it —
`test_the_largest_logit_in_every_position`, a seven by seven matrix whose maximum
walks the diagonal at five group sizes that do not divide it. Three of the five
catch it, at 3, 5 and 7, where the dropped lanes own real columns. `LESSONS.md`
L38.

**The missing barrier before the second scratch reuse was caught, and L32 says
it should not have been.** Eight runs of eight, 21 to 23 of the 81 tests failing
from run to run: the verdict is stable and the blast radius is not, which is what
a real race looks like. The same mutant one reduction earlier survives all 81 in
the same pass. The difference is not the device: at the maximum, every lane walks
its whole slice of the row between reading `scratch[0]` and writing to it, so the
read is finished everywhere before the first write lands; at the denominator the
value being published is already in a register and the write follows the read
immediately. L32's evidence stands and its title does not. `LESSONS.md` L39.

## The §2.0 gate, applied for the first time, refused all three candidates

`ROADMAP.md` §1.3 held three laptop candidates and L37 had just sharpened §2.0
into a question that can be answered before paying for a calibration: *does the
framework have a page about exactly this mistake?* It was answered against the
installed libraries rather than from memory, and all three failed.

| Candidate | Verdict |
|---|---|
| `byte_level_bpe_roundtrip` | Fails, and worse than the others: the byte-to-codepoint table is a published function copied verbatim across the ecosystem, and `tiktoken` and `tokenizers` are installed on this machine. A task about it measures recall. |
| `pairwise_sum_error_bound` | Fails. `np.sum`'s own Notes describe pairwise summation and the precision consequence of not using it. |
| `strided_view_aliasing` | Fails twice. `sliding_window_view` documents the aliasing and the read-only default in its `writeable` parameter; `as_strided` carries a `.. warning::` and a Notes section about unpredictable writes. |

The quotes are in `ROADMAP.md` §1.3 beside each verdict.

**Why this is a result and not a stall.** §1.2's reopening condition is two
independent attempts *written against §2.0*. A candidate that fails §2.0's own
gate before it is written is not one of those attempts, so calibrating it would
buy a fifth `warmup` task and leave the count at one. Thirty draws would not have
advanced the branch.

### The counterexample was already in `v2`

Three tasks here discriminate at the top. Two satisfy §2.0 and both are
accelerated. The third is `flash_attention_backward` — laptop, in `v2`, Opus
15/15 against Sonnet 8/10 — and it satisfies §2.0 not at all. Flash attention has
papers, tutorials and reference implementations. What makes it hard is L28's
mechanism: the graded function is handed one block of keys and never the rest, so
the state it needs is structurally unavailable rather than merely discouraged.

§2.0 was derived from the two accelerated tasks and generalised to the laptop
tier without a laptop example. The laptop example exists and works differently.
So three refusals in a row may mean the tier is saturated, or may mean four
candidates were chosen against the wrong criterion for that tier, and those are
not the same conclusion. `ROADMAP.md` §9.1 lays out the three options with a
recommendation and leaves the decision where it belongs.

## The task is finished and cannot be published

`validate` was green the first time and the second time. The third time it said
`BROKEN  starter timed out instead of failing`, on a task whose starter fails in
under two seconds.

The first cause I found was real: `ccusage`, which draws my terminal's status
line, at 748 percent CPU on a ten-core machine. Killing it turned a run that
would not finish in ten minutes into 1.98 s. I wrote it up, disabled the status
line, and moved on — and that was the mistake, because the symptom came back with
the machine idle. The four slow runs I already had in hand were 1.98 s, 33 s,
278 s and 2920 s, and a single cause that switches on and off does not produce
three orders of magnitude. L40 is that lesson; L41 is the problem.

**The problem.** About half of this suite's failing runs enter a state where
every MPS operation in the process is roughly a hundred times slower, for the
life of the process. The untouched starter, same command, same idle machine:

```
1.0s   1.4s   33s   124s   278s   312s   344s   2920s
```

The reference measured 1.2 s to 3.7 s in every one of those runs, and
`metal_cross_entropy_kernel`'s starter — which also fails all of its tests — ran
ten times out of ten under 1.4 s. So it is this task, not the machine, and not
"tests that fail" in general. Inside a slow run `--durations` shows no hanging
test: every test is uniformly slower.

Seven suspects, each killed by a measurement rather than an argument: the status
line, the largest shapes, `torch.autograd.grad` in the expected value, memory
pressure (the slow run peaked at 261 MB against the fast runs' 330), the NaN
sentinel, a kernel that writes nothing, and the size of the compared tensors. No
cause. The table is in L41.

**Why that stops the calibration.** `time_limit_s` is 300 s and `STATUSES` calls
a killed run `timeout` — evidence, and a failure. A pass rate computed over
affected draws would therefore still be sound. The failure *shapes* would not be,
and the failure shapes are what this repository publishes instead of partial
credit; they are most of what journals 09 through 13 are made of. Thirty draws
would have cost about two dollars and bought a number I could defend beside a
table I could not.

So the task ships `frozen_set: unvalidated`, uncalibrated, with the reason in
`meta.yaml` rather than in my head, and `ROADMAP.md` section 9 item 3 carries it
as the thing to fix before the `v2` sweep — which is the run it would corrupt.
The next step there is not more theorising: bisect this suite against the forward
task's structure until the smallest reproducer exists.

## Two defects in the mutation script, found by using it

Both were in `tools/mutate_metal_task.py` and both mattered.

It waited **1800 seconds** per mutant, and the mutant that returns out-of-range
lanes before a barrier — undefined behaviour in Metal Shading Language — spent
every second of it before the guard fired. It now reads the task's own
`time_limit_s` out of `meta.yaml` and calls a timeout `CAUGHT`, which is what the
graded harness does. The limit is read rather than repeated, because a second
copy of a number is how L11 and L20 happened.

The second is worse and quieter. The gate that proves the untouched starter fails
read any non-zero exit as "fails cleanly" — including a timeout. A starter that
never answered would have certified the tests, and this task's starter times out
about half the time. `TIMED_OUT = 124` is now distinct from a real failure and
stops the run with its own message.

## Verified

| Claim | Level | Evidence |
|---|---|---|
| Reference passes, starter fails cleanly | L2 | `validate --tier all` → `metal_softmax_backward_kernel  81 passed  81 failed  52.43s  ok` |
| Every task still validates | L2 | `validate --tier all` → `14 task(s) validated`, `1 task(s) not checked here: fused_rmsnorm_kernel` |
| The forward task's mutants survive the refactor | L2 | `metal_cross_entropy_kernel: all 13 mutants behaved as expected` |
| The new task's mutants all match | L2 | `metal_softmax_backward_kernel: all 19 mutants behaved as expected` |
| The reference's worst deviation over the shipped grid | L2 | probe → `worst absolute error 9.909e-08`, `worst row sum 1.974e-07` |
| `exp` on this path is not a fast-math approximation | L2 | single-lane launch vs float64 CPU → `1.56e-09` |
| The barrier race is caught, repeatedly | L2 | 6 trials → `CAUGHT` 6 of 6, 21 to 23 failures; 8 of 8 across the session |
| The new test catches the power-of-two fold | L2 | `CAUGHT  power-of-two fold  3 failed, 78 passed` |
| Harness suite | L1 | `pytest -q` → `105 passed in 45.76s` |
| Every checked-in cost re-derives | L2 | `check_cost.py` → `152 file(s) checked` |
| Every calibration block re-derives | L2 | `check_calibration.py` → `26 calibration entries re-derived from 152 draw(s)` |
| Fifteen tasks, 704 hidden tests | L2 | 680 counted by `validate --tier all` here, plus `fused_rmsnorm_kernel`'s 24 from journal 06 |
| The instability is this task, not the machine | L2 | forward starter 10 runs of 10 under 1.4 s; this one 1.0 s to 2920 s |
| The three laptop candidates fail the §2.0 gate | L2 | docstrings of `np.sum`, `sliding_window_view`, `as_strided` on numpy 1.23.5, quoted in `ROADMAP.md` §1.3 |
| Spend | L2 | $0. No model was asked anything this session. |

## Not verified

- **The cause of the instability.** Seven suspects are dead; none of them is a
  cause. The next step is a bisection against the forward task's structure, not
  another theory.
- **The task has never been run by a model.** No calibration block, no draws, and
  therefore no evidence about what it measures — only evidence that it is
  correctly built.
- **`validate --tier all` is not reproducible on this task today.** The run
  pasted above is green; an earlier one on the same working tree returned
  `BROKEN  starter timed out instead of failing`. Both are real.
- **Nothing in this session is committed.** `COMMITS.md` carries the queue and
  `~/scratchbench-session14-commits.sh` runs it.
