# 09 — The accelerated tier answers (2026-08-09)

## What this step was

The kernel task had been verified on hardware in session 06 and never asked of a
model. This step asks it, and asks the laptop tier of Haiku five times, on a
rented RTX 4000 Ada. It is the first time in this project that both tiers have
been measured in the same place.

Four things came out of it, and only one was expected.

## 1. The accelerated tier discriminates where the laptop tier does not

Session 08 established that `claude-opus-5` solves the laptop tier forty times
out of forty. The same model, same settings, same day, on `fused_rmsnorm_kernel`:

```
draw1: FAIL  1 failed, 23 passed
draw2: FAIL  1 failed, 23 passed
draw3: FAIL  1 failed, 23 passed
draw4: FAIL  1 failed, 23 passed
draw5: pass  24 passed

fused_rmsnorm_kernel  1/5  SPLIT
```

**One out of five.** The category that was nearly cut for violating the
founding constraint, and that needed a whole tier built beside it to exist at
all (L10), is the only part of this benchmark still asking a question Opus 5
cannot answer.

Haiku 4.5 on the same task went 3 out of 5. Two rates from five draws of one
task are not distinguishable and the leaderboard says so. What is not a
coincidence is what the two failures look like.

## 2. The same verdict, opposite information

| | draws that failed | tests failed, out of 24 |
|---|---:|---|
| `claude-opus-5` | 4 of 5 | **1** every time |
| `claude-haiku-4-5` | 2 of 5 | 24, and 23 |

Opus fails one test out of twenty-four. Haiku, when it fails, fails everything.
Binary scoring calls both of those `FAIL`, and they are not the same event: one
is a working kernel with an edge missed, the other is a kernel that does not run.

The first version of that sentence I wrote said Opus "consistently" misses the
same test, which was an inference from four identical count lines. It is a
measurement now. Four more draws, each keeping its workdir and re-running pytest
against it:

```
FAIL  -> test_rmsnorm_kernel.py::test_a_single_column
FAIL  -> test_rmsnorm_kernel.py::test_a_single_column
FAIL  -> test_rmsnorm_kernel.py::test_a_single_column
```

Every captured failure, four for four, is the same test.

## 3. The edge, and whether the task is allowed to ask about it

```
E   triton.compiler.errors.CompilationError: at 26:23:
E       mean_sq = sum_sq / n_cols.to(tl.float32)
E                          ^
E   AttributeError("'int' object has no attribute 'to'")
```

Triton specializes integer kernel arguments whose value is `1` into compile-time
constants. At every row width but one, `n_cols` arrives as a runtime scalar and
`.to(tl.float32)` is a reasonable thing to write; at `N = 1` it arrives as a
Python `int`, `.to` does not exist, and the kernel never compiles. The reference
divides by `n_cols` directly and never meets it.

Before any of that is worth reporting, the task has to be allowed to ask. A
hidden test that can fail for a reason `prompt.md` never states is a broken task,
and a broken task does not produce a lower score, it produces a wrong one. The
prompt says:

> `x` may be a view whose rows are strided rather than a contiguous tensor: its
> columns are adjacent, but consecutive rows need not be. **M may be zero and N
> is at least one.**

Licensed, in as many words. The test stands and so does the result.

This is the clearest example this repository has produced of the distinction the
next task set is designed around. It is not a convention to guess. It is a place
where the fluent answer and the correct answer differ, and where knowing the
difference requires knowing how the compiler treats its arguments.

## 4. `solution_error` earned itself, in its first real use

Haiku's laptop sweep produced three `NOIMPORT` outcomes across five draws:

```
draw1  bpe_merge_order     SyntaxError: unterminated string literal
draw2  bpe_merge_order     IndentationError: unexpected indent
draw3  softmax_stability   SyntaxError: unmatched '}'
```

The model emitted Python that does not parse. Session 07 moved that case from
`collection_error` (no evidence, out of every rate) to `solution_error`
(evidence, and a failure), on the argument that the starter imports cleanly by
contract so anything that does not was written by the solver. It had never
happened in a real run.

What the old classification would have published, computed from the same files:

```
draw1: real 4/8 = 50%   old code 4/7 = 57%   (+ run exits non-zero)
draw2: real 4/8 = 50%   old code 4/7 = 57%
draw3: real 4/8 = 50%   old code 4/7 = 57%
```

Seven points of inflation, in the model's favour, in three draws out of five,
plus three sweeps that would have exited non-zero and looked like harness
failures rather than results. The bias runs exactly the way L22 predicted:
unparseable output comes from the weaker model, so dropping it from the
denominator flatters the model it should be catching.

## The tier arithmetic, in the wild

The first run in this project's history where both tiers were measured at once:

```
$ python -m runner.cli run --model reference --tasks all --tier all --no-write

accelerated  100% (1/1)
laptop       100% (8/8)  ·  headline
10.5s wall clock  ·  1 attempt(s) per task
attention 100%  data 100%  numerics 100%  tokenization 100%  training 100%
```

Two lines, the headline one labelled, and `kernels` absent from the category
line because that category belongs to the other tier. Before session 07 this
would have been a single blended percentage. Every accelerated-only results file
from today also carries `"pass_rate": null` rather than the accelerated number
promoted into the headline slot, which is the same fix seen from the other side.

## Hardware

Second card this task has been proved on. Everything green:

```
NVIDIA RTX 4000 Ada Generation | torch 2.8.0+cu128 | triton 3.4.0
Linux 6.8.0 | Python 3.12.3

77 passed, 1 skipped          # the skip is the accelerated-hardware-missing test,
                              # correctly inapplicable on a box that has the hardware
fused_rmsnorm_kernel  24 passed  24 failed  6.22s  ok
all 6 mutants caught
```

The A4500 result from session 06 reproduces on a different generation of card,
which is worth more than repeating it on the same one.

## Cost

$3.30 for everything on the leaderboard: $2.64 Opus laptop, $0.34 Haiku laptop,
$0.28 Opus accelerated, $0.04 Haiku accelerated. Twenty-two results files, and
every cost in them re-derives from its own tokens.

## What this changes

`docs/V2_DESIGN.md` was written against a suspicion that mechanism-dense tasks
discriminate. There is now a task that does it, its failure mode is understood
to the line, and it is on the tier that was almost cut. The design is updated
accordingly.

## Still open

- **v2 has no tasks written.** The two named in the design are still the two to
  write.
- **The kernels category has one task.** The README promises Metal as well as
  Triton and only Triton exists.
- **Only two models have been asked anything.** A third would tell us whether
  `attention_causal_mask` at 3/5 is a property of Haiku or of the task.
