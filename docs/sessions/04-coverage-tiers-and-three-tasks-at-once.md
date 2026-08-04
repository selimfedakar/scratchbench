# 04 — Coverage, tiers, and three tasks written at once (2026-07-27)

## What this step was

Five tasks left three holes: the `data` and `kernels` categories were empty,
and nothing was harder than a 3. This step closed two of them, added three
tasks, and built the mechanism that lets the third one be closed without
breaking the promise on the tin.

The task set is eight now:

| task | category | diff | tests |
|---|---|---|---|
| `softmax_stability` | numerics | 1 | 29 |
| `bpe_merge_order` | tokenization | 2 | 30 |
| `attention_causal_mask` | attention | 2 | 26 |
| `kv_cache_equivalence` | attention | 3 | 21 |
| `grad_accumulation` | training | 3 | 27 |
| `sharded_dataloader` | **data** | **4** | 53 |
| `quantization_error_bounds` | numerics | **4** | 65 |
| `online_softmax_attention` | attention | **5** | 57 |

## Running three authors at once

The two new hard tasks were written in parallel by forked agents carrying this
session's full context — the conventions, the authoring order, the L2 bar. That
is the part that made it work: an agent started cold would have needed the
whole briefing and would still have written in a different voice.

It was not free. One fork died twice — a server error, then a session limit —
and the second death happened after it had written all five of its files and
before it had validated any of them. So `online_softmax_attention` arrived as
something that *looked* finished and had never been run.

Which is exactly the state a benchmark must not accept. I validated it myself
and ran the mutation pass its author never got to: six wrong implementations,
all caught. The lesson is not about agents. It is that "the files exist" and
"the task works" are different claims, and only one of them is checkable by
looking at the directory.

## The tiers

`kernels` was the hole I could not close by writing a task, because closing it
honestly meant choosing between two things the repository promises.

The founding constraint is a laptop, no GPU, five minutes, reproducible by
anyone. A CUDA kernel task breaks all of that. But leaving CUDA out cuts the
most interesting question in ML engineering out of a benchmark about ML
engineering, and it is the question I care most about personally.

The way out is not to pick one. It is to stop pretending they are the same
measurement:

- **`tier: laptop`** — the default and the headline. `requires_gpu` must be
  false, enforced by the loader. The leaderboard pass rate is computed over
  this tier and nothing else, so the number stays reproducible by construction.
- **`tier: accelerated`** — may set `requires_gpu: true` and must declare
  `accelerator: cuda | metal`. Reported separately, never folded in. The
  five-minute limit still applies: a GPU is not a licence to need an hour.

On a machine without the hardware, an accelerated task returns
`needs_accelerator` — neither a pass nor a failure. Rounding an absence of
evidence down to zero is how a leaderboard starts reporting on hardware it
never ran on.

And the rule that governs everything else does not bend for hardware either:
until an accelerated reference has been run on real hardware, the task stays
out of the frozen set. If the reference has never passed, it is not a task yet.

`scratchbench validate` lists such a task as `UNCHECKED` rather than `ok`,
because the alternative is a table that says everything is fine while one row
has never been executed.

## Technologies

- `runner/tasks.py` gained `tier`, `accelerator`, and an
  `available_accelerators()` probe built on `torch.cuda.is_available()` and
  `torch.backends.mps.is_available()`. The field defaults to `laptop`, so the
  six existing tasks needed no edits.
- `sharded_dataloader` is standard library only — no numpy. It is bookkeeping,
  not arithmetic, and `random.Random(f"{seed}:{epoch}")` is a better shuffle
  than anything I would have written on top of a generator I had to seed twice.
- `quantization_error_bounds` and `online_softmax_attention` are numpy.

## Verified

```
scratchbench validate --tier all

task                       reference  untouched starter  time   verdict
-------------------------  ---------  -----------------  -----  -------
attention_causal_mask      26 passed  26 failed          0.27s  ok
bpe_merge_order            30 passed  30 failed          0.21s  ok
grad_accumulation          27 passed  27 failed          1.93s  ok
kv_cache_equivalence       21 passed  21 failed          0.27s  ok
online_softmax_attention   57 passed  57 failed          0.30s  ok
quantization_error_bounds  65 passed  65 failed          0.30s  ok
sharded_dataloader         53 passed  53 failed          0.22s  ok
softmax_stability          29 passed  29 failed          0.28s  ok

8 task(s) validated: reference passes, starter fails cleanly.
```

```
scratchbench run --model reference --tasks all --tier all

laptop       100% (8/8)
4.0s wall clock
attention 100%  data 100%  numerics 100%  tokenization 100%  training 100%
```

Harness suite: `45 passed in 7.31s`.

Mutation passes on all three new tasks, every mutant caught:

| task | mutants | worst case |
|---|---|---|
| `sharded_dataloader` | 6 | contiguous slab instead of a stride: 2 failed |
| `quantization_error_bounds` | 6 | shift-then-round formulation: 5 failed |
| `online_softmax_attention` | 6 | no `-inf` guard in the merge: 1 failed |

The worst-case column is the one worth reading. Binary scoring means one
failure is a failed task, so all eighteen are caught — but a mutant caught by a
single test is a task with one thread holding it, and if that test is ever
weakened the hole opens silently.

## Still open

`kernels` has the mechanism and no task. The first one is a CUDA kernel, it
cannot be validated on this laptop, and it does not enter the frozen set until
it has been run on real hardware.

## Next

The Anthropic adapter, one attempt per task — because a model that passes on
the eleventh try is measuring the scaffolding, not the model.
