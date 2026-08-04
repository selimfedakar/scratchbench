# 05 — The first model adapter, and the first kernel (2026-07-28)

## What this step was

Three things, in the order they had to happen: the Anthropic adapter, so the
harness can grade something other than its own reference; the first `kernels`
task, so the category that motivated the tier mechanism stops being empty; and
a CI job that can run the accelerated tier on hardware this laptop does not
have.

The kernel task is the one with an asterisk on it, and the asterisk is the
point of the whole section below: it is authored, it is not validated, and it
is marked `frozen_set: unvalidated` until its reference has passed on a real
GPU.

## The Anthropic adapter

`adapters/anthropic_api.py`. One model call per task, streamed, with the answer
constrained to a JSON schema that can only describe the files the task
declared.

Four decisions worth writing down, because each one was a fork:

**One attempt.** `max_attempts = 1`, and it is written into every results file.
A harness that re-prompts until something passes is measuring its own retry
loop. Transport retries the SDK performs on a 429 or a 5xx are not attempts in
this sense — the model gets no feedback from them, so they cannot teach it
anything about the task — and the results file says which number it is
reporting.

**No server-side fallbacks.** The Anthropic SDK will re-run a declined request
on another model for you, and for an application that is the right default:
the caller wants an answer. Here it is exactly wrong. A silently substituted
model would publish one model's score under another model's name, which is the
single failure this repository cannot survive. So fallbacks stay off, the
adapter checks that the model that answered is the model that was asked, and a
refusal is reported as an absence of evidence instead of as a failure.

**Anything that is not a measurement raises.** A refusal, a response truncated
at the token ceiling, a model substitution, a response that is not JSON: all of
them raise, `grade` turns that into `adapter_error`, and a run containing one
exits non-zero. The alternative is scoring a harness problem as a model
failure, which is a number that looks exactly like a real one.

**The model may only write files the task declared.** The response schema pins
the filename to an enum of the starter's own names, and the writer checks again
before anything lands. This is not tidiness. `solve` runs in the directory the
hidden tests are about to be copied into, so a model that could drop a
`conftest.py` there could rig its own score.

Cost is computed from the published per-token prices and the usage the API
returns, including the cache multipliers even though this adapter does not
cache. A model whose price is not in the table reports `usd_cost: null` rather
than a plausible-looking number.

`attempts` and `usd_cost` now reach the results file, taken off the adapter
rather than off a flag. The reference solver is a plain function with neither
attribute and its control run correctly records one attempt at no cost.

The SDK is an optional dependency (`pip install 'scratchbench[anthropic]'`).
The laptop tier's claim is that anyone can clone this and re-run it with no
account and no key; making an API client a hard dependency would quietly
contradict that.

## What the adapter tests actually cover

Offline, all of it. There is no network in this session's evidence, and the
report says so. What the tests do cover is the part that would corrupt a score
without looking like a bug: what the model is shown, and what is allowed to
land in the graded directory.

- the prompt contains `prompt.md` and the starter files and none of the hidden
  test function names;
- the response schema admits exactly the declared filenames;
- `conftest.py` and `../escaped.py` are refused, and a response with no usable
  file raises rather than grading an untouched starter;
- the price arithmetic, including the 1.25× cache write and 0.1× cache read
  multipliers, and an unpriced model reporting nothing;
- a run records the adapter's `max_attempts` and `usd_cost`.

## The first kernel task

`fused_rmsnorm_kernel` — category `kernels`, difficulty 4, tier `accelerated`,
`accelerator: cuda`, `deps: [torch, triton]`.

Triton rather than raw CUDA C, for two reasons. The README already promises
"Metal and Triton kernels" in the `kernels` row, so this is the thing that was
advertised. And a Triton kernel needs no `nvcc`, no compile flags and no build
step — `pip install -e .` on a box whose torch already ships triton is the
entire setup, which keeps the reproducibility bar as close to the laptop tier's
as hardware allows.

The task is RMSNorm forward, fused, one program per row. What it actually
probes is four things a wrapper cannot fake:

- **float32 accumulation under a float16 input.** The prompt says the rows are
  long enough that a float16 accumulator will not survive one, and one test
  means it: 16384 values with a standard deviation of two put the sum of
  squares within a factor of two of the largest number float16 can represent.
  In float32 that is unremarkable. In float16 it saturates to infinity and the
  row comes back as zeros.
- **A row walked in blocks, with the tail masked.** `BLOCK_SIZE` is chosen by
  the caller and has no fixed relationship to N, so the kernel is launched with
  blocks smaller than the row, larger than the row, and exactly equal to it.
- **Two independent row strides.** The input may be a strided view and the
  output may be a window into a wider buffer. One test hands the kernel a slice
  of a padded buffer and checks the padding is still there afterwards.
- **That the kernel is where the work happens.** The hidden tests launch
  `rmsnorm_fwd_kernel` directly, with their own grid and their own block size.
  This is why the prompt fixes the kernel's parameter order and calls it part
  of the interface: it is the only way to grade "write a Triton kernel" instead
  of "produce a correct tensor somehow".

The licensing pass ran both ways. Every assertion has a sentence in
`prompt.md` behind it, and every promise in the prompt has a test behind it,
with two deliberate exceptions recorded here rather than left implicit:

- *"more than one kernel launch per call"* is in a sentence about what the task
  does **not** need. It is scope guidance, not a claim, and nothing enforces
  it. A promise that cannot be tested does not belong in a prompt (L8); this
  one is not a promise.
- Nothing forces the **wrapper** to call the kernel. A solution with a correct
  kernel and a PyTorch wrapper would pass. That is the intended reading — the
  kernel is the task and the wrapper is glue — and the direct-launch tests are
  what make the kernel unavoidable.

## What is not verified, and how it gets verified

This laptop has no CUDA device, so the reference has never run. That is the
whole reason `frozen_set: unvalidated` exists:

```
$ python -m runner.cli validate --tier all
task                       reference  untouched starter  time   verdict
-------------------------  ---------  -----------------  -----  -----------------------------------------
attention_causal_mask      26 passed  26 failed          0.25s  ok
bpe_merge_order            30 passed  30 failed          0.20s  ok
fused_rmsnorm_kernel       -          -                  -      UNCHECKED  no cuda device on this machine
grad_accumulation          27 passed  27 failed          1.70s  ok
kv_cache_equivalence       21 passed  21 failed          0.25s  ok
online_softmax_attention   57 passed  57 failed          0.29s  ok
quantization_error_bounds  65 passed  65 failed          0.29s  ok
sharded_dataloader         53 passed  53 failed          0.22s  ok
softmax_stability          29 passed  29 failed          0.26s  ok

8 task(s) validated: reference passes, starter fails cleanly.
1 task(s) not checked here: fused_rmsnorm_kernel
```

Three specific things need hardware to settle, and they are listed so nobody
has to rediscover them:

1. that the reference passes at all — Triton's type promotion, the
   `out_ptr.dtype.element_ty` cast on the store, and a `range` over a runtime
   bound are all things I have written from the documented semantics and not
   executed;
2. that the untouched starter fails with test failures rather than a collection
   error. Its kernel body raises `NotImplementedError`, which Triton will hit
   at compile time rather than at import time, so the failure arrives inside a
   test where pytest counts it as a failure — that is the expectation, not an
   observation;
3. that the five mutants are caught.

## The accelerated CI job

A second job in `ci.yml`, manual (`workflow_dispatch` with an input) and
pinned to a `[self-hosted, gpu]` runner, because GitHub's hosted runners have
no NVIDIA device.

The step that matters is the one before the validation: it asserts
`torch.cuda.is_available()` and exits if not. Without it the job is worse than
useless — `validate` prints UNCHECKED and exits zero on a machine with no
accelerator, so a runner that quietly lost its GPU would produce a green tick
against a task nothing had run.

The laptop job is untouched and still defaults to the laptop tier, which is why
the new task cannot break it.

Verification level: **L0**. No such runner exists yet and this has never
executed.

## The bug this session found in last session's work

Adding a real accelerated task immediately produced two false statements from
code written last session, neither of which could exist while the tier had no
tasks in it:

```
accelerated  0% (0/1)  ·  1 not run: no accelerator on this machine
...
attention 100%  data 100%  kernels 0%  numerics 100%  tokenization 100%  training 100%
```

and, from `validate`, `9 task(s) validated` when eight had been.

`needs_accelerator` was being kept out of the *verdict* and folded into the
*rates*. A tier where nothing ran reported 0%, a category where nothing ran
reported 0%, and the summary counted a row it had just labelled UNCHECKED. Now
outcomes with no evidence are excluded from every rate, a tier with nothing
measured has a `pass_rate` of `null` rather than zero, and both counts are
counted rather than inferred:

```
accelerated  not run    (1 task(s): no accelerator on this machine)
laptop       100% (8/8)
4.5s wall clock  ·  1 attempt(s) per task
attention 100%  data 100%  numerics 100%  tokenization 100%  training 100%
```

## Verified this session

| claim | level | evidence |
|---|---|---|
| Harness suite passes | L1 | `python -m pytest -q` → 56 passed in 16.64s |
| Eight laptop tasks still validate | L2 | `validate --tier all` output pasted above |
| Reference run unaffected | L2 | `run --model reference --tasks all --tier all` → laptop 100% (8/8), accelerated not run |
| Adapter contract (prompt, allowlist, cost) | L1 | eight offline tests, in the suite above |
| `attempts` and `usd_cost` reach the results file | L2 | a run with a fake adapter writes `attempts: 3`, `usd_cost: 1.25` |
| Kernel task files parse | L0 | `py_compile` on reference, starter and hidden tests |
| Kernel task is correct | **none** | no CUDA device here; `frozen_set: unvalidated` |
| Anthropic adapter against the live API | **none** | no request was made this session |
| Accelerated CI job | L0 | written, never executed, no runner exists |

## Next

Run the kernel task on hardware — validation, then the mutation pass — and
either fix what it finds or move it into the frozen set. Then the leaderboard
entry that a real model run produces, which is the first number this repository
will publish about somebody else's work.
