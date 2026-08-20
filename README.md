# scratchbench

[![ci](https://github.com/selimfedakar/scratchbench/actions/workflows/ci.yml/badge.svg)](https://github.com/selimfedakar/scratchbench/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Can the model actually implement it?**

Machine learning machinery — attention masks, KV-caches, gradient accumulation, quantization, a fused Triton kernel, a Metal threadgroup reduction — written from a written specification and graded by hidden pytest suites. On the laptop you already own. Under five minutes a task. No GPU, no Kaggle account, no cloud bill.

```
$ scratchbench run --model claude-haiku-4-5 --tasks softmax_stability
  softmax_stability ... pass  (29 passed)

claude-haiku-4-5  ·  2026-08-03T05:43:14+00:00  ·  task set v1

task               category  tier    diff  result  time   tests
-----------------  --------  ------  ----  ------  -----  ---------
softmax_stability  numerics  laptop  1     pass    0.34s  29 passed

laptop       100% (1/1)
10.3s wall clock  ·  1 attempt(s) per task  ·  $0.01
```

A model was handed a specification, wrote a file, and twenty-nine tests it never saw agreed. Ten seconds and eight tenths of a cent.

---

## The gap this fills

I like MLE-bench. I like KernelBench. I could not run either of them.

MLE-bench asks agents to place in Kaggle competitions: real datasets, real training runs, hours of compute per task. KernelBench wants an NVIDIA GPU. SWE-bench and the harnesses around it grade something else entirely — fixing issues in an existing repository, which is a different skill from building the thing in the first place.

So there is a gap, and it is a specific one. Nobody is measuring whether a model can implement the *machinery* of machine learning — the causal mask, the KV-cache, the gradient accumulation, the quantization round-trip — correctly, from a specification, with nothing to copy from.

And here is the part that makes closing it possible: **you do not need a training run to grade an implementation.** You need tests. A wrong attention mask fails a shape-and-values assertion in eleven milliseconds. The loss curve was never the referee.

## The rule everything else follows from

> **Every task must be gradeable on a laptop, without a GPU, in under five minutes.**

That single constraint is the whole design:

- **Anyone can reproduce the leaderboard.** If you cannot re-run a benchmark, you are trusting a number, and trusting numbers is how benchmarks rot.
- **Grading is objective.** Tests pass or they do not. No judge model, no rubric, no argument.
- **A full sweep costs tens of dollars, not thousands** — so it can be re-run the day a new model ships, not six months later.

Thirteen tasks, 507 hidden tests, and the entire reference sweep finishes in **sixteen seconds**.

## The first rows

**Laptop tier**, five independent draws each, one attempt per task, both models at their own defaults:

| Model | Pass rate | Draws | Solved | Cost per draw | Wall clock |
|---|---:|---:|---:|---:|---:|
| `claude-opus-5` | **100%**, every draw | 5 | 40/40 | $0.49 to $0.56 | 187 to 215s |
| `claude-haiku-4-5` | **38% to 50%** | 5 | 18/40 | $0.059 to $0.077 | 62 to 90s |

Read that first row as a limit as much as a result: **the laptop tier has stopped measuring at the top.** The sampling is real and it shows up in the same files — the prompt is byte-identical each time so the input is fixed at 15858 tokens, while the output ranges over 16476 to 19070, a 16% spread in what the model actually wrote. Five materially different answers, one verdict every time. A ceiling hit once is a good day; a ceiling hit five times out of five is a ceiling.

**The accelerated tier has not stopped measuring.** One Triton task, same model, same day:

| Model | Pass rate | Draws | Solved | Cost per draw |
|---|---:|---:|---:|---:|
| `claude-opus-5` | **20%** | 5 | 1/5 | $0.049 to $0.061 |
| `claude-haiku-4-5` | **60%** | 5 | 3/5 | $0.008 |

That is not Haiku beating Opus — five draws of one task cannot tell 20% from 60%. It is what sits underneath: **Opus fails one test out of twenty-four, the same one every time; Haiku, when it fails, fails all twenty-four.** A working kernel with an edge missed and a kernel that does not run are the same word under binary scoring and completely different results. The edge is that Triton specializes an integer argument whose value is `1` into a compile-time constant, so `n_cols.to(tl.float32)` compiles at every row width except a single-column one — which `prompt.md` licenses explicitly.

Every results file is checked in at [`leaderboard/`](leaderboard/) and every cost re-derives from the tokens beside it. The whole table cost $3.30. What all of it implies about the next task set is [`docs/V2_DESIGN.md`](docs/V2_DESIGN.md).

## Run it

```bash
git clone https://github.com/selimfedakar/scratchbench
cd scratchbench
pip install -e .

scratchbench validate                          # every task, both halves of the check
scratchbench run --model reference --tasks all # the control: known-correct solutions
scratchbench report                            # your results, next to everyone else's
```

To grade a real model, add the SDK and a key:

```bash
pip install 'scratchbench[anthropic]'
export ANTHROPIC_API_KEY=...
scratchbench run --model claude-opus-5 --tasks all
```

Start with `--model reference`, though, and not out of politeness. A harness that has never been proven against a correct solution will happily report that a model failed when what actually failed was the harness — and a benchmark gets exactly one chance to publish a wrong number about somebody else's work.

## What "verified" means here, and why it is two things

Every task in this repository has to clear a bar with two halves:

1. the reference solution passes the hidden tests, **and**
2. the untouched starter fails them with real assertion errors — not import errors, not collection errors.

Half of that pair is worthless. A suite that passes against a correct solution proves nothing if it also passes against an empty one, and an import error and a wrong answer score identically while meaning completely different things.

```
$ scratchbench validate --tier all
task                          reference  untouched starter  time   verdict
----------------------------  ---------  -----------------  -----  -----------------------------------------
activation_checkpointing_rng  34 passed  34 failed          1.31s  ok
attention_causal_mask         26 passed  26 failed          0.28s  ok
bpe_merge_order               30 passed  30 failed          0.20s  ok
flash_attention_backward      49 passed  49 failed          1.00s  ok
fused_rmsnorm_kernel          -          -                  -      UNCHECKED  no cuda device on this machine
grad_accumulation             27 passed  27 failed          1.61s  ok
kv_cache_equivalence          21 passed  21 failed          0.30s  ok
metal_cross_entropy_kernel    68 passed  68 failed          1.41s  ok
online_softmax_attention      57 passed  57 failed          0.31s  ok
quantization_error_bounds     65 passed  65 failed          0.29s  ok
sharded_dataloader            53 passed  53 failed          0.22s  ok
softmax_stability             29 passed  29 failed          0.26s  ok
speculative_decoding_verify   24 passed  24 failed          4.72s  ok

12 task(s) validated: reference passes, starter fails cleanly.
1 task(s) not checked here: fused_rmsnorm_kernel
```

That last line is the tier separation working rather than a gap: this machine has Metal and no CUDA, so the Triton task is `UNCHECKED` instead of counted.

There is a third check that does not fit in a table, and it is the one that has found the most: **the reference gets mutated on purpose.** Passing tests prove the reference is right; only mutants prove the tests would catch anything else. Every wrong version a real implementation would plausibly produce — the off-by-one, the wrong axis, the missing rescale — gets written and run, and the suite has to catch each one.

That check has changed this repository three times, and [`docs/LESSONS.md`](docs/LESSONS.md) has all three. The shortest version of why it exists: `sharded_dataloader` passed fifty-one tests, and a wrong implementation that gave each rank a contiguous slab instead of every `world_size`-th sample passed all fifty-one of them too.

## Two claims, and the receipts

**A GPU task does not get into the frozen set on the strength of its directory listing.** `fused_rmsnorm_kernel` — a fused RMSNorm forward pass in Triton, one program per row, float32 accumulation under a float16 input — was written on a laptop with no CUDA device and marked `frozen_set: unvalidated` until it had run on hardware. When it finally did, on a rented RTX A4500, it came back with two real defects: a mutant that survived, and two tests that could not fail. Both are fixed, and the run that says so is re-runnable:

```
$ bash tools/verify_accelerated.sh
=== 1. hardware ===
NVIDIA RTX A4500 | torch 2.8.0+cu128 | triton 3.4.0
...
fused_rmsnorm_kernel  24 passed  24 failed          6.71s  ok
...
all 6 mutants caught
```

The mutant that survived the first time was the one the task is named after — a float16 accumulator instead of a float32 one. The test written to catch it was arithmetic about an implementation I was not running. [`docs/LESSONS.md`](docs/LESSONS.md) L14 is the whole story, and the reason the accelerated tier exists at all is L10.

**A cost on a leaderboard has to be reproducible, not plausible.** Every results file carries the tokens the price was computed from — input, output, cache read, cache write — because a stale price table and a run that used more tokens than expected produce the same figure, and afterwards nothing tells them apart. Not for you and not for me. That gap was found by trying to check my own number and discovering I had not kept the means to.

```
$ python tools/check_cost.py
ok           claude-haiku-4-5-20260803.json  12103 in + 11947 out = $0.071838
ok           claude-opus-5-20260803.json  15858 in + 18453 out = $0.540615
```

CI runs that on every push, because a check done once by hand in a README is a check that stops running the day it is written.

## How a task works

Each task hands the model a specification and a stub. It never sees the tests.

```
tasks/attention_causal_mask/
  meta.yaml          category, difficulty, time limit, what it probes
  prompt.md          the specification the model receives
  starter/           the files it edits — signatures fixed, bodies empty
  hidden_tests/      never shown to the model; the only thing that decides the score
  reference/         a correct solution, for maintainers only
```

Scoring is binary per task: every hidden test passes, or the task is failed. Partial credit hides exactly the failures worth knowing about.

The graded environment is pinned — one thread, fixed hash seed, no inherited `PYTHONPATH` — because numpy and torch change the *order* of their floating-point reductions with the number of cores available, and a task that flips from pass to fail depending on which machine graded it is a benchmark quietly lying in the direction that is hardest to notice.

See [`TASK_FORMAT.md`](TASK_FORMAT.md) for the full contract, and [`CONTRIBUTING.md`](CONTRIBUTING.md) if you want to add a task.

## The task set

| Task | Set | Category | Difficulty | Tests | What it actually probes |
|---|---|---|:---:|---:|---|
| `softmax_stability` | v1 | numerics | 1 | 29 | the max-subtraction, and what happens at ±1e4 |
| `bpe_merge_order` | v1 | tokenization | 2 | 30 | merge ranks applied in order, round-trip exactness |
| `attention_causal_mask` | warmup | attention | 2 | 26 | masking, and what a fully masked row returns |
| `kv_cache_equivalence` | v1 | attention | 3 | 21 | token-by-token decoding equals one pass, with RoPE |
| `grad_accumulation` | v1 | training | 3 | 27 | N micro-batches equal one big batch, in autograd |
| `sharded_dataloader` | warmup | data | 4 | 53 | stride not slab, disjoint ranks, resumable mid-epoch |
| `quantization_error_bounds` | warmup | numerics | 4 | 65 | per-channel scales, clipping, the error you promised |
| `online_softmax_attention` | v1 | attention | 5 | 57 | tiled attention with a running max and rescale |
| `flash_attention_backward` | v2 | attention | 5 | 49 | a derived backward, and the row term that is not blockwise |
| `fused_rmsnorm_kernel` ⚡ | v2 | kernels | 4 | 24 | a real Triton reduction, not a PyTorch one-liner |
| `metal_cross_entropy_kernel` ⚡ | v2 | kernels | 4 | 68 | a Metal threadgroup reduction at a group size it does not choose |

**`v1` is five laptop tasks and it was eight, plus the CUDA kernel, when the rows above were measured.** `attention_causal_mask`, `sharded_dataloader` and `quantization_error_bounds` moved to `warmup` on 2026-08-13, on the draws already published rather than on new ones: three models, six draws each for two of them, and only Haiku ever failed one. A task that everything solves is not wrong, it has stopped measuring, and the rule that says so is below. `fused_rmsnorm_kernel` moved the other way on 2026-08-17, out of `v1` and into `v2`, on its own published draws: Opus solves 1 of 5, which is the lowest frontier rate in the repository.

⚡ = `accelerated` tier: needs a GPU — CUDA for the Triton task, Metal for the other, which is to say any Apple silicon Mac. Reported beside the laptop rate and **never folded into it**. On a machine without the hardware it comes back `needs_accelerator` — not a pass, not a failure, an absence of evidence, and it stays out of every percentage rather than being rounded down to zero.

## The next set has to earn its place

Every difficulty number in the table above was assigned by how hard the task felt to write, before a single model had been asked. Then a model was asked, and Claude Opus 5 solved all eight laptop tasks in every one of five draws. Those numbers were a measurement of the author.

So from `v2` onward a task carries what models actually scored on it, and the rule is mechanical rather than editorial:

```yaml
calibration:
  - model: claude-opus-5
    draws: 15
    passed: 15
    date: 2026-08-09
```

**A task the top of the field clears cannot be in a numbered set.** The loader refuses it — not a warning, a `TaskError` — because a task that the frontier never fails costs a sweep and returns no information. It goes to `frozen_set: warmup`, where it still separates a small model from a large one and stays out of the headline. Concretely: a numbered set needs two calibration entries of five draws or more, and the highest two pass rates among them must not both be 100%. `TASK_FORMAT.md` has the table.

That rule read *one* entry until 2026-08-17 — the single best model — and this is what happened to the first three tasks written against it:

| Task | Set | Tests | Opus 5 | Sonnet 5 | Haiku 4.5 | What it probes |
|---|---|---:|---:|---:|---:|---|
| `speculative_decoding_verify` | warmup | 24 | 15/15 | 10/10 | 10/20 | accept-and-correct, graded on the output distribution |
| `flash_attention_backward` | **v2** | 49 | 15/15 | **8/10** | 3/19 | a derived backward, and the row term that is not blockwise |
| `activation_checkpointing_rng` | warmup | 34 | 10/10 | 10/10 | 1/10 | recomputation that puts the generator back where it found it |

All three were refused. Two of them deserved it: Opus and Sonnet both sweep them, and a task two frontier models clear has stopped measuring the frontier. The middle one did not. Sonnet loses two draws of ten to it, which makes it the only task on the laptop tier that separates one frontier model from another — and the rule threw it out on Opus's account, because it equated *the top of the field* with *the strongest model in it*. Reading the top two entries instead moves exactly one task in the whole repository, which is that one. The uncomfortable part — that a rule was rewritten after it refused something its author liked, and that an earlier journal had already decided the other way — is written up as `docs/LESSONS.md` L35 rather than smoothed over here.

Every draw behind those numbers is checked in under [`calibration/`](calibration/) and [`leaderboard/`](leaderboard/), and `tools/check_calibration.py` re-derives all twenty-three entries from them on every push. A figure that decides whether a task is allowed into a frozen set is not allowed to be the one figure nobody can reproduce.

All three were built the way the design document asked for: the obvious answer forbidden, a property that is provable rather than plausible, mechanism that composes. `speculative_decoding_verify` never states the acceptance rule, only the guarantee it exists to provide, and grades the emitted tokens against the target distribution over two hundred thousand draws. `flash_attention_backward` hands the graded function one block of keys and never the rest, so the correction term that belongs to a whole query row cannot be accumulated from the block in hand. `activation_checkpointing_rng` compares generator state byte for byte, so an implementation that recomputes correctly and leaves the random stream rewound fails while its gradients are perfect. All three catch every wrong implementation written against them — twenty-eight mutants, twenty-eight caught.

And Opus 5 passed all three, forty draws out of forty. The other two did not, and the *shape* of how they failed is the part worth reading. On `speculative_decoding_verify`, Haiku's wrong versions run, accept at plausible rates, and emit plausible tokens — what fails is the distribution those tokens are drawn from, which no comparison against a reference array on one input would ever have noticed. On `flash_attention_backward` the three models fail at three different depths:

| | fails | what the failure is |
|---|---:|---|
| `claude-opus-5` | 0 of 15 | nothing |
| `claude-sonnet-5` | 2 of 10 | the decode-time query offset, the same two tests both times |
| `claude-haiku-4-5` | 16 of 19 | the row correction, or the `1/sqrt(d)` scale |

Sonnet's `2 failed, 47 passed` is the signature of a mutant written for this task weeks before Sonnet was asked anything — *queries not put at the end of the key range*. A real model reproduced one of the invented wrong implementations, exactly, twice.

### The first task the rule let in on its own numbers

`metal_cross_entropy_kernel`, written on 2026-08-13 against the corrected criterion — the difficulty is a fact about a tool, and an obscure one — is the first task in this repository that a frontier model fails.

| Model | Passed | Cost per draw |
|---|---:|---|
| `claude-opus-5` | **8 / 10** | $0.078 to $0.108 |
| `claude-sonnet-5` | **4 / 10** | $0.033 to $0.091 |
| `claude-haiku-4-5` | **1 / 9** | $0.006 to $0.117 |

Three models, three rates, in the order the models are usually put in — which no other single task here manages. The kernel is a Metal source string compiled at runtime by `torch.mps.compile_shader`, so the tier that needed a rented GPU box now has half of itself reproducible on any Apple silicon Mac. The harness compiles the string and dispatches it, one threadgroup per row, at a threadgroup size the kernel never chooses: from 1 to 1024, narrower than the row or wider than it, not necessarily a power of two, not necessarily a multiple of the simdgroup width.

**And the failure has a name.** Every one of Opus's two failures and five of Sonnet's six are the same compile error, on the same identifier:

```
E   SyntaxError: program_source:39:14: error: cannot combine with previous 'type-name' declaration specifier
E           uint half = (n + 1u) / 2u;
E                ^
```

`half` is a type in Metal Shading Language, so it cannot name a variable — and the reduction underneath it is correct in every one of those draws. I made the identical mistake in the first draft of the reference, an hour before either model was asked ([`docs/LESSONS.md`](docs/LESSONS.md) L34). Haiku fails a third way: six draws call Metal functions that do not exist, and one wrote a fold whose stride never reaches zero, hung the GPU, and was killed by the task's own time limit.

That is exactly the difficulty L30 says survives, and exactly the difficulty L30 says is a moving target. This row is a measurement of one day. The full failure tables are in [`leaderboard/`](leaderboard/).

Forty out of forty is the result. Not a disappointment: it is the first number this repository has produced that it could not have produced by asking the author how hard something felt, and the tasks it refuses are ones their author spent a day on. What it says is in [`docs/LESSONS.md`](docs/LESSONS.md) L30 and it is uncomfortable — the property that makes the one discriminating task hard is not on the list these three were built from, and naming it correctly makes the laptop tier's remaining difficulty look like unfamiliarity rather than depth.

## Honesty about contamination

Any public benchmark leaks into the next training run. Pretending otherwise is worse than the leak.

So: **v1 is frozen and dated.** Every result records the task-set version and the date it was run. When contamination becomes visible in the numbers, a fresh held-out set ships and the old one stays published as a historical record. You will be able to watch the drift, which is more useful than a benchmark that quietly stops meaning anything.

## Honesty about everything else

- **Every row is one draw, not a converged number.** The same model on the same task passed one run and failed the next, four hours apart, same prompt, same settings. The harness is pinned to be deterministic; the model is sampled, and none of the pinning reaches it. Two models one task apart are not distinguishable, and this repository will not pretend otherwise until there are repeated runs. `--repeat N` runs N independent sweeps and `report --variance` prints how many of them each task passed; the two rows above predate it and have no error bar.
- **The headline rate is the laptop tier and nothing else.** Not an average over the tiers that happened to run: the accelerated tier is reported in full beside it, never inside it, and the results file names which tier the percentage came from.
- **One attempt per task.** A model that passes on the eleventh try is measuring the scaffolding, not the model. The number is written into every results file so nobody has to take it on trust. Repeated *independent* runs are a different thing from retries, and welcome.
- **No sampling, thinking or effort knobs are set.** Every model is asked at its own defaults, deliberately: a score at one effort setting and a score at another are not comparable, and nothing in a results file would say so.
- **No server-side fallbacks.** If the model that answers is not the model that was asked, the run raises rather than publishing one model's work under another's name.
- **[`docs/LESSONS.md`](docs/LESSONS.md) is the other half of this repository.** Thirty entries, first person, newest first: what I expected, what happened, and what it cost. Including the ones where a wrong number nearly shipped and something mechanical caught it. A benchmark that is wrong looks exactly like a benchmark that is right, which is why the mistakes are documented rather than quietly fixed.
- **A correctness benchmark cannot see a slow kernel.** `metal_cross_entropy_kernel` grades the answer, and a kernel that ignores its threadgroup entirely — one lane, a serial loop over the row, no reduction — passes all sixty-eight tests. The only assertion that would catch it is a wall-clock one, and those are banned here in every tier because timing on unknown hardware is not reproducible. The degenerate kernel is checked in as a mutant with `SURVIVES` as its expected verdict, so the hole is a re-runnable fact rather than a footnote, and all ten Opus draws wrote a real threadgroup reduction anyway. `docs/LESSONS.md` L33.

## What this repository does not contain

No course materials, no assignment code, no test suites from anyone's curriculum. Every task here is written from scratch for this repository. Inspiration is not redistribution, and the difference matters.

## Who is writing this

I am Ahmet Selim Fedakar, a computer science undergraduate in Los Angeles. I co-founded EXCAR and Auris, two AI voice startups, where I build the machine learning and speech systems; separately I have shipped two apps to the App Store on my own.

This came out of working through Stanford's CS336 alone, and one recurring question: when a model tells me my implementation is wrong, is it right?

Now there is a way to find out.

## License

MIT.

---

*Bir başka gün, bir başka yerde, bir başka zaman ve bir başka mekânda, tekrar görüşünceye kadar kendinize çok iyi bakın.*
