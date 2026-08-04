# scratchbench

**Can the model actually implement it?**

Machine learning machinery — attention masks, KV-caches, gradient accumulation, quantization, a fused Triton kernel — written from a written specification and graded by hidden pytest suites. On the laptop you already own. Under five minutes a task. No GPU, no Kaggle account, no cloud bill.

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

Nine tasks, 332 hidden tests, and the entire reference sweep finishes in **under six seconds**.

## The first two rows

| Model | Pass rate | Solved | Cost | Wall clock | Attempts |
|---|---:|---:|---:|---:|---:|
| `claude-opus-5` | **100%** | 8/8 | $0.54 | 213.1s | 1 |
| `claude-haiku-4-5` | **38%** | 3/8 | $0.07 | 116.6s | 1 |

Laptop tier, task set v1, 2026-08-03, one attempt per task, both models at their own defaults. The results files are checked in at [`leaderboard/`](leaderboard/), and the costs reproduce from the tokens in them to the last digit.

Read that first row as a limit as much as a result: **v1 does not discriminate at the top.** It separates a small model from a large one cleanly and it cannot yet tell two large models apart. What that implies about the next task set is written up in [`leaderboard/README.md`](leaderboard/README.md).

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
$ scratchbench validate
task                       reference  untouched starter  time   verdict
-------------------------  ---------  -----------------  -----  -------
attention_causal_mask      26 passed  26 failed          0.27s  ok
bpe_merge_order            30 passed  30 failed          0.20s  ok
grad_accumulation          27 passed  27 failed          1.48s  ok
kv_cache_equivalence       21 passed  21 failed          0.26s  ok
online_softmax_attention   57 passed  57 failed          0.28s  ok
quantization_error_bounds  65 passed  65 failed          0.30s  ok
sharded_dataloader         53 passed  53 failed          0.23s  ok
softmax_stability          29 passed  29 failed          0.34s  ok
```

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

## The task set — v1, frozen

| Task | Category | Difficulty | Tests | What it actually probes |
|---|---|:---:|---:|---|
| `softmax_stability` | numerics | 1 | 29 | the max-subtraction, and what happens at ±1e4 |
| `bpe_merge_order` | tokenization | 2 | 30 | merge ranks applied in order, round-trip exactness |
| `attention_causal_mask` | attention | 2 | 26 | masking, and what a fully masked row returns |
| `kv_cache_equivalence` | attention | 3 | 21 | token-by-token decoding equals one pass, with RoPE |
| `grad_accumulation` | training | 3 | 27 | N micro-batches equal one big batch, in autograd |
| `sharded_dataloader` | data | 4 | 53 | stride not slab, disjoint ranks, resumable mid-epoch |
| `quantization_error_bounds` | numerics | 4 | 65 | per-channel scales, clipping, the error you promised |
| `online_softmax_attention` | attention | 5 | 57 | tiled attention with a running max and rescale |
| `fused_rmsnorm_kernel` ⚡ | kernels | 4 | 24 | a real Triton reduction, not a PyTorch one-liner |

⚡ = `accelerated` tier: needs CUDA, reported beside the laptop rate and **never folded into it**. On a machine without the hardware it comes back `needs_accelerator` — not a pass, not a failure, an absence of evidence, and it stays out of every percentage rather than being rounded down to zero.

## Honesty about contamination

Any public benchmark leaks into the next training run. Pretending otherwise is worse than the leak.

So: **v1 is frozen and dated.** Every result records the task-set version and the date it was run. When contamination becomes visible in the numbers, a fresh held-out set ships and the old one stays published as a historical record. You will be able to watch the drift, which is more useful than a benchmark that quietly stops meaning anything.

## Honesty about everything else

- **Every row is one draw, not a converged number.** The same model on the same task passed one run and failed the next, four hours apart, same prompt, same settings. The harness is pinned to be deterministic; the model is sampled, and none of the pinning reaches it. Two models one task apart are not distinguishable, and this repository will not pretend otherwise until there are repeated runs. `--repeat N` runs N independent sweeps and `report --variance` prints how many of them each task passed; the two rows above predate it and have no error bar.
- **The headline rate is the laptop tier and nothing else.** Not an average over the tiers that happened to run: the accelerated tier is reported in full beside it, never inside it, and the results file names which tier the percentage came from.
- **One attempt per task.** A model that passes on the eleventh try is measuring the scaffolding, not the model. The number is written into every results file so nobody has to take it on trust. Repeated *independent* runs are a different thing from retries, and welcome.
- **No sampling, thinking or effort knobs are set.** Every model is asked at its own defaults, deliberately: a score at one effort setting and a score at another are not comparable, and nothing in a results file would say so.
- **No server-side fallbacks.** If the model that answers is not the model that was asked, the run raises rather than publishing one model's work under another's name.
- **[`docs/LESSONS.md`](docs/LESSONS.md) is the other half of this repository.** Twenty-five entries, first person, newest first: what I expected, what happened, and what it cost. Including the ones where a wrong number nearly shipped and something mechanical caught it. A benchmark that is wrong looks exactly like a benchmark that is right, which is why the mistakes are documented rather than quietly fixed.
- **The `kernels` row promises Metal as well as Triton.** Only Triton exists today.

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
