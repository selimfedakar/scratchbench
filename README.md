# scratchbench

**Can the model actually implement it?**

Machine learning tasks with hidden tests, graded by pytest, on your laptop, in under five minutes per task. No GPU. No Kaggle account. No cloud bill.

<!-- TODO(launch): leaderboard screenshot -->

---

## The problem with existing ML benchmarks

I like MLE-bench. I like KernelBench. I could not run either of them.

MLE-bench asks agents to place in Kaggle competitions, which means real datasets and real training runs — hours of compute per task, at Kaggle scale. KernelBench wants an NVIDIA GPU. SWE-bench and the agent harnesses around it grade something else entirely: fixing issues in existing repositories, which is a different skill from building the thing in the first place.

So there is a gap, and it is a specific one. Nobody is measuring whether a model can implement the *machinery* of machine learning — the attention mask, the RoPE rotation, the KV-cache, the gradient accumulation, the quantization round-trip — correctly, from a specification, with nothing to copy from.

That gap is worth closing, and here is the part that makes it possible: **you do not need a training run to grade an implementation.** You need tests. A wrong attention mask fails a shape-and-values assertion in eleven milliseconds. The loss curve was never the referee. The tests were.

## The rule everything else follows from

> **Every task must be gradeable on a laptop, without a GPU, in under five minutes.**

That single constraint is the whole design:

- **Anyone can reproduce the leaderboard.** If you cannot re-run a benchmark, you are trusting a number, and trusting numbers is how benchmarks rot.
- **Grading is objective.** Tests pass or they do not. No judge model, no rubric, no argument.
- **A full sweep costs tens of dollars, not thousands.** Which means it can be re-run on the day a new model ships, instead of six months later.

## Run it

```bash
git clone https://github.com/selimfedakar/scratchbench
cd scratchbench
pip install -e .

scratchbench validate                     # every task, both halves of the check
scratchbench run --model reference --tasks all
scratchbench report                       # your results, next to everyone else's
```

The model adapters are skeletons today, so `--model reference` is the run that
works: it puts the known-correct solutions through the same harness a model
would face. That is deliberate ordering rather than an omission. A harness that
has never been proven against a correct solution will happily report that a
model failed when what actually failed was the harness, and a benchmark only
gets one chance to publish a wrong number about somebody's model.

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

See [`TASK_FORMAT.md`](TASK_FORMAT.md) for the full contract, and
[`CONTRIBUTING.md`](CONTRIBUTING.md) if you want to add a task.

## Categories

| Category | Examples |
|---|---|
| `tokenization` | BPE merge order, special-token handling, round-trip exactness |
| `attention` | causal masking, GQA, RoPE, KV-cache correctness under a growing sequence |
| `training` | gradient accumulation equivalence, LR schedules, gradient checkpointing |
| `data` | deterministic sharded dataloaders, packing, dedup |
| `numerics` | mixed precision, quantization error bounds, softmax stability |
| `kernels` | Metal and Triton kernels checked against a reference implementation |

## Honesty about contamination

Any public benchmark leaks into the next training run. Pretending otherwise is worse than the leak.

So: **v1 is frozen and dated.** Every result records the task-set version and the date it was run. When contamination becomes visible in the numbers, a fresh held-out set ships and the old one stays published as a historical record. You will be able to see the drift, which is more useful than a benchmark that quietly stops meaning anything.

## What this repository does not contain

No course materials, no assignment code, no test suites from anyone's curriculum. Every task here is written from scratch for this repository. Inspiration is not redistribution, and the difference matters.

## Who is writing this

Selim Fedakar — computer science student in Los Angeles, co-founder and CTO of two hardware-and-AI companies, two apps live on the App Store. I have been working through Stanford's CS336 on my own, and the recurring question that produced this repository was a simple one: when a model tells me my implementation is wrong, is it right?

Now there is a way to find out.

## License

MIT.

---

*Bir başka gün, bir başka yerde, bir başka zaman ve bir başka mekânda, tekrar görüşünceye kadar kendinize çok iyi bakın.*
