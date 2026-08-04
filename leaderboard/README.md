# Leaderboard — task set v1

The laptop tier only. That is the rate anyone can reproduce with a clone, an
API key and about four minutes, which is the entire argument this repository is
making. The accelerated tier is reported separately below and never folded in.

Every row is one run, one attempt per task, at the model's own default
settings. The results file each row was read from is checked in beside this
page, so nothing here has to be taken on trust.

## Task set v1 — laptop tier, 8 tasks, 308 hidden tests

| Model | Pass rate | Solved | Cost | Wall clock | Attempts | Run |
|---|---:|---:|---:|---:|---:|---|
| `claude-opus-5` | **100%** | 8/8 | $0.54 | 213.1s | 1 | [2026-08-03](claude-opus-5-20260803.json) |
| `claude-haiku-4-5` | **38%** | 3/8 | $0.07 | 116.6s | 1 | [2026-08-03](claude-haiku-4-5-20260803.json) |

308 rather than 332: the 24 tests in `fused_rmsnorm_kernel` belong to the
accelerated tier, which is reported below and is not part of this rate. The
header used to fold them in, which is the same mistake one line up from the
number it describes.

Cost is computed from the tokens in the results file and the published list
price. Both check out to the last digit: Opus 5 is
`(15858 × 5 + 18453 × 25) / 1e6 = 0.540615`, Haiku 4.5 is
`(12103 × 1 + 11947 × 5) / 1e6 = 0.071838`. That arithmetic is not a paragraph
any more, it is a command, and CI runs it on every push:

```bash
python tools/check_cost.py
ok           claude-haiku-4-5-20260803.json  12103 in + 11947 out = $0.071838
ok           claude-opus-5-20260803.json  15858 in + 18453 out = $0.540615
```

### Task by task

| Task | Difficulty | `claude-opus-5` | `claude-haiku-4-5` |
|---|:---:|:---:|:---:|
| `softmax_stability` | 1 | pass | fail (1 of 29) |
| `bpe_merge_order` | 2 | pass | fail (1 of 30) |
| `attention_causal_mask` | 2 | pass | pass |
| `kv_cache_equivalence` | 3 | pass | fail (19 of 21) |
| `grad_accumulation` | 3 | pass | fail (17 of 27) |
| `sharded_dataloader` | 4 | pass | pass |
| `quantization_error_bounds` | 4 | pass | pass |
| `online_softmax_attention` | 5 | pass | fail (15 of 57) |

The counts in parentheses are hidden tests failed. They are worth reading even
though scoring is binary: one failed test out of thirty is a convention the
model guessed differently, and nineteen out of twenty-one is a mechanism it did
not build.

## Accelerated tier — 1 task, 24 hidden tests

Reported beside the laptop rate, never inside it. Verified with
`tools/verify_accelerated.sh` on hardware rather than in CI.

| Task | Accelerator | Verified on | Reference | Untouched starter | Mutants |
|---|---|---|---:|---:|---:|
| `fused_rmsnorm_kernel` | cuda | NVIDIA RTX A4500, torch 2.8.0+cu128, triton 3.4.0 | 24 passed | 24 failed | 6 of 6 caught |

No model has been graded on it yet.

## What a row is, and what it is not

**A row is one draw, not a converged number.** The same model on the same task
passed one run and failed the next, four hours apart, with the same prompt and
the same settings. Everything in this harness is pinned to be deterministic and
none of that pinning reaches the model, which is sampled. So a single sweep is a
sample: two models one task apart on eight tasks are not distinguishable, and
nothing here should be read as though they were.

Both rows above are single draws, and that is now a gap with a command behind
it rather than a caveat:

```bash
scratchbench run --model claude-opus-5 --tasks all --repeat 5
scratchbench report --variance
```

Each draw writes its own complete results file — they are independent runs, not
retries, and no draw sees another's results. The per-task column that comes back
is `k/N`, which separates a task a model reliably solves from one it solves half
the time. Those two are the same row today.

**No row here has an error bar yet.** Until one does, treat any two rows within
a task or two of each other as indistinguishable.

**One attempt per task, and that is a measurement decision rather than a budget
one.** A harness that re-prompts with the test results until something passes is
reporting how good its own retry loop is. Repeated *independent* runs are a
different thing from retries and are welcome; retries within a run are not.

**No knobs.** Every model is asked at its own defaults: no sampling parameters,
no thinking configuration, no effort setting. A score measured at one effort
level and a score measured at another are not comparable, and nothing in a
results file would say so.

**Contamination is assumed, not denied.** v1 is frozen and dated. Every result
records the task-set version and the date it was run. When contamination shows
up in the numbers a fresh held-out set ships and this page stays as a historical
record.

## The first thing these two rows say about the benchmark

A frontier model solves the laptop tier completely, on the first attempt, for
half a dollar. That is a real result and it is also a limit: **v1 does not
discriminate at the top.** It separates a small model from a large one clearly —
38% against 100% — and it cannot yet tell two large models apart.

The next task set has to be harder, and "harder" here has a specific meaning
that the failures above point at. Haiku lost `kv_cache_equivalence` on nineteen
tests out of twenty-one and `grad_accumulation` on seventeen out of twenty-seven:
tasks where a mechanism either exists or does not. It lost `softmax_stability`
and `bpe_merge_order` on exactly one test each, which is a different kind of
miss. Difficulty that comes from more mechanism to get right is the direction;
difficulty that comes from more conventions to guess is not.

## Sending a result

Run the set and open a pull request with the results file. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Include the harness version, the task
set version, and the attempt count: a pass on the eleventh try is a different
result from a pass on the first, and this page says which one it is.
