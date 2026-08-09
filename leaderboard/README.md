# Leaderboard — task set v1

The laptop tier only. That is the rate anyone can reproduce with a clone, an
API key and about four minutes, which is the entire argument this repository is
making. The accelerated tier is reported separately below and never folded in.

Every row is one run, one attempt per task, at the model's own default
settings. The results file each row was read from is checked in beside this
page, so nothing here has to be taken on trust.

## Task set v1 — laptop tier, 8 tasks, 308 hidden tests

| Model | Pass rate | Draws | Solved | Cost per draw | Wall clock | Attempts | Runs |
|---|---:|---:|---:|---:|---:|---:|---|
| `claude-opus-5` | **100%**, in all five | 5 | 40/40 | $0.49 to $0.56 | 187 to 215s | 1 | [2026-08-09](claude-opus-5-20260809-draw1.json) ×5 |
| `claude-haiku-4-5` | **38%** | 1 | 3/8 | $0.07 | 117s | 1 | [2026-08-03](claude-haiku-4-5-20260803.json) |

The Opus row was a single draw until 2026-08-09 and is now five independent
sweeps: [1](claude-opus-5-20260809-draw1.json) ·
[2](claude-opus-5-20260809-draw2.json) ·
[3](claude-opus-5-20260809-draw3.json) ·
[4](claude-opus-5-20260809-draw4.json) ·
[5](claude-opus-5-20260809-draw5.json). The earlier single draw
([2026-08-03](claude-opus-5-20260803.json)) is still checked in; it agreed, at
$0.54 and 213.1s. The Haiku row is still one draw and is labelled as one.

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

**A row is a sample, and now some of it has an error bar.** The same model on
the same task passed one run and failed the next, four hours apart, with the
same prompt and the same settings. Everything in this harness is pinned to be
deterministic and none of that pinning reaches the model, which is sampled. So a
single sweep is one draw:

```bash
scratchbench run --model claude-opus-5 --tasks all --repeat 5
scratchbench report --variance
```

Each draw writes its own complete results file. They are independent runs, not
retries: no draw sees another's results and `max_attempts` is still 1.

```
claude-opus-5  ·  task set v1  ·  5 draw(s)  ·  laptop tier  ·  8 task(s) asked

task                       passed  spread
-------------------------  ------  ------
attention_causal_mask      5/5     always
bpe_merge_order            5/5     always
grad_accumulation          5/5     always
kv_cache_equivalence       5/5     always
online_softmax_attention   5/5     always
quantization_error_bounds  5/5     always
sharded_dataloader         5/5     always
softmax_stability          5/5     always

set pass rate: 100% in every draw
```

**Forty out of forty.** The sampling is real and it is visible in the same
files: the prompt is byte-identical every time, so all five draws billed exactly
15858 input tokens, while the output ran from 16476 to 19070 tokens — a 16%
spread in what the model wrote. Five materially different answers to each task,
and the same verdict every time.

That is the strongest statement this page can make about v1, and it is not a
compliment to the benchmark. A ceiling reached once is a good day; a ceiling
reached five times out of five, with the model visibly sampling underneath, is a
ceiling.

**The cost column is the one carrying the error bar.** $0.49 to $0.56 across the
five, entirely from output length, because input is fixed and output is not. A
single-draw cost is a 7% estimate presented as a figure, so this table reports
the range rather than one of the five.

**The Haiku row is still one draw.** Until it is five, do not read the gap
between it and anything else as anything but a gap between a sample and a range.

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
half a dollar, and does it five times out of five. That is a real result and it
is also a limit: **v1 does not discriminate at the top.** It separates a small
model from a large one clearly — 38% against 100% — and it cannot tell two large
models apart. The five draws move that sentence from a suspicion to a
measurement: there is no draw in which v1 finds anything to say about Opus 5.

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
