# Leaderboard — task set v1

Every row is five independent draws unless it says otherwise. They are repeated
samples, not retries: no draw sees another's results and each task gets one
attempt. Every results file a row was computed from is checked in beside this
page, so nothing here has to be taken on trust.

Two tiers, reported separately and never folded together. `laptop` is the
headline: no GPU, reproducible by anyone with a clone and an API key.
`accelerated` needs the hardware it names.

## Laptop tier — 8 tasks, 308 hidden tests

| Model | Pass rate | Draws | Solved | Cost per draw | Wall clock | Attempts |
|---|---:|---:|---:|---:|---:|---:|
| `claude-opus-5` | **100%**, every draw | 5 | 40/40 | $0.49 to $0.56 | 187 to 215s | 1 |
| `claude-sonnet-5` | **100%**, every draw | 5 | 39/39 | $0.30 to $0.43 | 176 to 751s | 1 |
| `claude-haiku-4-5` | **38% to 50%**, mean 45% | 5 | 18/40 | $0.059 to $0.077 | 62 to 90s | 1 |

Files: opus [1](claude-opus-5-20260809-draw1.json)
[2](claude-opus-5-20260809-draw2.json) [3](claude-opus-5-20260809-draw3.json)
[4](claude-opus-5-20260809-draw4.json) [5](claude-opus-5-20260809-draw5.json) ·
sonnet [1](claude-sonnet-5-20260809-draw1.json)
[2](claude-sonnet-5-20260809-draw2.json)
[3](claude-sonnet-5-20260809-draw3.json)
[4](claude-sonnet-5-20260809-draw4.json)
[5](claude-sonnet-5-20260809-draw5.json) ·
haiku [1](claude-haiku-4-5-20260809-draw1.json)
[2](claude-haiku-4-5-20260809-draw2.json)
[3](claude-haiku-4-5-20260809-draw3.json)
[4](claude-haiku-4-5-20260809-draw4.json)
[5](claude-haiku-4-5-20260809-draw5.json). The earlier single draws from
2026-08-03 are still checked in and agreed: Opus at 100%, Haiku at 38%, which
was the bottom of the range it turned out to have.

Sonnet's fourth draw measured seven tasks rather than eight: its adapter never
answered on `online_softmax_attention`, and an absence of evidence is not a
failure, so it is out of both the numerator and the denominator. The draw is
still published and the table says the set rate came from four complete draws
rather than five. Its wall clock of 751 seconds is the same event seen from the
other side, and it is why that column is a range and not a figure.

**What the second 100% costs this page.** Two of the three models tried now clear
the v1 laptop tier completely, and the cheaper one does it for $0.30 a draw. The
row is worth having as a record and it is not a ranking: `docs/V2_DESIGN.md` §5
concludes from it that `attention_causal_mask`, `sharded_dataloader` and
`quantization_error_bounds` belong in the warm-up set with the three tasks
written on 2026-08-09, which is where a task goes when it has stopped
discriminating rather than when it is wrong.

**On 2026-08-13 those three moved**, on the strength of the draws already on this
page rather than new ones: six for Opus and Haiku, five for Sonnet, with the
counts written into each `meta.yaml` and re-derived from these files by
`tools/check_calibration.py`. So `v1` is five laptop tasks today and it was eight
when every row above was measured. The rows are unchanged and so are the files
behind them; what changed is which set the tasks belong to, and a sweep of
`--set v1` after that date is not comparable with the ones here.

### Task by task, k out of 5 draws

| Task | Difficulty | `claude-opus-5` | `claude-sonnet-5` | `claude-haiku-4-5` |
|---|:---:|:---:|:---:|:---:|
| `softmax_stability` | 1 | 5/5 | 5/5 | 0/5 |
| `bpe_merge_order` | 2 | 5/5 | 5/5 | 0/5 |
| `attention_causal_mask` | 2 | 5/5 | 5/5 | **3/5** |
| `kv_cache_equivalence` | 3 | 5/5 | 5/5 | 5/5 |
| `grad_accumulation` | 3 | 5/5 | 5/5 | 0/5 |
| `sharded_dataloader` | 4 | 5/5 | 5/5 | 5/5 |
| `quantization_error_bounds` | 4 | 5/5 | 5/5 | 5/5 |
| `online_softmax_attention` | 5 | 5/5 | 4/4 | 0/5 |

## Accelerated tier, Metal — `metal_cross_entropy_kernel`, 68 hidden tests

Verified on an **Apple M1 Pro**, macOS 15.5, torch 2.8.0, Metal 32023.883,
Python 3.10. Reference 68 passed, untouched starter 68 failed, thirteen mutants
each returning the verdict it was written to get. No rented hardware: the kernel
is a Metal source string compiled at runtime by `torch.mps.compile_shader`, so
anyone with an Apple silicon Mac can re-run every number in this section.

Ten independent draws per model, one attempt each.

| Model | Pass rate | Draws | Solved | Cost per draw | Wall clock | Attempts |
|---|---:|---:|---:|---:|---:|---:|
| `claude-opus-5` | **80%** | 10 | 8/10 | $0.078 to $0.108 | 31 to 48s | 1 |
| `claude-sonnet-5` | **40%** | 10 | 4/10 | $0.033 to $0.091 | 19 to 52s | 1 |
| `claude-haiku-4-5` | **11%** | 9 | 1/9 | $0.006 to $0.117 | 8 to 310s | 1 |

Files: opus [1](accelerated-metal-claude-opus-5-20260813-draw1.json)
[2](accelerated-metal-claude-opus-5-20260813-draw2.json)
[3](accelerated-metal-claude-opus-5-20260813-draw3.json)
[4](accelerated-metal-claude-opus-5-20260813-draw4.json)
[5](accelerated-metal-claude-opus-5-20260813-draw5.json)
[6](accelerated-metal-claude-opus-5-20260813-draw6.json)
[7](accelerated-metal-claude-opus-5-20260813-draw7.json)
[8](accelerated-metal-claude-opus-5-20260813-draw8.json)
[9](accelerated-metal-claude-opus-5-20260813-draw9.json)
[10](accelerated-metal-claude-opus-5-20260813-draw10.json) · sonnet
[1](accelerated-metal-claude-sonnet-5-20260813-draw1.json)
[2](accelerated-metal-claude-sonnet-5-20260813-draw2.json)
[3](accelerated-metal-claude-sonnet-5-20260813-draw3.json)
[4](accelerated-metal-claude-sonnet-5-20260813-draw4.json)
[5](accelerated-metal-claude-sonnet-5-20260813-draw5.json)
[6](accelerated-metal-claude-sonnet-5-20260813-draw6.json)
[7](accelerated-metal-claude-sonnet-5-20260813-draw7.json)
[8](accelerated-metal-claude-sonnet-5-20260813-draw8.json)
[9](accelerated-metal-claude-sonnet-5-20260813-draw9.json)
[10](accelerated-metal-claude-sonnet-5-20260813-draw10.json) · haiku
[1](accelerated-metal-claude-haiku-4-5-20260813-draw1.json)
[2](accelerated-metal-claude-haiku-4-5-20260813-draw2.json)
[3](accelerated-metal-claude-haiku-4-5-20260813-draw3.json)
[4](accelerated-metal-claude-haiku-4-5-20260813-draw4.json)
[5](accelerated-metal-claude-haiku-4-5-20260813-draw5.json)
[6](accelerated-metal-claude-haiku-4-5-20260813-draw6.json)
[7](accelerated-metal-claude-haiku-4-5-20260813-draw7.json)
[8](accelerated-metal-claude-haiku-4-5-20260813-draw8.json)
[9](accelerated-metal-claude-haiku-4-5-20260813-draw9.json)
[10](accelerated-metal-claude-haiku-4-5-20260813-draw10.json).

Haiku's tenth draw came back `adapter_error` — the API returned `overloaded` —
so it measured nothing and is in neither the numerator nor the denominator. Its
rate is nine draws and the table says so.

**This is the first task in the repository a frontier model fails**, and the
three rates are ordered the way the models are, which no other single task here
does. It is also the reason the `kernels` category no longer promises Metal
without delivering it.

### The failures, by name rather than by count

| | draws that failed | what happened |
|---|---:|---|
| `claude-opus-5` | 2 of 10 | both `uint half = ...`, which does not compile |
| `claude-sonnet-5` | 6 of 10 | five the same `half`, one a power-of-two fold that drops entries |
| `claude-haiku-4-5` | 8 of 9 | six call Metal functions that do not exist, one loops forever, one is wrong |

`half` is a type in Metal Shading Language, the 16-bit float, so it cannot name
a variable. Every model that reached for it was writing the same correct
reduction — fold the live range in half, hence the name — and the compiler
rejected the source before a single test ran:

```
E   SyntaxError: program_source:39:14: error: cannot combine with previous
    'type-name' declaration specifier
E           uint half = (n + 1u) / 2u;
E                ^
```

Sonnet's one draw that compiled and still failed is `25 failed, 43 passed`, with
`for (uint offset = tpg >> 1; offset > 0; offset >>= 1)` as its fold: the
power-of-two halving that silently drops entries whenever the threadgroup size
is not a power of two. That implementation is mutant three in
`tools/mutate_metal_task.py`, written before any model was asked, and its
failures land on the same tests the mutant's do.

Haiku fails a third way, and it is the same way it failed the CUDA task: the
kernels call things that are not there — `simdgroup_max`, `atomic_fetch_max` on
a float, `atomic_compare_exchange_weak_explicit` with the wrong argument types.
One draw wrote `for (uint step = (tpg + 1) / 2; step > 0; step = (step + 1) / 2)`,
which never reaches zero, hung the GPU, and was killed by the task's own 300
second limit — the only `timeout` this repository has ever recorded.

**What that means for the number.** The signal separating these three models
today is dominated by one reserved word, and a model that learns it moves
several draws up without having become better at threadgroup reductions. The
rest of the difficulty is real — the group size the kernel does not choose, the
idle lanes, the fold over a range that is not a power of two, the shift that
keeps a vocabulary row finite — but this row is a measurement of 2026-08-13 and
it will not age well. That is what `docs/LESSONS.md` L30 predicted about
designing against unfamiliarity, now with a number attached.

## Accelerated tier, CUDA — 1 task, 24 hidden tests

Verified on an **NVIDIA RTX 4000 Ada Generation**, torch 2.8.0+cu128,
triton 3.4.0, Linux 6.8, Python 3.12. Reference 24 passed, untouched starter 24
failed, six mutants out of six caught — the second distinct card this task has
been proved on, after an RTX A4500.

| Model | Pass rate | Draws | Solved | Cost per draw | Wall clock | Attempts |
|---|---:|---:|---:|---:|---:|---:|
| `claude-opus-5` | **20%** | 5 | 1/5 | $0.049 to $0.061 | 24 to 29s | 1 |
| `claude-haiku-4-5` | **60%** | 5 | 3/5 | $0.008 | 12 to 17s | 1 |

Files: opus [1](accelerated-claude-opus-5-20260809-draw1.json)
[2](accelerated-claude-opus-5-20260809-draw2.json)
[3](accelerated-claude-opus-5-20260809-draw3.json)
[4](accelerated-claude-opus-5-20260809-draw4.json)
[5](accelerated-claude-opus-5-20260809-draw5.json) · haiku
[1](accelerated-claude-haiku-4-5-20260809-draw1.json)
[2](accelerated-claude-haiku-4-5-20260809-draw2.json)
[3](accelerated-claude-haiku-4-5-20260809-draw3.json)
[4](accelerated-claude-haiku-4-5-20260809-draw4.json)
[5](accelerated-claude-haiku-4-5-20260809-draw5.json).

**Do not read that table as Haiku beating Opus.** Five draws of one task is a
tiny sample and the two rates are not distinguishable. The rows are here because
of what is underneath them, which is not a rate at all.

### The failures are opposite, and the verdict column cannot show it

| | draws that failed | tests failed, out of 24 |
|---|---:|---|
| `claude-opus-5` | 4 of 5 | **1** every time |
| `claude-haiku-4-5` | 2 of 5 | 24, and 23 |

Opus writes a kernel that works. Four failures out of four are the same single
test, `test_a_single_column`, captured by name four separate times. Haiku, when
it fails, fails everything: the kernel does not run.

Under binary scoring both of those are the word `FAIL`, and they are completely
different events. One is an engineer who shipped something that works and missed
an edge; the other is code that does not compile into a working kernel.

### The edge Opus misses

```
E   triton.compiler.errors.CompilationError: at 26:23:
E       mean_sq = sum_sq / n_cols.to(tl.float32)
E                          ^
E   AttributeError("'int' object has no attribute 'to'")
```

Triton specializes integer kernel arguments whose value is `1` into compile-time
constants. For every row width but one, `n_cols` arrives as a runtime scalar and
`.to(tl.float32)` is valid; at `N = 1` it arrives as a Python `int`, which has
no `.to`, and the kernel fails to compile. The reference divides by `n_cols`
directly and never meets the problem.

The task's `prompt.md` licenses the case in as many words: *"M may be zero and
**N is at least one**."* A hidden test that can fail for a reason the prompt
never states is a broken task, and this one is stated.

That is the shape of difficulty the next task set is being designed around: not
more conventions to guess, but a mechanism where the fluent answer and the
correct answer differ.

## What a row is, and what it is not

**A row is a sample.** The harness is pinned to be deterministic and none of
that pinning reaches the model, which is sampled. Opus's laptop draws billed
exactly 15858 input tokens every time and produced between 16476 and 19070
output tokens, a 16% spread in what it actually wrote, and still scored 40 out
of 40. The score converged; the process did not.

**The cost column carries the error bar, not the score.** Input is fixed and
output is not, so cost inherits all of the model's variance. A cost quoted to
the cent from one draw is a 7% estimate presented as a figure, which is why
every cost here is a range.

**One attempt per task, and that is a measurement decision.** A harness that
re-prompts with the test results until something passes is reporting how good
its own retry loop is. Repeated independent runs are a different thing and are
what this page is made of.

**No knobs.** Every model is asked at its own defaults: no sampling parameters,
no thinking configuration, no effort setting. A score measured at one effort
level and a score measured at another are not comparable, and nothing in a
results file would say so.

**Contamination is assumed, not denied.** v1 is frozen and dated. Every result
records the task-set version and the date. When contamination shows up in the
numbers a fresh held-out set ships and this page stays as a historical record.

## What these rows say about the benchmark

**The laptop tier has stopped measuring at the top.** Forty tasks asked of
Opus 5, forty passed, in every one of five draws, with the model visibly
sampling underneath. A ceiling reached once is a good day. A ceiling reached
five times out of five is a ceiling.

**The accelerated tier has not.** One task, and it puts a frontier model at 1
out of 5. The kernels category was the last one added, was nearly cut for
violating the laptop constraint, and is now the only part of this benchmark
still asking a question Opus 5 cannot answer.

**And the useful signal was never the rate.** Haiku loses
`kv_cache_equivalence` and `grad_accumulation` outright — a mechanism it does
not have — while losing `softmax_stability` and `bpe_merge_order` by one test —
a convention it guessed differently. Opus loses one test out of twenty-four on a
kernel it otherwise wrote correctly. Three different things, one word.
Difficulty that comes from more mechanism to build is the direction;
difficulty that comes from more conventions to guess is not.

The design that follows from all of this is [`../docs/V2_DESIGN.md`](../docs/V2_DESIGN.md).

## Cost, reproduced rather than trusted

Every figure on this page re-derives from the token counts in the file beside
it, and CI checks it on every push:

```
$ python tools/check_cost.py
...
22 file(s) checked: every cost reproduces from its own tokens.
```

The whole of this page cost **$3.30**: $2.64 for Opus on the laptop tier, $0.34
for Haiku, $0.28 and $0.04 for the two accelerated sets.

## Sending a result

Run the set and open a pull request with the results files. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Send more than one draw if you can: the
per-task `k/N` column is the difference between a task a model reliably solves
and one it solves half the time, and a single run cannot tell them apart.
