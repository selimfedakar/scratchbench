# Task set v2 — design

v1 is frozen and stays published. This is the design for the set that replaces
it as the headline, and the argument for why each part of it is shaped the way
it is.

It was written before the tasks, so that the tasks would be written against a
stated criterion rather than against my sense of hard, which is the mistake v1
was calibrated with (`LESSONS.md` L21). Three of them have since been written and
all three were refused by the rule in section 4. The document is updated in place
rather than rewritten, and the sections that turned out to be wrong say so:
**2.0** is the property the criterion was missing, **3** is a structural claim no
test could have kept (L28), and **5** is what v2 actually contains, which is
nothing.

---

## 1. What v1 measured, and where it stopped

The first full sweep, 2026-08-03, laptop tier, one attempt per task:

| Model | Pass rate | Cost |
|---|---:|---:|
| `claude-opus-5` | 100% (8/8) | $0.54 |
| `claude-haiku-4-5` | 38% (3/8) | $0.07 |

That is a working benchmark and a real result. It is also the ceiling: a set
that a frontier model clears completely, first attempt, has stopped measuring
frontier models. It still separates a small model from a large one, and it will
keep doing that as a historical record, but it cannot rank the top.

The useful signal is not the rate, it is the **shape of the failures**. Haiku
lost four tasks in two clearly different ways:

| Task | Tests failed | What the failure was |
|---|---:|---|
| `kv_cache_equivalence` | 19 of 21 | the mechanism was not built |
| `grad_accumulation` | 17 of 27 | the mechanism was not built |
| `online_softmax_attention` | 15 of 57 | the mechanism was partly built |
| `softmax_stability` | 1 of 29 | a convention was guessed differently |
| `bpe_merge_order` | 1 of 30 | a convention was guessed differently |

A near-total failure means the model did not have the thing the task is about.
A single failed test out of thirty means it had the thing and disagreed about
an edge the prompt underspecified. Only the first kind of difficulty is worth
scaling, and the distinction is the whole design input for v2:

> **Difficulty must come from mechanism to build, not from conventions to
> guess.** More conventions make a task longer and its score noisier. More
> mechanism makes it harder.

### 1b. One task already discriminates, and it is on the other tier

Written after the first accelerated model runs, which changed the shape of this
document.

`fused_rmsnorm_kernel` — a fused RMSNorm forward pass in Triton, 24 hidden
tests, the `kernels` category that was nearly cut for violating the laptop
constraint — puts `claude-opus-5` at **1 out of 5 draws** on the same day the
laptop tier put it at 40 out of 40.

More useful than the rate is the failure. Four of the four captured failures are
the same test, `test_a_single_column`, and the cause is one line:

```
mean_sq = sum_sq / n_cols.to(tl.float32)
AttributeError("'int' object has no attribute 'to'")
```

Triton specializes an integer kernel argument whose value is `1` into a
compile-time constant, so that line compiles at every row width except `N = 1`.
The prompt licenses the case in as many words. Haiku 4.5, meanwhile, went 3 of 5
on the same task, and its two failures were 24 of 24 tests and 23 of 24: not an
edge missed, a kernel that does not run.

**That contrast is the specification for v2, stated by an existing task rather
than by me.** The task discriminates because:

- the fluent answer and the correct answer differ, and the difference is not a
  convention anyone could guess — it is a fact about how the compiler treats its
  arguments;
- the failure is a single test out of twenty-four, so the task separates *how*
  two models fail rather than only whether they do;
- nothing about it rewards writing more code.

Everything in section 2 was reasoned from Haiku's laptop failures. This is the
first time the reasoning has been checked against a task that actually separates
frontier models.

## 2. What actually discriminates at the top

> **Read 2.0 first. It was written after the four properties below had been
> satisfied in full by three tasks, and a frontier model had passed all three
> forty draws out of forty.** The four are real, they are worth having, and they
> are not sufficient. Keeping them here with the result attached is more useful
> than editing them out.

**2.0 The difficulty is a fact about a tool, and an obscure one.** This is what
`fused_rmsnorm_kernel` actually has and what section 1b failed to name. RMSNorm
is four lines; the task is hard because Triton specialises an integer argument
whose value is `1` into a compile-time constant, and no amount of reasoning about
the normalisation reaches that. The three laptop tasks written against 2.1
through 2.4 are all *reasoning*, and a model that can carry a derivation carries
all of them. The obscurity clause is load-bearing and it is uncomfortable:
`activation_checkpointing_rng` was written deliberately around a runtime fact,
and lost, because bracketing an RNG around a recomputation is documented in every
framework that implements checkpointing. See `LESSONS.md` L30, including what
this implies about designing against a moving target.

Mechanism density alone is not sufficient either — `online_softmax_attention` is
difficulty 5, mechanism dense, and Opus 5 passed it first try. Four further
properties separate frontier models where a single hard mechanism does not:

**2.1 The obvious correct answer is forbidden.** The task states a constraint
that rules out the textbook one-liner, and a hidden test enforces the
constraint rather than only the output. A model that reaches for the standard
implementation gets a correct answer that fails. This is the single strongest
discriminator, because fluency stops being enough.

**2.2 Mechanisms compose and errors interact.** Three or four pieces that are
each individually familiar, wired together so that a mistake in one shows up as
a wrong answer attributed to another. Getting each piece right in isolation is
not enough; the model has to hold the whole thing in view.

**2.3 A backward pass, derived rather than recalled.** v1 has no hand-written
backward at all — `grad_accumulation` uses autograd. A derived backward is
mechanism-dense, exactly checkable against autograd to 1e-6, and it is the
canonical place where implementations are subtly wrong in a way that passes a
casual test.

**2.4 A property that is provable, not merely plausible.** The best tasks have
a correctness statement stronger than "matches my reference on these inputs":
an exact distributional identity, an equivalence across two very different
computations, an invariant under a parameter the answer must not depend on.
Those catch the wrong-but-close implementation that a value comparison at a
loose tolerance lets through.

### What is deliberately not on that list

- **Underspecified prompts.** That is convention-guessing wearing a costume.
- **Sheer volume of code.** It measures context handling and inflates cost.
- **Repair tasks** (here is a broken implementation, fix it). A real and
  interesting axis, and a different one: it is closer to what SWE-bench already
  measures. If it ships it ships as a separate track, not inside v2.
- **Wall-clock performance targets on the laptop tier.** Timing on unknown
  hardware is not reproducible, and reproducibility is the founding claim. A
  complexity constraint has to be tested structurally, not with a stopwatch.

## 3. Candidate tasks

Eight candidates, scored against the four properties above. The plan is to
write the top two first, ask Opus 5, and only escalate if it passes them.

| Candidate | Category | Forbids the easy answer | Composes | Backward | Provable property |
|---|---|:---:|:---:|:---:|:---:|
| `speculative_decoding_verify` | attention | yes | partly | no | **exact** |
| `flash_attention_backward` | attention | yes | yes | yes | exact vs autograd |
| `activation_checkpointing_rng` | training | yes | yes | yes | exact |
| `paged_kv_cache` | attention | yes | yes | no | equivalence |
| `topk_moe_routing` | training | partly | yes | no | equivalence |
| `mixed_precision_step` | numerics | partly | yes | partly | exact |
| `gqa_rope_sliding_window` | attention | no | yes | no | equivalence |
| `adamw_schedule_trajectory` | training | no | partly | no | trajectory |

### The two to write first

**`speculative_decoding_verify`** (attention, laptop, numpy). Given a draft
model's proposed tokens with their probabilities and the target model's
probabilities, implement the accept/reject step: accept token `i` with
probability `min(1, p_target/p_draft)`, and on rejection sample from the
normalized positive residual `max(0, p_target - p_draft)`.

Why it discriminates: the whole point of the algorithm is that the output
distribution is **exactly** the target model's, and almost every plausible wrong
version — normalizing the wrong thing, clamping instead of taking the positive
part, resampling from the target directly, accepting on the wrong ratio —
produces something that looks right and is not. The test is not a tolerance on
one array: with a fixed seed and a large number of draws, the empirical output
distribution has to match the target's to within a sampling bound that a wrong
renormalization misses by orders of magnitude. That is a provable property, and
no amount of code fluency substitutes for having the derivation.

**`flash_attention_backward`** (attention, laptop, torch without autograd). The
forward is given. Implement the backward: the recomputation, the `D = rowsum(dO
∘ O)` term, and the correct rescaling per block.

Why it discriminates: the `D` term is the exact spot implementations get wrong,
it does not fall out of pattern-matching the forward, and it is checkable
against `torch.autograd` in float64 — measured at 2.3e-15 across every block
size, against a tolerance of 1e-10.

Non-materialization is enforced by the shape of the interface, not by a test.
The first version of this paragraph claimed the tests would catch it: identical
results across several block sizes, which a materialising implementation
supposedly could not fake. That was wrong, and it was wrong before a line of the
task existed — block-size invariance is a property of every correct
implementation, including one that builds the whole N×N matrix, computes `D`
correctly over the whole row, and slices it (L28). So the task splits in two:
`key_block_gradients` is handed one block of keys and never the rest, which
makes materialisation unavailable rather than detectable, and the hidden tests
grade that function directly against autograd's own score gradient. The
top-level driver that walks the blocks can still cheat, and the prompt claims
nothing about it that is not enforced.

There is a second thing the split buys, and it turned out to be the sharper
one. `D` is a property of a whole query row and the block function can only see
one block of keys, so an implementation that accumulates the correction from the
columns in hand is not merely inelegant, it is wrong — and wrong in the exact
way real blocked implementations are: right shape, right magnitude, and correct
whenever there happens to be a single block.

### The rest, in one line each

- **`activation_checkpointing_rng`** — recompute-in-backward with dropout, so
  the RNG state has to be captured and restored or the recomputed activations
  differ from the forward's. The state handling is the mechanism nobody gets
  right by accident, and equivalence with the non-checkpointed gradient is
  exact.
- **`paged_kv_cache`** — block table indirection, allocation and free, and
  copy-on-write when a sequence forks. Pure Python, laptop-cheap, and dense
  with bookkeeping that has to be exactly right.
- **`topk_moe_routing`** — top-k gating with a capacity factor, token dropping,
  the auxiliary load-balancing loss, and dispatch/combine that has to round-trip.
- **`mixed_precision_step`** — fp32 master weights, dynamic loss scaling, and
  the skip-on-inf/nan path, which is where the trajectory diverges.
- **`gqa_rope_sliding_window`** — four attention mechanisms at once with a
  decode-time equivalence property. Weakest of the eight: it composes but does
  not forbid the obvious answer.
- **`adamw_schedule_trajectory`** — decoupled weight decay, bias correction,
  warmup then cosine, global-norm clipping, graded on the trajectory over many
  steps. Risk: convention-heavy, so the prompt has to pin the ordering in prose
  and the difficulty has to come from the trajectory rather than the guesses.

Every one of them obeys the founding constraint: laptop, no GPU, under five
minutes. Anything that needs hardware goes to the accelerated tier and is
reported separately, as it is today.

## 4. The process change that matters more than the tasks

v1's difficulty numbers were assigned by how hard each task felt to write, and
`meta.yaml` presented them as properties of the task. They were a measurement of
me. This is the fix, and it is mechanical:

**A task does not enter a v2 frozen set until it has been calibrated against
models.** In addition to the three existing admission checks — reference
passes, untouched starter fails with real assertions, mutants caught — a task
must carry:

```yaml
calibration:
  - model: claude-opus-5
    draws: 5
    passed: 2
    date: 2026-08-10
  - model: claude-haiku-4-5
    draws: 5
    passed: 0
    date: 2026-08-10
```

and it is **refused** if the strongest model tried passed every draw. A task
that everything solves adds cost and no information; it can live in a warm-up
set, not in the headline.

Note that `difficulty` then stops being an opinion and becomes a summary of
that block. The number in v1 stays as it is, labelled for what it was.

**Built, alongside the first three candidates rather than before them.**
Infrastructure written before its first user has tests that share the user's
blind spot — that is L11, exactly, and it published wrong numbers twice. So the
`calibration` key, its validation and the admission rule were written once there
were real tasks to hold them to, and the rule then refused the first two tasks
it was pointed at, which is the only way to know it does anything.

As implemented, "the strongest model tried" is read off the block rather than
declared: the entry with the highest pass rate is the one the rule looks at, and
if that entry passed every draw the loader refuses the task outright. A numbered
set from `v2` onward also needs at least one entry of five draws or more,
because one draw is not a measurement of a sampled process (L19). `v1` is exempt
by name — it predates the rule and is published as what it was. Tasks that fail
the bar go to `frozen_set: warmup`, which keeps the calibration block and stays
out of the headline. `TASK_FORMAT.md` has the table.

## 5. What v2 is made of

- **Carried over from v1:** `kv_cache_equivalence`, `grad_accumulation`,
  `online_softmax_attention`. These are the three that discriminated.
- **Retired from the headline:** `softmax_stability` and `bpe_merge_order`.
  Both were lost on a single test, which is convention noise rather than
  capability, and both are solved by everything. They stay in v1, published.
- **Under review, and the review came back. Moved on 2026-08-13.**
  `attention_causal_mask`, `sharded_dataloader` and `quantization_error_bounds`
  needed a third model before they could earn a place. `claude-sonnet-5` solved
  all three. Three models, no failures, and the four extra draws this was waiting
  on turned out to be already published: the 2026-08-09 sweeps carry five each
  and the 2026-08-03 sweep a sixth for Opus and Haiku. All three are
  `frozen_set: warmup` with the block re-derived from `leaderboard/`, so `v1` is
  five laptop tasks now and was eight when its rows were measured.
- **New: nothing yet, and that is a measurement rather than a delay.** Three of
  the candidates above were written end to end on 2026-08-09 —
  `speculative_decoding_verify`, `flash_attention_backward` and
  `activation_checkpointing_rng` — and all three were refused by the admission
  rule in section 4, because `claude-opus-5` passed 15, 15 and 10 draws
  respectively without a single failure. They are `frozen_set: warmup`. They
  separate Opus from Haiku by a wide margin, their failure shapes are the most
  informative thing in the repository after the kernel task, and they do not
  measure at the top. Section 2.0 is what that cost buys.

  The five remaining candidates are all reasoning tasks of the same kind, so
  writing them against the same criterion has a known answer. Anything new on
  the laptop tier has to clear 2.0 first or it is a fourth warm-up task.
- **Accelerated tier: promoted from a side quest to the part that works.** It is
  the only tier currently discriminating at the top, so it gets more tasks
  rather than fewer: a Metal kernel (so the README's claim stops being half
  true), a backward kernel, and a reduction whose correctness depends on the
  launch configuration. Still reported beside the headline and never inside it —
  that separation is what makes the laptop claim true, and it does not bend
  because the accelerated tier turned out to be the interesting one.

  **The Metal kernel was written on 2026-08-13 and it is all three of those at
  once.** `metal_cross_entropy_kernel` is a reduction whose correctness depends
  on the launch configuration, because the caller chooses the threadgroup size
  and the kernel never sees it until it runs. It is the first task in the
  repository written against 2.0 rather than 2.1 through 2.4, and it is the
  first one a frontier model has failed on the laptop this repository is
  developed on.

  The honest tension has moved. The tier that discriminates was the tier nobody
  could reproduce without renting hardware; half of it is now reproducible on any
  Apple silicon Mac, because `torch.mps.compile_shader` compiles a Metal source
  string at runtime and needs no toolchain beyond torch. That is an argument for
  writing the next accelerated task in Metal rather than CUDA, and it does not
  weaken the case in section 3 for making the laptop tier harder.

## 6. Measurement changes that ship with v2

Harder tasks alone would produce noisier single draws, which is a worse
benchmark, not a better one. Three changes go with them.

1. **Repeated draws are the default for a published row.** `--repeat 5`
   minimum. The leaderboard reports `k/N` per task and a range on the set rate.
   A single-draw row is admissible only when it is labelled as one, which is
   what the two v1 rows are.
2. **Cost per solved task becomes a column.** It is the axis on which frontier
   models genuinely differ and the one that does not saturate: two models can
   both score 100% and be an order of magnitude apart in what that cost.
3. **The failure shape is reported, not just the verdict.** Scoring stays
   binary — partial credit hides the failures worth knowing about — but the
   per-task hidden-test counts are already in the results file, and the
   leaderboard should show them. `1 of 29` and `19 of 21` are the same "fail"
   and completely different information, and that distinction is what produced
   this document.

## 7. Cost

At v1 prices, one Opus 5 sweep of eight tasks is $0.54 and takes three and a
half minutes. Five draws is $2.70. A v2 set of ten harder tasks, five draws,
across four models is roughly $30 to $60 per full refresh — which is the number
that makes the founding constraint worth having, and it is unchanged by
anything in this document.
