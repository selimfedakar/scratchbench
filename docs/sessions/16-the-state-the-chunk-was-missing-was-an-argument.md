# 16 — The state the chunk was missing was an argument

The roadmap's instruction for this session was one line. Selim had chosen option
A in `ROADMAP.md` §9.1 the day before: laptop candidate 2 gets written against
`LESSONS.md` L28's mechanism rather than `V2_DESIGN.md` §2.0's hazard clause,
because §2.0 was derived from two accelerated tasks and generalised to a tier
whose one discriminating example works differently. Write the task, calibrate it,
and add the sentence to §2.0 that says which criterion belongs to which tier.

All of that happened. The task is finished, verified, mutation-tested and
calibrated, and it is `warmup`: `claude-opus-5` 10 of 10, `claude-sonnet-5` 10 of
10, `claude-haiku-4-5` 2 of 10. The interesting part of the session is the
diagnosis of why, which is that I implemented half of the mechanism I had written
down and handed the other half to the model in an argument list.

## The task

`chunked_batchnorm_backward`, category `training`, laptop tier, torch, 44 hidden
tests, 180 second limit.

Batch normalisation is the layer whose output at one position is a function of
every other position in the batch, so it is the obvious place to look for a
decomposition a caller can impose. The forward pass survives the split easily —
sums and sums of squares add up — and the backward pass is where the coupling
shows. Writing `M` for the number of positions the statistics were taken over
and `s` for `sqrt(var + eps)`:

    dx = (gamma / s) * (dy - (1/M) * sum(dy) - xhat * (1/M) * sum(dy * xhat))

Both sums run over the whole batch. A micro-batch holds neither, and neither can
be accumulated on the way past, because each is needed in full before the first
`dx` can be written down. That is why the real distributed version of this layer
reduces twice before it computes a single input gradient, and it is why the task
has three functions: one that produces a chunk's share of the parameter
gradients, one that produces a chunk's input gradients, and a driver that walks
the chunks twice.

Padding is the second mechanism and it interacts with the first. A dropped
position contributed nothing to the mean and nothing to the variance and takes
exactly zero gradient, while the correction terms are per-channel constants and
are *not* zero there. The mask has to be applied before the sums and again after
the arithmetic, and the mutation run below shows that neither placement is
sufficient on its own.

The interface is the enforcement, exactly as in `flash_attention_backward`:
`chunk_parameter_gradients` and `chunk_input_gradients` are handed one chunk and
are never handed another one, so holding the whole batch is unavailable rather
than detectable.

## Both halves of L2

The three gates, in the order `CLAUDE.md` mandates, with the reference written
first and the prompt written last.

```
chunked_batchnorm_backward       44 passed   44 failed          2.62s   ok
```

That line is from `validate --tier all`, which reported **15 task(s) validated,
1 task(s) not checked here** — `fused_rmsnorm_kernel`, no CUDA on this machine,
which is the correct behaviour rather than a gap. The starter's 44 failures are
`NotImplementedError` from three function bodies, not collection errors.

Mutants went into `tools/mutate_v2_tasks.py` beside the existing ones, sixteen of
them, each with an expected verdict checked in both directions:

```
chunked_batchnorm_backward: all 16 mutants behaved as expected
```

Fourteen were expected to be caught and were: averaging the corrections over the
chunk in hand, dropping either correction term, dropping both, running one pass
over the chunks instead of two, counting padded positions in the denominator,
subtracting one from it, leaving padding in the result, leaving padding in the
parameter gradients, putting the epsilon outside the square root in each of the
two places it appears, correlating the upstream gradient with the raw deviation
instead of the standardised input, returning the two parameter gradients the
other way round, and leaving `gamma` out of the input gradient.

Two were expected to survive and did, and they are a fact about the task rather
than a hole in it. The reference masks the standardised input inside its helper,
and masks the upstream gradient again inside `chunk_input_gradients`; both are
redundant, because the callers multiply by an already-masked quantity and the
answer is masked again at the end. Only the first mask in
`chunk_parameter_gradients` and the last mask in `chunk_input_gradients` are
load-bearing. The reference's docstring now says so, and points at the mutant
that proves it. Defence in depth means most of the defences are untested, and the
honest way to hold that is an expectation checked in both directions rather than
a quiet deletion.

Tolerances were measured rather than chosen. Swept over thirty-nine seeds, seven
splits, four mask densities and four epsilons, the chunked answer and autograd
differ by at most **7.105e-15**. The comparisons use `atol=1e-10`, and the
mistake the task is named after moves the first case's answer by **3.872**. The
two tests that use no reference at all — a constant upstream gradient leaves the
input alone, a zero scale leaves the input alone — compare against a true zero at
`atol=1e-12`, where the reference lands at **5.551e-16** and at exactly zero.

## Thirty draws

Statistics line off, `results/` empty, ten draws per model at `--tasks
chunked_batchnorm_backward`.

| Model | Draws | Passed | Spread |
|---|---|---|---|
| `claude-opus-5` | 10 | 10 | always |
| `claude-sonnet-5` | 10 | 10 | always |
| `claude-haiku-4-5` | 10 | 2 | 0% to 100% |

$1.7124 for the thirty, no `adapter_error`, no `timeout`, no
`collection_error`. The admission rule reads the top two entries and refuses a
task when both are at the ceiling, so this one is `warmup` — decided by the
loader, not by me.

## Why it did not discriminate, which is the session

The mechanism I set out to reproduce, in the words I wrote before designing
anything: *the graded unit is handed one piece of a decomposition the caller
chose, the correct answer needs state that piece does not have, and materialising
the whole is unavailable rather than discouraged.* Every clause of that is true
of this task.

Here is Opus's answer:

```python
dx = gamma_b * inv_std * (dy_masked - (dbeta_b + x_hat * dgamma_b) / n)
```

`dgamma_b`, `dbeta_b` and `n` are `dgamma`, `dbeta` and `total_count` — three of
`chunk_input_gradients`'s own arguments. The state the chunk was missing arrives
in the signature, so there is nothing to reconstruct and what remains is one
algebra step.

I had reasoned my way into it by an analogy that does not hold.
`flash_attention_backward` hands its block function `o` and `do`, so handing this
one `dgamma` and `dbeta` looked like the same move. It is not: `o` and `do` are
forward artefacts, and the whole difficulty of that task is *recognising* that
the row correction it needs is `rowsum(dO * O)`, a quantity nobody passes it. I
passed the correction itself.

The failure shapes say it independently, and they say it in a way the counts
cannot. All eight of Haiku's losses fail `test_matches_autograd[chunk_sizes0]` —
the case where the whole batch is a **single chunk**, which is precisely where
the chunked mistake is correct by construction. So not one of the eight is the
mistake this task was built around. One draw dropped `gamma` from the scale
correction; one summed a pointwise term over the chunk and so was also the only
one to fail split invariance; the rest are ordinary algebra. Reading `2/10` would
have told me none of that, which is L27 one project over.

`LESSONS.md` **L44** is the entry. The criterion now has both halves in
`ROADMAP.md` §9.3, and the check that would have cost a sentence is the first
thing it asks of any candidate: name the quantity the graded unit cannot compute,
then look for it in the argument list; if it is there, there is no task.

## What else changed

`V2_DESIGN.md` §2.0 gained the scoping paragraph §9.1 asked for: the hazard
question is the **accelerated** tier's criterion, L28's mechanism is the
**laptop** tier's, and the three refusals recorded in §1.3 are a fact about numpy
and torch being exhaustively documented rather than a fact about the tier.

`ROADMAP.md` §2.3 says two laptop candidates clearing the admission rule reopens
the tier question, and two have now cleared it. The branch is deliberately not
taken, and the reason is written down rather than assumed: that sentence holds
only when each attempt carried the property it was written to isolate, and this
one did not. Candidate 3 in §9.3 is designed against both halves — the chunk
returns a reduction buffer whose contents are the solution's own choice, the
caller sums it, and no signature says which sums the answer needs or that the
count of kept positions is one of them.

## Numbers from this session's runs

- `python -m pytest -q` — **110 passed**.
- `validate --tier all` — **15 task(s) validated, 1 task(s) not checked here**.
- `tools/mutate_v2_tasks.py --task chunked_batchnorm_backward` — **all 16 mutants
  behaved as expected**.
- `tools/check_cost.py` — **212 file(s) checked**.
- `tools/check_calibration.py` — **32 calibration entries re-derived from 212
  draw(s)**.
- Sixteen tasks, **748** hidden tests. Laptop thirteen, accelerated three.
- Set counts: `v1` five, `v2` four, `warmup` **seven**, `unvalidated` empty.
- Spend: **$1.7124**. Running total roughly $16.

---

## Candidate 3, written the same session, and the credit ran out

The diagnosis above was cheap enough to act on immediately, so
`chunked_batchnorm_reduction` was written before the session ended: the same
layer, the same decomposition, and the second half of the criterion restored by
**deleting arguments** rather than adding mechanism. The mean, the variance, the
count and both parameter gradients are absent from every signature. The caller
reduces exactly once, over a buffer whose contents the solution designs, and
hands the total back on a second walk.

That constraint is the whole task, because the obvious route needs two
reductions and is not on offer: `sum(dy * xhat)` wants an `xhat` that wants a
mean that does not exist while the reduction is running. The way through is to
stop reducing centred quantities:

    dgamma = (sum(dy * x) - mean * sum(dy)) / sqrt(var + eps)

Five raw sums per channel — count, `sum(x)`, `sum(x*x)`, `sum(dy)`,
`sum(dy*x)` — carry the mean, the variance and both parameter gradients out the
other side. Nothing in either signature says so.

L2 on both halves, `57 passed  57 failed  ok`, and sixteen mutants behaving as
expected, fifteen caught and one expected survivor (the same redundant mask the
sibling task documents). Tolerances measured rather than chosen: worst
disagreement with autograd **1.472e-13** over thirty-nine seeds, seven splits,
four mask densities and four epsilons, against `atol=1e-9`, with the chunk-local
mistake moving the first case by **3.180**. The tolerance is two orders looser
than the sibling task's for a reason that is the task: a single reduction cannot
carry centred moments, so the variance comes out of a subtraction that cancels.

**The measurement is incomplete and the reason is billing.** `claude-opus-5`
went **9 of 10** — not the ceiling, and the first laptop task since
`flash_attention_backward` that a frontier model loses a draw to.
`claude-sonnet-5` produced one graded draw, which passed, and then nine
`adapter_error`s: *Your credit balance is too low to access the Anthropic API*.
Those nine measured nothing and are in neither side of the rate, which is the
`STATUSES` table doing its job and the third time this repository has had a
sweep cut off by billing and reported the hole rather than averaging over it.
Haiku was never asked. The task ships `frozen_set: unvalidated` with a two-entry
block that says exactly what was measured.

**What Opus's one loss was, and it is not the mechanism.** Its reduction and its
recovery are correct — `var = sum_xx / safe_count - mean * mean`, then
`dgamma = (sum_dyx - mean * sum_dy) * inv_std`, which is the identity the task is
about. What it lost on is broadcasting:

```python
count = m.sum()
dx = gamma_b * inv_std_b * (dym - (dbeta_b + xhat * dgamma_b) / count)
# RuntimeError: The size of tensor a (4) must match the size of tensor b (3)
#               at non-singleton dimension 2
```

`mean`, `inv_std`, `dgamma` and `dbeta` are all reshaped to `(1, channels, 1)`
on the four lines above; `count` is not, so it aligns with the position axis
instead of the channel axis and the whole draw dies at the first gradient. One
draw of one model is not a shape, and until Sonnet and Haiku are asked it is not
known whether this task discriminates on its mechanism or on arithmetic care —
which is the same question candidate 2 answered badly, so it is the one to
answer with evidence rather than hope.
