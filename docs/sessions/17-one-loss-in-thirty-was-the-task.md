# 17 — One loss in thirty was the task

Session 16 ended mid-measurement. Laptop candidate 3,
`chunked_batchnorm_reduction`, was written, verified and mutation-tested, and its
sweep stopped after `claude-opus-5` and one `claude-sonnet-5` draw because the
Anthropic credit balance ran out. Nine `adapter_error`s followed, which measured
nothing and were recorded as measuring nothing. The roadmap's instruction for
this session was money rather than code: nine more Sonnet draws, ten Haiku
draws, about $2.

It cost $1.20 and it produced a result the count does not describe.

## The sweep

| model | draws | passed |
|---|---|---|
| `claude-opus-5` | 10 | 9 |
| `claude-sonnet-5` | 10 | 10 |
| `claude-haiku-4-5` | 10 | 0 |

Thirty draws in `calibration/`, re-derived by `tools/check_calibration.py`, which
now reports 35 entries over 251 files. Sonnet's ten are one graded draw from
2026-09-06 and nine from today; the nine adapter errors are in neither the
numerator nor the denominator, which is why the resumed sweep asked for nine and
not ten.

Read as a rate, that is a task which separates Haiku from the frontier and takes
a draw off Opus — the first laptop task since `flash_attention_backward` that a
frontier model loses one to. The admission rule agrees. It refuses a task the
top two measured entries both clear, and Sonnet 10/10 beside Opus 9/10 does not
clear.

## The parameter id that decides it

`SPLITS` in the hidden tests is:

```python
SPLITS = [(6,), (3, 3), (1, 5), (4, 2), (2, 1, 3), (1, 1, 1, 1, 1, 1)]
```

`chunk_sizes0` is `(6,)`: one chunk holding the whole batch. The mistake this
task is named after — computing a whole-batch quantity over the chunk in front
of you — is *correct by construction* when the chunk is the batch. So a draw that
fails `chunk_sizes0` failed for some other reason, whatever the total count of
failures says.

Eleven draws were lost across the thirty. Ten of them fail `chunk_sizes0`.

Opus's single loss is one of the ten:

```
FAILED test_matches_autograd[chunk_sizes0]
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 2
```

Four of the five quantities it needed had been reshaped to `(1, channels, 1)`
and `count` had not. The reduction it designed was correct in all ten draws,
including this one. Nine of Haiku's ten losses are the same category or worse:
three fail `test_the_buffer_has_one_shape_and_it_does_not_move_with_the_chunk`
because the buffer's first dimension moves with the chunk, one raises
`IndexError`, two never implement the functions at all and raise
`NotImplementedError` out of the starter.

## The one draw that is the task

Haiku's tenth. It is the only loss in thirty that passes `chunk_sizes0` and
fails `chunk_sizes1` through `chunk_sizes5`, which is the signature of the
chunked mistake and nothing else.

It designs the buffer correctly. Four rows, and the docstring names them:

```
Row 0: sum of dy values at kept positions per channel
Row 1: sum of x values at kept positions per channel
Row 2: sum of dy * x values at kept positions per channel
Row 3: count of kept positions per channel
```

Those are raw moments, which is the insight the task is built on: the mean does
not exist yet while the reduction runs, so anything centred is unavailable and
`sum(dy * xhat)` has to be assembled afterwards out of quantities that are
merely summed. What it leaves out is `sum(x * x)`, and the variance has to come
from somewhere:

```python
sum_x_sq = (x_masked ** 2).sum(dim=(0, 2))   # this chunk only
variance = (sum_x_sq / N) - mean_sq          # N is the whole batch's count
```

A chunk-local numerator over a whole-batch denominator. The same two lines
appear again for the two gradient sums it also failed to carry. With one chunk
the numerator and the denominator agree and the answer is exact to `float64`;
with any split at all it is wrong. That is the failure the task exists to catch,
and it happened once.

## What that means, and it is not session 16 again

Session 16's candidate did not discriminate because I built half the mechanism
and handed the model the other half in an argument list (L44). It is tempting to
file this session under the same heading and it would be wrong.

Candidate 3's design is sound and the claim is checkable rather than
argued — here are the three signatures the prompt fixes:

```python
def chunk_reduction(x_chunk, dy_chunk, mask_chunk) -> Tensor

def chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, gamma, eps,
                          reduction_total) -> Tensor

def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, gamma, eps
                               ) -> tuple[list[Tensor], Tensor, Tensor]
```

The mean, the variance, the count and both parameter gradients are in none of
them. `reduction_total` is the sum of a buffer whose contents the solution
designs, so it is not a channel for handing anything over. Both halves of
`ROADMAP.md` §9.3's criterion hold, by construction.

So the measurement is valid, and what it reports is that **the frontier is 20 for
20 on the mechanism**. That is a different result from session 16's and a more
useful one: session 16 measured nothing about the field, and this session
measured something about the field that I did not want to hear.

The lesson (L45) is that satisfying a discriminating criterion is not
discriminating. §9.3 was written in response to a failure, and a rule written in
response to a failure reads like a guarantee. It is a filter. It says a task
*can* separate models; only a sweep says whether the models being asked are in
fact separated by it.

## Why it is `warmup` when the rule would have taken it

This is the part I want on the record, because the cheap move was available and
it was defensible on paper. `_check_admission` reads the top two measured
entries and refuses only a task both clear. Sonnet 10/10 and Opus 9/10 do not
both clear. `frozen_set: v2` would have passed validation, CI, and every check
in the repository.

It would also have published a headline number about a reduction, sourced from a
task whose reduction nothing at the top of the field got wrong, on the strength
of one draw that mis-shaped a tensor. That is `LESSONS.md` L21 wearing a new
costume: L21 is about difficulty numbers assigned by feel and then contradicted
by a sweep, and this would be an admission decision assigned by a counter and
then contradicted by the same draws it counted.

**A rule that counts cannot see a shape, so passing it is a floor and never a
finding.** The task ships as `warmup`: real, calibrated, thirty draws checked
in, and it separates Haiku cleanly. It is not evidence about the laptop tier.

## What else changed

`ROADMAP.md` gains §9.4. It records the result, states that §9.3's criterion was
neither wrong nor misapplied, and is deliberately exact about §1.2: option B
becomes revisitable when two independent attempts *fail admission*, and that has
not happened. Candidate 2 was an invalid measurement and candidate 3 is a valid
measurement of a task the frontier clears. Only the second is evidence about the
tier, and one is not two. The branch stays closed.

The order of the remaining work changed for a reason this session supplied
itself. Finding the paragraph above took an afternoon of running `pytest` inside
`--keep` directories by hand and grepping for one parameter id, to answer a
question §4.2 exists to make the report answer. A fourth laptop candidate
designed before §4 lands would be evaluated the same slow, skippable way. **§4
is no longer only a precondition for §5; it is the tool the next candidate
needs.**

`unvalidated` is now empty. The sets are `v1` 5, `v2` 4, `warmup` 8, seventeen
tasks over 805 hidden tests.

## Numbers from this session's runs

```
110 passed in 67.88s
16 task(s) validated: reference passes, starter fails cleanly.
1 task(s) not checked here: fused_rmsnorm_kernel
35 calibration entries re-derived from 251 draw(s) in calibration/ and leaderboard/
251 file(s) checked: every cost reproduces from its own tokens.
$1.1997 spent across 19 draws
```

## Unverified

`OPENAI_API_KEY` is on the environment the adapters already read and the credit
is loaded, so §3's precondition is closed. **No OpenAI request has been made from
this repository.** The adapter does not exist yet, and nothing here has
established that the key works for the models §3 will ask for.
