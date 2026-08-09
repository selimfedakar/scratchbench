# Verifying a draft without changing what the model says

Fill in `speculative_decoding.py`. numpy only:

```python
def verify_draft(draft_tokens, draft_probs, target_probs, rng) -> np.ndarray
```

Speculative decoding puts a small fast model in front of a large slow one. The
small model runs ahead and proposes the next several tokens; the large model
scores all of them in a single forward pass, which costs about what one token
used to cost. Whatever survives that check is emitted, and the round starts
again from wherever it got to.

The check is the part that has to be right. A user who asks for the large model
is entitled to the large model's output distribution, and a scheme that emits
something merely close to it has bought its speedup by quietly swapping the
model out. So the requirement is exact, not approximate, and it is the whole
specification of this function:

> Take the proposal apart in such a way that the token this function emits at
> any given position is distributed **exactly** as the large model's own
> distribution for that position, and accept the small model's guesses as often
> as it is possible to do that.

Both halves matter. Rejecting everything and sampling from the large model
satisfies the first half perfectly and is worth nothing, because the point of
the exercise is the tokens that survive. Your acceptance rule has to be the most
generous one that does not break the guarantee.

## What you are handed

`draft_probs` has shape `(n_draft, vocab)`: row `i` is the small model's
distribution over the vocabulary at proposal position `i`. `draft_tokens` has
shape `(n_draft,)` and holds what the small model actually drew — token
`draft_tokens[i]` is a sample from `draft_probs[i]`, so the small model gave it
positive probability.

`target_probs` has shape `(n_draft + 1, vocab)`. Its first `n_draft` rows are
the large model's distributions for the same positions, computed in one pass
over the proposal. The extra row is the large model's distribution for the
position immediately after the proposal, and it is there because that pass
produced it whether or not anyone uses it.

Every row of both arrays is a probability distribution: non-negative, summing to
one. They may be `float32` or `float64`, and neither array nor `draft_tokens` is
yours to modify. The two models can agree closely or barely at all, and either
can give a token no mass at all.

`rng` is a `numpy.random.Generator`. Every random choice the function makes
comes out of it and nowhere else, so two generators in the same state produce
the same round. How much of it you consume is up to you.

## What a round produces

Walk the proposal from the front. A position you accept contributes the proposed
token itself, unchanged, and you carry on to the next one. The first position
you reject ends the round: everything the small model drafted after it was
conditioned on a token that is not being emitted, so none of it can be used.
Either way the round emits **one** token after the accepted prefix — a
replacement for the proposal you turned down, or, if you got all the way to the
end without rejecting anything, a token for the position after the proposal,
which is what the extra row of `target_probs` is for.

So the return value is a one-dimensional array of token ids of length between
one and `n_draft + 1`, all but the last of them equal to the proposal they came
from, with an integer dtype. A round is allowed to be handed an empty proposal,
in which case there is nothing to check and the only thing left to do is the
token after it.

The guarantee above applies at every position, conditional on the round reaching
it: given that the first `i` proposals were accepted, the token emitted at
position `i` is distributed exactly as `target_probs[i]`.

## Conventions

- numpy only. No `torch`, no `scipy`.
- The arrays you are given are read-only as far as this function is concerned.
- The vocabulary can be as small as one token, and `n_draft` as small as zero.
- Nothing here depends on the sequence the tokens belong to; the distributions
  are given, and this function is only the accept-and-correct step.
