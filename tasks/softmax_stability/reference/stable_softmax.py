"""Reference solution — numerically stable softmax, log-softmax and logsumexp.

Two ideas carry the whole file.

The first is the max shift. exp() overflows a float64 a little above 709, so a
logit of 10_000 is not a large number, it is infinity, and every probability in
that row becomes NaN. Subtracting the row maximum before exponentiating leaves
the result unchanged — softmax is invariant to a constant shift along its axis —
while guaranteeing the largest exponent is exactly exp(0).

The second is that -inf has to survive. Attention masks are additive -inf, so a
masked position arrives here as -inf and must leave as probability zero. That
works for free, since exp(-inf) is 0. What does not work for free is a slice
where *everything* is masked: the maximum is then -inf, the shift becomes
-inf - -inf, and the row turns to NaN. Every function below special-cases that
slice instead of letting the arithmetic decide.
"""

from __future__ import annotations

import numpy as np


def logsumexp(x, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log(sum(exp(x))) along `axis`, computed without overflowing."""
    x = np.asarray(x, dtype=np.float64)

    peak = np.max(x, axis=axis, keepdims=True)
    # An all -inf slice has an -inf peak; shifting by 0 there keeps the
    # subtraction finite and the sum below correctly collapses to zero.
    shift = np.where(np.isneginf(peak), 0.0, peak)

    total = np.sum(np.exp(x - shift), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        # total is zero exactly when the slice was entirely -inf, and log(0)
        # is the -inf this function is supposed to return there.
        out = shift + np.log(total)

    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x, axis: int = -1) -> np.ndarray:
    """Probabilities along `axis`, summing to one."""
    x = np.asarray(x, dtype=np.float64)

    peak = np.max(x, axis=axis, keepdims=True)
    shift = np.where(np.isneginf(peak), 0.0, peak)

    exponentiated = np.exp(x - shift)
    total = np.sum(exponentiated, axis=axis, keepdims=True)

    # Divide only where there is something to divide by; a fully masked slice
    # keeps the zeros it was initialised with.
    return np.divide(
        exponentiated,
        total,
        out=np.zeros_like(exponentiated),
        where=total > 0.0,
    )


def log_softmax(x, axis: int = -1) -> np.ndarray:
    """log(softmax(x)) along `axis`, without going through softmax.

    Taking the logarithm of the probabilities would throw away everything that
    underflowed: a position 800 nats below the maximum has a probability of
    exactly 0.0 in float64, so log() reports -inf where the true answer is
    -800. Subtracting logsumexp keeps that number.
    """
    x = np.asarray(x, dtype=np.float64)

    total = logsumexp(x, axis=axis, keepdims=True)
    # Where logsumexp is -inf the whole slice was -inf, so every entry of `x`
    # is -inf too and subtracting zero already gives the required -inf.
    return x - np.where(np.isneginf(total), 0.0, total)
