"""Reference solution — causal attention over a padded batch.

Three things have to line up, and each of them has a classic way of going
wrong.

The direction. A query may read the keys at or before its own position. Written
as an additive mask that is 0 where reading is allowed and -inf where it is
not, the allowed region is the lower triangle. Transposing it silently trains a
model that can see its own answer, and the loss curve looks *better*, which is
what makes the bug expensive.

The alignment. Once a KV-cache exists, the number of queries stops matching the
number of keys: one new query arrives against a history of hundreds. Those
queries belong at the *end* of the key range, so with n_queries queries and
n_keys keys, query row i sits at absolute position n_keys - n_queries + i.
Anchoring the queries at position 0 instead gives a mask that is correct
whenever the two lengths happen to be equal and wrong the moment they are not.

The padding. Batching sequences of different lengths means some key slots are
filler and must receive no attention at all. Masking them can produce a query
whose entire row is -inf, and the shifted softmax that keeps everything else
finite turns that row into NaN. One NaN in the attention output poisons every
parameter through the backward pass, so the fully masked row is handled
explicitly rather than left to the arithmetic.
"""

from __future__ import annotations

import numpy as np


def causal_mask(n_queries: int, n_keys: int) -> np.ndarray:
    """Additive mask of shape (n_queries, n_keys): 0 to look, -inf to not."""
    # The queries are the last n_queries positions of the key range.
    query_pos = np.arange(n_keys - n_queries, n_keys)[:, None]
    key_pos = np.arange(n_keys)[None, :]
    return np.where(key_pos <= query_pos, 0.0, -np.inf)


def attention(q, k, v, key_padding_mask=None) -> np.ndarray:
    """Scaled dot-product attention, causal, with optional key padding."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    n_queries, head_dim = q.shape[-2], q.shape[-1]
    n_keys = k.shape[-2]

    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
    scores = scores + causal_mask(n_queries, n_keys)

    if key_padding_mask is not None:
        # (batch, n_keys) -> (batch, 1, 1, n_keys): the same padding applies to
        # every head and every query of that sequence.
        keep = np.asarray(key_padding_mask, dtype=bool)[:, None, None, :]
        scores = np.where(keep, scores, -np.inf)

    peak = np.max(scores, axis=-1, keepdims=True)
    # A row that is entirely -inf would shift by -inf and become NaN; shifting
    # it by zero keeps the exponentials at zero, and the divide below then
    # leaves the row as the zeros it was initialised with.
    shift = np.where(np.isneginf(peak), 0.0, peak)

    exponentiated = np.exp(scores - shift)
    total = np.sum(exponentiated, axis=-1, keepdims=True)
    weights = np.divide(
        exponentiated,
        total,
        out=np.zeros_like(exponentiated),
        where=total > 0.0,
    )

    return weights @ v
