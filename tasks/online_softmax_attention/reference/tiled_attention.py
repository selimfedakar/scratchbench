"""Reference solution — attention streamed over blocks of keys.

Ordinary attention builds the whole score matrix before it can normalise
anything, because softmax needs the sum over every key before it knows what any
single weight is. That matrix is quadratic in the sequence length, and it is the
reason long contexts were expensive long before anyone wrote a fused kernel.

The way out is to stop treating the normaliser as something you need up front.
Walk the keys in blocks, and carry three numbers per query: the largest score
seen so far, the sum of the exponentials relative to that largest score, and the
same weighted sum applied to the values. Each of those is a partial answer that
can be corrected later. When a block arrives holding a score bigger than
anything before it, the maximum moves, and everything already accumulated was
measured against the old one — so it is rescaled by exp(old max - new max),
which is a number no larger than one, and the accumulation continues.

Three mistakes live in that sentence, and all three are quiet.

Rescale the numerator and forget the denominator, and the output is a weighted
average whose weights no longer sum to one: every value is scaled by a factor
that drifts with the data. Nothing overflows, nothing warns, the shapes are
right, and the model trains to something slightly wrong.

Skip the rescaling when the maximum moves and the answer is dominated by
whichever block happened to arrive first, which makes the result depend on the
block size — a performance knob silently changing the mathematics.

And in the causal case a block can lie entirely in the future of every query it
was scored against. Its maximum is -inf, correctly, and the rescaling factor is
then exp(-inf - -inf), which is NaN. One NaN in the accumulator reaches every
parameter on the next backward pass.

The merge is written as its own function on purpose. It is associative, which is
what lets the blocks be processed in any grouping — one machine walking them in
order, or many machines each taking a slice and combining the partial states
afterwards. That algebra is the whole idea; the loop below is bookkeeping
around it.
"""

from __future__ import annotations

import numpy as np

#: A partial result over some group of keys, for every query:
#: (running maximum, denominator, numerator).
State = tuple[np.ndarray, np.ndarray, np.ndarray]


def combine_states(left: State, right: State) -> State:
    """The state of the union of two groups of keys."""
    left_max, left_denominator, left_numerator = (
        np.asarray(part, dtype=np.float64) for part in left
    )
    right_max, right_denominator, right_numerator = (
        np.asarray(part, dtype=np.float64) for part in right
    )

    merged_max = np.maximum(left_max, right_max)
    # Two empty groups leave the maximum at -inf, and both rescaling factors
    # below would be exp(-inf - -inf), which is NaN. Shifting by zero there
    # keeps them at zero, and the empty state survives the merge intact.
    shift = np.where(np.isneginf(merged_max), 0.0, merged_max)

    left_scale = np.exp(left_max - shift)
    right_scale = np.exp(right_max - shift)

    denominator = left_denominator * left_scale + right_denominator * right_scale
    numerator = (
        left_numerator * left_scale[..., None] + right_numerator * right_scale[..., None]
    )
    return merged_max, denominator, numerator


def attention_tiled(q, k, v, block_size: int, causal: bool = False):
    """Attention over `q`, `k`, `v`, accumulated `block_size` keys at a time.

    Returns `(output, logsumexp)`.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    batch, heads, n_queries, head_dim = q.shape
    n_keys = k.shape[-2]
    head_dim_v = v.shape[-1]

    state: State = (
        np.full((batch, heads, n_queries), -np.inf),
        np.zeros((batch, heads, n_queries)),
        np.zeros((batch, heads, n_queries, head_dim_v)),
    )

    # The queries are the last n_queries positions of the key range, which is
    # the convention a KV-cache forces and the one the rest of this benchmark
    # uses: one fresh query against a long history sits at the end of it.
    query_pos = np.arange(n_keys - n_queries, n_keys)[:, None]

    for start in range(0, n_keys, block_size):
        stop = min(start + block_size, n_keys)

        scores = (q @ np.swapaxes(k[:, :, start:stop, :], -1, -2)) / np.sqrt(head_dim)
        if causal:
            key_pos = np.arange(start, stop)[None, :]
            scores = np.where(key_pos <= query_pos, scores, -np.inf)

        block_max = np.max(scores, axis=-1)
        # Same -inf guard as in the merge: a block wholly in the future of a
        # query contributes nothing, and must do so without arithmetic on
        # -inf - -inf.
        shift = np.where(np.isneginf(block_max), 0.0, block_max)
        weights = np.exp(scores - shift[..., None])

        state = combine_states(
            state,
            (block_max, np.sum(weights, axis=-1), weights @ v[:, :, start:stop, :]),
        )

    running_max, denominator, numerator = state
    return numerator / denominator[..., None], running_max + np.log(denominator)
