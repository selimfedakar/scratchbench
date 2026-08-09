"""Reference solution — the backward pass of blocked attention, derived by hand.

The forward pass of attention can be streamed over blocks of keys because the
softmax normaliser can be carried as a running quantity and corrected later.
The backward pass looks like it should follow the same shape, and mostly it
does: every gradient here is a sum over key blocks, and each block's
contribution can be computed from that block alone.

Except for one term, and it is the term everything turns on.

Write `S` for the scaled scores, `P = softmax(S)` row by row, `O = P V`. Then
the gradient of the loss with respect to the scores is

    dS = P * (dP - D)

with `dP = dO V^T` and `D` the row sums of `dO * O`. `dP` is blockwise: the
columns of `dP` belonging to a key block depend only on that block's values. `D`
is not. It is a property of the *whole* row — one number per query, the same
number in every block — and it comes from the softmax Jacobian, whose off
diagonal couples every key to every other one. That is why the forward pass gets
to keep a running correction and the backward pass does not: there is a quantity
here that cannot be accumulated block by block, and it has to be computed up
front from things that are already available.

The mistake that follows from missing this is not a crash. An implementation
that computes `D` from the block it is holding — the row sums of `P * dP`
restricted to those columns — produces gradients that are the right shape, the
right order of magnitude, and correct whenever there is exactly one block. It
gets quietly worse as the block size shrinks, which is the direction anyone
tuning for memory moves in.

`D` is cheap once it is recognised for what it is. `sum(dO * O)` along the value
dimension needs no scores, no matrix the size of the sequence, and no
recomputation: `O` and `dO` are both already in hand. Everything else falls out
of the chain rule on `O = P V` and `S = Q K^T / sqrt(d)`:

    dV = P^T dO
    dQ = (dS K) / sqrt(d)
    dK = (dS^T Q) / sqrt(d)

`P` itself is never stored between the two passes. It is recomputed from `Q`,
the key block, and the log-sum-exp the forward pass already returned — which is
the trade the whole method is built on, arithmetic in exchange for never holding
a matrix quadratic in the sequence length.

The masked case needs one more piece of care. A key block can lie entirely in
the future of every query it was scored against, and that block must contribute
nothing rather than a NaN. Masking the scores to negative infinity before the
exponential does it: the weights come out as exact zeros, and zeros propagate
through all three products without arithmetic on infinities.
"""

from __future__ import annotations

import math

import torch


def key_block_gradients(
    q,
    k_block,
    v_block,
    o,
    lse,
    do,
    key_offset: int,
    query_offset: int,
    causal: bool = False,
):
    """What one block of keys contributes to `(dq, dk, dv)`.

    Returns `(dq_contribution, dk_block, dv_block)`.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])

    scores = (q @ k_block.transpose(-1, -2)) * scale
    if causal:
        query_positions = query_offset + torch.arange(q.shape[-2], device=q.device)
        key_positions = key_offset + torch.arange(k_block.shape[-2], device=q.device)
        # Masking before the exponential rather than after keeps a block that
        # is wholly in the future at exact zeros instead of NaN.
        scores = scores.masked_fill(key_positions > query_positions[:, None], -math.inf)

    # The forward pass kept the log-sum-exp precisely so this line can exist
    # without the full score matrix ever being stored.
    weights = torch.exp(scores - lse[..., None])

    dv_block = weights.transpose(-1, -2) @ do
    d_weights = do @ v_block.transpose(-1, -2)

    # One number per query, over every key there is. Computing it from this
    # block's columns instead is the mistake this task is about.
    delta = (do * o).sum(dim=-1)

    d_scores = weights * (d_weights - delta[..., None])
    return (d_scores @ k_block) * scale, (d_scores.transpose(-1, -2) @ q) * scale, dv_block


def flash_attention_backward(q, k, v, o, lse, do, block_size: int, causal: bool = False):
    """Gradients of the attention forward pass, one block of keys at a time.

    Returns `(dq, dk, dv)`.
    """
    n_keys = k.shape[-2]
    # The queries are the last n_queries positions of the key range.
    query_offset = n_keys - q.shape[-2]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for start in range(0, n_keys, block_size):
        stop = min(start + block_size, n_keys)
        dq_contribution, dk_block, dv_block = key_block_gradients(
            q,
            k[..., start:stop, :],
            v[..., start:stop, :],
            o,
            lse,
            do,
            key_offset=start,
            query_offset=query_offset,
            causal=causal,
        )
        dq = dq + dq_contribution
        dk[..., start:stop, :] = dk_block
        dv[..., start:stop, :] = dv_block

    return dq, dk, dv
