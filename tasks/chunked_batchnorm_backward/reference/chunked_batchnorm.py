"""Reference solution — batch normalisation's backward pass, one micro-batch at a time.

Batch normalisation is the layer whose output for one position depends on every
other position in the batch, which is exactly the property that makes it awkward
the moment the batch stops fitting in one place. The forward pass survives the
split easily: sums and sums of squares add up, so the mean and the variance can
be assembled from per-chunk partials and handed back. The backward pass is
where the coupling shows itself.

Write `M` for the number of positions the statistics were taken over, `s` for
`sqrt(var + eps)`, and `xhat` for `(x - mean) / s`. Differentiating through the
statistics rather than treating them as constants gives, for every position the
mask keeps,

    dx = (gamma / s) * (dy - (1/M) * sum(dy) - xhat * (1/M) * sum(dy * xhat))

and both of those sums run over the **whole** batch. A chunk holds neither of
them. They cannot be accumulated on the way past either, because each one is
needed in full before the first input gradient can be written down: this is why
the caller makes two passes, and why a real implementation of this layer across
devices reduces twice before it computes a single `dx`.

The two sums are already named. `sum(dy)` over the whole batch is the gradient
of the loss with respect to the shift, and `sum(dy * xhat)` is the gradient with
respect to the scale. Every implementation computes both anyway, so the
information the chunk is missing is not extra work — it is the parameter
gradients, arriving one pass early.

The mistake that follows from missing this is not a crash. An implementation
that takes the two averages over the chunk it is holding produces gradients of
the right shape, the right order of magnitude, and correct whenever there is
exactly one chunk. It gets worse as the chunks get smaller, which is the
direction anyone splitting a batch for memory moves in, and it is the reason
this layer has a synchronising variant at all.

Padding is the second mechanism and it interacts with the first. A position the
mask drops contributed nothing to the mean and nothing to the variance, its
output was not a function of anything, and it therefore takes exactly zero
gradient — while the correction terms above are not zero there, since they are
per-channel constants. Zeroing the upstream gradient before the sums and zeroing
the result after the arithmetic are both necessary, and neither is sufficient on
its own.
"""

from __future__ import annotations

import torch


def _standardise(x_chunk, mask_chunk, mean, var, eps):
    """`(xhat, keep)` for one chunk, both broadcast to the chunk's own shape.

    `xhat` is zero wherever the mask is, and that is belt and braces rather than
    a load-bearing line: both callers multiply it by an upstream gradient that
    has already been masked, so either mask alone would keep padding out of the
    sums. `tools/mutate_v2_tasks.py` says so in the only way that counts, with a
    mutant that removes this one and survives.
    """
    keep = mask_chunk[:, None, :].to(x_chunk.dtype)
    deviation = x_chunk - mean[None, :, None]
    return deviation / torch.sqrt(var + eps)[None, :, None] * keep, keep


def chunk_parameter_gradients(x_chunk, dy_chunk, mask_chunk, mean, var, eps):
    """One chunk's share of `(dgamma, dbeta)`.

    Both are plain sums over the positions this chunk keeps, so the chunks add
    up to the whole batch's answer and no chunk needs to know about any other.
    """
    x_hat, keep = _standardise(x_chunk, mask_chunk, mean, var, eps)
    dy = dy_chunk * keep
    return (dy * x_hat).sum(dim=(0, 2)), dy.sum(dim=(0, 2))


def chunk_input_gradients(
    x_chunk,
    dy_chunk,
    mask_chunk,
    mean,
    var,
    gamma,
    eps,
    dgamma,
    dbeta,
    total_count,
):
    """The gradient with respect to this chunk's inputs.

    `dgamma`, `dbeta` and `total_count` are the whole batch's; everything else
    is this chunk's alone.
    """
    x_hat, keep = _standardise(x_chunk, mask_chunk, mean, var, eps)
    dy = dy_chunk * keep

    # The two whole-batch averages. Computing them from this chunk's own
    # positions instead is the mistake this task is about.
    correction = (dbeta[None, :, None] + x_hat * dgamma[None, :, None]) / total_count

    scale = (gamma / torch.sqrt(var + eps))[None, :, None]
    # The correction is a per-channel constant and is not zero at a padded
    # position, so the mask has to be applied after the subtraction as well as
    # before it.
    return scale * (dy - correction) * keep


def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, mean, var, gamma, eps):
    """Walk the chunks twice: parameter gradients first, then input gradients.

    Returns `(dx_chunks, dgamma, dbeta)`.
    """
    total_count = sum(int(mask_chunk.sum()) for mask_chunk in mask_chunks)

    dgamma = torch.zeros_like(gamma)
    dbeta = torch.zeros_like(gamma)
    for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks):
        chunk_dgamma, chunk_dbeta = chunk_parameter_gradients(
            x_chunk, dy_chunk, mask_chunk, mean, var, eps
        )
        dgamma = dgamma + chunk_dgamma
        dbeta = dbeta + chunk_dbeta

    dx_chunks = [
        chunk_input_gradients(
            x_chunk,
            dy_chunk,
            mask_chunk,
            mean,
            var,
            gamma,
            eps,
            dgamma,
            dbeta,
            total_count,
        )
        for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks)
    ]
    return dx_chunks, dgamma, dbeta
