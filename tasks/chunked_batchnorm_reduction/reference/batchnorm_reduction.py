"""Reference solution — batch normalisation's backward pass in one reduction.

The caller has cut the batch into chunks and will walk them exactly twice. On
the first walk it asks each chunk for a buffer and adds those buffers together.
On the second it hands the total back and asks for that chunk's input
gradients. Nothing else crosses between chunks, and in particular there is no
second reduction: whatever the answer needs from the whole batch has to be in
that one buffer.

That constraint is the task, because the obvious route needs two reductions and
is not available. Normalising needs the mean and the variance, which are sums
over the batch; the input gradient needs, on top of them,

    dbeta = sum(dy)        and        dgamma = sum(dy * xhat)

and `xhat` is not computable until the mean and the variance already exist. So a
first pass that returns the statistics and a second that returns the gradient
sums is the natural design, and the caller does not offer it.

The way out is to stop reducing centred quantities and reduce raw ones. Over the
kept positions of a channel, carrying

    count,  sum(x),  sum(x*x),  sum(dy),  sum(dy*x)

is enough for all of it, because each of the four things the answer wants falls
out of them afterwards:

    mean   = sum(x) / count
    var    = sum(x*x) / count - mean * mean
    dbeta  = sum(dy)
    dgamma = (sum(dy*x) - mean * sum(dy)) / sqrt(var + eps)

The last line is the one that has to be seen. `sum(dy * xhat)` looks like it
needs `xhat`, which needs the mean, which is why the two-pass shape suggests
itself — but `xhat` is affine in `x`, so the sum of `dy * xhat` is an affine
combination of two sums that were available before the mean was. Five numbers per
channel, one pass, and none of them is centred on anything.

Everything after that is the ordinary backward pass. Writing `M` for the count
and `s` for `sqrt(var + eps)`, a kept position gets

    dx = (gamma / s) * (dy - dbeta / M - xhat * dgamma / M)

and a padded position gets exactly zero: it contributed to none of the five sums,
its output was not a function of anything, and the correction terms are
per-channel constants that are emphatically not zero there.
"""

from __future__ import annotations

import torch

# Rows of the buffer, in the order this solution chose to stack them. The caller
# adds buffers together and never looks inside, so the only real constraint is
# that every entry is a sum over the chunk's kept positions.
COUNT, SUM_X, SUM_XX, SUM_DY, SUM_DYX = range(5)


def chunk_reduction(x_chunk, dy_chunk, mask_chunk):
    """The five per-channel sums this solution needs from the whole batch."""
    keep = mask_chunk[:, None, :].to(x_chunk.dtype)
    x = x_chunk * keep
    dy = dy_chunk * keep
    return torch.stack(
        [
            keep.expand_as(x_chunk).sum(dim=(0, 2)),
            x.sum(dim=(0, 2)),
            (x * x).sum(dim=(0, 2)),
            dy.sum(dim=(0, 2)),
            (dy * x).sum(dim=(0, 2)),
        ]
    )


def _unpack(reduction_total, eps):
    """`(count, mean, inv_std, dgamma, dbeta)` recovered from the raw moments."""
    count = reduction_total[COUNT]
    mean = reduction_total[SUM_X] / count
    variance = reduction_total[SUM_XX] / count - mean * mean
    inv_std = 1.0 / torch.sqrt(variance + eps)

    dbeta = reduction_total[SUM_DY]
    # sum(dy * xhat) without ever having formed xhat during the reduction.
    dgamma = (reduction_total[SUM_DYX] - mean * dbeta) * inv_std
    return count, mean, inv_std, dgamma, dbeta


def chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, gamma, eps, reduction_total):
    """The gradient with respect to this chunk's inputs."""
    count, mean, inv_std, dgamma, dbeta = _unpack(reduction_total, eps)

    keep = mask_chunk[:, None, :].to(x_chunk.dtype)
    x_hat = (x_chunk - mean[None, :, None]) * inv_std[None, :, None] * keep
    dy = dy_chunk * keep

    correction = (dbeta[None, :, None] + x_hat * dgamma[None, :, None]) / count[
        None, :, None
    ]
    # The correction is a per-channel constant and is not zero at a padded
    # position, so the mask is applied after the subtraction as well as before.
    return (gamma * inv_std)[None, :, None] * (dy - correction) * keep


def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, gamma, eps):
    """Reduce once over the chunks, then walk them again.

    Returns `(dx_chunks, dgamma, dbeta)`.
    """
    reduction_total = None
    for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks):
        buffer = chunk_reduction(x_chunk, dy_chunk, mask_chunk)
        reduction_total = buffer if reduction_total is None else reduction_total + buffer

    dx_chunks = [
        chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, gamma, eps, reduction_total)
        for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks)
    ]

    _, _, _, dgamma, dbeta = _unpack(reduction_total, eps)
    return dx_chunks, dgamma, dbeta
