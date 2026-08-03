"""Reference solution — integer quantization and the bound it promises.

Quantization is the one step in a model that is *allowed* to lose information,
which makes it the one place a bug can hide inside an error budget that was
expected anyway. A wrong scale does not raise and does not produce NaN. It costs
a point of accuracy, and a point of accuracy is indistinguishable from the
honest price of moving to eight bits.

Three things have to be right, and each has a way of going wrong that survives
every shape check.

The range. Symmetric quantization uses the signed integers and pins the zero
point at zero; asymmetric quantization uses the unsigned integers and moves the
zero point to wherever zero happens to land. Mixing them — computing an
unsigned scale and then clamping into the signed range — clips the top half of
the distribution to a constant. Every activation above the midpoint becomes the
same number.

Zero. An exact zero has to come back as an exact zero. Padding is zero, masks
are zero, ReLU outputs are mostly zero, pruned weights are zero, and there are
a great many of them. If the zero point is off by one, every one of those zeros
returns as the same small non-zero constant with the same sign — which is a bias
added to the whole tensor rather than noise sprinkled across it. Noise averages
out over a layer. Bias accumulates through it. That is why the asymmetric range
below is widened to include zero even when the data never reaches it, and why
the integer that represents zero is the zero point itself rather than something
rounded near it.

The axis. Per-channel quantization exists because a single outlier channel drags
a per-tensor scale wide enough to flatten every other channel into a handful of
levels. Each channel therefore gets its own scale, computed from its own values,
which means reducing over every axis *except* the channel axis. Reducing over
the channel axis instead returns arrays of exactly the right shape holding
entirely the wrong numbers, and the shape is the part people check.

What ties it together is one inequality. Rounding to the nearest multiple of the
scale moves a value by at most half a scale, so for anything inside the
representable range the round trip satisfies

    |x - dequantize(quantize(x))| <= scale / 2

and that bound is the entire guarantee quantization offers. If it is violated,
the scale, the range, or the clamp is wrong — there is nothing else it could be.
"""

from __future__ import annotations

import numpy as np


def _integer_range(num_bits: int, symmetric: bool) -> tuple[int, int]:
    """The smallest and largest integer a code word may take."""
    if symmetric:
        return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    return 0, 2**num_bits - 1


def _parameters(values: np.ndarray, num_bits: int, symmetric: bool) -> tuple[float, int]:
    """The scale and zero point for one group of values."""
    qmin, qmax = _integer_range(num_bits, symmetric)

    if symmetric:
        peak = float(np.max(np.abs(values)))
        # All zeros: there is no scale that carries information, and any
        # non-zero one would still send every element to the zero point. One
        # keeps the round trip exact.
        return (1.0, 0) if peak == 0.0 else (peak / qmax, 0)

    # Zero is forced into the range. Without this an all-positive tensor would
    # place its zero point outside the interval it can represent, and zero —
    # the most common value in the network — would not round-trip.
    low = min(float(np.min(values)), 0.0)
    high = max(float(np.max(values)), 0.0)
    if high == low:  # both are zero, by construction of low and high
        return 1.0, 0

    scale = (high - low) / (qmax - qmin)
    zero_point = int(np.rint(qmin - low / scale))
    return scale, int(np.clip(zero_point, qmin, qmax))


def quantize_per_tensor(x, num_bits: int = 8, symmetric: bool = True):
    """Quantize the whole tensor with one scale. Returns (q, scale, zero_point)."""
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = _integer_range(num_bits, symmetric)
    scale, zero_point = _parameters(x, num_bits, symmetric)

    codes = np.rint(x / scale) + zero_point
    return np.clip(codes, qmin, qmax).astype(np.int32), scale, zero_point


def dequantize(q, scale, zero_point) -> np.ndarray:
    """Undo the mapping: scale * (q - zero_point), broadcasting as numpy does."""
    return np.asarray(scale, dtype=np.float64) * (
        np.asarray(q, dtype=np.float64) - np.asarray(zero_point, dtype=np.float64)
    )


def quantize_per_channel(x, axis: int, num_bits: int = 8, symmetric: bool = True):
    """Quantize with one scale per slice along `axis`. Returns (q, scale, zero_point)."""
    x = np.asarray(x, dtype=np.float64)
    axis = axis % x.ndim
    qmin, qmax = _integer_range(num_bits, symmetric)

    n_channels = x.shape[axis]
    scales = np.empty(n_channels, dtype=np.float64)
    zero_points = np.empty(n_channels, dtype=np.int32)
    for channel in range(n_channels):
        # Everything the channel contains, with the channel axis removed: the
        # reduction runs over all the other axes.
        group = np.take(x, channel, axis=axis)
        scales[channel], zero_points[channel] = _parameters(group, num_bits, symmetric)

    broadcast = [1] * x.ndim
    broadcast[axis] = n_channels
    codes = np.rint(x / scales.reshape(broadcast)) + zero_points.reshape(broadcast)
    return np.clip(codes, qmin, qmax).astype(np.int32), scales, zero_points
