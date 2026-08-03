"""Integer quantization: the affine mapping, and the bound it guarantees."""

from __future__ import annotations

import numpy as np


def quantize_per_tensor(x, num_bits: int = 8, symmetric: bool = True):
    """Quantize the whole tensor with one scale. Returns (q, scale, zero_point)."""
    raise NotImplementedError


def dequantize(q, scale, zero_point) -> np.ndarray:
    """Undo the mapping: scale * (q - zero_point), broadcasting as numpy does."""
    raise NotImplementedError


def quantize_per_channel(x, axis: int, num_bits: int = 8, symmetric: bool = True):
    """Quantize with one scale per slice along `axis`. Returns (q, scale, zero_point)."""
    raise NotImplementedError
