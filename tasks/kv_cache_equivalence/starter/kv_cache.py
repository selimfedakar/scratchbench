"""Causal attention with rotary positions and a KV-cache."""

from __future__ import annotations

import numpy as np


class CachedAttention:
    """One causal attention layer that can decode with or without a cache."""

    def __init__(self, wq, wk, wv, wo, n_heads: int, rope_base: float = 10_000.0) -> None:
        self.wq = np.asarray(wq, dtype=np.float64)
        self.wk = np.asarray(wk, dtype=np.float64)
        self.wv = np.asarray(wv, dtype=np.float64)
        self.wo = np.asarray(wo, dtype=np.float64)
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.head_dim = self.wq.shape[0] // n_heads

    def forward(self, x, cache=None):
        """Run the layer over `x`, continuing from `cache` if one is given.

        Returns `(output, (keys, values))`.
        """
        raise NotImplementedError
