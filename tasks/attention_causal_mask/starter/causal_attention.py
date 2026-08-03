"""Causal attention over a padded batch."""

from __future__ import annotations

import numpy as np


def causal_mask(n_queries: int, n_keys: int) -> np.ndarray:
    """Additive mask of shape (n_queries, n_keys): 0 to look, -inf to not."""
    raise NotImplementedError


def attention(q, k, v, key_padding_mask=None) -> np.ndarray:
    """Scaled dot-product attention, causal, with optional key padding."""
    raise NotImplementedError
