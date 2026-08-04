"""Attention streamed over blocks of keys, with an online softmax."""

from __future__ import annotations

import numpy as np

#: A partial result over some group of keys, for every query:
#: (running maximum, denominator, numerator).
State = tuple[np.ndarray, np.ndarray, np.ndarray]


def combine_states(left: State, right: State) -> State:
    """The state of the union of two groups of keys."""
    raise NotImplementedError


def attention_tiled(q, k, v, block_size: int, causal: bool = False):
    """Attention over `q`, `k`, `v`, accumulated `block_size` keys at a time.

    Returns `(output, logsumexp)`.
    """
    raise NotImplementedError
