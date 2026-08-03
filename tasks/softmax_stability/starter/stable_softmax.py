"""Numerically stable softmax, log-softmax and logsumexp."""

from __future__ import annotations

import numpy as np


def logsumexp(x, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log(sum(exp(x))) along `axis`, computed without overflowing."""
    raise NotImplementedError


def softmax(x, axis: int = -1) -> np.ndarray:
    """Probabilities along `axis`, summing to one."""
    raise NotImplementedError


def log_softmax(x, axis: int = -1) -> np.ndarray:
    """log(softmax(x)) along `axis`, without going through softmax."""
    raise NotImplementedError
