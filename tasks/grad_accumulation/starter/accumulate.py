"""Gradient accumulation: a large-batch step taken a few samples at a time."""

from __future__ import annotations


def accumulated_step(model, optimizer, loss_fn, x, y, micro_batch_size: int) -> float:
    """One optimiser step over `x`, taken `micro_batch_size` samples at a time.

    Returns the mean loss over the whole batch.
    """
    raise NotImplementedError
