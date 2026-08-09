"""Activation checkpointing across a block that contains dropout."""

from __future__ import annotations

import torch


def dropout_mask(shape, p: float, generator) -> torch.Tensor:
    """The multiplier inverted dropout applies to a block's activations."""
    raise NotImplementedError


def forward_and_backward(x, blocks, p: float, generator, dy):
    """Run the stack forward, then backward, recomputing each block as it goes.

    Returns `(y, dx, grads)`.
    """
    raise NotImplementedError
