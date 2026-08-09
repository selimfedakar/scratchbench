"""The backward pass of blocked attention, derived rather than differentiated."""

from __future__ import annotations

import math

import torch


def key_block_gradients(
    q,
    k_block,
    v_block,
    o,
    lse,
    do,
    key_offset: int,
    query_offset: int,
    causal: bool = False,
):
    """What one block of keys contributes to `(dq, dk, dv)`.

    Returns `(dq_contribution, dk_block, dv_block)`.
    """
    raise NotImplementedError


def flash_attention_backward(q, k, v, o, lse, do, block_size: int, causal: bool = False):
    """Gradients of the attention forward pass, one block of keys at a time.

    Returns `(dq, dk, dv)`.
    """
    raise NotImplementedError
