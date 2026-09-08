"""Batch normalisation's backward pass in one reduction over the chunks."""

from __future__ import annotations

import torch


def chunk_reduction(x_chunk, dy_chunk, mask_chunk):
    """This chunk's contribution to the buffer the caller sums."""
    raise NotImplementedError


def chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, gamma, eps, reduction_total):
    """The gradient with respect to this chunk's inputs."""
    raise NotImplementedError


def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, gamma, eps):
    """Returns `(dx_chunks, dgamma, dbeta)`."""
    raise NotImplementedError
