"""Batch normalisation's backward pass, one micro-batch at a time."""

from __future__ import annotations

import torch


def chunk_parameter_gradients(x_chunk, dy_chunk, mask_chunk, mean, var, eps):
    """One chunk's share of `(dgamma, dbeta)`."""
    raise NotImplementedError


def chunk_input_gradients(
    x_chunk,
    dy_chunk,
    mask_chunk,
    mean,
    var,
    gamma,
    eps,
    dgamma,
    dbeta,
    total_count,
):
    """The gradient with respect to this chunk's inputs."""
    raise NotImplementedError


def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, mean, var, gamma, eps):
    """Returns `(dx_chunks, dgamma, dbeta)`."""
    raise NotImplementedError
