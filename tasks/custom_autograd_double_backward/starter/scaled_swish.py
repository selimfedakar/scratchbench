"""A custom autograd Function whose backward is itself differentiable."""

from __future__ import annotations

import torch


class ScaledSwish(torch.autograd.Function):
    """`w * x * sigmoid(alpha * x)`, differentiated by hand."""

    @staticmethod
    def forward(ctx, x, w, alpha):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


def scaled_swish(x, w, alpha):
    """`w * x * sigmoid(alpha * x)`, through the Function above."""
    return ScaledSwish.apply(x, w, alpha)
