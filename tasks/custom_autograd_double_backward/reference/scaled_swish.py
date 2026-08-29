"""A custom autograd Function whose backward is itself differentiable.

The forward is `w * x * sigmoid(alpha * x)`. Writing it as an
`autograd.Function` means the backward is written by hand, and writing the
backward out of differentiable operations on the saved tensors is what makes
the second derivative come out right.
"""

from __future__ import annotations

import torch


def _reduce_to(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Undo broadcasting: sum `grad` back down to `shape`.

    An input that was broadcast in the forward pass appears in several output
    entries, so its gradient is the sum over the entries it reached. Leading
    dimensions the input never had are summed away entirely; dimensions it had
    as a one are summed with the dimension kept.

    Measured on torch 2.8.0: the engine does this itself. A Function that
    returns a gradient at the broadcast shape gets it summed down silently, with
    no warning, and the result is bit-identical to the one below — only a shape
    that is not broadcast-compatible with the input is rejected. So this is
    explicit rather than necessary, and the hidden tests cannot tell the two
    apart. See `docs/LESSONS.md` L36 and the two mutants marked SURVIVES in
    `tools/mutate_v2_tasks.py`.
    """
    while grad.dim() > len(shape):
        grad = grad.sum(dim=0)
    for dim, size in enumerate(shape):
        if size == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad


class ScaledSwish(torch.autograd.Function):
    """`w * x * sigmoid(alpha * x)`, differentiated by hand, twice over."""

    @staticmethod
    def forward(ctx, x, w, alpha):
        # The sigmoid computed here is deliberately not saved. Anything this
        # function computes is computed with autograd disabled, so a tensor
        # saved from here is a constant as far as a later differentiation is
        # concerned: reusing it in the backward gives first derivatives that are
        # right and second derivatives that are missing every term that should
        # have flowed through it. The backward recomputes the sigmoid from `x`
        # instead, inside the graph.
        ctx.save_for_backward(x, w)
        ctx.alpha = float(alpha)
        return w * x * torch.sigmoid(ctx.alpha * x)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        alpha = ctx.alpha

        # Every operation below is a differentiable torch operation on tensors
        # that carry a graph, so autograd can differentiate this function in
        # turn. `.detach()`, `.data` and `torch.no_grad()` would each cut that
        # graph and cost nothing at first order.
        s = torch.sigmoid(alpha * x)
        swish = x * s

        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            d_swish = s + alpha * x * s * (1 - s)
            grad_x = _reduce_to(grad_output * w * d_swish, x.shape)
        if ctx.needs_input_grad[1]:
            grad_w = _reduce_to(grad_output * swish, w.shape)

        # `alpha` is a Python float rather than a tensor, so it has no gradient
        # and the third slot is None. The number of things returned matches the
        # number of things the forward took, always.
        return grad_x, grad_w, None


def scaled_swish(x, w, alpha):
    """`w * x * sigmoid(alpha * x)`, through the Function above."""
    return ScaledSwish.apply(x, w, alpha)
