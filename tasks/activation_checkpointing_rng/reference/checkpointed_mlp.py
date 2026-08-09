"""Reference solution — activation checkpointing across a stochastic block.

Checkpointing trades arithmetic for memory: the forward pass keeps only the
boundaries between blocks and throws the inside of each one away, and the
backward pass runs each block forward again to rebuild what it needs. The trade
is only honest if the second forward produces the *same* activations as the
first, and the moment a block contains dropout that stops being automatic.
Randomness is state, the first forward consumed some, and the recomputation has
to consume the same draws rather than the next ones.

Restoring the generator before each recomputation is the half everybody writes.
There is a second half, and it is where implementations quietly go wrong: once
the recomputation is done, the generator is sitting where that block's *forward*
left it, several blocks back. Whatever draws next — the next training step's
dropout, an augmentation, a sampled batch — gets randomness that has already
been used. Nothing about the gradients this step says so; they are perfect. The
divergence starts on the step after, and it looks like a training run that will
not reproduce.

So the bracket is the mechanism: remember where the forward ended, restore for
each recomputation, and put the generator back at the end. Two states, not one.

Everything else is small on purpose. The block is a residual MLP with a ReLU and
a dropout in the middle, and its backward is four matrix products and two masks.
The mask is drawn whether or not it will change anything — at `p` of zero it is
all ones and the draw still happens, because the amount of randomness a forward
pass consumes cannot depend on the value of a hyper-parameter, or two runs of
the same model become unreproducible against each other.
"""

from __future__ import annotations

import torch


def dropout_mask(shape, p: float, generator) -> torch.Tensor:
    """The multiplier inverted dropout applies to a block's activations."""
    keep = torch.rand(tuple(shape), generator=generator, dtype=torch.float64) >= p
    return keep.to(torch.float64) / (1.0 - p)


def _block_forward(u, w1, w2, p, generator):
    """One block, keeping everything. Used by the forward and the recompute."""
    h = u @ w1
    a = torch.relu(h)
    mask = dropout_mask(a.shape, p, generator)
    dropped = a * mask
    return h, mask, dropped, dropped @ w2 + u


def forward_and_backward(x, blocks, p: float, generator, dy):
    """Run the stack forward, then backward, recomputing each block as it goes.

    Returns `(y, dx, grads)`.
    """
    # Only the boundaries survive the forward. Everything inside a block is
    # rebuilt from its input during the backward, which is the whole point.
    boundaries = [x]
    states = []
    for w1, w2 in blocks:
        states.append(generator.get_state())
        _, _, _, out = _block_forward(boundaries[-1], w1, w2, p, generator)
        boundaries.append(out)

    # Where the forward pass finished. The recomputations below are about to
    # rewind the generator repeatedly, and this is what gets put back.
    after_forward = generator.get_state()

    grads: list[tuple[torch.Tensor, torch.Tensor]] = []
    d_out = dy
    for index in reversed(range(len(blocks))):
        w1, w2 = blocks[index]
        u = boundaries[index]

        generator.set_state(states[index])
        h, mask, dropped, _ = _block_forward(u, w1, w2, p, generator)

        d_dropped = d_out @ w2.T
        d_w2 = dropped.T @ d_out
        d_a = d_dropped * mask
        d_h = d_a * (h > 0)
        d_w1 = u.T @ d_h
        # The residual adds the block's input to its output, so the gradient
        # arrives at the input by both routes.
        d_out = d_h @ w1.T + d_out

        grads.append((d_w1, d_w2))

    generator.set_state(after_forward)
    return boundaries[-1], d_out, list(reversed(grads))
