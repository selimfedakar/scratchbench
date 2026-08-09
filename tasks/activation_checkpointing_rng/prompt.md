# Recomputing a block that was not deterministic the first time

Fill in `checkpointed_mlp.py`. torch only:

```python
def dropout_mask(shape, p: float, generator) -> Tensor
def forward_and_backward(x, blocks, p: float, generator, dy) -> tuple[Tensor, Tensor, list]
```

Activation checkpointing is the trade that lets a model be deeper than its
memory: the forward pass keeps only the boundaries between blocks and throws
away everything inside them, and the backward pass runs each block forward again
to rebuild what it needs. It costs one extra forward and it is supposed to be
otherwise invisible — same outputs, same gradients, same everything.

Dropout is what makes that interesting. The recomputation has to reproduce the
activations the first pass produced, and half of what the first pass produced
came out of a random number generator that has since moved on. Getting the same
numbers back is not a matter of doing the same arithmetic.

## The block

`blocks` is a list of `(w1, w2)` pairs. A block takes its input `u`, of shape
`(rows, width)`, and computes

```
relu(u @ w1)  ->  apply the dropout mask  ->  @ w2  ->  add u back
```

so `w1` is `(width, hidden)`, `w2` is `(hidden, width)`, and the block's output
has the same shape as its input. The stack runs the blocks in order, first to
last. Everything is `float64`.

## The mask

`dropout_mask(shape, p, generator)` returns the multiplier for a tensor of that
shape: inverted dropout, so a surviving position is scaled up by one over the
probability of surviving and a dropped one is zero. A position survives when its
own uniform draw is **at least** `p`, and the draws come from one call for the
whole tensor, in `float64`, out of `generator` and nowhere else.

`p` is at least zero and less than one. The mask is drawn whether or not it can
change anything — at `p` of zero every position survives and the draw still
happens, because how much randomness a forward pass consumes must not depend on
the value of a hyper-parameter.

## What the step has to produce

`forward_and_backward(x, blocks, p, generator, dy)` runs the stack forward from
`x`, then backward from `dy`, which is the gradient of the loss with respect to
the stack's output. It returns `(y, dx, grads)`, where `grads` holds one
`(dw1, dw2)` pair per block, in the same order `blocks` came in.

The result has to be the one you would have got from a stack that kept every
activation and never recomputed anything: same `y`, same gradients, to the
precision `float64` allows.

And one more thing, which is the same claim from the other side. When the call
returns, `generator` must be in exactly the state a plain forward pass over this
stack would have left it in — no further along, and not rewound. Anything that
draws randomness after this step, including the next step, has to receive what
it would have received if checkpointing were not in the picture. A step that
gets the gradients right and the generator wrong is a training run that produces
correct-looking numbers and cannot be reproduced.

## Conventions

- torch only, on the CPU, and `torch.autograd` will not help you here: the
  activations the backward needs no longer exist when it starts.
- `generator` is a `torch.Generator`. It is the only source of randomness.
- Nothing returned may carry a gradient history, and none of `x`, `dy` or the
  weights is yours to modify.
- There is at least one block, at least one row of input, and every dimension is
  at least one.
