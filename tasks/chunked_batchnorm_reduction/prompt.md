# Batch normalisation backward, with one reduction over the batch

Fill in `batchnorm_reduction.py`. torch only:

```python
def chunk_reduction(x_chunk, dy_chunk, mask_chunk) -> Tensor

def chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, gamma, eps,
                          reduction_total) -> Tensor

def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, gamma, eps
                               ) -> tuple[list[Tensor], Tensor, Tensor]
```

Batch normalisation is the layer whose output at one position is a function of
every other position in the batch, so a caller that cannot hold the batch has to
reduce across it. This one reduces **once**, and the shape of that is fixed:

1. It walks the chunks and asks each for a buffer from `chunk_reduction`. It adds
   those buffers together and never looks inside one.
2. It walks the chunks again and asks `chunk_input_gradients` for each chunk's
   gradients, handing back the total from the first walk.

There is no third walk and no second reduction. Whatever the answer needs from
the whole batch is what the buffer has to carry, and what the buffer holds is
entirely yours to decide.

## The forward pass you are differentiating

A chunk of the batch is a tensor of shape `(rows, channels, positions)`, and its
mask has shape `(rows, positions)` — one flag per row and position, shared by
every channel, true where there is real data and false where there is padding.
The chunks are what the caller chose to cut the batch into, they need not be the
same size, and the batch is their concatenation along the first dimension in the
order they are given.

Each channel was normalised on its own, over every kept position of every row of
the whole batch. The mean is the average of those values and the variance is the
average of their squared deviations from it, with no correction to the
denominator. A kept position's output is its deviation from the channel's mean,
divided by the square root of the channel's variance plus `eps`, scaled by
`gamma` and shifted by a `beta` you are not given; a padded position's output is
zero and is not a function of anything. Note what is *not* in any signature
above: the mean and the variance are not handed to you either, and neither is the
number of positions they were taken over.

`dy_chunk` has the shape of `x_chunk` and is the gradient of a scalar loss with
respect to that chunk's output. You are asked for the gradient of that same loss
with respect to the inputs, and with respect to `gamma` and `beta` — the last two
in that order, as `(dgamma, dbeta)`, both of shape `(channels,)`.
`torch.autograd` is no help here and there is no version of it that would be:
neither chunk function ever holds enough of the batch to rebuild the forward pass.

## The three functions

`chunk_reduction` returns a two-dimensional tensor whose last dimension is
`channels`. Its first dimension, and the meaning of everything in it, is your
design. The caller adds these together across the chunks, so whatever you put in
one has to be a quantity that is summed rather than averaged or compared, and
every chunk has to return the same shape.

`chunk_input_gradients` returns the finished gradient for the rows of one chunk,
with the shape of `x_chunk`. `reduction_total` is the sum of every chunk's
buffer, including this one's. Everything else it is given belongs to the chunk in
front of it, and it is never given another one.

`chunked_batchnorm_backward` performs the two walks over a batch it has been
handed as lists — `x_chunks`, `dy_chunks` and `mask_chunks`, of equal length and
aligned — and returns `(dx_chunks, dgamma, dbeta)`, where `dx_chunks` is a list
holding one gradient per chunk, in the order the chunks arrived. How the batch
was cut up is a memory knob and nothing else: the numbers coming out must not
depend on it.

## Conventions

- torch only, on the CPU, everything `float64` except the masks, which are
  `bool`. Nothing you return may carry a gradient history, and none of the
  tensors you are handed is yours to modify.
- The mask is arbitrary. Nothing promises that a row has a kept position in it,
  or that a chunk does; the batch as a whole has at least one, and `eps` is a
  positive number, so nothing here divides by zero. A padded position takes a
  gradient of exactly zero rather than something that is not a number.
- There is at least one chunk, every chunk has at least one row, and every
  dimension is at least one. Channels do not interact.
- The gradients are exact, not approximate: whatever this returns should agree
  with the analytic answer to the precision `float64` allows over quantities of
  order one, which is many digits rather than a few.
