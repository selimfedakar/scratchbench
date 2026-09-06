# Batch normalisation backward, one micro-batch at a time

Fill in `chunked_batchnorm.py`. torch only:

```python
def chunk_parameter_gradients(x_chunk, dy_chunk, mask_chunk, mean, var, eps
                              ) -> tuple[Tensor, Tensor]

def chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, mean, var, gamma, eps,
                          dgamma, dbeta, total_count) -> Tensor

def chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, mean, var,
                               gamma, eps) -> tuple[list[Tensor], Tensor, Tensor]
```

Batch normalisation is the layer whose output at one position is a function of
every other position in the batch, which is what makes it interesting the moment
the batch stops fitting anywhere. The forward pass takes the split in its
stride, because sums and sums of squares add up: a caller can walk the batch in
micro-batches, accumulate the partial statistics, and hand back a mean and a
variance for the whole thing. The backward pass is what this task is about, and
the caller has arranged it so that no function here ever sees more than one
micro-batch.

## The forward pass you are differentiating

A chunk of the batch is a tensor of shape `(rows, channels, positions)`, and its
mask has shape `(rows, positions)` — one flag per row and position, shared by
every channel, true where there is real data and false where there is padding.
The chunks are what the caller chose to cut the batch into, they need not be the
same size, and the batch is their concatenation along the first dimension in the
order they are given.

Each channel is normalised on its own, over every kept position of every row of
the whole batch. `mean` and `var`, both of shape `(channels,)`, are those
statistics: the average of the kept values, and the average of their squared
deviations from that average, with no correction to the denominator. They were
computed from the same batch you are now differentiating, which is the reason
this is not a pointwise operation. A kept position's output is its deviation
from the channel's mean, divided by the square root of the channel's variance
plus `eps`, scaled by `gamma` and shifted by a `beta` you are not given; a padded
position's output is zero and is not a function of anything.

`dy_chunk` has the shape of `x_chunk` and is the gradient of a scalar loss with
respect to that chunk's output. You are asked for the gradient of that same loss
with respect to the inputs, and with respect to `gamma` and `beta` — the last
two in that order, as `(dgamma, dbeta)`, both of shape `(channels,)`.
`torch.autograd` is no help here and there is no version of it that would be:
none of these functions holds enough of the batch to rebuild the forward pass
that produced the statistics it was handed.

## The three functions

`chunk_parameter_gradients` returns one chunk's **contribution** to
`(dgamma, dbeta)` — the chunks add up to the whole batch's answer, they do not
each compute it.

`chunk_input_gradients` returns the finished gradient for the rows of one chunk,
with the shape of `x_chunk`. Its `dgamma`, `dbeta` and `total_count` arguments
describe the **whole** batch: the first two are the assembled parameter
gradients, and `total_count` is the number of kept positions the statistics were
taken over. Everything else it is given belongs to the chunk in front of it, and
it is never given another one.

`chunked_batchnorm_backward` drives the two above over a batch it has been
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
  with the analytic answer to the precision `float64` allows, which is many
  digits rather than a few.
