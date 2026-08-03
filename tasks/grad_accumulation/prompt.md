# Gradient accumulation that changes only the memory

Fill in `accumulate.py`. PyTorch, CPU:

```python
def accumulated_step(model, optimizer, loss_fn, x, y, micro_batch_size: int) -> float
```

Take one optimiser step over the batch `(x, y)` while only ever putting
`micro_batch_size` samples through the model at a time, and return the mean
loss over the whole batch.

This is the trick that lets a large batch fit on a small machine: activations
dominate memory, parameters do not, so a batch of 32 split into four passes of
8 costs a quarter of the activation memory. What it must not cost is a
different answer. After `accumulated_step` returns, every parameter must hold
exactly the value it would have held after the obvious version — zero the
gradients, one forward and backward over all of `x`, one `optimizer.step()` —
down to floating-point noise.

`loss_fn(predictions, targets)` returns a scalar that is the **mean** over the
samples it was given. That is the detail the arithmetic turns on: backward
passes add into `.grad`, so several micro-batches accumulate on their own, but
a sum of means is not the mean of the whole. The batch is split into contiguous
chunks taken in order, and the last one is short whenever `micro_batch_size`
does not divide the batch — every real training run hits that case on the last
batch of an epoch.

## Conventions

- `x` and `y` are tensors whose first dimension is the batch. They have the
  same number of samples, at least one, and they are not modified.
- `micro_batch_size` is at least one and may exceed the batch size, in which
  case there is a single chunk.
- The optimiser is a normal `torch.optim` optimiser and may be stateful —
  momentum, Adam, weight decay. It is stepped exactly once per call.
- Whatever was in `.grad` before the call has nothing to do with this step.
- The gradients are left in place afterwards, holding the full-batch gradient,
  the way they would be after an ordinary backward pass.
- The return value is a plain Python `float`, the mean loss over all of `x`
  measured with the parameters as they were on entry — the loss the step was
  taken from, not a re-measurement afterwards.
- The model is deterministic in training mode: no dropout, no batch
  normalisation, nothing whose output depends on how the batch was split. Leave
  its mode alone.
- Everything runs on CPU in float64.
