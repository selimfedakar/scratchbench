"""Reference solution — gradient accumulation that is exactly one large step.

Gradient accumulation exists because activations, not parameters, are what fill
the memory. Splitting a batch into micro-batches and summing their gradients
buys the large-batch update at the memory cost of the small one, and the only
thing it is allowed to change is peak memory.

The arithmetic that makes that true is one factor. `backward` accumulates into
`.grad` rather than overwriting it, so the gradients of several micro-batches
add up on their own. But each micro-batch loss is a *mean* over its own
samples, and the sum of means is not the mean of the whole batch. Scaling each
micro-batch loss by its share of the batch — its sample count over the total —
is what turns the sum back into the gradient of the full-batch mean.

Dividing by the number of micro-batches instead is the version everyone writes
first. It agrees with this one whenever the batch divides evenly and is wrong
whenever it does not, which is every last batch of every epoch, where a short
micro-batch quietly gets the same weight as a full one.

The other two requirements are structural rather than numerical: the gradients
must start at zero, or whatever was left over from a previous step is folded
into this one, and `step` runs once at the end. Stepping inside the loop makes
each micro-batch a small update of its own, and with a stateful optimiser like
Adam or SGD with momentum, no rescaling recovers the single large step that was
supposed to happen.
"""

from __future__ import annotations


def accumulated_step(model, optimizer, loss_fn, x, y, micro_batch_size: int) -> float:
    """One optimiser step over `x`, taken `micro_batch_size` samples at a time.

    Returns the mean loss over the whole batch.
    """
    n_samples = x.shape[0]

    # Nothing from an earlier step is allowed to survive into this one.
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    for start in range(0, n_samples, micro_batch_size):
        stop = min(start + micro_batch_size, n_samples)
        # Weight by the share of the batch, not by the number of chunks: the
        # last chunk is usually shorter and must not count as a whole one.
        share = (stop - start) / n_samples

        loss = loss_fn(model(x[start:stop]), y[start:stop])
        (loss * share).backward()

        total_loss += loss.item() * share

    optimizer.step()
    return total_loss
