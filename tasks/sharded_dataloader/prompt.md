# A sharded sampler that survives resharding and resuming

Fill in `sharded_sampler.py`. Standard library only:

```python
class ShardedSampler:
    def __init__(self, shard_sizes, batch_size: int, world_size: int, seed: int,
                 drop_last: bool = False)
    def epoch_order(self, epoch: int) -> list[tuple[int, int]]
    def num_batches(self) -> int
    def batches(self, epoch: int, rank: int, start_batch: int = 0) -> list[list[tuple[int, int]]]
```

This is the part of a training run that has no loss curve of its own. A
dataloader that hands out the wrong samples does not crash and does not spike
the loss — the model simply learns from something other than what the
experiment says it learned from, and nothing in the logs mentions it.

The dataset is a list of shards. `shard_sizes[i]` is how many samples shard `i`
holds, and a sample is identified by the pair `(shard_index, offset)`. There is
no data here, only bookkeeping: this class decides who sees what, in which
order, in which batch.

## The order

`epoch_order(epoch)` returns every sample in the dataset exactly once, shuffled.

The shuffle is a function of the seed and the epoch, and of nothing else. Not
the world size, not the batch size, not the rank, not whether the tail is
dropped. A job checkpointed on eight machines and resumed on four has to see the
same data in the same order, and a shuffle that happens per rank after the split
quietly makes the data a function of the cluster shape. Different epochs give
different orders; the same seed and epoch give the same order every time, from
any instance.

Any shuffle satisfying that is acceptable — the specific permutation is yours.

## The split

Rank `r` takes every `world_size`-th sample of the epoch order, starting at
index `r`, and cuts its share into contiguous batches of `batch_size`. Ranks are
therefore disjoint: no position in the order goes to two workers.

Every rank receives the same number of batches, and `num_batches()` is that
number. This is not a nicety. Distributed training synchronises at every step,
so a rank that runs out of data one batch early hangs the whole job waiting at a
collective for a gradient that is never coming. Every batch is full;
`batch_size` samples, never fewer.

Since the dataset does not generally divide by `batch_size * world_size`, the
tail is handled one of two ways and `drop_last` chooses which:

- **`drop_last=True`** — keep the largest prefix of the epoch order that is a
  multiple of `batch_size * world_size` and drop the rest. Fewer than one full
  step of samples is lost, and no sample is seen twice.
- **`drop_last=False`** — extend the epoch order by repeating samples from its
  own front until the length is a multiple of `batch_size * world_size`. Fewer
  than one full step of samples is repeated, every sample is seen at least once,
  and the repeats are drawn from this epoch's order rather than from anywhere
  else. This is the mode where the dataset may be smaller than a single step, in
  which case the padding wraps around it as many times as it needs to.

## Resuming

`start_batch` is what makes a crashed run resumable: `batches(epoch, rank,
start_batch=k)` returns the batches from `k` onwards, identical to the full list
sliced from `k`. A job that dies eighty percent through an epoch and comes back
at batch zero trains four fifths of that epoch twice.

Nothing is stored on the instance between calls. The epoch, the rank and the
batch index are the whole state, which is why a fresh `ShardedSampler` built
from the same arguments resumes correctly.

## Conventions

- `shard_sizes` holds non-negative integers and is not modified. A shard may be
  empty. The dataset as a whole has at least one sample.
- `batch_size`, `world_size` and `epoch` are at least one, at least one, and at
  least zero. `rank` is in `range(world_size)`. `start_batch` is between zero
  and `num_batches()` inclusive.
- Returned samples are plain `(int, int)` tuples; `batches` returns a list of
  lists of them. An empty result is an empty list, not `None`.
- `total_samples` is available as a property and is the sum of the shard sizes.
- With `drop_last=True` and a step larger than the dataset, there are zero
  batches, and that answer is the same for every rank.
- No numpy, no torch. The standard library shuffles well enough and this task is
  bookkeeping, not arithmetic.
