"""Reference solution — a sharded sampler that survives resharding and resuming.

A dataloader is the part of a training run nobody reads and everybody trusts.
It has no loss curve of its own, so when it is wrong the model simply learns
from the wrong data and the run looks normal.

Four things have to hold at once, and each one is a real bug that has shipped
in real training code.

The order must depend on the epoch and the seed, and on nothing else. It is
tempting to shuffle after splitting, one stream per rank, because that is the
shorter program. Then the sample order becomes a function of the world size, a
job resumed on a different number of machines sees different data, and the
run is no longer the run that was checkpointed.

The ranks must be disjoint. Each sample belongs to one worker per epoch or the
gradient double-counts it.

Every rank must receive the same number of batches. Distributed training
synchronises at every step, so a rank that runs out first hangs the job at a
collective while the others wait for a gradient that is never coming. That is
why the tail is either dropped or padded, and never left ragged.

And resuming must continue rather than restart. A job that dies eighty percent
through an epoch and comes back at batch zero silently trains four fifths of
that epoch twice.
"""

from __future__ import annotations

import random

#: One sample: which shard it lives in, and its offset inside that shard.
Sample = tuple[int, int]


class ShardedSampler:
    """Decides which samples each rank sees, in which order, in which batch."""

    def __init__(
        self,
        shard_sizes,
        batch_size: int,
        world_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.shard_sizes = list(shard_sizes)
        self.batch_size = batch_size
        self.world_size = world_size
        self.seed = seed
        self.drop_last = drop_last

    @property
    def total_samples(self) -> int:
        return sum(self.shard_sizes)

    def epoch_order(self, epoch: int) -> list[Sample]:
        """The whole dataset, shuffled by (seed, epoch) and nothing else."""
        samples = [
            (shard, offset)
            for shard, size in enumerate(self.shard_sizes)
            for offset in range(size)
        ]
        # A string seed keeps the derivation obvious and avoids the collisions
        # that seed + epoch invites: seed 1 epoch 2 and seed 2 epoch 1 are
        # different runs and must not share a shuffle.
        random.Random(f"{self.seed}:{epoch}").shuffle(samples)
        return samples

    def num_batches(self) -> int:
        """Batches per rank. Identical for every rank, by construction."""
        per_step = self.batch_size * self.world_size
        if self.drop_last:
            return self.total_samples // per_step
        return -(-self.total_samples // per_step)  # ceiling division

    def batches(self, epoch: int, rank: int, start_batch: int = 0) -> list[list[Sample]]:
        """The batches `rank` should see in `epoch`, from `start_batch` on."""
        order = self.epoch_order(epoch)
        per_step = self.batch_size * self.world_size
        needed = self.num_batches() * per_step

        if needed <= len(order):
            order = order[:needed]
        else:
            # Pad from the front of this epoch's own order, so the duplicates
            # are drawn from the same distribution and the same shuffle.
            shortfall = needed - len(order)
            order = order + [order[i % len(order)] for i in range(shortfall)]

        # Stride, not contiguous chunks: rank r takes every world_size-th
        # sample. Disjoint by construction, and equal in length because the
        # length is now a multiple of the world size.
        mine = order[rank :: self.world_size]
        batches = [
            mine[start : start + self.batch_size]
            for start in range(0, len(mine), self.batch_size)
        ]
        return batches[start_batch:]
