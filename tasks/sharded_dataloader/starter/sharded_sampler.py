"""A sharded, resumable, reshardable sampler for distributed training."""

from __future__ import annotations

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
        raise NotImplementedError

    def num_batches(self) -> int:
        """Batches per rank. Identical for every rank."""
        raise NotImplementedError

    def batches(self, epoch: int, rank: int, start_batch: int = 0) -> list[list[Sample]]:
        """The batches `rank` should see in `epoch`, from `start_batch` on."""
        raise NotImplementedError
