"""Hidden tests — sharded_dataloader.

Every assertion is licensed by a sentence in prompt.md. No tolerances: this
task is about which samples go where, and that is exact.

The tests never assume a particular shuffle. They check the properties the
prompt promises — that the order is a permutation, that it depends on the seed
and the epoch and on nothing else, that the ranks partition it, that every rank
gets the same count, and that resuming continues. Any correct shuffle passes.
"""

from collections import Counter

import pytest

from sharded_sampler import ShardedSampler


SHARDS = [7, 3, 11, 1, 8]  # 30 samples across five shards of unequal size


def every_sample(shard_sizes) -> list[tuple[int, int]]:
    return [(shard, offset) for shard, size in enumerate(shard_sizes) for offset in range(size)]


def collect(sampler: ShardedSampler, epoch: int) -> list[tuple[int, int]]:
    """Every sample handed out in one epoch, across all ranks."""
    seen = []
    for rank in range(sampler.world_size):
        for batch in sampler.batches(epoch, rank):
            seen.extend(batch)
    return seen


# -- the epoch order -------------------------------------------------------


def test_the_order_is_a_permutation_of_the_dataset():
    order = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=0).epoch_order(0)
    assert sorted(order) == sorted(every_sample(SHARDS))


def test_samples_are_shard_and_offset_pairs():
    order = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=0).epoch_order(0)
    for shard, offset in order:
        assert isinstance(shard, int) and isinstance(offset, int)
        assert 0 <= shard < len(SHARDS)
        assert 0 <= offset < SHARDS[shard]


def test_the_same_seed_and_epoch_give_the_same_order():
    first = ShardedSampler(SHARDS, 4, 2, seed=7).epoch_order(3)
    second = ShardedSampler(SHARDS, 4, 2, seed=7).epoch_order(3)
    assert first == second


def test_the_order_is_stable_across_repeated_calls():
    sampler = ShardedSampler(SHARDS, 4, 2, seed=7)
    assert sampler.epoch_order(1) == sampler.epoch_order(1)


def test_different_epochs_shuffle_differently():
    sampler = ShardedSampler(SHARDS, 4, 2, seed=7)
    orders = [tuple(sampler.epoch_order(epoch)) for epoch in range(5)]
    assert len(set(orders)) == 5


def test_different_seeds_shuffle_differently():
    orders = {tuple(ShardedSampler(SHARDS, 4, 2, seed=seed).epoch_order(0)) for seed in range(5)}
    assert len(orders) == 5


def test_the_order_does_not_depend_on_the_world_size():
    # A job resumed on a different number of machines must see the same data.
    single = ShardedSampler(SHARDS, 4, world_size=1, seed=11).epoch_order(2)
    many = ShardedSampler(SHARDS, 4, world_size=5, seed=11).epoch_order(2)
    assert single == many


def test_the_order_does_not_depend_on_the_batch_size_or_the_tail_policy():
    small = ShardedSampler(SHARDS, batch_size=1, world_size=2, seed=11, drop_last=True)
    large = ShardedSampler(SHARDS, batch_size=16, world_size=2, seed=11, drop_last=False)
    assert small.epoch_order(2) == large.epoch_order(2)


def test_the_shard_sizes_are_not_modified():
    sizes = list(SHARDS)
    sampler = ShardedSampler(sizes, 4, 2, seed=0)
    sampler.batches(0, 0)
    assert sizes == SHARDS


# -- every rank gets the same number of batches ---------------------------


@pytest.mark.parametrize("batch_size,world_size", [(1, 1), (4, 2), (3, 4), (5, 3), (8, 2), (16, 2)])
def test_every_rank_gets_the_same_number_of_batches(batch_size, world_size):
    for drop_last in (True, False):
        sampler = ShardedSampler(SHARDS, batch_size, world_size, seed=1, drop_last=drop_last)
        counts = {len(sampler.batches(0, rank)) for rank in range(world_size)}
        assert counts == {sampler.num_batches()}


@pytest.mark.parametrize("batch_size,world_size", [(1, 1), (4, 2), (3, 4), (5, 3), (7, 2)])
def test_every_batch_is_full(batch_size, world_size):
    for drop_last in (True, False):
        sampler = ShardedSampler(SHARDS, batch_size, world_size, seed=1, drop_last=drop_last)
        for rank in range(world_size):
            for batch in sampler.batches(0, rank):
                assert len(batch) == batch_size


def test_num_batches_agrees_with_what_is_handed_out():
    sampler = ShardedSampler(SHARDS, 4, 3, seed=1)
    assert len(sampler.batches(0, 0)) == sampler.num_batches()


# -- dropping the tail -----------------------------------------------------


def test_dropping_the_tail_never_repeats_a_sample():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=2, drop_last=True)
    seen = collect(sampler, 0)
    assert len(seen) == len(set(seen))


def test_dropping_the_tail_loses_less_than_one_full_step():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=2, drop_last=True)
    seen = collect(sampler, 0)
    dropped = sampler.total_samples - len(seen)
    assert 0 <= dropped < sampler.batch_size * sampler.world_size


def test_dropped_samples_are_a_subset_of_the_dataset():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=2, drop_last=True)
    assert set(collect(sampler, 0)) <= set(every_sample(SHARDS))


def test_a_batch_larger_than_the_dataset_yields_nothing_when_dropping():
    sampler = ShardedSampler(SHARDS, batch_size=64, world_size=2, seed=2, drop_last=True)
    assert sampler.num_batches() == 0
    for rank in range(2):
        assert sampler.batches(0, rank) == []


def test_dropping_is_exact_when_the_batch_divides_the_dataset():
    # 30 samples, 3 per rank, 2 ranks: nothing to drop, nothing to pad.
    sampler = ShardedSampler(SHARDS, batch_size=3, world_size=2, seed=2, drop_last=True)
    seen = collect(sampler, 0)
    assert sorted(seen) == sorted(every_sample(SHARDS))


# -- padding the tail ------------------------------------------------------


def test_padding_shows_every_sample_at_least_once():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=3, drop_last=False)
    seen = collect(sampler, 0)
    assert set(seen) == set(every_sample(SHARDS))


def test_padding_adds_less_than_one_full_step():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=3, drop_last=False)
    seen = collect(sampler, 0)
    added = len(seen) - sampler.total_samples
    assert 0 <= added < sampler.batch_size * sampler.world_size


def test_padding_repeats_only_samples_that_are_already_in_the_epoch():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=3, drop_last=False)
    counts = Counter(collect(sampler, 0))
    assert set(counts) == set(every_sample(SHARDS))
    assert max(counts.values()) <= 2


def test_padding_is_drawn_from_the_front_of_this_epochs_order():
    sampler = ShardedSampler(SHARDS, batch_size=4, world_size=2, seed=3, drop_last=False)
    counts = Counter(collect(sampler, 0))
    repeated = {sample for sample, count in counts.items() if count > 1}
    order = sampler.epoch_order(0)
    assert repeated == set(order[: len(repeated)])


def test_padding_covers_a_dataset_smaller_than_one_step():
    sampler = ShardedSampler([3], batch_size=4, world_size=2, seed=3, drop_last=False)
    assert sampler.num_batches() == 1
    seen = collect(sampler, 0)
    assert len(seen) == 8
    assert set(seen) == {(0, 0), (0, 1), (0, 2)}


def test_no_padding_when_the_batch_divides_the_dataset():
    sampler = ShardedSampler(SHARDS, batch_size=5, world_size=3, seed=3, drop_last=False)
    seen = collect(sampler, 0)
    assert sorted(seen) == sorted(every_sample(SHARDS))


# -- the ranks partition the epoch ----------------------------------------


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5])
def test_no_two_ranks_receive_the_same_position(world_size):
    sampler = ShardedSampler(SHARDS, batch_size=2, world_size=world_size, seed=4, drop_last=True)
    seen_by_rank = [
        [sample for batch in sampler.batches(0, rank) for sample in batch]
        for rank in range(world_size)
    ]
    flat = [sample for samples in seen_by_rank for sample in samples]
    assert len(flat) == len(set(flat))


def test_the_ranks_together_cover_the_kept_portion():
    sampler = ShardedSampler(SHARDS, batch_size=3, world_size=2, seed=4, drop_last=True)
    order = sampler.epoch_order(0)
    kept = sampler.num_batches() * sampler.batch_size * sampler.world_size
    assert sorted(collect(sampler, 0)) == sorted(order[:kept])


def test_a_rank_takes_every_world_size_th_position():
    # Stride, not a contiguous slab: rank r starts at index r and steps by the
    # world size. Contiguous chunks would also be disjoint and equal in size,
    # so nothing else in this file separates the two.
    sampler = ShardedSampler(SHARDS, batch_size=1, world_size=3, seed=5, drop_last=True)
    order = sampler.epoch_order(0)
    for rank in range(3):
        flat = [sample for batch in sampler.batches(0, rank) for sample in batch]
        assert flat == order[rank :: 3][: len(flat)]


def test_batches_are_contiguous_within_a_rank():
    sampler = ShardedSampler(SHARDS, batch_size=3, world_size=2, seed=5, drop_last=True)
    order = sampler.epoch_order(0)
    mine = order[0::2]
    assert sampler.batches(0, 0)[0] == mine[:3]
    assert sampler.batches(0, 0)[1] == mine[3:6]


def test_one_rank_sees_the_order_in_order():
    sampler = ShardedSampler(SHARDS, batch_size=5, world_size=1, seed=5, drop_last=True)
    flat = [sample for batch in sampler.batches(0, 0) for sample in batch]
    assert flat == sampler.epoch_order(0)[: len(flat)]


def test_resharding_changes_who_sees_what_but_not_what_is_seen():
    two = ShardedSampler(SHARDS, batch_size=3, world_size=2, seed=6, drop_last=True)
    three = ShardedSampler(SHARDS, batch_size=2, world_size=3, seed=6, drop_last=True)
    assert two.epoch_order(0) == three.epoch_order(0)
    assert sorted(collect(two, 0)) == sorted(collect(three, 0))


# -- resuming --------------------------------------------------------------


@pytest.mark.parametrize("start_batch", [0, 1, 2, 3])
def test_resuming_returns_the_remaining_batches(start_batch):
    sampler = ShardedSampler(SHARDS, batch_size=2, world_size=2, seed=8)
    for rank in range(2):
        full = sampler.batches(0, rank)
        assert sampler.batches(0, rank, start_batch=start_batch) == full[start_batch:]


def test_resuming_at_the_end_yields_nothing():
    sampler = ShardedSampler(SHARDS, batch_size=2, world_size=2, seed=8)
    assert sampler.batches(0, 0, start_batch=sampler.num_batches()) == []


def test_resuming_does_not_restart_the_epoch():
    sampler = ShardedSampler(SHARDS, batch_size=2, world_size=2, seed=8)
    full = sampler.batches(0, 0)
    resumed = sampler.batches(0, 0, start_batch=1)
    assert resumed[0] != full[0]
    assert resumed[0] == full[1]


def test_resuming_survives_a_new_sampler_object():
    # The state that matters is the seed, the epoch and the batch index —
    # nothing lives inside the instance.
    first = ShardedSampler(SHARDS, batch_size=2, world_size=2, seed=8)
    restarted = ShardedSampler(SHARDS, batch_size=2, world_size=2, seed=8)
    assert restarted.batches(4, 1, start_batch=2) == first.batches(4, 1)[2:]


# -- awkward shapes --------------------------------------------------------


def test_an_empty_shard_contributes_nothing_and_breaks_nothing():
    with_empty = ShardedSampler([4, 0, 4], batch_size=2, world_size=2, seed=9)
    assert sorted(with_empty.epoch_order(0)) == sorted(every_sample([4, 0, 4]))
    assert with_empty.total_samples == 8


def test_a_single_sample_dataset():
    sampler = ShardedSampler([1], batch_size=1, world_size=1, seed=9, drop_last=False)
    assert sampler.num_batches() == 1
    assert sampler.batches(0, 0) == [[(0, 0)]]


def test_a_single_shard_behaves_like_any_other():
    sampler = ShardedSampler([12], batch_size=3, world_size=2, seed=9, drop_last=True)
    assert sampler.num_batches() == 2
    assert sorted(collect(sampler, 0)) == [(0, offset) for offset in range(12)]


def test_more_ranks_than_samples_still_gives_every_rank_a_batch():
    sampler = ShardedSampler([2], batch_size=1, world_size=4, seed=9, drop_last=False)
    assert sampler.num_batches() == 1
    for rank in range(4):
        assert len(sampler.batches(0, rank)) == 1
        assert len(sampler.batches(0, rank)[0]) == 1
