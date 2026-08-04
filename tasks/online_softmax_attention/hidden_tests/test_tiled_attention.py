"""Hidden tests — online_softmax_attention.

Every assertion is licensed by a sentence in prompt.md, and every promise in
prompt.md has a test here that enforces it.

Tolerances. Quantities of order one are compared at `atol=1e-12, rtol=0`.
Streaming the keys in blocks legitimately differs from a single-pass softmax:
each merge adds a multiplication and an addition, so a dozen blocks drift by a
few ULPs of one, around 1e-15. A margin of three orders of magnitude leaves that
alone while still catching any algebraic mistake, which is wrong by percent
rather than by ULPs. Where a logsumexp of order 1e4 is involved the comparison
is relative instead, because float64 preserves relative precision and 1e4 has
already spent four of its fifteen digits.

`plain_attention` is the independent answer: the full score matrix, one softmax,
no blocks. Several tests also pin the output to values computed by hand with
`math.exp`, so that a mistake made identically in both the tiled and the naive
path still fails.
"""

import math

import numpy as np
import pytest

from tiled_attention import attention_tiled, combine_states


# -- independent references, not tests ------------------------------------


def plain_attention(q, k, v, causal=False):
    """Attention the ordinary way: whole matrix, one softmax."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    n_queries, head_dim = q.shape[-2], q.shape[-1]
    n_keys = k.shape[-2]

    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
    if causal:
        query_pos = np.arange(n_keys - n_queries, n_keys)[:, None]
        key_pos = np.arange(n_keys)[None, :]
        scores = np.where(key_pos <= query_pos, scores, -np.inf)

    peak = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - peak)
    total = np.sum(weights, axis=-1, keepdims=True)
    return (weights / total) @ v, peak[..., 0] + np.log(total[..., 0])


def empty_state(batch=1, heads=1, n_queries=1, head_dim_v=2):
    return (
        np.full((batch, heads, n_queries), -np.inf),
        np.zeros((batch, heads, n_queries)),
        np.zeros((batch, heads, n_queries, head_dim_v)),
    )


def one_key_state(score, value):
    """The state of a group holding exactly one key."""
    value = np.asarray(value, dtype=np.float64)
    return (
        np.array([[[score]]], dtype=np.float64),
        np.array([[[1.0]]]),
        value.reshape(1, 1, 1, -1).copy(),
    )


def one_hot_values(batch, heads, n_keys):
    """Values that make the output equal to the attention weights."""
    return np.tile(np.eye(n_keys), (batch, heads, 1, 1))


def random_inputs(batch=2, heads=2, n_queries=6, n_keys=12, head_dim=4, head_dim_v=3, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(batch, heads, n_queries, head_dim)) * spread,
        rng.normal(size=(batch, heads, n_keys, head_dim)) * spread,
        rng.normal(size=(batch, heads, n_keys, head_dim_v)),
    )


# -- the merge -------------------------------------------------------------


def test_merging_two_single_key_groups_matches_a_hand_computed_state():
    left = one_key_state(0.5, [1.0, 0.0])
    right = one_key_state(2.0, [0.0, 1.0])

    merged_max, denominator, numerator = combine_states(left, right)

    rescale = math.exp(0.5 - 2.0)
    assert np.allclose(merged_max, [[[2.0]]], atol=1e-12, rtol=0.0)
    assert np.allclose(denominator, [[[rescale + 1.0]]], atol=1e-12, rtol=0.0)
    assert np.allclose(
        numerator, [[[[rescale, 1.0]]]], atol=1e-12, rtol=0.0
    )


def test_the_merged_maximum_is_the_larger_of_the_two():
    left = (np.array([[[1.0, 9.0]]]), np.array([[[1.0, 1.0]]]), np.zeros((1, 1, 2, 2)))
    right = (np.array([[[4.0, 3.0]]]), np.array([[[1.0, 1.0]]]), np.zeros((1, 1, 2, 2)))
    merged_max, _, _ = combine_states(left, right)
    assert np.allclose(merged_max, [[[4.0, 9.0]]], atol=1e-12, rtol=0.0)


def test_the_denominator_is_rescaled_onto_the_new_maximum():
    # Left holds three keys measured against a maximum of 0, right holds two
    # measured against 5. After the merge everything is relative to 5.
    left = (np.array([[[0.0]]]), np.array([[[3.0]]]), np.zeros((1, 1, 1, 2)))
    right = (np.array([[[5.0]]]), np.array([[[2.0]]]), np.zeros((1, 1, 1, 2)))
    _, denominator, _ = combine_states(left, right)
    assert np.allclose(
        denominator, [[[3.0 * math.exp(-5.0) + 2.0]]], atol=1e-12, rtol=0.0
    )


def test_the_numerator_is_rescaled_onto_the_new_maximum():
    left = (
        np.array([[[0.0]]]),
        np.array([[[1.0]]]),
        np.array([[[[4.0, 0.0]]]]),
    )
    right = (
        np.array([[[5.0]]]),
        np.array([[[1.0]]]),
        np.array([[[[0.0, 7.0]]]]),
    )
    _, _, numerator = combine_states(left, right)
    assert np.allclose(
        numerator, [[[[4.0 * math.exp(-5.0), 7.0]]]], atol=1e-12, rtol=0.0
    )


def test_the_side_holding_the_maximum_is_carried_through_unscaled():
    # Its scores are already measured against the merged maximum, so nothing
    # about it changes.
    left = (np.array([[[5.0]]]), np.array([[[2.0]]]), np.array([[[[6.0, 1.0]]]]))
    right = (np.array([[[1.0]]]), np.array([[[1.0]]]), np.array([[[[0.0, 0.0]]]]))
    _, denominator, numerator = combine_states(left, right)
    assert np.allclose(denominator, [[[2.0 + math.exp(-4.0)]]], atol=1e-12, rtol=0.0)
    assert np.allclose(numerator, [[[[6.0, 1.0]]]], atol=1e-12, rtol=0.0)


def test_an_empty_group_on_the_left_is_the_identity():
    state = one_key_state(3.0, [2.0, -1.0])
    merged = combine_states(empty_state(), state)
    for part, expected in zip(merged, state):
        assert np.allclose(part, expected, atol=1e-12, rtol=0.0)


def test_an_empty_group_on_the_right_is_the_identity():
    state = one_key_state(3.0, [2.0, -1.0])
    merged = combine_states(state, empty_state())
    for part, expected in zip(merged, state):
        assert np.allclose(part, expected, atol=1e-12, rtol=0.0)


def test_merging_two_empty_groups_stays_empty_and_never_nan():
    merged_max, denominator, numerator = combine_states(empty_state(), empty_state())
    assert not np.any(np.isnan(merged_max))
    assert not np.any(np.isnan(denominator))
    assert not np.any(np.isnan(numerator))
    assert np.all(np.isneginf(merged_max))
    assert np.all(denominator == 0.0)
    assert np.all(numerator == 0.0)


def test_the_merge_is_associative():
    rng = np.random.default_rng(1)

    def state():
        return (
            rng.normal(size=(2, 2, 3)) * 4.0,
            # At least one, because the key holding the group's maximum always
            # contributes exp(0) to its denominator.
            np.abs(rng.normal(size=(2, 2, 3))) + 1.0,
            rng.normal(size=(2, 2, 3, 2)),
        )

    a, b, c = state(), state(), state()
    left_first = combine_states(combine_states(a, b), c)
    right_first = combine_states(a, combine_states(b, c))
    for one, other in zip(left_first, right_first):
        assert np.allclose(one, other, atol=1e-12, rtol=0.0)


def test_the_merge_preserves_the_shapes():
    left = (np.zeros((2, 3, 4)), np.ones((2, 3, 4)), np.zeros((2, 3, 4, 5)))
    right = (np.ones((2, 3, 4)), np.ones((2, 3, 4)), np.ones((2, 3, 4, 5)))
    merged_max, denominator, numerator = combine_states(left, right)
    assert merged_max.shape == (2, 3, 4)
    assert denominator.shape == (2, 3, 4)
    assert numerator.shape == (2, 3, 4, 5)


def test_the_merge_returns_float64_even_from_float32():
    left = (
        np.zeros((1, 1, 1), dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.float32),
        np.zeros((1, 1, 1, 2), dtype=np.float32),
    )
    for part in combine_states(left, left):
        assert part.dtype == np.float64


def test_the_merge_does_not_modify_its_inputs():
    left = one_key_state(0.5, [1.0, 0.0])
    right = one_key_state(2.0, [0.0, 1.0])
    originals = [part.copy() for part in left + right]

    combine_states(left, right)

    for original, current in zip(originals, left + right):
        assert np.array_equal(original, current)


# -- the tiled attention, pinned to absolute answers ----------------------


def test_a_hand_computed_case():
    q = np.array([[[[1.0, 0.0, 0.0, 0.0]]]])
    k = np.array([[[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]])
    v = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])

    # Dot products 2, 0, 1 over a head dimension of 4, so the scaled scores are
    # 1, 0 and 0.5.
    weights = [math.exp(1.0), math.exp(0.0), math.exp(0.5)]
    total = sum(weights)
    expected = [
        (weights[0] * 1.0 + weights[2] * 1.0) / total,
        (weights[1] * 1.0 + weights[2] * 1.0) / total,
    ]

    output, logsumexp = attention_tiled(q, k, v, block_size=2)
    assert np.allclose(output[0, 0, 0], expected, atol=1e-12, rtol=0.0)
    assert np.allclose(logsumexp, [[[math.log(total)]]], atol=1e-12, rtol=0.0)


def test_the_scale_is_one_over_root_head_dim():
    # Dot products 2 and 0 over a head dimension of 4: scaled scores 1 and 0.
    q = np.array([[[[1.0, 0.0, 0.0, 0.0]]]])
    k = np.array([[[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]]])
    v = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])

    output, _ = attention_tiled(q, k, v, block_size=1)
    total = math.exp(1.0) + 1.0
    assert np.allclose(
        output[0, 0, 0], [math.exp(1.0) / total, 1.0 / total], atol=1e-12, rtol=0.0
    )


def test_uniform_scores_average_the_values():
    # Zero queries make every score zero, so every key gets the same weight.
    q = np.zeros((1, 2, 3, 4))
    k = np.zeros((1, 2, 5, 4))
    v = np.random.default_rng(2).normal(size=(1, 2, 5, 3))

    output, logsumexp = attention_tiled(q, k, v, block_size=2)
    assert np.allclose(output, np.mean(v, axis=-2, keepdims=True), atol=1e-12, rtol=0.0)
    assert np.allclose(logsumexp, math.log(5.0), atol=1e-12, rtol=0.0)


def test_the_first_causal_query_reads_only_the_first_value():
    q, k, v = random_inputs(batch=1, heads=1, n_queries=4, n_keys=4, seed=3)
    output, logsumexp = attention_tiled(q, k, v, block_size=2, causal=True)
    assert np.allclose(output[0, 0, 0], v[0, 0, 0], atol=1e-12, rtol=0.0)
    # One visible key means the denominator is exactly one, so the logsumexp is
    # that key's own score.
    own_score = float(q[0, 0, 0] @ k[0, 0, 0]) / math.sqrt(q.shape[-1])
    assert np.allclose(logsumexp[0, 0, 0], own_score, atol=1e-12, rtol=0.0)


# -- agreement with a single-pass softmax ---------------------------------

BLOCK_SIZES = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 64]


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_matches_a_single_pass_softmax(block_size):
    q, k, v = random_inputs(seed=4)
    expected_output, expected_logsumexp = plain_attention(q, k, v)
    output, logsumexp = attention_tiled(q, k, v, block_size)
    assert np.allclose(output, expected_output, atol=1e-12, rtol=0.0)
    assert np.allclose(logsumexp, expected_logsumexp, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_matches_a_single_pass_softmax_with_causal_masking(block_size):
    q, k, v = random_inputs(seed=5)
    expected_output, expected_logsumexp = plain_attention(q, k, v, causal=True)
    output, logsumexp = attention_tiled(q, k, v, block_size, causal=True)
    assert np.allclose(output, expected_output, atol=1e-12, rtol=0.0)
    assert np.allclose(logsumexp, expected_logsumexp, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("causal", [False, True])
def test_the_answer_does_not_depend_on_the_block_size(causal):
    q, k, v = random_inputs(seed=6)
    reference = attention_tiled(q, k, v, block_size=12, causal=causal)
    for block_size in BLOCK_SIZES:
        output, logsumexp = attention_tiled(q, k, v, block_size, causal=causal)
        assert np.allclose(output, reference[0], atol=1e-12, rtol=0.0)
        assert np.allclose(logsumexp, reference[1], atol=1e-12, rtol=0.0)


def test_a_block_size_past_the_end_is_a_single_block():
    q, k, v = random_inputs(seed=7)
    expected = plain_attention(q, k, v)
    output, logsumexp = attention_tiled(q, k, v, block_size=1000)
    assert np.allclose(output, expected[0], atol=1e-12, rtol=0.0)
    assert np.allclose(logsumexp, expected[1], atol=1e-12, rtol=0.0)


def test_causal_is_off_unless_it_is_asked_for():
    q, k, v = random_inputs(seed=8)
    default = attention_tiled(q, k, v, 5)
    explicit = attention_tiled(q, k, v, 5, causal=False)
    assert np.allclose(default[0], explicit[0], atol=1e-12, rtol=0.0)
    assert not np.allclose(
        default[0], attention_tiled(q, k, v, 5, causal=True)[0], atol=1e-6, rtol=0.0
    )


# -- shapes and the returned pair -----------------------------------------


def test_the_output_and_the_logsumexp_have_the_documented_shapes():
    q, k, v = random_inputs(batch=2, heads=3, n_queries=4, n_keys=9, head_dim=8, head_dim_v=5, seed=9)
    output, logsumexp = attention_tiled(q, k, v, block_size=4)
    assert output.shape == (2, 3, 4, 5)
    assert logsumexp.shape == (2, 3, 4)


def test_the_values_may_be_wider_or_narrower_than_the_keys():
    q, k, v = random_inputs(head_dim=8, head_dim_v=1, seed=10)
    output, _ = attention_tiled(q, k, v, block_size=5)
    assert output.shape == (2, 2, 6, 1)


def test_the_weights_are_a_distribution_over_the_keys():
    q, k, _ = random_inputs(batch=2, heads=2, n_queries=6, n_keys=12, seed=11)
    weights, _ = attention_tiled(q, k, one_hot_values(2, 2, 12), block_size=5)
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0)


def test_the_logsumexp_is_the_normaliser_the_output_was_divided_by():
    q, k, _ = random_inputs(batch=1, heads=1, n_queries=4, n_keys=9, seed=12)
    n_keys = k.shape[-2]
    weights, logsumexp = attention_tiled(q, k, one_hot_values(1, 1, n_keys), block_size=4)

    scores = (q @ np.swapaxes(k, -1, -2)) / math.sqrt(q.shape[-1])
    # Recovering the scores from the weights: weight = exp(score - logsumexp).
    assert np.allclose(weights, np.exp(scores - logsumexp[..., None]), atol=1e-12, rtol=0.0)


# -- causal geometry -------------------------------------------------------


def test_causal_queries_sit_at_the_end_of_the_key_range():
    # Two queries against five keys are positions 3 and 4, so the first cannot
    # see the last key and the second can see everything.
    q, k, _ = random_inputs(batch=1, heads=1, n_queries=2, n_keys=5, seed=13)
    weights, _ = attention_tiled(q, k, one_hot_values(1, 1, 5), block_size=2, causal=True)
    assert weights[0, 0, 0, 4] == 0.0
    assert np.allclose(weights[0, 0, 0, :4].sum(), 1.0, atol=1e-12, rtol=0.0)
    assert np.allclose(weights[0, 0, 1].sum(), 1.0, atol=1e-12, rtol=0.0)


def test_no_causal_weight_lands_on_a_future_key():
    q, k, _ = random_inputs(batch=1, heads=1, n_queries=6, n_keys=6, seed=14)
    weights, _ = attention_tiled(q, k, one_hot_values(1, 1, 6), block_size=2, causal=True)
    assert np.all(weights[0, 0][np.triu_indices(6, k=1)] == 0.0)


def test_a_block_wholly_in_the_future_does_not_poison_the_answer():
    # Eight queries, eight keys, blocks of two: the first query has three
    # entire blocks it may not look at.
    q, k, v = random_inputs(batch=1, heads=2, n_queries=8, n_keys=8, seed=15)
    output, logsumexp = attention_tiled(q, k, v, block_size=2, causal=True)
    assert not np.any(np.isnan(output))
    assert not np.any(np.isnan(logsumexp))
    expected_output, expected_logsumexp = plain_attention(q, k, v, causal=True)
    assert np.allclose(output, expected_output, atol=1e-12, rtol=0.0)
    assert np.allclose(logsumexp, expected_logsumexp, atol=1e-12, rtol=0.0)


def test_changing_a_later_key_leaves_earlier_causal_outputs_untouched():
    q, k, v = random_inputs(batch=1, heads=1, n_queries=6, n_keys=6, seed=16)
    before, _ = attention_tiled(q, k, v, block_size=4, causal=True)

    rng = np.random.default_rng(17)
    k[:, :, 4:, :] = rng.normal(size=(1, 1, 2, k.shape[-1]))
    v[:, :, 4:, :] = rng.normal(size=(1, 1, 2, v.shape[-1]))
    after, _ = attention_tiled(q, k, v, block_size=4, causal=True)

    assert np.allclose(before[:, :, :4, :], after[:, :, :4, :], atol=1e-12, rtol=0.0)
    assert not np.allclose(before[:, :, 4:, :], after[:, :, 4:, :], atol=1e-6, rtol=0.0)


# -- magnitudes ------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_large_scores_do_not_overflow(causal):
    # A spread of 100 over a head dimension of four puts the scaled scores in
    # the tens of thousands, where exp() is infinity in float64.
    q, k, v = random_inputs(batch=1, heads=2, n_queries=5, n_keys=11, seed=18, spread=100.0)
    output, logsumexp = attention_tiled(q, k, v, block_size=3, causal=causal)

    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(logsumexp))

    expected_output, expected_logsumexp = plain_attention(q, k, v, causal=causal)
    assert np.allclose(output, expected_output, atol=1e-9, rtol=0.0)
    # The logsumexp here is of order 1e4, so relative precision is the right
    # frame: float64 keeps about fifteen significant digits either way.
    assert np.allclose(logsumexp, expected_logsumexp, rtol=1e-12, atol=0.0)


def test_very_negative_scores_stay_finite():
    q, k, v = random_inputs(batch=1, heads=1, n_queries=4, n_keys=9, seed=19)
    output, logsumexp = attention_tiled(q - 300.0, k + 300.0, v, block_size=2)
    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(logsumexp))


# -- independence and hygiene ---------------------------------------------


def test_batch_entries_do_not_mix():
    q, k, v = random_inputs(batch=3, seed=20)
    together = attention_tiled(q, k, v, block_size=5)
    for i in range(3):
        alone = attention_tiled(q[i : i + 1], k[i : i + 1], v[i : i + 1], block_size=5)
        assert np.allclose(together[0][i : i + 1], alone[0], atol=1e-12, rtol=0.0)
        assert np.allclose(together[1][i : i + 1], alone[1], atol=1e-12, rtol=0.0)


def test_heads_do_not_mix():
    q, k, v = random_inputs(batch=1, heads=4, seed=21)
    together = attention_tiled(q, k, v, block_size=5)
    for head in range(4):
        alone = attention_tiled(
            q[:, head : head + 1], k[:, head : head + 1], v[:, head : head + 1], block_size=5
        )
        assert np.allclose(together[0][:, head : head + 1], alone[0], atol=1e-12, rtol=0.0)


def test_float32_input_is_accepted_and_the_output_is_float64():
    q, k, v = random_inputs(seed=22)
    output, logsumexp = attention_tiled(
        q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), block_size=5
    )
    assert output.dtype == np.float64
    assert logsumexp.dtype == np.float64
    assert np.all(np.isfinite(output))


def test_the_inputs_are_left_alone():
    q, k, v = random_inputs(seed=23)
    originals = [q.copy(), k.copy(), v.copy()]
    attention_tiled(q, k, v, block_size=5, causal=True)
    for original, current in zip(originals, [q, k, v]):
        assert np.array_equal(original, current)
