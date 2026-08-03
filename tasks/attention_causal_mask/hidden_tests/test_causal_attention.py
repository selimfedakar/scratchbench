"""Hidden tests — attention_causal_mask.

Every assertion is licensed by a sentence in prompt.md. Tolerances are 1e-12:
these are float64 dot products over head dimensions of at most eight, so the
accumulated error is a few ULPs and anything larger is a real disagreement.

Several tests recover the attention weights instead of guessing at them, by
passing an identity matrix as the values. Row i of the output is then exactly
the distribution query i placed over the keys, which lets a test inspect the
weights without depending on anything private to the implementation.
"""

import math

import numpy as np

from causal_attention import attention, causal_mask


def one_hot_values(batch: int, heads: int, n_keys: int) -> np.ndarray:
    """Values that make the output equal to the attention weights."""
    return np.tile(np.eye(n_keys), (batch, heads, 1, 1))


# -- the mask --------------------------------------------------------------


def test_mask_is_lower_triangular_when_the_lengths_match():
    mask = causal_mask(3, 3)
    assert mask.shape == (3, 3)
    assert mask[0, 0] == 0.0
    assert mask[0, 1] == -np.inf
    assert mask[0, 2] == -np.inf
    assert mask[1, 0] == 0.0
    assert mask[1, 1] == 0.0
    assert mask[1, 2] == -np.inf
    assert np.all(mask[2] == 0.0)


def test_mask_blocks_with_negative_infinity_and_not_a_large_number():
    mask = causal_mask(4, 4)
    blocked = mask[np.triu_indices(4, k=1)]
    assert np.all(np.isneginf(blocked))
    assert np.all(mask[np.tril_indices(4)] == 0.0)


def test_a_single_cached_query_may_read_the_whole_history():
    mask = causal_mask(1, 6)
    assert mask.shape == (1, 6)
    assert np.all(mask == 0.0)


def test_queries_sit_at_the_end_of_the_key_range():
    # Two queries against five keys are positions 3 and 4.
    mask = causal_mask(2, 5)
    assert mask.shape == (2, 5)
    assert np.all(mask[0, :4] == 0.0)
    assert np.isneginf(mask[0, 4])
    assert np.all(mask[1] == 0.0)


def test_mask_of_one_by_one():
    assert causal_mask(1, 1).shape == (1, 1)
    assert causal_mask(1, 1)[0, 0] == 0.0


# -- the scaled dot product ------------------------------------------------


def test_output_keeps_the_query_shape():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 3, 5, 8))
    k = rng.normal(size=(2, 3, 5, 8))
    v = rng.normal(size=(2, 3, 5, 4))
    assert attention(q, k, v).shape == (2, 3, 5, 4)


def test_values_may_have_their_own_width():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(1, 1, 3, 8))
    k = rng.normal(size=(1, 1, 3, 8))
    v = rng.normal(size=(1, 1, 3, 2))
    assert attention(q, k, v).shape == (1, 1, 3, 2)


def test_the_scale_is_one_over_root_head_dim():
    # One query, two keys, dot products of 2 and 0 over a head dimension of 4,
    # so the scaled scores are 1 and 0.
    q = np.array([[[[1.0, 0.0, 0.0, 0.0]]]])
    k = np.array([[[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]]])
    v = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])

    out = attention(q, k, v)
    total = math.exp(1.0) + math.exp(0.0)
    expected = [math.exp(1.0) / total, math.exp(0.0) / total]
    assert np.allclose(out[0, 0, 0], expected, atol=1e-12)


def test_weights_are_a_distribution_over_the_visible_keys():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(2, 2, 4, 8))
    k = rng.normal(size=(2, 2, 4, 8))
    weights = attention(q, k, one_hot_values(2, 2, 4))
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-12)


def test_uniform_scores_give_a_uniform_distribution():
    # Zero queries make every score zero, so each query splits its attention
    # evenly over everything it is allowed to see.
    q = np.zeros((1, 1, 4, 8))
    k = np.zeros((1, 1, 4, 8))
    weights = attention(q, k, one_hot_values(1, 1, 4))
    assert np.allclose(weights[0, 0, 0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(weights[0, 0, 1], [0.5, 0.5, 0.0, 0.0], atol=1e-12)
    assert np.allclose(weights[0, 0, 3], [0.25] * 4, atol=1e-12)


# -- causality -------------------------------------------------------------


def test_no_weight_lands_on_a_future_key():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(1, 1, 5, 8))
    k = rng.normal(size=(1, 1, 5, 8))
    weights = attention(q, k, one_hot_values(1, 1, 5))[0, 0]
    assert np.all(weights[np.triu_indices(5, k=1)] == 0.0)


def test_changing_a_later_token_leaves_earlier_outputs_untouched():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(1, 2, 6, 8))
    k = rng.normal(size=(1, 2, 6, 8))
    v = rng.normal(size=(1, 2, 6, 8))

    before = attention(q, k, v)
    k[:, :, 4:, :] = rng.normal(size=(1, 2, 2, 8))
    v[:, :, 4:, :] = rng.normal(size=(1, 2, 2, 8))
    after = attention(q, k, v)

    assert np.allclose(before[:, :, :4, :], after[:, :, :4, :], atol=1e-12)
    assert not np.allclose(before[:, :, 4:, :], after[:, :, 4:, :], atol=1e-6)


def test_the_first_query_reads_only_the_first_value():
    rng = np.random.default_rng(5)
    q = rng.normal(size=(1, 1, 3, 8))
    k = rng.normal(size=(1, 1, 3, 8))
    v = rng.normal(size=(1, 1, 3, 8))
    out = attention(q, k, v)
    assert np.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-12)


# -- the cached-decode alignment ------------------------------------------

def test_a_trailing_chunk_matches_the_same_rows_of_a_full_pass():
    # Two queries against five keys must reproduce rows 3 and 4 of the full
    # five-query attention. This only holds if the queries are anchored at the
    # end of the key range.
    rng = np.random.default_rng(6)
    q = rng.normal(size=(1, 2, 5, 8))
    k = rng.normal(size=(1, 2, 5, 8))
    v = rng.normal(size=(1, 2, 5, 8))

    full = attention(q, k, v)
    chunk = attention(q[:, :, 3:, :], k, v)
    assert np.allclose(chunk, full[:, :, 3:, :], atol=1e-12)


def test_one_query_against_a_history_matches_the_last_row():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 1, 7, 8))
    k = rng.normal(size=(1, 1, 7, 8))
    v = rng.normal(size=(1, 1, 7, 8))

    full = attention(q, k, v)
    last = attention(q[:, :, -1:, :], k, v)
    assert np.allclose(last[0, 0, 0], full[0, 0, -1], atol=1e-12)


# -- padding ---------------------------------------------------------------


def test_a_padded_key_receives_no_attention():
    rng = np.random.default_rng(8)
    q = rng.normal(size=(1, 1, 4, 8))
    k = rng.normal(size=(1, 1, 4, 8))
    keep = np.array([[True, False, True, True]])

    weights = attention(q, k, one_hot_values(1, 1, 4), key_padding_mask=keep)[0, 0]
    assert np.all(weights[:, 1] == 0.0)
    # The rows that can still see something remain distributions.
    assert np.allclose(weights[[0, 2, 3]].sum(axis=-1), 1.0, atol=1e-12)


def test_padding_at_the_end_matches_a_shorter_unpadded_sequence():
    rng = np.random.default_rng(9)
    q = rng.normal(size=(1, 2, 5, 8))
    k = rng.normal(size=(1, 2, 5, 8))
    v = rng.normal(size=(1, 2, 5, 8))
    keep = np.array([[True, True, True, False, False]])

    padded = attention(q, k, v, key_padding_mask=keep)
    short = attention(q[:, :, :3, :], k[:, :, :3, :], v[:, :, :3, :])
    assert np.allclose(padded[:, :, :3, :], short, atol=1e-12)


def test_padding_is_per_sequence_not_per_batch():
    rng = np.random.default_rng(10)
    q = rng.normal(size=(2, 1, 4, 8))
    k = rng.normal(size=(2, 1, 4, 8))
    v = rng.normal(size=(2, 1, 4, 8))
    keep = np.array([[True, True, True, True], [True, True, False, False]])

    both = attention(q, k, v, key_padding_mask=keep)
    first = attention(q[:1], k[:1], v[:1], key_padding_mask=keep[:1])
    second = attention(q[1:], k[1:], v[1:], key_padding_mask=keep[1:])

    assert np.allclose(both[:1], first, atol=1e-12)
    assert np.allclose(both[1:], second, atol=1e-12)


def test_every_head_sees_the_same_padding():
    rng = np.random.default_rng(11)
    q = rng.normal(size=(1, 3, 4, 8))
    k = rng.normal(size=(1, 3, 4, 8))
    keep = np.array([[True, False, True, True]])
    weights = attention(q, k, one_hot_values(1, 3, 4), key_padding_mask=keep)
    assert np.all(weights[:, :, :, 1] == 0.0)


def test_a_fully_masked_query_gives_zeros_and_never_nan():
    # Left padding: the first key is filler, and causality means the first
    # query has nothing else to look at.
    rng = np.random.default_rng(12)
    q = rng.normal(size=(1, 2, 3, 8))
    k = rng.normal(size=(1, 2, 3, 8))
    v = rng.normal(size=(1, 2, 3, 8))
    keep = np.array([[False, True, True]])

    out = attention(q, k, v, key_padding_mask=keep)
    assert not np.any(np.isnan(out))
    assert np.all(out[:, :, 0, :] == 0.0)
    assert not np.allclose(out[:, :, 1, :], 0.0)


def test_everything_masked_is_all_zeros():
    rng = np.random.default_rng(13)
    q = rng.normal(size=(1, 1, 3, 8))
    k = rng.normal(size=(1, 1, 3, 8))
    v = rng.normal(size=(1, 1, 3, 8))
    keep = np.array([[False, False, False]])

    out = attention(q, k, v, key_padding_mask=keep)
    assert not np.any(np.isnan(out))
    assert np.all(out == 0.0)


def test_a_mask_that_keeps_everything_changes_nothing():
    rng = np.random.default_rng(14)
    q = rng.normal(size=(2, 2, 4, 8))
    k = rng.normal(size=(2, 2, 4, 8))
    v = rng.normal(size=(2, 2, 4, 8))
    keep = np.ones((2, 4), dtype=bool)
    assert np.allclose(attention(q, k, v, keep), attention(q, k, v), atol=1e-12)


# -- independence and hygiene ---------------------------------------------


def test_batch_entries_do_not_mix():
    rng = np.random.default_rng(15)
    q = rng.normal(size=(3, 2, 4, 8))
    k = rng.normal(size=(3, 2, 4, 8))
    v = rng.normal(size=(3, 2, 4, 8))
    both = attention(q, k, v)
    for i in range(3):
        assert np.allclose(both[i : i + 1], attention(q[i : i + 1], k[i : i + 1], v[i : i + 1]), atol=1e-12)


def test_heads_do_not_mix():
    rng = np.random.default_rng(16)
    q = rng.normal(size=(1, 4, 5, 8))
    k = rng.normal(size=(1, 4, 5, 8))
    v = rng.normal(size=(1, 4, 5, 8))
    both = attention(q, k, v)
    for h in range(4):
        one = attention(q[:, h : h + 1], k[:, h : h + 1], v[:, h : h + 1])
        assert np.allclose(both[:, h : h + 1], one, atol=1e-12)


def test_float32_input_is_accepted_and_the_output_is_float64():
    rng = np.random.default_rng(17)
    q = rng.normal(size=(1, 1, 3, 8)).astype(np.float32)
    k = rng.normal(size=(1, 1, 3, 8)).astype(np.float32)
    v = rng.normal(size=(1, 1, 3, 8)).astype(np.float32)
    out = attention(q, k, v)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


def test_the_inputs_are_left_alone():
    rng = np.random.default_rng(18)
    q = rng.normal(size=(1, 1, 4, 8))
    k = rng.normal(size=(1, 1, 4, 8))
    v = rng.normal(size=(1, 1, 4, 8))
    keep = np.array([[True, True, False, True]])
    originals = [q.copy(), k.copy(), v.copy(), keep.copy()]

    attention(q, k, v, key_padding_mask=keep)

    for original, current in zip(originals, [q, k, v, keep]):
        assert np.array_equal(original, current)
