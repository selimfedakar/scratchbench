"""Hidden tests — kv_cache_equivalence.

Every assertion is licensed by a sentence in prompt.md.

Tolerances: 1e-11 wherever a cached path is compared against a full pass. The
two paths perform the same float64 arithmetic in a different order — one long
matrix multiply against a concatenation of several short ones — so they agree
to a handful of ULPs on values of order one, and a disagreement worth catching
is many orders of magnitude larger than that.

`slow_forward` recomputes the layer with explicit loops and explicit
trigonometry. It exists so that a solution which skips the rotation entirely,
and would therefore be self-consistent between its own cached and uncached
paths, still fails.
"""

import math

import numpy as np
import pytest

from kv_cache import CachedAttention


D_MODEL = 8
N_HEADS = 2
BASE = 10_000.0


def build(n_heads: int = N_HEADS, d_model: int = D_MODEL, seed: int = 0):
    rng = np.random.default_rng(seed)
    weights = [rng.normal(size=(d_model, d_model)) / math.sqrt(d_model) for _ in range(4)]
    layer = CachedAttention(*weights, n_heads=n_heads, rope_base=BASE)
    return layer, weights


def sequence(batch: int, seq_len: int, d_model: int = D_MODEL, seed: int = 99) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(batch, seq_len, d_model))


def slow_forward(weights, n_heads: int, x: np.ndarray) -> np.ndarray:
    """The same layer, written position by position with no cache at all."""
    wq, wk, wv, wo = weights
    batch, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    half = head_dim // 2

    def rotate(vectors: np.ndarray) -> np.ndarray:
        rotated = np.empty_like(vectors)
        for position in range(vectors.shape[0]):
            for i in range(half):
                theta = position * BASE ** (-i / half)
                first, second = vectors[position, i], vectors[position, i + half]
                rotated[position, i] = first * math.cos(theta) - second * math.sin(theta)
                rotated[position, i + half] = second * math.cos(theta) + first * math.sin(theta)
        return rotated

    merged = np.zeros((batch, seq_len, d_model))
    for b in range(batch):
        q, k, v = x[b] @ wq, x[b] @ wk, x[b] @ wv
        for h in range(n_heads):
            columns = slice(h * head_dim, (h + 1) * head_dim)
            q_rot, k_rot, v_head = rotate(q[:, columns]), rotate(k[:, columns]), v[:, columns]
            for position in range(seq_len):
                scores = np.array(
                    [q_rot[position] @ k_rot[j] / math.sqrt(head_dim) for j in range(position + 1)]
                )
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                merged[b, position, columns] = probabilities @ v_head[: position + 1]

    return merged @ wo


# -- shapes ----------------------------------------------------------------


def test_output_and_cache_shapes():
    layer, _ = build()
    out, (keys, values) = layer.forward(sequence(2, 5))
    assert out.shape == (2, 5, D_MODEL)
    assert keys.shape == (2, N_HEADS, 5, D_MODEL // N_HEADS)
    assert values.shape == keys.shape


def test_the_cache_grows_by_the_length_of_the_chunk():
    layer, _ = build()
    x = sequence(1, 6)
    _, cache = layer.forward(x[:, :4])
    assert cache[0].shape[2] == 4
    _, cache = layer.forward(x[:, 4:], cache)
    assert cache[0].shape[2] == 6
    assert cache[1].shape[2] == 6


# -- the layer itself ------------------------------------------------------


def test_a_full_pass_matches_an_explicit_position_by_position_computation():
    layer, weights = build()
    x = sequence(2, 6)
    out, _ = layer.forward(x)
    assert np.allclose(out, slow_forward(weights, N_HEADS, x), atol=1e-11)


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_the_head_split_is_contiguous(n_heads):
    layer, weights = build(n_heads=n_heads)
    x = sequence(1, 5)
    out, _ = layer.forward(x)
    assert np.allclose(out, slow_forward(weights, n_heads, x), atol=1e-11)


def test_a_lone_token_is_just_the_value_path():
    # One token has exactly one key to attend to, so its weight is one and the
    # output is the value projection followed by the output projection.
    layer, (_, _, wv, wo) = build()
    x = sequence(1, 1)
    out, _ = layer.forward(x)
    assert np.allclose(out, (x @ wv) @ wo, atol=1e-12)


def test_uniform_attention_averages_the_visible_values():
    # Zero query weights make every score zero, so each position averages the
    # values it is allowed to see, and causality decides which those are.
    rng = np.random.default_rng(3)
    wq = np.zeros((D_MODEL, D_MODEL))
    wk, wv, wo = (rng.normal(size=(D_MODEL, D_MODEL)) for _ in range(3))
    layer = CachedAttention(wq, wk, wv, wo, n_heads=N_HEADS, rope_base=BASE)

    x = sequence(1, 5)
    out, _ = layer.forward(x)

    values = x @ wv
    running = np.cumsum(values, axis=1) / np.arange(1, 6)[None, :, None]
    assert np.allclose(out, running @ wo, atol=1e-12)


def test_changing_a_later_token_leaves_earlier_outputs_untouched():
    layer, _ = build()
    x = sequence(1, 6)
    before, _ = layer.forward(x)

    changed = x.copy()
    changed[:, 4:, :] = np.random.default_rng(77).normal(size=(1, 2, D_MODEL))
    after, _ = layer.forward(changed)

    assert np.allclose(before[:, :4, :], after[:, :4, :], atol=1e-12)
    assert not np.allclose(before[:, 4:, :], after[:, 4:, :], atol=1e-6)


# -- the equivalence this task is named after -----------------------------


def test_decoding_one_token_at_a_time_matches_a_full_pass():
    layer, _ = build()
    x = sequence(1, 7)

    full, _ = layer.forward(x)

    cache = None
    steps = []
    for position in range(x.shape[1]):
        step, cache = layer.forward(x[:, position : position + 1], cache)
        steps.append(step)
    incremental = np.concatenate(steps, axis=1)

    assert incremental.shape == full.shape
    assert np.allclose(incremental, full, atol=1e-11)


def test_a_prefill_followed_by_single_tokens_matches_a_full_pass():
    layer, _ = build()
    x = sequence(1, 8)
    full, _ = layer.forward(x)

    prefill, cache = layer.forward(x[:, :5])
    steps = [prefill]
    for position in range(5, 8):
        step, cache = layer.forward(x[:, position : position + 1], cache)
        steps.append(step)

    assert np.allclose(np.concatenate(steps, axis=1), full, atol=1e-11)


def test_uneven_chunks_match_a_full_pass():
    layer, _ = build()
    x = sequence(1, 6)
    full, _ = layer.forward(x)

    cache = None
    steps = []
    start = 0
    for length in (3, 2, 1):
        step, cache = layer.forward(x[:, start : start + length], cache)
        steps.append(step)
        start += length

    assert np.allclose(np.concatenate(steps, axis=1), full, atol=1e-11)


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_incremental_decoding_holds_for_every_head_count(n_heads):
    layer, _ = build(n_heads=n_heads)
    x = sequence(1, 6)
    full, _ = layer.forward(x)

    cache = None
    steps = []
    for position in range(6):
        step, cache = layer.forward(x[:, position : position + 1], cache)
        steps.append(step)

    assert np.allclose(np.concatenate(steps, axis=1), full, atol=1e-11)


def test_a_batch_decodes_incrementally_too():
    layer, _ = build()
    x = sequence(3, 5)
    full, _ = layer.forward(x)

    cache = None
    steps = []
    for position in range(5):
        step, cache = layer.forward(x[:, position : position + 1], cache)
        steps.append(step)

    assert np.allclose(np.concatenate(steps, axis=1), full, atol=1e-11)


def test_a_long_sequence_stays_aligned():
    layer, _ = build()
    x = sequence(1, 24)
    full, _ = layer.forward(x)

    cache = None
    steps = []
    for position in range(24):
        step, cache = layer.forward(x[:, position : position + 1], cache)
        steps.append(step)

    assert np.allclose(np.concatenate(steps, axis=1), full, atol=1e-11)


# -- the cache is data, not state -----------------------------------------


def test_the_cache_it_was_given_is_not_modified():
    layer, _ = build()
    x = sequence(1, 4)
    _, cache = layer.forward(x[:, :2])
    keys_before, values_before = cache[0].copy(), cache[1].copy()

    layer.forward(x[:, 2:], cache)

    assert np.array_equal(cache[0], keys_before)
    assert np.array_equal(cache[1], values_before)


def test_the_same_cache_can_be_continued_twice():
    # Branching from one prefix is what beam search and speculative decoding
    # do, and it only works if the cache is left intact.
    layer, _ = build()
    x = sequence(1, 4)
    _, cache = layer.forward(x[:, :2])

    first, _ = layer.forward(x[:, 2:3], cache)
    second, _ = layer.forward(x[:, 2:3], cache)
    assert np.array_equal(first, second)


def test_the_input_is_not_modified():
    layer, _ = build()
    x = sequence(1, 4)
    before = x.copy()
    layer.forward(x)
    assert np.array_equal(before, x)


def test_float32_input_and_weights_give_a_float64_output():
    rng = np.random.default_rng(5)
    weights = [rng.normal(size=(D_MODEL, D_MODEL)).astype(np.float32) for _ in range(4)]
    layer = CachedAttention(*weights, n_heads=N_HEADS, rope_base=BASE)

    out, (keys, _) = layer.forward(sequence(1, 3).astype(np.float32))
    assert out.dtype == np.float64
    assert keys.dtype == np.float64
    assert np.all(np.isfinite(out))
