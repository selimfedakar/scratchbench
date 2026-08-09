"""Hidden tests — flash_attention_backward.

Every assertion is licensed by a sentence in prompt.md, and every promise in
prompt.md has a test here that enforces it.

The independent answer is `torch.autograd` over a plain materialised attention:
build the whole score matrix, softmax it, multiply by the values, and
differentiate. That is a completely different computation from the blocked one
under test, and in particular the row-coupling term this task is about lives
inside PyTorch's own softmax backward rather than in anything written here — so
a mistake shared between the reference solution and these tests has nowhere to
hide.

One test does not use autograd at all. If every row of `v` is the same vector,
the output is that vector for every query, and the correction term cancels the
whole of `dP` exactly: `dq` and `dk` are then zero on paper, for any `q`, `k` and
`do`. An implementation that drops the correction, or computes it over one block
instead of the whole row, fails that with no reference of any kind involved.

Tolerances. Everything is float64 and every gradient here is of order one, so
the only disagreement between two correct implementations is the order they sum
in: measured across every block size below, the blocked answer and autograd
differ by at most 2.3e-15. The comparisons use `atol=1e-10, rtol=0`, which is
four and a half orders of headroom over that and at least eight below the size
of any mistake this task exists to catch. The exact cancellation test uses
`atol=1e-12` against a true answer of zero, where the reference lands at
3.5e-16.
"""

import math

import pytest
import torch

from flash_backward import flash_attention_backward, key_block_gradients


# -- inputs and the independent answers ------------------------------------


def random_case(
    batch=2, heads=2, n_queries=6, n_keys=6, head_dim=4, head_dim_v=3, seed=0
):
    """`(q, k, v, do)`, all float64, none of them requiring grad."""
    generator = torch.Generator().manual_seed(seed)

    def sample(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=generator)

    return (
        sample(batch, heads, n_queries, head_dim),
        sample(batch, heads, n_keys, head_dim),
        sample(batch, heads, n_keys, head_dim_v),
        sample(batch, heads, n_queries, head_dim_v),
    )


def scores_of(q, k, causal):
    """The scaled, masked score matrix. Materialised on purpose: this is the
    thing the function under test is not allowed to build."""
    n_queries, n_keys = q.shape[-2], k.shape[-2]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        query_positions = torch.arange(n_keys - n_queries, n_keys)[:, None]
        key_positions = torch.arange(n_keys)[None, :]
        scores = scores.masked_fill(key_positions > query_positions, -math.inf)
    return scores


def forward(q, k, v, causal=False):
    """`(o, lse)`, the pair a blocked forward pass would have returned."""
    scores = scores_of(q, k, causal)
    lse = torch.logsumexp(scores, dim=-1)
    return torch.exp(scores - lse[..., None]) @ v, lse


def autograd_gradients(q, k, v, do, causal=False):
    """`(dq, dk, dv)` from PyTorch, over the whole matrix at once."""
    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = k.detach().clone().requires_grad_(True)
    v_leaf = v.detach().clone().requires_grad_(True)

    weights = torch.softmax(scores_of(q_leaf, k_leaf, causal), dim=-1)
    ((weights @ v_leaf) * do).sum().backward()
    return q_leaf.grad, k_leaf.grad, v_leaf.grad


def autograd_score_gradient(q, k, v, do, causal=False):
    """The loss gradient with respect to the scores, from PyTorch.

    Used only to slice a single key block's share of `dq` out of it, which the
    top-level gradients cannot be sliced into.
    """
    scores = scores_of(q, k, causal).detach().clone().requires_grad_(True)
    ((torch.softmax(scores, dim=-1) @ v) * do).sum().backward()
    return scores.grad


BLOCK_SIZES = [1, 2, 3, 4, 5, 7, 8, 16]


# -- agreement with autograd -----------------------------------------------


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("causal", [False, True])
def test_matches_autograd(block_size, causal):
    q, k, v, do = random_case(seed=1)
    o, lse = forward(q, k, v, causal)

    dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do, block_size, causal=causal)
    expected_dq, expected_dk, expected_dv = autograd_gradients(q, k, v, do, causal)

    assert torch.allclose(dq, expected_dq, atol=1e-10, rtol=0.0)
    assert torch.allclose(dk, expected_dk, atol=1e-10, rtol=0.0)
    assert torch.allclose(dv, expected_dv, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("causal", [False, True])
def test_the_answer_does_not_depend_on_the_block_size(causal):
    q, k, v, do = random_case(n_queries=9, n_keys=9, seed=2)
    o, lse = forward(q, k, v, causal)

    baseline = flash_attention_backward(q, k, v, o, lse, do, 9, causal=causal)
    for block_size in [1, 2, 4, 5, 8, 9, 32]:
        current = flash_attention_backward(q, k, v, o, lse, do, block_size, causal=causal)
        for one, other in zip(current, baseline):
            assert torch.allclose(one, other, atol=1e-10, rtol=0.0)


def test_a_block_size_past_the_end_is_a_single_block():
    q, k, v, do = random_case(seed=3)
    o, lse = forward(q, k, v)
    dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do, 1000)
    expected = autograd_gradients(q, k, v, do)
    for one, other in zip((dq, dk, dv), expected):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)


def test_causal_is_off_unless_it_is_asked_for():
    q, k, v, do = random_case(seed=4)
    o, lse = forward(q, k, v)

    default = flash_attention_backward(q, k, v, o, lse, do, 3)
    explicit = flash_attention_backward(q, k, v, o, lse, do, 3, causal=False)
    for one, other in zip(default, explicit):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)

    o_causal, lse_causal = forward(q, k, v, causal=True)
    masked = flash_attention_backward(q, k, v, o_causal, lse_causal, do, 3, causal=True)
    assert not torch.allclose(default[0], masked[0], atol=1e-6, rtol=0.0)


# -- the correction term, without a reference implementation ---------------


@pytest.mark.parametrize("block_size", [1, 2, 3, 6])
@pytest.mark.parametrize("causal", [False, True])
def test_identical_values_make_the_query_and_key_gradients_vanish(block_size, causal):
    # Every key carries the same value vector, so every query's output is that
    # vector and the row correction cancels dP exactly. dq and dk are zero on
    # paper, for any q, k and do; dv is not.
    q, k, _, do = random_case(seed=5)
    generator = torch.Generator().manual_seed(6)
    one_value = torch.randn(2, 2, 1, 3, dtype=torch.float64, generator=generator)
    v = one_value.expand(2, 2, k.shape[-2], 3).contiguous()

    o, lse = forward(q, k, v, causal)
    dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do, block_size, causal=causal)

    assert torch.allclose(dq, torch.zeros_like(dq), atol=1e-12, rtol=0.0)
    assert torch.allclose(dk, torch.zeros_like(dk), atol=1e-12, rtol=0.0)
    assert dv.abs().max() > 1e-6


# -- one block on its own --------------------------------------------------


@pytest.mark.parametrize("bounds", [(0, 2), (2, 5), (4, 8), (0, 8)])
@pytest.mark.parametrize("causal", [False, True])
def test_one_block_carries_exactly_its_share_of_the_gradients(bounds, causal):
    start, stop = bounds
    q, k, v, do = random_case(n_queries=8, n_keys=8, seed=7)
    o, lse = forward(q, k, v, causal)

    dq_part, dk_block, dv_block = key_block_gradients(
        q,
        k[..., start:stop, :],
        v[..., start:stop, :],
        o,
        lse,
        do,
        key_offset=start,
        query_offset=k.shape[-2] - q.shape[-2],
        causal=causal,
    )

    _, expected_dk, expected_dv = autograd_gradients(q, k, v, do, causal)
    d_scores = autograd_score_gradient(q, k, v, do, causal)
    expected_dq_part = (d_scores[..., start:stop] @ k[..., start:stop, :]) / math.sqrt(
        q.shape[-1]
    )

    assert torch.allclose(dq_part, expected_dq_part, atol=1e-10, rtol=0.0)
    assert torch.allclose(dk_block, expected_dk[..., start:stop, :], atol=1e-10, rtol=0.0)
    assert torch.allclose(dv_block, expected_dv[..., start:stop, :], atol=1e-10, rtol=0.0)


def test_the_block_contributions_add_up_to_the_whole_query_gradient():
    q, k, v, do = random_case(n_queries=6, n_keys=6, seed=8)
    o, lse = forward(q, k, v, causal=True)

    total = torch.zeros_like(q)
    for start in range(0, 6, 2):
        contribution, _, _ = key_block_gradients(
            q,
            k[..., start : start + 2, :],
            v[..., start : start + 2, :],
            o,
            lse,
            do,
            key_offset=start,
            query_offset=0,
            causal=True,
        )
        total = total + contribution

    dq, _, _ = flash_attention_backward(q, k, v, o, lse, do, 2, causal=True)
    assert torch.allclose(total, dq, atol=1e-10, rtol=0.0)


def test_a_key_block_a_query_may_not_read_contributes_nothing_to_it():
    # Eight queries against eight keys: the first query sits at position zero
    # and may read one key, so the block holding keys four to seven is entirely
    # out of its reach and must contribute exact zeros rather than a NaN.
    q, k, v, do = random_case(n_queries=8, n_keys=8, seed=9)
    o, lse = forward(q, k, v, causal=True)

    dq_part, dk_block, dv_block = key_block_gradients(
        q,
        k[..., 4:8, :],
        v[..., 4:8, :],
        o,
        lse,
        do,
        key_offset=4,
        query_offset=0,
        causal=True,
    )

    for tensor in (dq_part, dk_block, dv_block):
        assert torch.isfinite(tensor).all()
    assert torch.allclose(dq_part[..., 0, :], torch.zeros_like(dq_part[..., 0, :]), atol=1e-12)


def test_the_key_offset_is_what_places_the_block_in_the_sequence():
    # The same four keys, described as the first block and as the last block of
    # an eight-key range, are masked differently and must not agree.
    q, k, v, do = random_case(n_queries=8, n_keys=8, seed=10)
    o, lse = forward(q, k, v, causal=True)

    as_first, _, _ = key_block_gradients(
        q, k[..., :4, :], v[..., :4, :], o, lse, do, key_offset=0, query_offset=0, causal=True
    )
    as_last, _, _ = key_block_gradients(
        q, k[..., :4, :], v[..., :4, :], o, lse, do, key_offset=4, query_offset=0, causal=True
    )
    assert not torch.allclose(as_first, as_last, atol=1e-6, rtol=0.0)


# -- geometry --------------------------------------------------------------


def test_fewer_queries_than_keys_puts_the_queries_at_the_end():
    # The decode shape: one fresh query against a long history, which under the
    # trailing-query convention may read every key there is.
    q, k, v, do = random_case(n_queries=1, n_keys=16, seed=11)
    o, lse = forward(q, k, v, causal=True)

    dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do, 5, causal=True)
    expected = autograd_gradients(q, k, v, do, causal=True)
    for one, other in zip((dq, dk, dv), expected):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("causal", [False, True])
def test_a_short_query_window_over_a_long_history(causal):
    q, k, v, do = random_case(n_queries=3, n_keys=11, seed=12)
    o, lse = forward(q, k, v, causal)

    result = flash_attention_backward(q, k, v, o, lse, do, 4, causal=causal)
    expected = autograd_gradients(q, k, v, do, causal)
    for one, other in zip(result, expected):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)


def test_a_single_key_and_a_single_query():
    q, k, v, do = random_case(n_queries=1, n_keys=1, seed=13)
    o, lse = forward(q, k, v, causal=True)

    result = flash_attention_backward(q, k, v, o, lse, do, 1, causal=True)
    expected = autograd_gradients(q, k, v, do, causal=True)
    for one, other in zip(result, expected):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)


def test_the_values_may_be_wider_or_narrower_than_the_keys():
    for head_dim_v in (1, 7):
        q, k, v, do = random_case(head_dim=8, head_dim_v=head_dim_v, seed=14)
        o, lse = forward(q, k, v)
        result = flash_attention_backward(q, k, v, o, lse, do, 4)
        assert result[2].shape == v.shape
        expected = autograd_gradients(q, k, v, do)
        for one, other in zip(result, expected):
            assert torch.allclose(one, other, atol=1e-10, rtol=0.0)


def test_batch_entries_do_not_mix():
    q, k, v, do = random_case(batch=3, seed=15)
    o, lse = forward(q, k, v, causal=True)
    together = flash_attention_backward(q, k, v, o, lse, do, 4, causal=True)

    for index in range(3):
        cut = slice(index, index + 1)
        o_alone, lse_alone = forward(q[cut], k[cut], v[cut], causal=True)
        alone = flash_attention_backward(
            q[cut], k[cut], v[cut], o_alone, lse_alone, do[cut], 4, causal=True
        )
        for one, other in zip(alone, together):
            assert torch.allclose(one, other[cut], atol=1e-10, rtol=0.0)


def test_heads_do_not_mix():
    q, k, v, do = random_case(batch=1, heads=4, seed=16)
    o, lse = forward(q, k, v)
    together = flash_attention_backward(q, k, v, o, lse, do, 4)

    for head in range(4):
        cut = slice(head, head + 1)
        o_alone, lse_alone = forward(q[:, cut], k[:, cut], v[:, cut])
        alone = flash_attention_backward(
            q[:, cut], k[:, cut], v[:, cut], o_alone, lse_alone, do[:, cut], 4
        )
        for one, other in zip(alone, together):
            assert torch.allclose(one, other[:, cut], atol=1e-10, rtol=0.0)


# -- interface -------------------------------------------------------------


def test_the_gradients_have_the_shapes_and_dtype_of_what_they_differentiate():
    q, k, v, do = random_case(batch=2, heads=3, n_queries=4, n_keys=9, head_dim=8, head_dim_v=5, seed=17)
    o, lse = forward(q, k, v)
    dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do, 4)

    assert dq.shape == q.shape and dq.dtype == torch.float64
    assert dk.shape == k.shape and dk.dtype == torch.float64
    assert dv.shape == v.shape and dv.dtype == torch.float64


def test_the_gradients_carry_no_autograd_history():
    q, k, v, do = random_case(seed=18)
    o, lse = forward(q, k, v)
    for tensor in flash_attention_backward(q, k, v, o, lse, do, 3):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_the_inputs_are_left_alone():
    q, k, v, do = random_case(seed=19)
    o, lse = forward(q, k, v, causal=True)
    originals = [tensor.clone() for tensor in (q, k, v, o, lse, do)]

    flash_attention_backward(q, k, v, o, lse, do, 4, causal=True)

    for original, current in zip(originals, (q, k, v, o, lse, do)):
        assert torch.equal(original, current)
