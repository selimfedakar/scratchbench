"""Hidden tests — chunked_batchnorm_backward.

Every assertion is licensed by a sentence in prompt.md, and every promise in
prompt.md has a test here that enforces it.

The independent answer is `torch.autograd` over the whole batch at once: build
the statistics from the concatenated input, standardise, scale and shift, and
differentiate. That is a different computation from the chunked one under test,
and in particular the whole-batch coupling this task is about lives inside
PyTorch's own backward for the mean and the variance rather than in anything
written here, so a mistake shared between the reference solution and these tests
has nowhere to hide.

Two tests use no reference of any kind. If the upstream gradient is the same
number at every position of a channel, that channel's input gradient is exactly
zero, because the standardised input sums to zero over the batch and the mean of
a constant is itself. And a scale of zero leaves nothing for the input gradient
to be. Both hold on paper for any input, and neither can be satisfied by an
implementation that has the wrong denominator in front of a correction it does
compute.

Invariance under the split is a test here rather than a decoration, and that is
worth one sentence because the sibling task in this repository states the
opposite. `flash_attention_backward` cannot use block-size invariance to catch
the mistake it is about: an implementation that materialises the whole score
matrix is invariant too. Here the mistake is the other kind. Averaging over the
chunk in hand rather than over the batch is *exactly* a dependence on the split,
correct at one chunk and wrong at two, so the invariance test bites.

Tolerances. Everything is float64 and every gradient here is of order one, so
the only disagreement between two correct implementations is the order they sum
in: swept over thirty-nine seeds, seven splits, four mask densities and four
epsilons, the chunked answer and autograd differ by at most 7.1e-15. The
comparisons use `atol=1e-10, rtol=0`, which is four orders of headroom over that
and ten below the size of the mistake this task exists to catch — averaging the
corrections over the chunk in hand moves the answer by 3.9 on the first case
here. The two exact tests compare against a true answer of zero at `atol=1e-12`,
where the reference lands at 5.6e-16 and at exactly zero.
"""

import pytest
import torch

from chunked_batchnorm import (
    chunk_input_gradients,
    chunk_parameter_gradients,
    chunked_batchnorm_backward,
)


EPS = 1e-5


# -- inputs and the independent answers ------------------------------------


def random_case(chunk_sizes=(2, 3), channels=3, length=4, seed=0, keep=0.75):
    """`(x_chunks, dy_chunks, mask_chunks, gamma)`, all float64, no grad."""
    generator = torch.Generator().manual_seed(seed)

    def sample(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=generator)

    x_chunks = [sample(rows, channels, length) for rows in chunk_sizes]
    dy_chunks = [sample(rows, channels, length) for rows in chunk_sizes]
    mask_chunks = [
        torch.rand((rows, length), generator=generator, dtype=torch.float64) < keep
        for rows in chunk_sizes
    ]
    # The statistics are taken over the kept positions, so at least one of them
    # has to exist for the case to mean anything.
    mask_chunks[0][0, 0] = True
    return x_chunks, dy_chunks, mask_chunks, sample(channels)


def joined(chunks):
    return torch.cat(list(chunks), dim=0)


def split_like(tensor, chunks):
    """`tensor` cut back into the rows each chunk owns."""
    pieces = []
    start = 0
    for chunk in chunks:
        pieces.append(tensor[start : start + chunk.shape[0]])
        start += chunk.shape[0]
    return pieces


def statistics(x, mask):
    """`(mean, var)` over the kept positions, the pair the forward pass saved."""
    keep = mask[:, None, :].to(x.dtype)
    count = mask.sum().to(x.dtype)
    mean = (x * keep).sum(dim=(0, 2)) / count
    var = (((x - mean[None, :, None]) * keep) ** 2).sum(dim=(0, 2)) / count
    return mean, var


def autograd_gradients(x, dy, mask, gamma, eps=EPS):
    """`(dx, dgamma, dbeta)` from PyTorch, over the whole batch at once."""
    x_leaf = x.detach().clone().requires_grad_(True)
    gamma_leaf = gamma.detach().clone().requires_grad_(True)
    beta_leaf = torch.zeros_like(gamma).requires_grad_(True)

    keep = mask[:, None, :].to(x.dtype)
    count = mask.sum().to(x.dtype)
    mean = (x_leaf * keep).sum(dim=(0, 2)) / count
    var = (((x_leaf - mean[None, :, None]) * keep) ** 2).sum(dim=(0, 2)) / count
    x_hat = (x_leaf - mean[None, :, None]) / torch.sqrt(var + eps)[None, :, None]
    y = (gamma_leaf[None, :, None] * x_hat + beta_leaf[None, :, None]) * keep

    (y * dy).sum().backward()
    return x_leaf.grad, gamma_leaf.grad, beta_leaf.grad


def backward_of(case, eps=EPS):
    """Run the whole pipeline on a case, with the statistics it implies."""
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mean, var = statistics(joined(x_chunks), joined(mask_chunks))
    return chunked_batchnorm_backward(
        x_chunks, dy_chunks, mask_chunks, mean, var, gamma, eps
    )


SPLITS = [(6,), (3, 3), (1, 5), (4, 2), (2, 1, 3), (1, 1, 1, 1, 1, 1)]


# -- agreement with autograd -----------------------------------------------


@pytest.mark.parametrize("chunk_sizes", SPLITS)
def test_matches_autograd(chunk_sizes):
    case = random_case(chunk_sizes=chunk_sizes, seed=1)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    for produced, expected in zip(dx_chunks, split_like(expected_dx, x_chunks)):
        assert torch.allclose(produced, expected, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("eps", [1e-8, 1e-5, 1e-2, 0.5])
def test_the_epsilon_sits_under_the_square_root_with_the_variance(eps):
    case = random_case(chunk_sizes=(3, 2), seed=2)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case, eps=eps)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma, eps=eps
    )

    for produced, expected in zip(dx_chunks, split_like(expected_dx, x_chunks)):
        assert torch.allclose(produced, expected, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("keep", [0.2, 0.5, 1.0])
def test_the_statistics_are_over_the_kept_positions_and_nothing_else(keep):
    case = random_case(chunk_sizes=(4, 3), channels=2, length=5, seed=3, keep=keep)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    for produced, expected in zip(dx_chunks, split_like(expected_dx, x_chunks)):
        assert torch.allclose(produced, expected, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


# -- the split is a memory knob and nothing else ---------------------------


@pytest.mark.parametrize("chunk_sizes", SPLITS)
def test_the_answer_does_not_depend_on_how_the_batch_is_split(chunk_sizes):
    # One batch of six rows, described six different ways. An implementation
    # that averages over the chunk it is holding agrees with the baseline at
    # one chunk and disagrees at every other split.
    whole = random_case(chunk_sizes=(6,), seed=4)
    x, dy, mask, gamma = joined(whole[0]), joined(whole[1]), joined(whole[2]), whole[3]

    baseline_dx, baseline_dgamma, baseline_dbeta = backward_of(whole)

    pieces = []
    start = 0
    for rows in chunk_sizes:
        stop = start + rows
        pieces.append((x[start:stop], dy[start:stop], mask[start:stop]))
        start = stop
    split_case = (
        [piece[0] for piece in pieces],
        [piece[1] for piece in pieces],
        [piece[2] for piece in pieces],
        gamma,
    )

    dx_chunks, dgamma, dbeta = backward_of(split_case)

    assert torch.allclose(joined(dx_chunks), baseline_dx[0], atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, baseline_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, baseline_dbeta, atol=1e-10, rtol=0.0)


def test_one_chunk_of_one_row_at_a_time():
    case = random_case(chunk_sizes=(1,) * 7, channels=2, length=3, seed=5)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


# -- what happens without a reference implementation -----------------------


@pytest.mark.parametrize("chunk_sizes", [(6,), (3, 3), (2, 1, 3)])
def test_an_upstream_gradient_that_is_constant_leaves_the_input_alone(chunk_sizes):
    # The standardised input sums to zero over the batch, so a constant
    # upstream gradient correlates with it not at all and matches its own
    # average exactly: dx is zero on paper, for any input and any scale.
    case = random_case(chunk_sizes=chunk_sizes, channels=3, length=4, seed=6)
    x_chunks, _, mask_chunks, gamma = case
    generator = torch.Generator().manual_seed(7)
    per_channel = torch.randn(3, dtype=torch.float64, generator=generator)
    dy_chunks = [
        per_channel[None, :, None].expand_as(x_chunk).contiguous()
        for x_chunk in x_chunks
    ]

    dx_chunks, dgamma, _ = backward_of((x_chunks, dy_chunks, mask_chunks, gamma))

    for dx_chunk in dx_chunks:
        assert torch.allclose(dx_chunk, torch.zeros_like(dx_chunk), atol=1e-12, rtol=0.0)
    assert torch.allclose(dgamma, torch.zeros_like(dgamma), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("chunk_sizes", [(6,), (3, 3)])
def test_a_scale_of_zero_leaves_the_input_alone(chunk_sizes):
    x_chunks, dy_chunks, mask_chunks, gamma = random_case(
        chunk_sizes=chunk_sizes, seed=8
    )
    zero_gamma = torch.zeros_like(gamma)

    dx_chunks, dgamma, dbeta = backward_of(
        (x_chunks, dy_chunks, mask_chunks, zero_gamma)
    )

    for dx_chunk in dx_chunks:
        assert torch.allclose(dx_chunk, torch.zeros_like(dx_chunk), atol=1e-12, rtol=0.0)
    # The parameter gradients do not depend on the scale at all.
    _, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), zero_gamma
    )
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


# -- one chunk on its own --------------------------------------------------


def test_the_chunk_parameter_gradients_add_up_to_the_whole_batch():
    case = random_case(chunk_sizes=(2, 1, 3), seed=9)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mean, var = statistics(joined(x_chunks), joined(mask_chunks))

    total_dgamma = torch.zeros_like(gamma)
    total_dbeta = torch.zeros_like(gamma)
    for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks):
        chunk_dgamma, chunk_dbeta = chunk_parameter_gradients(
            x_chunk, dy_chunk, mask_chunk, mean, var, EPS
        )
        assert chunk_dgamma.shape == gamma.shape
        assert chunk_dbeta.shape == gamma.shape
        total_dgamma = total_dgamma + chunk_dgamma
        total_dbeta = total_dbeta + chunk_dbeta

    _, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )
    assert torch.allclose(total_dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(total_dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_one_chunk_carries_exactly_its_share_of_the_input_gradients(index):
    # The whole batch's parameter gradients and count come from autograd here,
    # so this grades the chunk function against an answer it had no part in.
    case = random_case(chunk_sizes=(2, 1, 3), seed=10)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mean, var = statistics(joined(x_chunks), joined(mask_chunks))

    expected_dx, dgamma, dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )
    total_count = int(joined(mask_chunks).sum())

    produced = chunk_input_gradients(
        x_chunks[index],
        dy_chunks[index],
        mask_chunks[index],
        mean,
        var,
        gamma,
        EPS,
        dgamma,
        dbeta,
        total_count,
    )
    assert torch.allclose(
        produced, split_like(expected_dx, x_chunks)[index], atol=1e-10, rtol=0.0
    )


def test_the_count_is_the_number_of_kept_positions_in_the_whole_batch():
    # Two chunks of very different sizes and very different mask densities, so
    # a count taken from the chunk, from the rows, or from the number of
    # positions rather than from the mask all land somewhere else.
    x_chunks, dy_chunks, mask_chunks, gamma = random_case(
        chunk_sizes=(1, 7), channels=2, length=6, seed=11, keep=0.4
    )
    mask_chunks[0][:] = True
    mean, var = statistics(joined(x_chunks), joined(mask_chunks))

    expected_dx, dgamma, dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )
    produced = chunk_input_gradients(
        x_chunks[0],
        dy_chunks[0],
        mask_chunks[0],
        mean,
        var,
        gamma,
        EPS,
        dgamma,
        dbeta,
        int(joined(mask_chunks).sum()),
    )
    assert torch.allclose(
        produced, split_like(expected_dx, x_chunks)[0], atol=1e-10, rtol=0.0
    )


# -- padding ---------------------------------------------------------------


def test_a_padded_position_takes_no_gradient():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=5, seed=12, keep=0.5)
    x_chunks, _, mask_chunks, _ = case
    dx_chunks, _, _ = backward_of(case)

    for dx_chunk, mask_chunk in zip(dx_chunks, mask_chunks):
        padded = ~mask_chunk[:, None, :].expand_as(dx_chunk)
        assert torch.equal(dx_chunk[padded], torch.zeros_like(dx_chunk[padded]))


def test_what_sits_at_a_padded_position_changes_nothing():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=5, seed=13, keep=0.6)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    before = backward_of(case)

    generator = torch.Generator().manual_seed(14)
    noisy = []
    for x_chunk, mask_chunk in zip(x_chunks, mask_chunks):
        noise = torch.randn(
            x_chunk.shape, generator=generator, dtype=torch.float64
        ) * 100.0
        padded = ~mask_chunk[:, None, :].expand_as(x_chunk)
        noisy.append(torch.where(padded, noise, x_chunk))

    after = backward_of((noisy, dy_chunks, mask_chunks, gamma))

    for one, other in zip(after[0], before[0]):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)
    assert torch.allclose(after[1], before[1], atol=1e-10, rtol=0.0)
    assert torch.allclose(after[2], before[2], atol=1e-10, rtol=0.0)


def test_the_upstream_gradient_at_a_padded_position_changes_nothing():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=5, seed=15, keep=0.6)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    before = backward_of(case)

    generator = torch.Generator().manual_seed(16)
    noisy = []
    for dy_chunk, mask_chunk in zip(dy_chunks, mask_chunks):
        noise = torch.randn(
            dy_chunk.shape, generator=generator, dtype=torch.float64
        ) * 100.0
        padded = ~mask_chunk[:, None, :].expand_as(dy_chunk)
        noisy.append(torch.where(padded, noise, dy_chunk))

    after = backward_of((x_chunks, noisy, mask_chunks, gamma))

    for one, other in zip(after[0], before[0]):
        assert torch.allclose(one, other, atol=1e-10, rtol=0.0)
    assert torch.allclose(after[1], before[1], atol=1e-10, rtol=0.0)
    assert torch.allclose(after[2], before[2], atol=1e-10, rtol=0.0)


def test_a_row_that_is_entirely_padding():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=4, seed=17)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mask_chunks[0][1, :] = False
    mask_chunks[1][0, :] = False

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


def test_a_chunk_that_is_entirely_padding():
    case = random_case(chunk_sizes=(2, 2, 2), channels=2, length=4, seed=18)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mask_chunks[1][:] = False

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.isfinite(dx_chunks[1]).all()
    assert torch.equal(dx_chunks[1], torch.zeros_like(dx_chunks[1]))
    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


def test_one_kept_position_in_the_whole_batch():
    # The variance is exactly zero, so the epsilon is the only thing under the
    # square root and the standardised input is zero.
    case = random_case(chunk_sizes=(2, 2), channels=2, length=3, seed=19)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    for mask_chunk in mask_chunks:
        mask_chunk[:] = False
    mask_chunks[0][0, 0] = True

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    for dx_chunk in dx_chunks:
        assert torch.isfinite(dx_chunk).all()
    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


# -- geometry --------------------------------------------------------------


def test_a_single_channel():
    case = random_case(chunk_sizes=(2, 3), channels=1, length=4, seed=20)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


def test_a_single_position_per_row():
    case = random_case(chunk_sizes=(3, 2), channels=3, length=1, seed=21)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


def test_one_chunk_holding_one_row_of_one_position():
    case = random_case(chunk_sizes=(1,), channels=2, length=1, seed=22)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


def test_a_long_batch_of_many_small_chunks():
    case = random_case(chunk_sizes=(3, 1, 4, 2, 5), channels=4, length=7, seed=23)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=1e-10, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=1e-10, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=1e-10, rtol=0.0)


def test_channels_do_not_mix():
    case = random_case(chunk_sizes=(2, 3), channels=4, length=5, seed=24)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    together_dx, together_dgamma, together_dbeta = backward_of(case)

    for channel in range(4):
        cut = slice(channel, channel + 1)
        alone = (
            [x_chunk[:, cut] for x_chunk in x_chunks],
            [dy_chunk[:, cut] for dy_chunk in dy_chunks],
            mask_chunks,
            gamma[cut],
        )
        alone_dx, alone_dgamma, alone_dbeta = backward_of(alone)

        for one, other in zip(alone_dx, together_dx):
            assert torch.allclose(one, other[:, cut], atol=1e-10, rtol=0.0)
        assert torch.allclose(alone_dgamma, together_dgamma[cut], atol=1e-10, rtol=0.0)
        assert torch.allclose(alone_dbeta, together_dbeta[cut], atol=1e-10, rtol=0.0)


# -- interface -------------------------------------------------------------


def test_the_gradients_have_the_shapes_and_dtype_of_what_they_differentiate():
    case = random_case(chunk_sizes=(2, 1, 4), channels=5, length=3, seed=25)
    x_chunks, _, _, gamma = case
    dx_chunks, dgamma, dbeta = backward_of(case)

    assert len(dx_chunks) == len(x_chunks)
    for dx_chunk, x_chunk in zip(dx_chunks, x_chunks):
        assert dx_chunk.shape == x_chunk.shape
        assert dx_chunk.dtype == torch.float64
    assert dgamma.shape == gamma.shape and dgamma.dtype == torch.float64
    assert dbeta.shape == gamma.shape and dbeta.dtype == torch.float64


def test_the_gradients_carry_no_autograd_history():
    case = random_case(chunk_sizes=(2, 3), seed=26)
    dx_chunks, dgamma, dbeta = backward_of(case)

    for tensor in list(dx_chunks) + [dgamma, dbeta]:
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_the_inputs_are_left_alone():
    case = random_case(chunk_sizes=(2, 3), seed=27)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mean, var = statistics(joined(x_chunks), joined(mask_chunks))

    watched = list(x_chunks) + list(dy_chunks) + list(mask_chunks) + [mean, var, gamma]
    originals = [tensor.clone() for tensor in watched]

    chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, mean, var, gamma, EPS)

    for original, current in zip(originals, watched):
        assert torch.equal(original, current)
