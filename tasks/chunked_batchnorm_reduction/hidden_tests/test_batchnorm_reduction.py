"""Hidden tests — chunked_batchnorm_reduction.

Every assertion is licensed by a sentence in prompt.md, and every promise in
prompt.md has a test here that enforces it.

The independent answer is `torch.autograd` over the whole batch at once: build
the statistics from the concatenated input, standardise, scale and shift, and
differentiate. That is a different computation from the chunked one under test,
and the whole-batch coupling this task is about lives inside PyTorch's own
backward for the mean and the variance rather than in anything written here.

The buffer is the solution's own design, so these tests never look inside it.
What they do check is the contract the caller relies on: it has the same shape
for every chunk, and it is additive — the buffer of a batch held in one piece
equals the sum of the buffers of its pieces. Anything that is not a sum over the
chunk's kept positions fails that, and a solution whose second pass depends on
something the buffer did not carry fails the gradient comparisons instead.

Two tests use no reference of any kind. If the upstream gradient is the same
number at every position of a channel, that channel's input gradient is exactly
zero, because the standardised input sums to zero over the batch and the mean of
a constant is itself. And a scale of zero leaves nothing for the input gradient
to be.

Tolerances. Everything is float64. The comparisons use `atol=1e-9, rtol=0`; the
measured worst disagreement with autograd, swept over thirty-nine seeds, seven
splits, four mask densities and four epsilons, is 1.47e-13. It is two orders
larger than the sibling task's 7.1e-15 for a reason worth stating: a single
reduction cannot carry centred moments, so the variance comes out of
`sum(x*x)/M - mean*mean` and that subtraction cancels. The tolerance is nearly
four orders above the measured worst and nine below the size of the mistake this
task exists to catch, which moves the first case's answer by 3.18. The two exact
tests compare against a true answer of zero at `atol=1e-10`, where the reference
lands at 7.8e-16 and at exactly zero.
"""

import pytest
import torch

from batchnorm_reduction import (
    chunk_input_gradients,
    chunk_reduction,
    chunked_batchnorm_backward,
)


EPS = 1e-5
ATOL = 1e-9
ZERO_ATOL = 1e-10


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
    pieces = []
    start = 0
    for chunk in chunks:
        pieces.append(tensor[start : start + chunk.shape[0]])
        start += chunk.shape[0]
    return pieces


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
    x_chunks, dy_chunks, mask_chunks, gamma = case
    return chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, gamma, eps)


def total_of(case):
    """Walk the chunks once and add the buffers, the way the caller does."""
    x_chunks, dy_chunks, mask_chunks, _ = case
    total = None
    for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks):
        buffer = chunk_reduction(x_chunk, dy_chunk, mask_chunk)
        total = buffer if total is None else total + buffer
    return total


def protocol(case, eps=EPS):
    """The whole two-walk protocol, driven from outside the driver."""
    x_chunks, dy_chunks, mask_chunks, gamma = case
    total = total_of(case)
    return [
        chunk_input_gradients(x_chunk, dy_chunk, mask_chunk, gamma, eps, total)
        for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks)
    ]


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
        assert torch.allclose(produced, expected, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("chunk_sizes", SPLITS)
def test_the_two_walks_produce_what_the_driver_produces(chunk_sizes):
    # The caller reduces first and asks for gradients second; driving that from
    # outside must give the same answer the driver gives.
    case = random_case(chunk_sizes=chunk_sizes, seed=2)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    from_protocol = protocol(case)
    expected_dx, _, _ = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(from_protocol), expected_dx, atol=ATOL, rtol=0.0)
    for produced, expected in zip(from_protocol, backward_of(case)[0]):
        assert torch.allclose(produced, expected, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("eps", [1e-8, 1e-5, 1e-2, 0.5])
def test_the_epsilon_sits_under_the_square_root_with_the_variance(eps):
    case = random_case(chunk_sizes=(3, 2), seed=3)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case, eps=eps)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma, eps=eps
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("keep", [0.2, 0.5, 1.0])
def test_the_statistics_are_over_the_kept_positions_and_nothing_else(keep):
    case = random_case(chunk_sizes=(4, 3), channels=2, length=5, seed=4, keep=keep)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


# -- the buffer's contract, without looking inside it -----------------------


@pytest.mark.parametrize("chunk_sizes", SPLITS)
def test_the_buffer_is_additive_across_the_chunks(chunk_sizes):
    # One batch of six rows, described six ways. Whatever the buffer holds, the
    # caller adds it up, so the whole batch's buffer is the sum of its pieces'.
    whole = random_case(chunk_sizes=(6,), seed=5)
    x, dy, mask, gamma = joined(whole[0]), joined(whole[1]), joined(whole[2]), whole[3]

    pieces, start = [], 0
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

    assert torch.allclose(total_of(split_case), total_of(whole), atol=ATOL, rtol=0.0)


def test_the_buffer_has_one_shape_and_it_does_not_move_with_the_chunk():
    case = random_case(chunk_sizes=(1, 7, 3), channels=4, length=5, seed=6)
    x_chunks, dy_chunks, mask_chunks, _ = case

    buffers = [
        chunk_reduction(x_chunk, dy_chunk, mask_chunk)
        for x_chunk, dy_chunk, mask_chunk in zip(x_chunks, dy_chunks, mask_chunks)
    ]

    for buffer in buffers:
        assert buffer.dim() == 2
        assert buffer.shape[1] == 4
        assert buffer.shape == buffers[0].shape
        assert buffer.dtype == torch.float64
        assert buffer.requires_grad is False and buffer.grad_fn is None


@pytest.mark.parametrize("chunk_sizes", SPLITS)
def test_the_answer_does_not_depend_on_how_the_batch_is_split(chunk_sizes):
    whole = random_case(chunk_sizes=(6,), seed=7)
    x, dy, mask, gamma = joined(whole[0]), joined(whole[1]), joined(whole[2]), whole[3]
    baseline_dx, baseline_dgamma, baseline_dbeta = backward_of(whole)

    pieces, start = [], 0
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
    assert torch.allclose(joined(dx_chunks), baseline_dx[0], atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, baseline_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, baseline_dbeta, atol=ATOL, rtol=0.0)


def test_one_chunk_of_one_row_at_a_time():
    case = random_case(chunk_sizes=(1,) * 7, channels=2, length=3, seed=8)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


# -- what happens without a reference implementation -----------------------


@pytest.mark.parametrize("chunk_sizes", [(6,), (3, 3), (2, 1, 3)])
def test_an_upstream_gradient_that_is_constant_leaves_the_input_alone(chunk_sizes):
    case = random_case(chunk_sizes=chunk_sizes, channels=3, length=4, seed=9)
    x_chunks, _, mask_chunks, gamma = case
    generator = torch.Generator().manual_seed(10)
    per_channel = torch.randn(3, dtype=torch.float64, generator=generator)
    dy_chunks = [
        per_channel[None, :, None].expand_as(x_chunk).contiguous()
        for x_chunk in x_chunks
    ]

    dx_chunks, dgamma, _ = backward_of((x_chunks, dy_chunks, mask_chunks, gamma))

    for dx_chunk in dx_chunks:
        assert torch.allclose(
            dx_chunk, torch.zeros_like(dx_chunk), atol=ZERO_ATOL, rtol=0.0
        )
    assert torch.allclose(dgamma, torch.zeros_like(dgamma), atol=ZERO_ATOL, rtol=0.0)


@pytest.mark.parametrize("chunk_sizes", [(6,), (3, 3)])
def test_a_scale_of_zero_leaves_the_input_alone(chunk_sizes):
    x_chunks, dy_chunks, mask_chunks, gamma = random_case(
        chunk_sizes=chunk_sizes, seed=11
    )
    zero_gamma = torch.zeros_like(gamma)

    dx_chunks, dgamma, dbeta = backward_of(
        (x_chunks, dy_chunks, mask_chunks, zero_gamma)
    )

    for dx_chunk in dx_chunks:
        assert torch.allclose(
            dx_chunk, torch.zeros_like(dx_chunk), atol=ZERO_ATOL, rtol=0.0
        )
    _, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), zero_gamma
    )
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


# -- one chunk on its own --------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_one_chunk_carries_exactly_its_share_of_the_input_gradients(index):
    case = random_case(chunk_sizes=(2, 1, 3), seed=12)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    total = total_of(case)

    expected_dx, _, _ = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )
    produced = chunk_input_gradients(
        x_chunks[index], dy_chunks[index], mask_chunks[index], gamma, EPS, total
    )
    assert torch.allclose(
        produced, split_like(expected_dx, x_chunks)[index], atol=ATOL, rtol=0.0
    )


def test_a_lopsided_split_does_not_move_the_denominators():
    # One row against seven, and the small chunk fully kept while the large one
    # is mostly padding, so a count taken from the chunk or from the number of
    # positions lands somewhere else entirely.
    case = random_case(chunk_sizes=(1, 7), channels=2, length=6, seed=13, keep=0.4)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mask_chunks[0][:] = True

    expected_dx, _, _ = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )
    produced = chunk_input_gradients(
        x_chunks[0], dy_chunks[0], mask_chunks[0], gamma, EPS, total_of(case)
    )
    assert torch.allclose(
        produced, split_like(expected_dx, x_chunks)[0], atol=ATOL, rtol=0.0
    )


# -- padding ---------------------------------------------------------------


def test_a_padded_position_takes_no_gradient():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=5, seed=14, keep=0.5)
    x_chunks, _, mask_chunks, _ = case
    dx_chunks, _, _ = backward_of(case)

    for dx_chunk, mask_chunk in zip(dx_chunks, mask_chunks):
        padded = ~mask_chunk[:, None, :].expand_as(dx_chunk)
        assert torch.equal(dx_chunk[padded], torch.zeros_like(dx_chunk[padded]))


def test_what_sits_at_a_padded_position_changes_nothing():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=5, seed=15, keep=0.6)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    before = backward_of(case)

    generator = torch.Generator().manual_seed(16)
    noisy = []
    for x_chunk, mask_chunk in zip(x_chunks, mask_chunks):
        noise = torch.randn(x_chunk.shape, generator=generator, dtype=torch.float64) * 100.0
        padded = ~mask_chunk[:, None, :].expand_as(x_chunk)
        noisy.append(torch.where(padded, noise, x_chunk))

    after = backward_of((noisy, dy_chunks, mask_chunks, gamma))

    assert torch.allclose(joined(after[0]), joined(before[0]), atol=ATOL, rtol=0.0)
    assert torch.allclose(after[1], before[1], atol=ATOL, rtol=0.0)
    assert torch.allclose(after[2], before[2], atol=ATOL, rtol=0.0)


def test_the_upstream_gradient_at_a_padded_position_changes_nothing():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=5, seed=17, keep=0.6)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    before = backward_of(case)

    generator = torch.Generator().manual_seed(18)
    noisy = []
    for dy_chunk, mask_chunk in zip(dy_chunks, mask_chunks):
        noise = torch.randn(dy_chunk.shape, generator=generator, dtype=torch.float64) * 100.0
        padded = ~mask_chunk[:, None, :].expand_as(dy_chunk)
        noisy.append(torch.where(padded, noise, dy_chunk))

    after = backward_of((x_chunks, noisy, mask_chunks, gamma))

    assert torch.allclose(joined(after[0]), joined(before[0]), atol=ATOL, rtol=0.0)
    assert torch.allclose(after[1], before[1], atol=ATOL, rtol=0.0)
    assert torch.allclose(after[2], before[2], atol=ATOL, rtol=0.0)


def test_a_row_that_is_entirely_padding():
    case = random_case(chunk_sizes=(3, 2), channels=2, length=4, seed=19)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mask_chunks[0][1, :] = False
    mask_chunks[1][0, :] = False

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


def test_a_chunk_that_is_entirely_padding():
    case = random_case(chunk_sizes=(2, 2, 2), channels=2, length=4, seed=20)
    x_chunks, dy_chunks, mask_chunks, gamma = case
    mask_chunks[1][:] = False

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.isfinite(dx_chunks[1]).all()
    assert torch.equal(dx_chunks[1], torch.zeros_like(dx_chunks[1]))
    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


def test_one_kept_position_in_the_whole_batch():
    # The variance is exactly zero, so the epsilon is the only thing under the
    # square root and the standardised input is zero.
    case = random_case(chunk_sizes=(2, 2), channels=2, length=3, seed=21)
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
    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


# -- geometry --------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 4), (3, 1), (2, 2)], ids=["one_channel", "one_position", "square"]
)
def test_small_geometries(shape):
    channels, length = shape
    case = random_case(chunk_sizes=(2, 3), channels=channels, length=length, seed=22)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


def test_one_chunk_holding_one_row_of_one_position():
    case = random_case(chunk_sizes=(1,), channels=2, length=1, seed=23)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


def test_a_long_batch_of_many_small_chunks():
    case = random_case(chunk_sizes=(3, 1, 4, 2, 5), channels=4, length=7, seed=24)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    dx_chunks, dgamma, dbeta = backward_of(case)
    expected_dx, expected_dgamma, expected_dbeta = autograd_gradients(
        joined(x_chunks), joined(dy_chunks), joined(mask_chunks), gamma
    )

    assert torch.allclose(joined(dx_chunks), expected_dx, atol=ATOL, rtol=0.0)
    assert torch.allclose(dgamma, expected_dgamma, atol=ATOL, rtol=0.0)
    assert torch.allclose(dbeta, expected_dbeta, atol=ATOL, rtol=0.0)


def test_channels_do_not_mix():
    case = random_case(chunk_sizes=(2, 3), channels=4, length=5, seed=25)
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
            assert torch.allclose(one, other[:, cut], atol=ATOL, rtol=0.0)
        assert torch.allclose(alone_dgamma, together_dgamma[cut], atol=ATOL, rtol=0.0)
        assert torch.allclose(alone_dbeta, together_dbeta[cut], atol=ATOL, rtol=0.0)


# -- interface -------------------------------------------------------------


def test_the_gradients_have_the_shapes_and_dtype_of_what_they_differentiate():
    case = random_case(chunk_sizes=(2, 1, 4), channels=5, length=3, seed=26)
    x_chunks, _, _, gamma = case
    dx_chunks, dgamma, dbeta = backward_of(case)

    assert len(dx_chunks) == len(x_chunks)
    for dx_chunk, x_chunk in zip(dx_chunks, x_chunks):
        assert dx_chunk.shape == x_chunk.shape
        assert dx_chunk.dtype == torch.float64
    assert dgamma.shape == gamma.shape and dgamma.dtype == torch.float64
    assert dbeta.shape == gamma.shape and dbeta.dtype == torch.float64


def test_the_gradients_carry_no_autograd_history():
    case = random_case(chunk_sizes=(2, 3), seed=27)
    dx_chunks, dgamma, dbeta = backward_of(case)

    for tensor in list(dx_chunks) + [dgamma, dbeta]:
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_the_inputs_are_left_alone():
    case = random_case(chunk_sizes=(2, 3), seed=28)
    x_chunks, dy_chunks, mask_chunks, gamma = case

    watched = list(x_chunks) + list(dy_chunks) + list(mask_chunks) + [gamma]
    originals = [tensor.clone() for tensor in watched]

    chunked_batchnorm_backward(x_chunks, dy_chunks, mask_chunks, gamma, EPS)

    for original, current in zip(originals, watched):
        assert torch.equal(original, current)
