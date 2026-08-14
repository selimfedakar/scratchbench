"""Hidden tests for metal_cross_entropy_kernel.

Every expected value is computed on the CPU in float64 by PyTorch, so the kernel
is compared against an independent implementation rather than against itself.

Tolerance: the kernel accumulates in float32 and the reference in float64, so
the error grows with the number of terms in the sum and with the size of the
loss. Measured against the reference over every configuration in this file, the
worst absolute deviation is 2.1e-5, on the row whose logits sit around -800. The
pair below admits it — 2e-5 absolute plus 1e-5 of the value, which is 6e-5 there
— and is tight enough that every wrong implementation written against this task
in `tools/mutate_metal_task.py` fails rather than rounds into range.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from cross_entropy_kernel import SOURCE

ATOL = 2e-5
RTOL = 1e-5

#: The output buffer is filled with this before every launch, so a row nobody
#: wrote is a row that fails rather than a row that happens to hold a zero.
SENTINEL = float("nan")

KERNEL_NAME = "cross_entropy_rows"

SHAPES = [(1, 1), (4, 8), (5, 33), (7, 129), (2, 1000)]
GROUP_SIZES = [1, 7, 32, 33, 96, 1024]


@pytest.fixture(scope="module")
def kernel():
    """The compiled kernel, built once for the whole module."""
    library = torch.mps.compile_shader(SOURCE)
    return getattr(library, KERNEL_NAME)


def _inputs(rows, cols, seed=0, scale=1.0, shift=0.0):
    """Deterministic logits and targets on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, cols, generator=generator) * scale + shift
    targets = torch.randint(0, cols, (rows,), generator=generator)
    return logits, targets


def _expected(logits, targets):
    """The loss per row, in float64, on the CPU."""
    return F.cross_entropy(logits.double(), targets.long(), reduction="none")


def _launch(kernel, logits, targets, group_size, out_len=None):
    """Dispatch one threadgroup of `group_size` threads per row."""
    rows, cols = logits.shape
    out = torch.full((out_len or rows,), SENTINEL, device="mps", dtype=torch.float32)
    kernel(
        out,
        logits.contiguous().to("mps"),
        targets.to(torch.int32).to("mps"),
        cols,
        threads=rows * group_size,
        group_size=group_size,
    )
    torch.mps.synchronize()
    return out.cpu()


def _assert_close(got, expected):
    torch.testing.assert_close(got.double(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_matches_a_float64_cpu_loss(kernel, shape, group_size):
    rows, cols = shape
    logits, targets = _inputs(rows, cols, seed=rows * 100 + cols)
    got = _launch(kernel, logits, targets, group_size)
    _assert_close(got, _expected(logits, targets))


def test_a_row_wider_than_any_threadgroup(kernel):
    """20000 columns is fifteen passes even at the largest group size."""
    logits, targets = _inputs(2, 20000, seed=11)
    got = _launch(kernel, logits, targets, 256)
    _assert_close(got, _expected(logits, targets))


def test_many_rows_at_the_largest_group_size(kernel):
    """256 threadgroups of 1024 threads: eight simdgroups deep and wide."""
    logits, targets = _inputs(256, 4096, seed=12)
    got = _launch(kernel, logits, targets, 1024)
    _assert_close(got, _expected(logits, targets))


@pytest.mark.parametrize("group_size", [64, 256, 1024])
def test_a_threadgroup_wider_than_the_row(kernel, group_size):
    """Most of the group has no column of its own and must not corrupt the row."""
    logits, targets = _inputs(6, 3, seed=13)
    got = _launch(kernel, logits, targets, group_size)
    _assert_close(got, _expected(logits, targets))


@pytest.mark.parametrize("group_size", [7, 33, 96, 129, 1000])
def test_a_group_size_that_is_not_a_power_of_two(kernel, group_size):
    logits, targets = _inputs(4, 257, seed=14)
    got = _launch(kernel, logits, targets, group_size)
    _assert_close(got, _expected(logits, targets))


@pytest.mark.parametrize("group_size", [33, 64, 129, 512])
def test_a_group_wider_than_one_simdgroup(kernel, group_size):
    """A reduction that only ever crosses 32 lanes gets these rows wrong."""
    logits, targets = _inputs(3, 900, seed=15)
    got = _launch(kernel, logits, targets, group_size)
    _assert_close(got, _expected(logits, targets))


def test_a_single_thread_per_row(kernel):
    logits, targets = _inputs(5, 300, seed=16)
    got = _launch(kernel, logits, targets, 1)
    _assert_close(got, _expected(logits, targets))


def test_a_single_column(kernel):
    """One column means the loss is zero: the row's only class is the target."""
    logits, targets = _inputs(4, 1, seed=17)
    got = _launch(kernel, logits, targets, 64)
    _assert_close(got, _expected(logits, targets))
    torch.testing.assert_close(
        got, torch.zeros(4, dtype=torch.float32), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("group_size", [1, 32, 128])
def test_large_positive_logits_stay_finite(kernel, group_size):
    """exp(300) is past float32, so the shift is not optional."""
    logits = torch.tensor(
        [[300.0, 299.0, 100.0, -400.0], [88.0, 90.0, 91.5, 89.0]], dtype=torch.float32
    )
    targets = torch.tensor([1, 2])
    got = _launch(kernel, logits, targets, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, targets))


@pytest.mark.parametrize("group_size", [7, 33, 96, 1000])
def test_a_wide_row_of_large_logits(kernel, group_size):
    """A row whose spread is past float32's exponent range at every group size.

    A shift that is not the row's maximum is algebraically harmless and
    numerically fatal here: the largest element overflows the moment it is
    exponentiated against anything smaller than itself.
    """
    logits, targets = _inputs(3, 1000, seed=24, scale=120.0)
    got = _launch(kernel, logits, targets, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, targets))


@pytest.mark.parametrize("group_size", [1, 32, 128])
def test_large_negative_logits_stay_finite(kernel, group_size):
    """Every exponential underflows to zero unless the row is shifted first."""
    logits, targets = _inputs(3, 64, seed=18, shift=-800.0)
    got = _launch(kernel, logits, targets, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, targets))


@pytest.mark.parametrize("group_size", [7, 128])
def test_the_loss_does_not_move_when_a_row_is_shifted(kernel, group_size):
    """Adding a constant to a whole row leaves the softmax, and the loss, alone."""
    logits, targets = _inputs(5, 257, seed=19)
    plain = _launch(kernel, logits, targets, group_size)
    shifted = _launch(kernel, logits + 137.0, targets, group_size)
    torch.testing.assert_close(shifted, plain, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("group_size", [3, 64])
def test_a_flat_row_costs_the_log_of_its_width(kernel, group_size):
    """Equal logits make every class equally likely, whatever the target is."""
    cols = 512
    logits = torch.full((4, cols), 2.5, dtype=torch.float32)
    targets = torch.tensor([0, 1, 511, 300])
    got = _launch(kernel, logits, targets, group_size)
    expected = torch.full((4,), math.log(cols), dtype=torch.float64)
    torch.testing.assert_close(got.double(), expected, atol=ATOL, rtol=RTOL)


def test_a_target_the_row_is_certain_about_costs_nothing(kernel):
    logits = torch.full((3, 40), -50.0, dtype=torch.float32)
    logits[0, 0] = 50.0
    logits[1, 39] = 50.0
    logits[2, 17] = 50.0
    targets = torch.tensor([0, 39, 17])
    got = _launch(kernel, logits, targets, 96)
    _assert_close(got, _expected(logits, targets))
    assert (got.abs() < 1e-3).all(), got


@pytest.mark.parametrize("group_size", [5, 64])
def test_targets_at_both_ends_of_the_row(kernel, group_size):
    cols = 129
    logits, _ = _inputs(2, cols, seed=20)
    targets = torch.tensor([0, cols - 1])
    got = _launch(kernel, logits, targets, group_size)
    _assert_close(got, _expected(logits, targets))


def test_two_launches_agree_bit_for_bit(kernel):
    logits, targets = _inputs(64, 777, seed=21)
    first = _launch(kernel, logits, targets, 128)
    second = _launch(kernel, logits, targets, 128)
    assert torch.equal(first, second), (first - second).abs().max()
    _assert_close(first, _expected(logits, targets))


def test_nothing_is_written_past_the_last_row(kernel):
    """The output buffer is longer than the grid, and the tail is not the kernel's."""
    logits, targets = _inputs(9, 65, seed=22)
    got = _launch(kernel, logits, targets, 64, out_len=9 + 16)
    _assert_close(got[:9], _expected(logits, targets))
    assert torch.isnan(got[9:]).all(), got[9:]


@pytest.mark.parametrize("group_size", [1, 32, 1024])
def test_one_row_on_its_own(kernel, group_size):
    logits, targets = _inputs(1, 4096, seed=23)
    got = _launch(kernel, logits, targets, group_size)
    _assert_close(got, _expected(logits, targets))
