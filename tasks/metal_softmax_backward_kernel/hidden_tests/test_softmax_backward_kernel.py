"""Hidden tests for metal_softmax_backward_kernel.

Every expected value is computed on the CPU in float64 by PyTorch's own
autograd, so the kernel is compared against an independent implementation
rather than against itself.

Tolerance: the kernel accumulates in float32 and the reference in float64.
Measured against the reference over every configuration in this file, the worst
absolute deviation is 9.9e-8, on a four-by-eight row driven by a single thread,
and the worst absolute row sum is 2.0e-7. `ATOL` is 2e-6 — twenty times the
measured worst, so a different Apple GPU has room, and still four orders of
magnitude under the smallest gradient any test here looks at, which is 6.3e-3 on
the flat row. Every wrong implementation written against this task in
`tools/mutate_metal_task.py` fails rather than rounds into range.

`INVARIANT_ATOL` is looser at 1e-5, and for a reason: the two shift tests
compare two launches whose intermediate values differ by 137, so the
cancellation is real arithmetic rather than slack. The worst measured difference
there is 7.5e-7.
"""

import pytest
import torch

from softmax_backward_kernel import SOURCE

ATOL = 2e-6
RTOL = 1e-5
INVARIANT_ATOL = 1e-5

#: The output buffer is filled with this before every launch, so an element
#: nobody wrote is an element that fails rather than one that happens to hold a
#: zero — and zero is a legitimate answer here.
SENTINEL = float("nan")

KERNEL_NAME = "softmax_backward_rows"

SHAPES = [(1, 1), (4, 8), (5, 33), (7, 129), (2, 1000)]
GROUP_SIZES = [1, 7, 32, 33, 96, 1024]


@pytest.fixture(scope="module")
def kernel():
    """The compiled kernel, built once for the whole module."""
    library = torch.mps.compile_shader(SOURCE)
    return getattr(library, KERNEL_NAME)


def _inputs(rows, cols, seed=0, scale=1.0, shift=0.0):
    """Deterministic logits and upstream gradients on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, cols, generator=generator) * scale + shift
    upstream = torch.randn(rows, cols, generator=generator)
    return logits, upstream


def _expected(logits, upstream):
    """The gradient of the row-wise softmax, in float64, on the CPU."""
    x = logits.double().requires_grad_(True)
    y = torch.softmax(x, dim=-1)
    (grad,) = torch.autograd.grad(y, x, upstream.double())
    return grad


def _launch(kernel, logits, upstream, group_size, out_rows=None):
    """Dispatch one threadgroup of `group_size` threads per row."""
    rows, cols = logits.shape
    dx = torch.full(
        (out_rows or rows, cols), SENTINEL, device="mps", dtype=torch.float32
    )
    kernel(
        dx,
        logits.contiguous().to("mps"),
        upstream.contiguous().to("mps"),
        cols,
        threads=rows * group_size,
        group_size=group_size,
    )
    torch.mps.synchronize()
    return dx.cpu()


def _assert_close(got, expected):
    torch.testing.assert_close(got.double(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_matches_a_float64_cpu_gradient(kernel, shape, group_size):
    rows, cols = shape
    logits, upstream = _inputs(rows, cols, seed=rows * 100 + cols)
    got = _launch(kernel, logits, upstream, group_size)
    _assert_close(got, _expected(logits, upstream))


def test_a_row_wider_than_any_threadgroup(kernel):
    """20000 columns is fifteen passes even at the largest group size."""
    logits, upstream = _inputs(2, 20000, seed=11)
    got = _launch(kernel, logits, upstream, 256)
    _assert_close(got, _expected(logits, upstream))


def test_many_rows_at_the_largest_group_size(kernel):
    """128 threadgroups of 1024 threads: eight simdgroups deep and wide."""
    logits, upstream = _inputs(128, 4096, seed=12)
    got = _launch(kernel, logits, upstream, 1024)
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [64, 256, 1024])
def test_a_threadgroup_wider_than_the_row(kernel, group_size):
    """Most of the group has no column of its own and must not corrupt the row."""
    logits, upstream = _inputs(6, 3, seed=13)
    got = _launch(kernel, logits, upstream, group_size)
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [7, 33, 96, 129, 1000])
def test_a_group_size_that_is_not_a_power_of_two(kernel, group_size):
    logits, upstream = _inputs(4, 257, seed=14)
    got = _launch(kernel, logits, upstream, group_size)
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [33, 64, 129, 512])
def test_a_group_wider_than_one_simdgroup(kernel, group_size):
    """A reduction that only ever crosses 32 lanes gets these rows wrong."""
    logits, upstream = _inputs(3, 900, seed=15)
    got = _launch(kernel, logits, upstream, group_size)
    _assert_close(got, _expected(logits, upstream))


def test_a_single_thread_per_row(kernel):
    logits, upstream = _inputs(5, 300, seed=16)
    got = _launch(kernel, logits, upstream, 1)
    _assert_close(got, _expected(logits, upstream))


def test_a_single_column(kernel):
    """A row of one class has softmax 1 whatever the logit is, so it cannot move."""
    logits, upstream = _inputs(4, 1, seed=17)
    got = _launch(kernel, logits, upstream, 64)
    _assert_close(got, _expected(logits, upstream))
    torch.testing.assert_close(
        got, torch.zeros(4, 1, dtype=torch.float32), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("group_size", [1, 33, 128, 1000])
def test_the_gradients_of_a_row_sum_to_zero(kernel, group_size):
    """The softmax Jacobian's columns sum to zero, so every output row does too.

    This holds whatever the upstream gradient is, and it is the property an
    implementation that forgets to subtract the contraction loses first.
    """
    logits, upstream = _inputs(6, 513, seed=25)
    got = _launch(kernel, logits, upstream, group_size)
    torch.testing.assert_close(
        got.double().sum(dim=-1),
        torch.zeros(6, dtype=torch.float64),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("group_size", [1, 32, 128])
def test_large_positive_logits_stay_finite(kernel, group_size):
    """exp(300) is past float32, so the shift is not optional."""
    logits = torch.tensor(
        [[300.0, 299.0, 100.0, -400.0], [88.0, 90.0, 91.5, 89.0]], dtype=torch.float32
    )
    upstream = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]], dtype=torch.float32
    )
    got = _launch(kernel, logits, upstream, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [7, 33, 96, 1000])
def test_a_wide_row_of_large_logits(kernel, group_size):
    """A row whose spread is past float32's exponent range at every group size.

    A shift that is not the row's maximum is algebraically harmless and
    numerically fatal here: the largest element overflows the moment it is
    exponentiated against anything smaller than itself.
    """
    logits, upstream = _inputs(3, 1000, seed=24, scale=120.0)
    got = _launch(kernel, logits, upstream, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [3, 5, 7, 9, 33])
def test_the_largest_logit_in_every_position(kernel, group_size):
    """The row's maximum walks the row, at group sizes that do not divide it.

    A shift that is merely close to the maximum is good enough on a row of
    ordinary logits, so a reduction that quietly loses one lane's partial
    maximum can look correct everywhere. It stops looking correct when the lane
    it loses is the one holding a logit that overflows float32 on its own.
    """
    cols = 7
    logits = torch.zeros(cols, cols, dtype=torch.float32)
    for index in range(cols):
        logits[index, index] = 300.0
    upstream = torch.randn(cols, cols, generator=torch.Generator().manual_seed(41))
    got = _launch(kernel, logits, upstream, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [1, 32, 128])
def test_large_negative_logits_stay_finite(kernel, group_size):
    """Every exponential underflows to zero unless the row is shifted first."""
    logits, upstream = _inputs(3, 64, seed=18, shift=-800.0)
    got = _launch(kernel, logits, upstream, group_size)
    assert torch.isfinite(got).all(), got
    _assert_close(got, _expected(logits, upstream))


@pytest.mark.parametrize("group_size", [7, 128])
def test_the_gradient_does_not_move_when_a_row_of_logits_is_shifted(kernel, group_size):
    """Adding a constant to a whole row leaves the softmax, and its Jacobian, alone."""
    logits, upstream = _inputs(5, 257, seed=19)
    plain = _launch(kernel, logits, upstream, group_size)
    shifted = _launch(kernel, logits + 137.0, upstream, group_size)
    torch.testing.assert_close(shifted, plain, atol=INVARIANT_ATOL, rtol=RTOL)


@pytest.mark.parametrize("group_size", [7, 128])
def test_the_gradient_does_not_move_when_the_upstream_is_shifted(kernel, group_size):
    """The Jacobian annihilates a constant, so a constant added to the upstream
    gradient cancels against the contraction it changes."""
    logits, upstream = _inputs(5, 257, seed=19)
    plain = _launch(kernel, logits, upstream, group_size)
    shifted = _launch(kernel, logits, upstream + 137.0, group_size)
    torch.testing.assert_close(shifted, plain, atol=INVARIANT_ATOL, rtol=RTOL)


@pytest.mark.parametrize("group_size", [5, 96])
def test_a_uniform_upstream_gradient_produces_nothing(kernel, group_size):
    """The limiting case of the test above: a constant upstream gradient on its own."""
    logits, _ = _inputs(3, 257, seed=30)
    upstream = torch.full((3, 257), 0.75, dtype=torch.float32)
    got = _launch(kernel, logits, upstream, group_size)
    torch.testing.assert_close(
        got, torch.zeros(3, 257, dtype=torch.float32), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("group_size", [3, 64])
def test_a_flat_row_has_a_closed_form(kernel, group_size):
    """Equal logits make every class equally likely, and the Jacobian collapses
    to `(upstream - mean(upstream)) / n_cols`."""
    cols = 512
    logits = torch.full((4, cols), 2.5, dtype=torch.float32)
    upstream = torch.randn(4, cols, generator=torch.Generator().manual_seed(77))
    got = _launch(kernel, logits, upstream, group_size)
    closed = (upstream.double() - upstream.double().mean(dim=-1, keepdim=True)) / cols
    torch.testing.assert_close(got.double(), closed, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("group_size", [1, 33, 256])
def test_a_one_hot_upstream_gradient_is_a_column_of_the_jacobian(kernel, group_size):
    """One non-zero upstream element per row picks out one column of `dy_i/dx_j`,
    which is `y_j * (delta_ij - y_i)` and is checked against autograd's."""
    logits, _ = _inputs(3, 128, seed=31)
    upstream = torch.zeros(3, 128, dtype=torch.float32)
    upstream[0, 0] = 1.0
    upstream[1, 64] = 1.0
    upstream[2, 127] = 1.0
    got = _launch(kernel, logits, upstream, group_size)
    _assert_close(got, _expected(logits, upstream))


def test_two_launches_agree_bit_for_bit(kernel):
    logits, upstream = _inputs(64, 777, seed=21)
    first = _launch(kernel, logits, upstream, 128)
    second = _launch(kernel, logits, upstream, 128)
    assert torch.equal(first, second), (first - second).abs().max()
    _assert_close(first, _expected(logits, upstream))


def test_nothing_is_written_past_the_last_row(kernel):
    """The output buffer is longer than the grid, and the tail is not the kernel's."""
    logits, upstream = _inputs(9, 65, seed=22)
    got = _launch(kernel, logits, upstream, 64, out_rows=9 + 16)
    _assert_close(got[:9], _expected(logits, upstream))
    assert torch.isnan(got[9:]).all(), got[9:]


@pytest.mark.parametrize("group_size", [1, 32, 1024])
def test_one_row_on_its_own(kernel, group_size):
    logits, upstream = _inputs(1, 4096, seed=23)
    got = _launch(kernel, logits, upstream, group_size)
    _assert_close(got, _expected(logits, upstream))
