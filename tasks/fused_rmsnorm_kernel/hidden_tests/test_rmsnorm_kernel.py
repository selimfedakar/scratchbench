"""Hidden tests for the fused RMSNorm kernel.

Two things are graded here. The kernel is launched directly, with block sizes
chosen by the test rather than by the solution, because the kernel is the task
and a wrapper that quietly did the arithmetic in PyTorch would otherwise pass.
The wrapper is then graded on the contract it advertises.

There is deliberately no `skipif` on CUDA availability, and no test function for
it either. pytest exits zero when every test is skipped and the runner reads a
zero exit as a pass, so a skip would print a solved task on a machine that never
ran a line of it. A test would fail in the other direction: it is the one
assertion an untouched starter passes, and a test that cannot fail is noise in a
suite whose only job is to separate pass from fail. The module-level assertion
below is neither — missing hardware is a loud collection error, and every test
in this file still fails against the untouched starter. Deciding whether the
hardware exists at all stays the runner's job; it answers `needs_accelerator`,
which is neither a pass nor a failure.
"""

import torch

import rmsnorm_kernel
from rmsnorm_kernel import rmsnorm

assert torch.cuda.is_available(), "these tests grade a CUDA kernel and this machine has no CUDA device"

# float32 carries about 1.2e-7 of relative resolution. The reduction runs over
# at most a few thousand terms and is followed by a reciprocal square root, so
# a handful of ulps is expected; 1e-5 is two orders of magnitude looser than
# that and still far tighter than any real mistake in the mean or the scale.
FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6

# float16 resolves to about 4.9e-4 relative. The kernel computes in float32 and
# rounds once on the way out, so the error is dominated by that single store;
# 4e-3 is several times the rounding and nowhere near the error a float16
# accumulator or a missing rescale would produce.
FLOAT16_RTOL = 4e-3
FLOAT16_ATOL = 2e-3


def make_input(n_rows, n_cols, dtype, seed=0, scale=1.0):
    """Deterministic input, generated on the CPU and moved to the device."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n_cols, generator=generator, dtype=torch.float32) * scale
    return x.to(device="cuda", dtype=dtype)


def make_weight(n_cols, seed=1):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(n_cols, generator=generator, dtype=torch.float32)
    return weight.to(device="cuda")


def expected_rmsnorm(x, weight, eps):
    """The answer, computed in float64 on the CPU."""
    x64 = x.detach().to(device="cpu", dtype=torch.float64)
    weight64 = weight.detach().to(device="cpu", dtype=torch.float64)
    inv_rms = torch.rsqrt(x64.pow(2).mean(dim=1, keepdim=True) + eps)
    return x64 * inv_rms * weight64


def assert_matches(actual, x, weight, eps, rtol, atol):
    got = actual.detach().to(device="cpu", dtype=torch.float64)
    assert torch.isfinite(got).all(), "the result contains an infinity or a NaN"
    torch.testing.assert_close(got, expected_rmsnorm(x, weight, eps), rtol=rtol, atol=atol)


def run_kernel(x, weight, eps, block_size, out=None):
    """Launch the kernel directly, one program per row."""
    n_rows, n_cols = x.shape
    if out is None:
        # Zeros rather than `empty`, so a kernel that writes nothing produces a
        # deterministic wrong answer instead of whatever was in memory.
        out = torch.zeros((n_rows, n_cols), dtype=x.dtype, device=x.device)
    rmsnorm_kernel.rmsnorm_fwd_kernel[(n_rows,)](
        x,
        weight,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return out


# -- the kernel, launched directly ----------------------------------------
#
# Nothing here asserts that `rmsnorm_fwd_kernel` is a `@triton.jit` function.
# `run_kernel` subscripts it with a launch grid, which only a JITFunction
# supports, so a plain Python function fails every test below on its own. An
# explicit type assertion would say the same thing and would also be the one
# test an untouched starter passes, since the starter is already decorated.


def test_one_block_covers_the_whole_row():
    x = make_input(4, 128, torch.float32)
    weight = make_weight(128)
    out = run_kernel(x, weight, 1e-6, block_size=128)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_a_row_shorter_than_the_block_is_masked():
    x = make_input(3, 100, torch.float32)
    weight = make_weight(100)
    out = run_kernel(x, weight, 1e-6, block_size=128)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_a_row_longer_than_the_block_takes_several_passes():
    x = make_input(5, 1000, torch.float32)
    weight = make_weight(1000)
    out = run_kernel(x, weight, 1e-6, block_size=256)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_a_long_row_that_divides_evenly_into_blocks():
    x = make_input(2, 8192, torch.float32)
    weight = make_weight(8192)
    out = run_kernel(x, weight, 1e-6, block_size=1024)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_a_single_column():
    x = make_input(4, 1, torch.float32)
    weight = make_weight(1)
    out = run_kernel(x, weight, 1e-6, block_size=16)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_float16_input_gives_float16_output():
    x = make_input(4, 1024, torch.float16)
    weight = make_weight(1024)
    out = run_kernel(x, weight, 1e-6, block_size=1024)
    assert out.dtype == torch.float16
    assert_matches(out, x, weight, 1e-6, FLOAT16_RTOL, FLOAT16_ATOL)


def test_float16_values_whose_squares_are_past_the_float16_ceiling():
    # float16 stops at 65504, so any value past 256 squares to something the
    # format cannot hold. These rows sit around 300: ordinary float16 numbers
    # whose squares are around 9e4. Squared and accumulated in float32 that is
    # unremarkable; squared in the input dtype it is an infinity on the first
    # element, the mean square is infinite, its reciprocal root is zero, and
    # the row comes back as zeros rather than as an error.
    x = make_input(2, 4096, torch.float16, seed=7, scale=300.0)
    weight = make_weight(4096)
    out = run_kernel(x, weight, 1e-6, block_size=1024)
    assert_matches(out, x, weight, 1e-6, FLOAT16_RTOL, FLOAT16_ATOL)


def test_a_float16_row_long_enough_to_take_sixteen_blocks():
    # Sixteen passes over the row in each of the two loops, in float16. This
    # is about the loop rather than about the accumulator dtype: with a block
    # of 1024 no single accumulator lane sees more than sixteen squares, so a
    # float16 accumulator does not overflow here — the test above is the one
    # that pins the dtype.
    x = make_input(2, 16384, torch.float16, seed=18, scale=2.0)
    weight = make_weight(16384)
    out = run_kernel(x, weight, 1e-6, block_size=1024)
    assert_matches(out, x, weight, 1e-6, FLOAT16_RTOL, FLOAT16_ATOL)


def test_rows_of_the_input_may_be_strided():
    base = make_input(6, 300, torch.float32, seed=3)
    x = base[:, :192]
    assert x.stride(0) == 300 and x.stride(1) == 1
    weight = make_weight(192)
    out = run_kernel(x, weight, 1e-6, block_size=64)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_the_output_row_stride_is_respected():
    x = make_input(4, 96, torch.float32, seed=4)
    weight = make_weight(96)
    padded = torch.full((4, 160), -7.0, dtype=torch.float32, device="cuda")
    out = run_kernel(x, weight, 1e-6, block_size=128, out=padded[:, :96])

    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)
    # Everything past the row is somebody else's memory.
    assert torch.all(padded[:, 96:] == -7.0)


def test_each_column_is_scaled_by_its_own_weight():
    # A weight that increases along the row: an implementation that indexes it
    # by anything other than the column produces a different answer here even
    # though the norm itself is unchanged.
    x = make_input(4, 64, torch.float32, seed=5)
    weight = torch.arange(1, 65, dtype=torch.float32, device="cuda")
    out = run_kernel(x, weight, 1e-6, block_size=64)
    assert_matches(out, x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_rows_are_normalised_independently():
    # Rows at wildly different scales normalise to the same distribution. A
    # mean taken over the whole tensor instead of the row cannot do this.
    row = torch.randn(1, 256, generator=torch.Generator().manual_seed(11))
    x = torch.cat([row, row * 1000.0, row * 0.001]).to(device="cuda")
    weight = make_weight(256)
    out = run_kernel(x, weight, 1e-8, block_size=256)
    assert_matches(out, x, weight, 1e-8, FLOAT32_RTOL, FLOAT32_ATOL)


def test_shuffling_the_rows_shuffles_the_answers():
    x = make_input(8, 512, torch.float32, seed=6)
    weight = make_weight(512)
    order = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4], device="cuda")

    straight = run_kernel(x, weight, 1e-6, block_size=256)
    shuffled = run_kernel(x[order].contiguous(), weight, 1e-6, block_size=256)
    torch.testing.assert_close(shuffled, straight[order], rtol=0.0, atol=0.0)


def test_epsilon_is_added_to_the_mean_square():
    # A large epsilon changes the answer by a large, checkable amount; the
    # float64 expectation puts it in exactly one place.
    x = make_input(3, 128, torch.float32, seed=8)
    weight = make_weight(128)
    out = run_kernel(x, weight, 0.5, block_size=128)
    assert_matches(out, x, weight, 0.5, FLOAT32_RTOL, FLOAT32_ATOL)


def test_an_all_zero_row_is_finite():
    # Without epsilon inside the square root this is a division by zero and
    # the row comes back as NaN.
    x = torch.zeros(2, 64, dtype=torch.float32, device="cuda")
    weight = make_weight(64)
    out = run_kernel(x, weight, 1e-6, block_size=64)
    assert torch.all(out == 0.0)


def test_the_kernel_does_not_touch_its_inputs():
    x = make_input(4, 256, torch.float32, seed=9)
    weight = make_weight(256)
    x_before = x.clone()
    weight_before = weight.clone()

    run_kernel(x, weight, 1e-6, block_size=128)

    torch.testing.assert_close(x, x_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weight, weight_before, rtol=0.0, atol=0.0)


def test_the_same_input_gives_the_same_bits_twice():
    x = make_input(4, 1024, torch.float32, seed=10)
    weight = make_weight(1024)
    first = run_kernel(x, weight, 1e-6, block_size=256)
    second = run_kernel(x, weight, 1e-6, block_size=256)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


# -- the wrapper -----------------------------------------------------------


def test_the_wrapper_normalises():
    x = make_input(16, 4096, torch.float32, seed=12)
    weight = make_weight(4096)
    assert_matches(rmsnorm(x, weight, 1e-6), x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_the_wrapper_handles_a_row_length_that_is_not_a_power_of_two():
    x = make_input(7, 4095, torch.float16, seed=13)
    weight = make_weight(4095)
    assert_matches(rmsnorm(x, weight, 1e-6), x, weight, 1e-6, FLOAT16_RTOL, FLOAT16_ATOL)


def test_the_wrapper_returns_a_new_contiguous_tensor_on_the_same_device():
    x = make_input(4, 300, torch.float16, seed=14)
    weight = make_weight(300)
    out = rmsnorm(x, weight, 1e-6)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert out.is_contiguous()
    assert out.data_ptr() != x.data_ptr()


def test_the_wrapper_accepts_a_strided_view():
    base = make_input(5, 1000, torch.float32, seed=15)
    x = base[:, :768]
    weight = make_weight(768)
    assert_matches(rmsnorm(x, weight, 1e-6), x, weight, 1e-6, FLOAT32_RTOL, FLOAT32_ATOL)


def test_the_wrapper_leaves_its_arguments_alone():
    x = make_input(4, 512, torch.float32, seed=16)
    weight = make_weight(512)
    x_before = x.clone()
    weight_before = weight.clone()

    rmsnorm(x, weight, 1e-6)

    torch.testing.assert_close(x, x_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weight, weight_before, rtol=0.0, atol=0.0)


def test_the_wrapper_defaults_epsilon():
    x = make_input(4, 128, torch.float32, seed=17)
    weight = make_weight(128)
    torch.testing.assert_close(
        rmsnorm(x, weight), rmsnorm(x, weight, 1e-6), rtol=0.0, atol=0.0
    )


def test_no_rows_is_an_empty_result_rather_than_an_error():
    x = torch.empty(0, 128, dtype=torch.float32, device="cuda")
    weight = make_weight(128)
    out = rmsnorm(x, weight, 1e-6)
    assert out.shape == (0, 128)
    assert out.dtype == torch.float32
    assert out.device == x.device
