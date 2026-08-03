"""Hidden tests — quantization_error_bounds.

Every assertion is licensed by a sentence in prompt.md, and every promise the
prompt makes has a test behind it.

Tolerances: the round-trip bound is exact in real arithmetic, so the checks add
1e-12 absolute. One division and one multiplication in float64 on values of
order a hundred carry an error near 1e-14, which leaves two orders of magnitude
of headroom and is still many orders tighter than any wrong scale would produce.
Scales and dequantized values are compared with rtol=1e-12 for the same reason.

Several tests choose inputs whose peak is exactly a power-of-two-ish integer so
that the scale comes out at exactly 1.0 and the expected code words can be
written down. That is deliberate: it pins the rounding rule without depending on
any float that is not exactly representable.
"""

import numpy as np
import pytest

from quantize import dequantize, quantize_per_channel, quantize_per_tensor


def spread_per_channel(values, axis: int, ndim: int) -> np.ndarray:
    """Reshape a per-channel vector so it broadcasts against the tensor."""
    shape = [1] * ndim
    shape[axis] = -1
    return np.asarray(values).reshape(shape)


def round_trip_per_channel(x, axis, num_bits=8, symmetric=True):
    q, scale, zero_point = quantize_per_channel(x, axis, num_bits, symmetric)
    restored = dequantize(
        q,
        spread_per_channel(scale, axis % np.ndim(x), np.ndim(x)),
        spread_per_channel(zero_point, axis % np.ndim(x), np.ndim(x)),
    )
    return q, scale, zero_point, restored


# -- what comes back -------------------------------------------------------


def test_the_codes_keep_the_shape_and_are_int32():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5)) * 3.0
    q, _, _ = quantize_per_tensor(x)
    assert q.shape == x.shape
    assert q.dtype == np.int32


def test_the_scale_is_a_positive_scalar_and_the_zero_point_a_scalar_integer():
    rng = np.random.default_rng(1)
    q, scale, zero_point = quantize_per_tensor(rng.normal(size=7))
    assert np.ndim(scale) == 0 and np.ndim(zero_point) == 0
    assert float(scale) > 0.0
    assert float(zero_point) == int(zero_point)


def test_dequantize_returns_float64():
    q, scale, zero_point = quantize_per_tensor(np.array([-2.0, 5.0]))
    restored = dequantize(q, scale, zero_point)
    assert restored.dtype == np.float64
    assert restored.shape == q.shape


def test_dequantize_is_the_scale_times_the_offset_codes():
    # The mapping back is fully determined, so it can be checked directly.
    q = np.array([-4, 0, 7], dtype=np.int32)
    assert np.allclose(dequantize(q, 0.25, 2), [-1.5, -0.5, 1.25], rtol=0.0, atol=1e-15)


def test_dequantize_broadcasts_a_per_channel_scale():
    q = np.array([[0, 4], [0, 4]], dtype=np.int32)
    scale = np.array([[1.0], [10.0]])
    zero_point = np.array([[0], [2]])
    assert np.allclose(dequantize(q, scale, zero_point), [[0.0, 4.0], [-20.0, 20.0]], atol=1e-15)


# -- the symmetric mapping -------------------------------------------------


def test_symmetric_pins_the_zero_point_at_zero():
    rng = np.random.default_rng(2)
    _, _, zero_point = quantize_per_tensor(rng.normal(size=32) * 5.0, symmetric=True)
    assert int(zero_point) == 0


def test_the_symmetric_scale_comes_from_the_largest_magnitude():
    x = np.array([-4.0, 1.0, 2.0])  # peak 4, so the scale is 4/127
    _, scale, _ = quantize_per_tensor(x, num_bits=8, symmetric=True)
    assert scale == pytest.approx(4.0 / 127.0, rel=1e-12)


def test_the_symmetric_codes_are_written_down_exactly():
    # Peak 127 makes the scale exactly 1.0, so the code words are the rounded
    # inputs and nothing is hidden behind a float division.
    x = np.array([-127.0, -0.5, 0.0, 62.5, 63.5, 127.0])
    q, scale, zero_point = quantize_per_tensor(x, num_bits=8, symmetric=True)
    assert scale == pytest.approx(1.0, rel=1e-12)
    assert int(zero_point) == 0
    # Ties go to the even neighbour: -0.5 -> 0, 62.5 -> 62, 63.5 -> 64.
    assert q.tolist() == [-127, 0, 0, 62, 64, 127]


def test_the_symmetric_extremes_land_on_the_signed_limits():
    x = np.array([-9.0, 3.0, 9.0])
    q, _, _ = quantize_per_tensor(x, num_bits=8, symmetric=True)
    assert q.min() == -127
    assert q.max() == 127


# -- the asymmetric mapping ------------------------------------------------


def test_the_asymmetric_scale_comes_from_the_range_not_the_magnitude():
    # Range 2.0 over 255 steps. Using the largest magnitude, 1.5, would give a
    # different number.
    x = np.array([-0.5, 1.5])
    _, scale, _ = quantize_per_tensor(x, num_bits=8, symmetric=False)
    assert scale == pytest.approx(2.0 / 255.0, rel=1e-12)


def test_the_asymmetric_codes_and_zero_point_are_written_down_exactly():
    x = np.array([-0.5, 1.5])
    q, scale, zero_point = quantize_per_tensor(x, num_bits=8, symmetric=False)
    assert scale == pytest.approx(2.0 / 255.0, rel=1e-12)
    # 0.5 / scale is 63.75, which rounds to 64.
    assert int(zero_point) == 64
    assert q.tolist() == [0, 255]


def test_the_asymmetric_codes_are_unsigned():
    rng = np.random.default_rng(3)
    q, _, _ = quantize_per_tensor(rng.normal(size=64) * 7.0, num_bits=8, symmetric=False)
    assert q.min() >= 0
    assert q.max() <= 255


def test_zero_is_pulled_into_the_range_of_an_all_positive_tensor():
    # The interval is [0, 11], not [10, 11]: zero has to be representable.
    x = np.array([10.0, 11.0])
    _, scale, zero_point = quantize_per_tensor(x, num_bits=8, symmetric=False)
    assert scale == pytest.approx(11.0 / 255.0, rel=1e-12)
    assert int(zero_point) == 0


def test_zero_is_pulled_into_the_range_of_an_all_negative_tensor():
    x = np.array([-11.0, -10.0])
    q, scale, zero_point = quantize_per_tensor(x, num_bits=8, symmetric=False)
    assert scale == pytest.approx(11.0 / 255.0, rel=1e-12)
    assert int(zero_point) == 255
    assert q.min() == 0


def test_the_zero_point_dequantizes_to_exactly_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(size=48) * 4.0
    _, scale, zero_point = quantize_per_tensor(x, num_bits=8, symmetric=False)
    assert dequantize(np.array([zero_point], dtype=np.int32), scale, zero_point)[0] == 0.0


# -- zero survives ---------------------------------------------------------


@pytest.mark.parametrize("symmetric", [True, False])
def test_an_exact_zero_round_trips_to_exact_zero(symmetric):
    x = np.array([-3.0, 0.0, 0.0, 7.0, -0.0])
    q, scale, zero_point = quantize_per_tensor(x, num_bits=8, symmetric=symmetric)
    restored = dequantize(q, scale, zero_point)
    assert restored[1] == 0.0
    assert restored[2] == 0.0
    assert restored[4] == 0.0


@pytest.mark.parametrize("symmetric", [True, False])
def test_an_all_zero_tensor_round_trips_exactly(symmetric):
    x = np.zeros((3, 4))
    q, scale, zero_point = quantize_per_tensor(x, symmetric=symmetric)
    assert scale == 1.0
    assert int(zero_point) == 0
    assert np.all(q == 0)
    assert np.all(dequantize(q, scale, zero_point) == 0.0)


@pytest.mark.parametrize("symmetric", [True, False])
def test_a_mostly_zero_tensor_keeps_every_zero(symmetric):
    # The padded-tensor case: most of it is zero and none of those may drift.
    rng = np.random.default_rng(5)
    x = np.zeros(200)
    live = rng.integers(0, 200, size=20)
    x[live] = rng.normal(size=20) * 6.0

    q, scale, zero_point = quantize_per_tensor(x, symmetric=symmetric)
    restored = dequantize(q, scale, zero_point)
    assert np.all(restored[x == 0.0] == 0.0)


# -- the bound -------------------------------------------------------------


@pytest.mark.parametrize("num_bits", [2, 4, 8, 16])
@pytest.mark.parametrize("symmetric", [True, False])
def test_the_round_trip_stays_within_half_a_scale(num_bits, symmetric):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(6, 7)) * 20.0

    q, scale, zero_point = quantize_per_tensor(x, num_bits, symmetric)
    error = np.abs(x - dequantize(q, scale, zero_point))
    # Exact in real arithmetic; 1e-12 absorbs float64 rounding on values of
    # order twenty, which is about 1e-14.
    assert error.max() <= scale / 2.0 + 1e-12


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_bound_holds_for_a_lopsided_distribution(symmetric):
    rng = np.random.default_rng(7)
    x = rng.exponential(scale=3.0, size=500) + 1.0  # strictly positive, long tail
    q, scale, zero_point = quantize_per_tensor(x, symmetric=symmetric)
    error = np.abs(x - dequantize(q, scale, zero_point))
    assert error.max() <= scale / 2.0 + 1e-12


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_bound_holds_for_a_single_repeated_value(symmetric):
    for constant in (-4.0, 0.25, 12.0):
        x = np.full(5, constant)
        q, scale, zero_point = quantize_per_tensor(x, symmetric=symmetric)
        error = np.abs(x - dequantize(q, scale, zero_point))
        assert error.max() <= scale / 2.0 + 1e-12


def test_more_bits_cannot_be_worse():
    rng = np.random.default_rng(8)
    x = rng.normal(size=256) * 5.0

    def worst(num_bits):
        q, scale, zero_point = quantize_per_tensor(x, num_bits, symmetric=True)
        return np.abs(x - dequantize(q, scale, zero_point)).max()

    assert worst(8) < worst(4) < worst(2)


# -- the integer range for other widths -----------------------------------


@pytest.mark.parametrize("num_bits", [2, 4, 8, 16])
def test_the_symmetric_range_follows_the_bit_width(num_bits):
    rng = np.random.default_rng(9)
    q, _, _ = quantize_per_tensor(rng.normal(size=300) * 9.0, num_bits, symmetric=True)
    assert q.dtype == np.int32  # int32 at every width, not the narrowest that fits
    assert q.min() >= -(2 ** (num_bits - 1))
    assert q.max() <= 2 ** (num_bits - 1) - 1


@pytest.mark.parametrize("num_bits", [2, 4, 8, 16])
def test_the_asymmetric_range_follows_the_bit_width(num_bits):
    rng = np.random.default_rng(10)
    q, _, _ = quantize_per_tensor(rng.normal(size=300) * 9.0, num_bits, symmetric=False)
    assert q.dtype == np.int32  # int32 at every width, not an unsigned type
    assert q.min() >= 0
    assert q.max() <= 2**num_bits - 1


def test_four_bit_scales_use_the_narrower_range():
    x = np.array([-1.0, 1.0])
    _, symmetric_scale, _ = quantize_per_tensor(x, num_bits=4, symmetric=True)
    _, asymmetric_scale, _ = quantize_per_tensor(x, num_bits=4, symmetric=False)
    assert symmetric_scale == pytest.approx(1.0 / 7.0, rel=1e-12)
    assert asymmetric_scale == pytest.approx(2.0 / 15.0, rel=1e-12)


# -- per channel -----------------------------------------------------------


def test_per_channel_returns_one_scale_per_slice():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(4, 6)) * 2.0
    q, scale, zero_point = quantize_per_channel(x, axis=0)
    assert q.shape == x.shape
    assert q.dtype == np.int32
    assert scale.shape == (4,)
    assert zero_point.shape == (4,)
    assert scale.dtype == np.float64
    assert zero_point.dtype == np.int32


def test_each_channel_scale_comes_from_that_channel_alone():
    x = np.array([[1.0, -1.0, 0.5], [100.0, -100.0, 50.0]])
    _, scale, _ = quantize_per_channel(x, axis=0, num_bits=8, symmetric=True)
    assert scale == pytest.approx([1.0 / 127.0, 100.0 / 127.0], rel=1e-12)


def test_the_channel_axis_decides_the_reduction():
    x = np.array([[1.0, -1.0, 0.5], [100.0, -100.0, 50.0]])
    _, along_columns, _ = quantize_per_channel(x, axis=1, num_bits=8, symmetric=True)
    assert along_columns == pytest.approx(
        [100.0 / 127.0, 100.0 / 127.0, 50.0 / 127.0], rel=1e-12
    )


def test_a_negative_axis_counts_from_the_end():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, 4, 5)) * 3.0
    last = quantize_per_channel(x, axis=2)
    negative = quantize_per_channel(x, axis=-1)
    assert np.array_equal(last[0], negative[0])
    assert np.allclose(last[1], negative[1], rtol=1e-12, atol=0.0)
    assert np.array_equal(last[2], negative[2])


def test_per_channel_zero_points_are_computed_per_channel():
    x = np.array([[0.0, 1.0], [-1.0, 0.0]])
    _, scale, zero_point = quantize_per_channel(x, axis=0, num_bits=8, symmetric=False)
    assert scale == pytest.approx([1.0 / 255.0, 1.0 / 255.0], rel=1e-12)
    assert zero_point.tolist() == [0, 255]


def test_per_channel_symmetric_zero_points_are_all_zero():
    rng = np.random.default_rng(13)
    _, _, zero_point = quantize_per_channel(rng.normal(size=(5, 8)), axis=0, symmetric=True)
    assert zero_point.tolist() == [0] * 5


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("symmetric", [True, False])
def test_the_per_channel_bound_uses_that_channels_scale(axis, symmetric):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(3, 4, 5)) * 8.0
    _, scale, _, restored = round_trip_per_channel(x, axis, 8, symmetric)

    error = np.abs(x - restored)
    bound = spread_per_channel(scale, axis, x.ndim) / 2.0 + 1e-12
    assert np.all(error <= bound)


def test_per_channel_rescues_a_channel_a_single_scale_would_flatten():
    # This is why per-channel exists: one loud channel sets a per-tensor scale
    # so coarse that a quiet channel loses everything.
    rng = np.random.default_rng(15)
    x = np.empty((2, 64))
    x[0] = rng.uniform(-1.0, 1.0, size=64)
    x[1] = rng.uniform(-1000.0, 1000.0, size=64)

    q, scale, zero_point = quantize_per_tensor(x, symmetric=True)
    per_tensor_error = np.abs(x - dequantize(q, scale, zero_point))[0].max()

    _, _, _, restored = round_trip_per_channel(x, axis=0, symmetric=True)
    per_channel_error = np.abs(x - restored)[0].max()

    assert per_channel_error < per_tensor_error


def test_one_channel_is_the_same_as_per_tensor():
    rng = np.random.default_rng(16)
    x = rng.normal(size=(1, 40)) * 6.0
    per_tensor = quantize_per_tensor(x, symmetric=True)
    per_channel = quantize_per_channel(x, axis=0, symmetric=True)
    assert np.array_equal(per_tensor[0], per_channel[0])
    assert per_channel[1][0] == pytest.approx(per_tensor[1], rel=1e-12)


@pytest.mark.parametrize("symmetric", [True, False])
def test_an_all_zero_channel_beside_a_live_one(symmetric):
    x = np.array([[0.0, 0.0, 0.0], [2.0, -3.0, 1.0]])
    _, scale, zero_point, restored = round_trip_per_channel(x, axis=0, symmetric=symmetric)
    assert scale[0] == 1.0
    assert int(zero_point[0]) == 0
    assert np.all(restored[0] == 0.0)
    assert np.abs(x[1] - restored[1]).max() <= scale[1] / 2.0 + 1e-12


def test_per_channel_on_a_four_dimensional_weight_tensor():
    # The shape a convolution weight actually has, quantized per output filter.
    rng = np.random.default_rng(17)
    x = rng.normal(size=(6, 3, 2, 2)) * 4.0
    _, scale, _, restored = round_trip_per_channel(x, axis=0, symmetric=True)
    assert scale.shape == (6,)
    error = np.abs(x - restored)
    assert np.all(error <= spread_per_channel(scale, 0, x.ndim) / 2.0 + 1e-12)


# -- inputs ----------------------------------------------------------------


@pytest.mark.parametrize("symmetric", [True, False])
def test_float32_input_is_accepted(symmetric):
    rng = np.random.default_rng(18)
    x = (rng.normal(size=(3, 5)) * 4.0).astype(np.float32)
    q, scale, zero_point = quantize_per_tensor(x, symmetric=symmetric)
    restored = dequantize(q, scale, zero_point)
    assert restored.dtype == np.float64
    assert np.abs(np.asarray(x, dtype=np.float64) - restored).max() <= scale / 2.0 + 1e-12


def test_a_nested_list_is_accepted():
    q, scale, zero_point = quantize_per_tensor([[-2.0, 0.0], [1.0, 2.0]], symmetric=True)
    assert q.shape == (2, 2)
    assert scale == pytest.approx(2.0 / 127.0, rel=1e-12)


def test_the_input_is_left_alone():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(4, 4)) * 3.0
    before = x.copy()
    quantize_per_tensor(x, symmetric=True)
    quantize_per_tensor(x, symmetric=False)
    quantize_per_channel(x, axis=0)
    quantize_per_channel(x, axis=-1)
    assert np.array_equal(before, x)


def test_a_single_element_tensor_works():
    q, scale, zero_point = quantize_per_tensor(np.array([5.0]), symmetric=True)
    assert q.shape == (1,)
    assert np.abs(5.0 - dequantize(q, scale, zero_point)[0]) <= scale / 2.0 + 1e-12
