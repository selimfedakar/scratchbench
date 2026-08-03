"""Hidden tests — softmax_stability.

Every assertion here is licensed by a sentence in prompt.md. Tolerances are
1e-12 for quantities that should be exact to a few ULPs of a float64 sum, and
1e-9 where a value near 1e4 is involved, since float64 carries about 15
significant digits and 1e4 spends four of them before the fraction starts.
"""

import math

import numpy as np

from stable_softmax import log_softmax, logsumexp, softmax


# -- softmax, the ordinary cases -----------------------------------------


def test_softmax_matches_a_hand_computed_case():
    # exp(0) : exp(log 3) = 1 : 3
    out = softmax(np.array([0.0, math.log(3.0)]))
    assert np.allclose(out, [0.25, 0.75], atol=1e-12)


def test_softmax_sums_to_one_along_the_default_axis():
    rng = np.random.default_rng(0)
    out = softmax(rng.normal(size=(4, 7)) * 5.0)
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-12)


def test_softmax_is_invariant_to_a_constant_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5))
    assert np.allclose(softmax(x), softmax(x + 1000.0), atol=1e-12)


def test_softmax_is_never_negative():
    rng = np.random.default_rng(2)
    assert np.all(softmax(rng.normal(size=(6, 6)) * 20.0) >= 0.0)


# -- softmax, the reason this task exists ---------------------------------


def test_softmax_survives_logits_of_ten_thousand():
    # exp(1e4) is +inf in float64, so anything that exponentiates before
    # shifting produces NaN here.
    out = softmax(np.array([1e4, 1e4 + 1.0, 1e4 - 1.0]))
    assert np.all(np.isfinite(out))
    assert abs(out.sum() - 1.0) < 1e-12
    assert np.allclose(out, softmax(np.array([0.0, 1.0, -1.0])), atol=1e-12)


def test_softmax_survives_very_negative_logits():
    out = softmax(np.array([-1e4, -1e4 - 1.0]))
    assert np.all(np.isfinite(out))
    assert abs(out.sum() - 1.0) < 1e-12


def test_softmax_handles_a_mixed_batch_of_scales():
    x = np.array([[1e4, 1e4 + 2.0], [-1e4, -1e4 + 2.0], [0.0, 2.0]])
    out = softmax(x)
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-12)
    # Every row is the same two logits two apart, so every row is the same
    # distribution however far from zero it sits.
    assert np.allclose(out, out[2], atol=1e-12)


# -- masking ---------------------------------------------------------------


def test_masked_entries_get_exactly_zero_probability():
    out = softmax(np.array([1.0, -np.inf, 2.0]))
    assert out[1] == 0.0
    assert abs(out.sum() - 1.0) < 1e-12
    assert np.allclose(out[[0, 2]], softmax(np.array([1.0, 2.0])), atol=1e-12)


def test_a_fully_masked_slice_is_all_zeros():
    out = softmax(np.array([[-np.inf, -np.inf], [0.0, 0.0]]))
    assert np.all(out[0] == 0.0)
    assert np.allclose(out[1], [0.5, 0.5], atol=1e-12)


# -- axes ------------------------------------------------------------------


def test_softmax_along_the_first_axis():
    x = np.array([[0.0, 1.0], [math.log(3.0), 1.0]])
    out = softmax(x, axis=0)
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-12)
    assert np.allclose(out[:, 0], [0.25, 0.75], atol=1e-12)


def test_softmax_on_three_dimensions_with_a_middle_axis():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 5, 3)) * 10.0
    out = softmax(x, axis=1)
    assert out.shape == x.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_negative_and_positive_axis_agree():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 3, 4))
    assert np.allclose(softmax(x, axis=2), softmax(x, axis=-1), atol=1e-12)


# -- inputs ----------------------------------------------------------------


def test_float32_input_is_accepted_and_the_output_is_float64():
    out = softmax(np.array([1.0, 2.0], dtype=np.float32))
    assert out.dtype == np.float64
    assert abs(out.sum() - 1.0) < 1e-12


def test_a_nested_list_is_accepted():
    out = softmax([[0.0, math.log(3.0)]])
    assert np.allclose(out, [[0.25, 0.75]], atol=1e-12)


def test_the_input_array_is_left_alone():
    x = np.array([[1e4, 1.0], [-np.inf, 0.0]])
    before = x.copy()
    softmax(x)
    log_softmax(x)
    logsumexp(x)
    assert np.array_equal(before, x)


# -- log_softmax -----------------------------------------------------------


def test_log_softmax_agrees_with_the_logarithm_of_softmax():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 6)) * 3.0
    assert np.allclose(log_softmax(x), np.log(softmax(x)), atol=1e-12)


def test_log_softmax_exponentiates_back_to_softmax():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(3, 8)) * 8.0
    assert np.allclose(np.exp(log_softmax(x)), softmax(x), atol=1e-12)


def test_log_softmax_keeps_a_probability_that_underflowed():
    # exp(-800) is exactly 0.0 in float64, so log(softmax(x))[1] would be -inf
    # while the true log-probability is very close to -800.
    out = log_softmax(np.array([0.0, -800.0]))
    assert np.isfinite(out[1])
    assert abs(out[1] - (-800.0)) < 1e-9
    assert abs(out[0]) < 1e-9


def test_log_softmax_is_stable_at_large_magnitudes():
    out = log_softmax(np.array([1e4, 1e4 - 1.0, 1e4 - 2.0]))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, log_softmax(np.array([0.0, -1.0, -2.0])), atol=1e-9)


def test_log_softmax_sends_masked_entries_to_minus_infinity():
    out = log_softmax(np.array([1.0, -np.inf, 2.0]))
    assert out[1] == -np.inf
    assert np.allclose(out[[0, 2]], log_softmax(np.array([1.0, 2.0])), atol=1e-12)


def test_a_fully_masked_slice_log_softmaxes_to_minus_infinity():
    out = log_softmax(np.array([[-np.inf, -np.inf], [0.0, 0.0]]))
    assert np.all(out[0] == -np.inf)
    assert not np.any(np.isnan(out))


def test_log_softmax_along_the_first_axis():
    x = np.array([[0.0, 1.0], [math.log(3.0), 1.0]])
    assert np.allclose(log_softmax(x, axis=0), np.log(softmax(x, axis=0)), atol=1e-12)


# -- logsumexp -------------------------------------------------------------


def test_logsumexp_matches_a_hand_computed_case():
    # exp(0) + exp(log 3) = 4
    assert abs(logsumexp(np.array([0.0, math.log(3.0)])) - math.log(4.0)) < 1e-12


def test_logsumexp_survives_large_values():
    out = logsumexp(np.array([1e4, 1e4]))
    assert np.isfinite(out)
    assert abs(out - (1e4 + math.log(2.0))) < 1e-9


def test_logsumexp_drops_the_axis_by_default():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 4, 5))
    assert logsumexp(x, axis=1).shape == (3, 5)


def test_logsumexp_keepdims_holds_the_axis_open():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(3, 4, 5))
    assert logsumexp(x, axis=1, keepdims=True).shape == (3, 1, 5)


def test_logsumexp_along_the_first_axis():
    x = np.array([[0.0, 1.0], [math.log(3.0), 1.0]])
    out = logsumexp(x, axis=0)
    assert abs(out[0] - math.log(4.0)) < 1e-12
    assert abs(out[1] - (1.0 + math.log(2.0))) < 1e-12


def test_logsumexp_ignores_masked_entries():
    with_mask = logsumexp(np.array([1.0, -np.inf, 2.0]))
    without = logsumexp(np.array([1.0, 2.0]))
    assert abs(with_mask - without) < 1e-12


def test_logsumexp_of_a_fully_masked_slice_is_minus_infinity():
    out = logsumexp(np.array([[-np.inf, -np.inf], [0.0, 0.0]]))
    assert out[0] == -np.inf
    assert abs(out[1] - math.log(2.0)) < 1e-12
    assert not np.any(np.isnan(out))
