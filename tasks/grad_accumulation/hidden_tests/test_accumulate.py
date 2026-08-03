"""Hidden tests — grad_accumulation.

Every assertion is licensed by a sentence in prompt.md.

Everything runs in float64 so that "identical to the full-batch step" can be
asserted at 1e-12. The two paths sum the same terms in a different order, which
in float64 costs a few ULPs on values of order one; a wrong weighting or a
misplaced step is off by percent, not by 1e-12.
"""

import copy

import pytest
import torch
from torch import nn

from accumulate import accumulated_step


def build_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 3)).double()


def build_data(n_samples: int, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 6, generator=generator, dtype=torch.float64)
    y = torch.randn(n_samples, 3, generator=generator, dtype=torch.float64)
    return x, y


def full_batch_step(model, optimizer, loss_fn, x, y) -> float:
    """The single large step that accumulation has to reproduce."""
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def assert_same_parameters(left: nn.Module, right: nn.Module, atol: float = 1e-12) -> None:
    for a, b in zip(left.parameters(), right.parameters()):
        assert torch.allclose(a, b, atol=atol, rtol=0.0)


def compare(n_samples: int, micro_batch_size: int, make_optimizer, seed: int = 0):
    """Run both paths from the same start and hand back both models."""
    loss_fn = nn.MSELoss()
    x, y = build_data(n_samples)

    accumulated_model = build_model(seed)
    reference_model = copy.deepcopy(accumulated_model)

    accumulated_loss = accumulated_step(
        accumulated_model,
        make_optimizer(accumulated_model),
        loss_fn,
        x,
        y,
        micro_batch_size,
    )
    reference_loss = full_batch_step(
        reference_model, make_optimizer(reference_model), loss_fn, x, y
    )
    return accumulated_model, reference_model, accumulated_loss, reference_loss


def sgd(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


# -- the equivalence -------------------------------------------------------


def test_four_micro_batches_match_one_large_step():
    accumulated, reference, _, _ = compare(32, 8, sgd)
    assert_same_parameters(accumulated, reference)


@pytest.mark.parametrize("micro_batch_size", [1, 2, 3, 5, 8, 16, 32])
def test_every_micro_batch_size_gives_the_same_update(micro_batch_size):
    accumulated, reference, _, _ = compare(32, micro_batch_size, sgd)
    assert_same_parameters(accumulated, reference)


@pytest.mark.parametrize("n_samples,micro_batch_size", [(30, 8), (10, 4), (7, 3), (5, 2)])
def test_a_short_last_micro_batch_is_weighted_by_its_size(n_samples, micro_batch_size):
    # 30 samples in chunks of 8 is 8, 8, 8, 6 — and the 6 must not count as
    # much as the 8s do.
    accumulated, reference, _, _ = compare(n_samples, micro_batch_size, sgd)
    assert_same_parameters(accumulated, reference)


def test_a_micro_batch_larger_than_the_batch_is_a_single_pass():
    accumulated, reference, _, _ = compare(12, 64, sgd)
    assert_same_parameters(accumulated, reference)


def test_one_sample_at_a_time():
    accumulated, reference, _, _ = compare(9, 1, sgd)
    assert_same_parameters(accumulated, reference)


# -- optimisers with state -------------------------------------------------


def test_momentum_survives():
    # Stepping once per micro-batch cannot be rescaled into this answer.
    accumulated, reference, _, _ = compare(
        24, 6, lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    )
    assert_same_parameters(accumulated, reference)


def test_adam_survives():
    accumulated, reference, _, _ = compare(
        24, 6, lambda m: torch.optim.Adam(m.parameters(), lr=0.01)
    )
    assert_same_parameters(accumulated, reference, atol=1e-10)


def test_weight_decay_survives():
    accumulated, reference, _, _ = compare(
        20, 7, lambda m: torch.optim.SGD(m.parameters(), lr=0.1, weight_decay=0.01)
    )
    assert_same_parameters(accumulated, reference)


def test_the_optimiser_steps_exactly_once():
    class CountingSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0

        def step(self, *args, **kwargs):
            self.step_count += 1
            return super().step(*args, **kwargs)

    model = build_model()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    x, y = build_data(20)

    accumulated_step(model, optimizer, nn.MSELoss(), x, y, 6)
    assert optimizer.step_count == 1


# -- the gradients themselves ---------------------------------------------


def test_the_gradients_left_behind_are_the_full_batch_gradients():
    loss_fn = nn.MSELoss()
    x, y = build_data(24)

    accumulated_model = build_model()
    reference_model = copy.deepcopy(accumulated_model)

    accumulated_step(
        accumulated_model, torch.optim.SGD(accumulated_model.parameters(), lr=0.1), loss_fn, x, y, 5
    )
    loss_fn(reference_model(x), y).backward()

    for accumulated, expected in zip(accumulated_model.parameters(), reference_model.parameters()):
        assert accumulated.grad is not None
        assert torch.allclose(accumulated.grad, expected.grad, atol=1e-12, rtol=0.0)


def test_gradients_left_over_from_an_earlier_step_are_discarded():
    loss_fn = nn.MSELoss()
    x, y = build_data(16)

    polluted_model = build_model()
    clean_model = copy.deepcopy(polluted_model)
    for parameter in polluted_model.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)

    accumulated_step(
        polluted_model, torch.optim.SGD(polluted_model.parameters(), lr=0.1), loss_fn, x, y, 4
    )
    accumulated_step(
        clean_model, torch.optim.SGD(clean_model.parameters(), lr=0.1), loss_fn, x, y, 4
    )

    assert_same_parameters(polluted_model, clean_model)


# -- the returned loss -----------------------------------------------------


def test_the_returned_loss_is_the_full_batch_mean():
    _, _, accumulated_loss, reference_loss = compare(32, 7, sgd)
    assert isinstance(accumulated_loss, float)
    assert abs(accumulated_loss - reference_loss) < 1e-12


def test_the_returned_loss_is_measured_before_the_step():
    # The loss reported is the one the gradients came from, so it matches a
    # forward pass on the parameters as they were on the way in.
    loss_fn = nn.MSELoss()
    x, y = build_data(18)

    model = build_model()
    before = copy.deepcopy(model)
    with torch.no_grad():
        expected = loss_fn(before(x), y).item()

    reported = accumulated_step(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn, x, y, 5)
    assert abs(reported - expected) < 1e-12


def test_the_loss_is_a_plain_float_not_a_tensor():
    model = build_model()
    x, y = build_data(8)
    reported = accumulated_step(
        model, torch.optim.SGD(model.parameters(), lr=0.1), nn.MSELoss(), x, y, 3
    )
    assert not isinstance(reported, torch.Tensor)
    assert isinstance(reported, float)


# -- hygiene ---------------------------------------------------------------


def test_the_model_is_left_in_training_mode():
    model = build_model()
    model.train()
    x, y = build_data(8)
    accumulated_step(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.MSELoss(), x, y, 3)
    assert model.training


def test_the_data_is_not_modified():
    model = build_model()
    x, y = build_data(12)
    x_before, y_before = x.clone(), y.clone()
    accumulated_step(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.MSELoss(), x, y, 5)
    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)


def test_two_identical_runs_agree():
    first, _, first_loss, _ = compare(20, 6, sgd)
    second, _, second_loss, _ = compare(20, 6, sgd)
    assert_same_parameters(first, second, atol=0.0)
    assert first_loss == second_loss


def test_a_different_loss_function_works_too():
    loss_fn = nn.L1Loss()
    x, y = build_data(21)

    accumulated_model = build_model()
    reference_model = copy.deepcopy(accumulated_model)

    accumulated_step(
        accumulated_model, torch.optim.SGD(accumulated_model.parameters(), lr=0.1), loss_fn, x, y, 8
    )
    full_batch_step(
        reference_model, torch.optim.SGD(reference_model.parameters(), lr=0.1), loss_fn, x, y
    )
    assert_same_parameters(accumulated_model, reference_model)
