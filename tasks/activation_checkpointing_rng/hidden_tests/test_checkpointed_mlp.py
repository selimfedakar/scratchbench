"""Hidden tests — activation_checkpointing_rng.

Every assertion is licensed by a sentence in prompt.md, and every promise in
prompt.md has a test here that enforces it.

Two independent answers, and neither of them is the reference solution.

The first is a plain stack written here in the obvious way, keeping every
activation, differentiated by `torch.autograd`. That is what checkpointing is
supposed to be indistinguishable from, so it is what the outputs and the
gradients are compared against.

The second is the generator itself. `torch.Generator` state is a tensor of
bytes, so "the generator ended where it would have ended" is `torch.equal` on
two byte tensors — an exact check with no tolerance in it, and the one that
separates an implementation that restores the state before recomputing from one
that also puts it back afterwards. Both produce perfect gradients this step.

Tolerances. Everything is float64 and every quantity is of order one to a few
tens. The checkpointed path and the plain path do the same multiplications in
the same order, so they agree to a few units in the last place; `atol=1e-10,
rtol=0` is far above that and far below any mistake worth catching. The mask and
the generator-state comparisons are exact, because there is nothing approximate
about them.
"""

import pytest
import torch

from checkpointed_mlp import dropout_mask, forward_and_backward


# -- the plain stack, kept whole and differentiated by autograd ------------


def make_blocks(count=3, width=6, hidden=8, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def sample(rows, columns):
        return torch.randn(rows, columns, dtype=torch.float64, generator=generator)

    return [(sample(width, hidden), sample(hidden, width)) for _ in range(count)]


def make_input(rows=5, width=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(rows, width, dtype=torch.float64, generator=generator),
        torch.randn(rows, width, dtype=torch.float64, generator=generator),
    )


def plain_masks(x, blocks, p, generator):
    """The masks a forward over this stack draws, in order."""
    masks = []
    current = x
    for w1, w2 in blocks:
        activation = torch.relu(current @ w1)
        keep = torch.rand(activation.shape, generator=generator, dtype=torch.float64) >= p
        mask = keep.to(torch.float64) / (1.0 - p)
        masks.append(mask)
        current = (activation * mask) @ w2 + current
    return masks


def plain_reference(x, blocks, p, generator, dy):
    """`(y, dx, grads)` from a stack that keeps everything, via autograd."""
    masks = plain_masks(x, blocks, p, generator)

    leaf_x = x.detach().clone().requires_grad_(True)
    leaves = [
        (w1.detach().clone().requires_grad_(True), w2.detach().clone().requires_grad_(True))
        for w1, w2 in blocks
    ]

    current = leaf_x
    for (w1, w2), mask in zip(leaves, masks):
        current = (torch.relu(current @ w1) * mask) @ w2 + current

    output = current
    (output * dy).sum().backward()
    return (
        output.detach(),
        leaf_x.grad,
        [(w1.grad, w2.grad) for w1, w2 in leaves],
    )


# -- the mask ---------------------------------------------------------------


def test_the_mask_keeps_and_rescales():
    mask = dropout_mask((200, 200), 0.25, torch.Generator().manual_seed(2))
    kept = mask != 0.0
    # Inverted dropout: what survives is divided by the keep probability, so the
    # expectation of the mask is one.
    assert torch.allclose(mask[kept], torch.full_like(mask[kept], 1 / 0.75), atol=1e-12)
    assert torch.all(mask[~kept] == 0.0)
    # 40000 draws at p = 0.25 has a standard error of 0.002, so six of them is
    # 0.013 and this cannot flake on a correct implementation.
    assert abs(kept.to(torch.float64).mean().item() - 0.75) < 0.013


def test_a_dropout_probability_of_zero_keeps_everything_and_still_draws():
    generator = torch.Generator().manual_seed(3)
    mask = dropout_mask((4, 7), 0.0, generator)
    assert torch.all(mask == 1.0)
    # The draw happened: the generator moved. How much randomness a forward
    # consumes cannot depend on the value of p.
    assert not torch.equal(generator.get_state(), torch.Generator().manual_seed(3).get_state())


def test_the_mask_has_the_shape_and_dtype_it_was_asked_for():
    mask = dropout_mask((3, 5), 0.5, torch.Generator().manual_seed(4))
    assert mask.shape == (3, 5)
    assert mask.dtype == torch.float64


def test_the_mask_comes_out_of_the_generator_it_was_handed():
    one = dropout_mask((6, 6), 0.3, torch.Generator().manual_seed(5))
    other = dropout_mask((6, 6), 0.3, torch.Generator().manual_seed(5))
    assert torch.equal(one, other)
    different = dropout_mask((6, 6), 0.3, torch.Generator().manual_seed(6))
    assert not torch.equal(one, different)


# -- the outputs and the gradients ----------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.8])
def test_the_output_matches_a_stack_that_kept_everything(p):
    x, dy = make_input()
    blocks = make_blocks()

    y, _, _ = forward_and_backward(x, blocks, p, torch.Generator().manual_seed(7), dy)
    expected, _, _ = plain_reference(x, blocks, p, torch.Generator().manual_seed(7), dy)
    assert torch.allclose(y, expected, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.8])
def test_the_gradients_match_a_stack_that_kept_everything(p):
    # The whole claim of checkpointing: the answer is the one you would have got
    # without it. If the recomputed dropout mask is not the one the forward
    # drew, this is where it shows.
    x, dy = make_input()
    blocks = make_blocks()

    _, dx, grads = forward_and_backward(x, blocks, p, torch.Generator().manual_seed(8), dy)
    _, expected_dx, expected_grads = plain_reference(
        x, blocks, p, torch.Generator().manual_seed(8), dy
    )

    assert torch.allclose(dx, expected_dx, atol=1e-10, rtol=0.0)
    assert len(grads) == len(expected_grads)
    for (d_w1, d_w2), (expected_w1, expected_w2) in zip(grads, expected_grads):
        assert torch.allclose(d_w1, expected_w1, atol=1e-10, rtol=0.0)
        assert torch.allclose(d_w2, expected_w2, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_any_number_of_blocks_chains_correctly(count):
    x, dy = make_input()
    blocks = make_blocks(count=count)

    y, dx, grads = forward_and_backward(x, blocks, 0.4, torch.Generator().manual_seed(9), dy)
    expected_y, expected_dx, expected_grads = plain_reference(
        x, blocks, 0.4, torch.Generator().manual_seed(9), dy
    )

    assert len(grads) == count
    assert torch.allclose(y, expected_y, atol=1e-10, rtol=0.0)
    assert torch.allclose(dx, expected_dx, atol=1e-10, rtol=0.0)
    for (d_w1, d_w2), (expected_w1, expected_w2) in zip(grads, expected_grads):
        assert torch.allclose(d_w1, expected_w1, atol=1e-10, rtol=0.0)
        assert torch.allclose(d_w2, expected_w2, atol=1e-10, rtol=0.0)


def test_the_gradients_come_back_in_block_order():
    # Only the first block's weights can change the first block's output, so
    # zeroing the last block's incoming gradient must leave the first block's
    # gradient alone and not the other way round.
    x, dy = make_input()
    blocks = make_blocks(count=3)

    _, _, grads = forward_and_backward(x, blocks, 0.3, torch.Generator().manual_seed(10), dy)
    for (d_w1, d_w2), (w1, w2) in zip(grads, blocks):
        assert d_w1.shape == w1.shape
        assert d_w2.shape == w2.shape


def test_a_block_whose_output_is_ignored_still_passes_the_residual_through():
    # With w2 at zero the block is the identity, so the input gradient is the
    # output gradient exactly and the first weight gets nothing.
    x, dy = make_input()
    w1 = torch.randn(6, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    blocks = [(w1, torch.zeros(8, 6, dtype=torch.float64))]

    y, dx, grads = forward_and_backward(x, blocks, 0.3, torch.Generator().manual_seed(12), dy)
    assert torch.allclose(y, x, atol=1e-12, rtol=0.0)
    assert torch.allclose(dx, dy, atol=1e-12, rtol=0.0)
    assert torch.allclose(grads[0][0], torch.zeros_like(w1), atol=1e-12, rtol=0.0)


# -- the generator, which is the mechanism --------------------------------


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("count", [1, 3, 5])
def test_the_generator_ends_where_the_forward_pass_left_it(p, count):
    # The half that produces perfect gradients and a training run that will not
    # reproduce. After the recomputations the generator is sitting several
    # blocks back, and every draw after this step is randomness that has already
    # been used.
    x, dy = make_input()
    blocks = make_blocks(count=count)

    plain = torch.Generator().manual_seed(13)
    plain_masks(x, blocks, p, plain)
    expected_state = plain.get_state()

    generator = torch.Generator().manual_seed(13)
    forward_and_backward(x, blocks, p, generator, dy)
    assert torch.equal(generator.get_state(), expected_state)


def test_the_next_draw_after_a_step_is_the_one_the_plain_stack_would_have_made():
    # The same claim from the outside: whatever asks the generator for
    # randomness next gets what it would have got without checkpointing.
    x, dy = make_input()
    blocks = make_blocks(count=4)

    plain = torch.Generator().manual_seed(14)
    plain_masks(x, blocks, 0.35, plain)
    expected_next = torch.rand(3, 3, generator=plain, dtype=torch.float64)

    generator = torch.Generator().manual_seed(14)
    forward_and_backward(x, blocks, 0.35, generator, dy)
    assert torch.equal(torch.rand(3, 3, generator=generator, dtype=torch.float64), expected_next)


def test_two_steps_in_a_row_are_two_different_steps():
    # If the generator is left rewound, the second step draws the first step's
    # masks and the two come out identical.
    x, dy = make_input()
    blocks = make_blocks(count=3)
    generator = torch.Generator().manual_seed(15)

    first = forward_and_backward(x, blocks, 0.5, generator, dy)
    second = forward_and_backward(x, blocks, 0.5, generator, dy)
    assert not torch.allclose(first[0], second[0], atol=1e-6, rtol=0.0)


def test_a_second_step_matches_the_plain_stacks_second_step():
    # Not merely different from the first step: the *right* second step. An
    # implementation that leaves the generator anywhere other than where the
    # forward left it fails this even if it fails nothing else.
    x, dy = make_input()
    blocks = make_blocks(count=3)

    plain = torch.Generator().manual_seed(16)
    plain_masks(x, blocks, 0.5, plain)
    expected_second, expected_dx, _ = plain_reference(x, blocks, 0.5, plain, dy)

    generator = torch.Generator().manual_seed(16)
    forward_and_backward(x, blocks, 0.5, generator, dy)
    second_y, second_dx, _ = forward_and_backward(x, blocks, 0.5, generator, dy)

    assert torch.allclose(second_y, expected_second, atol=1e-10, rtol=0.0)
    assert torch.allclose(second_dx, expected_dx, atol=1e-10, rtol=0.0)


def test_every_block_gets_its_own_draw():
    # Three blocks consume three masks, so a stack of three advances the
    # generator exactly as far as three separate mask draws do.
    x, dy = make_input()
    blocks = make_blocks(count=3)

    counted = torch.Generator().manual_seed(17)
    for _ in range(3):
        dropout_mask((x.shape[0], blocks[0][0].shape[1]), 0.5, counted)
    expected_state = counted.get_state()

    generator = torch.Generator().manual_seed(17)
    forward_and_backward(x, blocks, 0.5, generator, dy)
    assert torch.equal(generator.get_state(), expected_state)


# -- interface -------------------------------------------------------------


def test_the_shapes_and_dtypes_are_what_they_differentiate():
    x, dy = make_input(rows=7, width=6)
    blocks = make_blocks(count=2, width=6, hidden=9)

    y, dx, grads = forward_and_backward(x, blocks, 0.3, torch.Generator().manual_seed(18), dy)
    assert y.shape == x.shape and y.dtype == torch.float64
    assert dx.shape == x.shape and dx.dtype == torch.float64
    for (d_w1, d_w2), (w1, w2) in zip(grads, blocks):
        assert d_w1.shape == w1.shape and d_w1.dtype == torch.float64
        assert d_w2.shape == w2.shape and d_w2.dtype == torch.float64


def test_nothing_returned_carries_a_gradient_history():
    x, dy = make_input()
    blocks = make_blocks()
    y, dx, grads = forward_and_backward(x, blocks, 0.3, torch.Generator().manual_seed(19), dy)
    for tensor in [y, dx, *[part for pair in grads for part in pair]]:
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_the_inputs_are_left_alone():
    x, dy = make_input()
    blocks = make_blocks()
    originals = [x.clone(), dy.clone()] + [w.clone() for pair in blocks for w in pair]

    forward_and_backward(x, blocks, 0.4, torch.Generator().manual_seed(20), dy)

    current = [x, dy] + [w for pair in blocks for w in pair]
    for original, now in zip(originals, current):
        assert torch.equal(original, now)


def test_a_single_row_of_input():
    x, dy = make_input(rows=1)
    blocks = make_blocks(count=2)
    y, dx, grads = forward_and_backward(x, blocks, 0.5, torch.Generator().manual_seed(21), dy)
    expected_y, expected_dx, expected_grads = plain_reference(
        x, blocks, 0.5, torch.Generator().manual_seed(21), dy
    )
    assert torch.allclose(y, expected_y, atol=1e-10, rtol=0.0)
    assert torch.allclose(dx, expected_dx, atol=1e-10, rtol=0.0)
    for (d_w1, _), (expected_w1, _) in zip(grads, expected_grads):
        assert torch.allclose(d_w1, expected_w1, atol=1e-10, rtol=0.0)
