"""Hidden tests for `custom_autograd_double_backward`.

Every tolerance here is float64 unless a test says otherwise, and every
comparison is against the same algebra evaluated by autograd on the plain
expression. The two differ only in the order the rounding happens, so the
tolerances sit a few orders of magnitude above float64 eps (2.2e-16) rather
than at it.
"""

from __future__ import annotations

import pytest
import torch

from scaled_swish import ScaledSwish, scaled_swish

# Doubles everywhere: the point of the comparison is the algebra, and float32
# noise would hide a wrong term behind a loose tolerance.
DTYPE = torch.float64
ATOL = 1e-10
RTOL = 1e-9

# (x shape, w shape). The last four broadcast, in both directions, because a
# gradient that is not summed back down to its input's shape is the mistake
# that survives every non-broadcasting test.
SHAPES = [
    ((4,), (4,)),
    ((3, 5), (3, 5)),
    ((3, 5), (5,)),
    ((3, 5), (1, 5)),
    ((3, 5), ()),
    ((1, 5), (3, 5)),
    ((2, 3, 4), (3, 1)),
]

ALPHAS = [1.0, 0.0, -2.5, 3.0]


def plain(x, w, alpha):
    """The expression the Function stands for, left to autograd."""
    return w * x * torch.sigmoid(alpha * x)


def make(x_shape, w_shape, seed=0, dtype=DTYPE, requires_grad=True):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(x_shape, generator=generator, dtype=dtype)
    w = torch.randn(w_shape, generator=generator, dtype=dtype)
    x.requires_grad_(requires_grad)
    w.requires_grad_(requires_grad)
    return x, w


def like(x, w):
    """Independent leaves holding the same numbers, for the autograd side."""
    return x.detach().clone().requires_grad_(x.requires_grad), w.detach().clone().requires_grad_(
        w.requires_grad
    )


# -- it is a Function at all -----------------------------------------------


def test_the_helper_goes_through_the_function():
    # The structural half of the task, in the one place it can also fail: a
    # solution that quietly replaces the class with a plain expression and lets
    # the framework differentiate it returns the right numbers everywhere else.
    assert issubclass(ScaledSwish, torch.autograd.Function)
    x, w = make((4,), (4,))
    out = scaled_swish(x, w, 1.0)
    assert out.grad_fn is not None
    assert "ScaledSwish" in type(out.grad_fn).__name__


# -- forward ----------------------------------------------------------------


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_forward_matches_the_expression(x_shape, w_shape, alpha):
    x, w = make(x_shape, w_shape)
    got = scaled_swish(x, w, alpha)
    expected = plain(x, w, alpha)
    assert torch.allclose(got, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
def test_forward_shape_is_the_broadcast_shape(x_shape, w_shape):
    x, w = make(x_shape, w_shape)
    expected = torch.broadcast_shapes(x_shape, w_shape)
    assert scaled_swish(x, w, 1.0).shape == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_keeps_the_dtype(dtype):
    x, w = make((3, 5), (5,), dtype=dtype)
    out = scaled_swish(x, w, 1.0)
    assert out.dtype == dtype
    # float32 carries about seven decimal digits, so the check is loosened to
    # match the type rather than the algebra.
    tol = 1e-6 if dtype is torch.float32 else ATOL
    assert torch.allclose(out, plain(x, w, 1.0), atol=tol, rtol=1e-5)


def test_alpha_zero_is_half():
    x, w = make((3, 5), (5,))
    # sigmoid(0) is exactly a half, so this one is an identity rather than an
    # approximation, and it catches an implementation that drops alpha.
    assert torch.allclose(scaled_swish(x, w, 0.0), 0.5 * w * x, atol=ATOL, rtol=RTOL)


def test_a_large_alpha_saturates_without_going_non_finite():
    x = torch.tensor([-40.0, -8.0, 0.0, 8.0, 40.0], dtype=DTYPE, requires_grad=True)
    w = torch.ones(5, dtype=DTYPE, requires_grad=True)
    out = scaled_swish(x, w, 25.0)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_an_integer_alpha_is_accepted():
    x, w = make((4,), (4,))
    assert torch.allclose(scaled_swish(x, w, 2), plain(x, w, 2.0), atol=ATOL, rtol=RTOL)


# -- first order ------------------------------------------------------------


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_first_order_gradients_match_autograd(x_shape, w_shape, alpha):
    x, w = make(x_shape, w_shape)
    ax, aw = like(x, w)

    scaled_swish(x, w, alpha).sum().backward()
    plain(ax, aw, alpha).sum().backward()

    assert torch.allclose(x.grad, ax.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(w.grad, aw.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
def test_each_gradient_has_the_shape_of_its_input(x_shape, w_shape):
    x, w = make(x_shape, w_shape)
    scaled_swish(x, w, 1.5).sum().backward()
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape


def test_a_weighted_output_gradient_is_carried_through():
    x, w = make((3, 5), (5,))
    ax, aw = like(x, w)
    generator = torch.Generator().manual_seed(7)
    upstream = torch.randn((3, 5), generator=generator, dtype=DTYPE)

    scaled_swish(x, w, 1.5).backward(upstream)
    plain(ax, aw, 1.5).backward(upstream)

    assert torch.allclose(x.grad, ax.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(w.grad, aw.grad, atol=ATOL, rtol=RTOL)


def test_a_non_contiguous_output_gradient_is_handled():
    x, w = make((4, 6), (6,))
    ax, aw = like(x, w)
    generator = torch.Generator().manual_seed(11)
    upstream = torch.randn((6, 4), generator=generator, dtype=DTYPE).t()
    assert not upstream.is_contiguous()

    scaled_swish(x, w, 1.5).backward(upstream)
    plain(ax, aw, 1.5).backward(upstream)

    assert torch.allclose(x.grad, ax.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(w.grad, aw.grad, atol=ATOL, rtol=RTOL)


def test_it_composes_inside_a_larger_graph():
    generator = torch.Generator().manual_seed(3)
    base = torch.randn((3, 5), generator=generator, dtype=DTYPE, requires_grad=True)
    w = torch.randn((5,), generator=generator, dtype=DTYPE, requires_grad=True)
    abase, aw = like(base, w)

    (scaled_swish(base * 2.0, w, 1.5) ** 2).sum().backward()
    (plain(abase * 2.0, aw, 1.5) ** 2).sum().backward()

    assert torch.allclose(base.grad, abase.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(w.grad, aw.grad, atol=ATOL, rtol=RTOL)


# -- which inputs asked for a gradient --------------------------------------


def test_only_x_asks_for_a_gradient():
    x, w = make((3, 5), (5,))
    w.requires_grad_(False)
    ax, aw = like(x, w)

    scaled_swish(x, w, 1.5).sum().backward()
    plain(ax, aw, 1.5).sum().backward()

    assert torch.allclose(x.grad, ax.grad, atol=ATOL, rtol=RTOL)
    assert w.grad is None


def test_only_w_asks_for_a_gradient():
    x, w = make((3, 5), (5,))
    x.requires_grad_(False)
    ax, aw = like(x, w)

    scaled_swish(x, w, 1.5).sum().backward()
    plain(ax, aw, 1.5).sum().backward()

    assert torch.allclose(w.grad, aw.grad, atol=ATOL, rtol=RTOL)
    assert x.grad is None


def test_neither_input_asks_for_a_gradient():
    x, w = make((3, 5), (5,), requires_grad=False)
    out = scaled_swish(x, w, 1.5)
    assert not out.requires_grad
    assert out.grad_fn is None
    assert torch.allclose(out, plain(x, w, 1.5), atol=ATOL, rtol=RTOL)


def test_alpha_never_collects_a_gradient():
    # alpha arrives as a number rather than a tensor, so the slot it occupies
    # has no gradient to return. An implementation that returns something there
    # fails here rather than somewhere unrelated.
    x, w = make((4,), (4,))
    out = scaled_swish(x, w, 2.0)
    out.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


# -- the tensors the backward reads -----------------------------------------


def test_an_input_modified_after_the_forward_is_detected():
    # The framework tracks a version on every tensor handed to it for the
    # backward, and refuses to compute a gradient from one that changed
    # underneath it. An implementation that keeps its own reference instead
    # gets no such check and quietly returns a gradient of the wrong numbers.
    generator = torch.Generator().manual_seed(5)
    base = torch.randn((4,), generator=generator, dtype=DTYPE, requires_grad=True)
    w = torch.randn((4,), generator=generator, dtype=DTYPE, requires_grad=True)

    x = base * 1.0
    out = scaled_swish(x, w, 1.5)
    x.mul_(2.0)

    with pytest.raises(RuntimeError):
        out.sum().backward()


def test_the_inputs_are_not_modified():
    x, w = make((3, 5), (5,))
    before_x = x.detach().clone()
    before_w = w.detach().clone()
    scaled_swish(x, w, 1.5).sum().backward()
    assert torch.equal(x.detach(), before_x)
    assert torch.equal(w.detach(), before_w)


# -- second order -----------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_gradcheck(alpha):
    x, w = make((3, 4), (4,), seed=13)
    assert torch.autograd.gradcheck(
        lambda a, b: scaled_swish(a, b, alpha), (x, w), eps=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("alpha", ALPHAS)
def test_gradgradcheck(alpha):
    # The centrepiece. A backward written out of operations that do not carry a
    # graph passes every test above and fails this one.
    x, w = make((3, 4), (4,), seed=17)
    assert torch.autograd.gradgradcheck(
        lambda a, b: scaled_swish(a, b, alpha), (x, w), eps=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
@pytest.mark.parametrize("alpha", [1.0, -2.5])
def test_second_derivative_in_x_matches_autograd(x_shape, w_shape, alpha):
    x, w = make(x_shape, w_shape, seed=19)
    ax, aw = like(x, w)

    (gx,) = torch.autograd.grad(scaled_swish(x, w, alpha).sum(), x, create_graph=True)
    (agx,) = torch.autograd.grad(plain(ax, aw, alpha).sum(), ax, create_graph=True)
    assert torch.allclose(gx, agx, atol=ATOL, rtol=RTOL)

    (gxx,) = torch.autograd.grad((gx**2).sum(), x)
    (agxx,) = torch.autograd.grad((agx**2).sum(), ax)
    assert torch.allclose(gxx, agxx, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
def test_the_mixed_second_derivative_matches_autograd(x_shape, w_shape):
    x, w = make(x_shape, w_shape, seed=23)
    ax, aw = like(x, w)

    (gx,) = torch.autograd.grad(scaled_swish(x, w, 1.5).sum(), x, create_graph=True)
    (agx,) = torch.autograd.grad(plain(ax, aw, 1.5).sum(), ax, create_graph=True)

    (gxw,) = torch.autograd.grad(gx.sum(), w)
    (agxw,) = torch.autograd.grad(agx.sum(), aw)
    assert torch.allclose(gxw, agxw, atol=ATOL, rtol=RTOL)
    assert gxw.shape == w.shape


def test_the_second_derivative_of_w_is_zero_and_present():
    # The forward is linear in w, so the second derivative in w is zero. It has
    # to be a tensor of zeros rather than a missing gradient: a backward that
    # cuts the graph produces None here and takes the whole chain with it.
    x, w = make((3, 5), (5,), seed=29)
    (gw,) = torch.autograd.grad(scaled_swish(x, w, 1.5).sum(), w, create_graph=True)
    (gww,) = torch.autograd.grad(gw.sum(), w, allow_unused=True, materialize_grads=True)
    assert gww is not None
    assert torch.allclose(gww, torch.zeros_like(w), atol=ATOL, rtol=RTOL)


def test_an_output_gradient_that_itself_carries_a_graph():
    # This is what create_graph does under the hood, written out: the tensor
    # handed to the backward requires grad, and the result has to stay
    # connected to it.
    x, w = make((3, 5), (5,), seed=31)
    ax, aw = like(x, w)
    generator = torch.Generator().manual_seed(37)
    upstream = torch.randn((3, 5), generator=generator, dtype=DTYPE, requires_grad=True)

    (gx,) = torch.autograd.grad(scaled_swish(x, w, 1.5), x, upstream, create_graph=True)
    (agx,) = torch.autograd.grad(plain(ax, aw, 1.5), ax, upstream, create_graph=True)
    assert gx.requires_grad

    (gu,) = torch.autograd.grad(gx.sum(), upstream)
    (agu,) = torch.autograd.grad(agx.sum(), upstream)
    assert torch.allclose(gu, agu, atol=ATOL, rtol=RTOL)
