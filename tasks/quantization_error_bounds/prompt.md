# Quantization, and the bound it is supposed to guarantee

Fill in `quantize.py`. numpy only:

```python
def quantize_per_tensor(x, num_bits: int = 8, symmetric: bool = True)
def dequantize(q, scale, zero_point) -> np.ndarray
def quantize_per_channel(x, axis: int, num_bits: int = 8, symmetric: bool = True)
```

Map a float tensor onto a small integer grid, and back. `quantize_per_tensor`
returns `(q, scale, zero_point)` with one scale for the whole tensor;
`quantize_per_channel` returns the same triple with one scale per slice along
`axis`. `dequantize` undoes either.

This is the compression step in a model that is *allowed* to lose information,
which is what makes it easy to get wrong quietly. A bad scale does not raise and
does not produce NaN — it costs a point of accuracy, and a point of accuracy is
indistinguishable from the honest price of dropping to eight bits. So the
specification below is written as arithmetic rather than as intent, and the thing
that holds it together is an inequality you can check.

## The two mappings

Both directions are affine. Forward: divide by the scale, round to the nearest
integer, add the zero point, and clamp into the integer range. Backward:

```
dequantize(q, scale, zero_point) = scale * (q - zero_point)
```

Rounding is to the nearest integer with ties going to the even neighbour, which
is what numpy does by default: −0.5 becomes 0, 62.5 becomes 62, 63.5 becomes 64.

**Symmetric** quantization uses the signed integers. For `num_bits = b` the
range is `-2**(b-1)` through `2**(b-1) - 1`, the zero point is fixed at zero,
and the scale is the largest magnitude in the data divided by the largest
positive code word — `max(abs(x)) / (2**(b-1) - 1)`.

**Asymmetric** quantization uses the unsigned integers, `0` through
`2**b - 1`, and earns its extra freedom by moving the zero point. The interval
it covers is the data's range **widened to include zero**: the low end is the
smaller of the tensor's minimum and zero, the high end the larger of its maximum
and zero. So an all-positive tensor is quantized over `[0, max]` rather than
`[min, max]`, which costs a little resolution and buys the property in the next
section. The scale is that interval divided by the number of steps in the
integer range, and the zero point is the integer that lands on zero — round
`qmin - low / scale` and clamp it into the integer range.

Both scales shrink as the bit width grows, and the bound below shrinks with
them.

## Zero has to survive

An element that is exactly zero must dequantize to exactly `0.0`. Not nearly.

Padding is zero, attention masks are zero, ReLU outputs are mostly zero, pruned
weights are zero, and in a real tensor there are a great many of them. If zero
comes back as a small non-zero constant, every one of those elements is off in
the same direction by the same amount, which is a bias added to the whole tensor
rather than noise spread across it. Noise averages out across a layer; bias
accumulates through it. That is the reason the asymmetric interval is stretched
to include zero even when the data never goes there.

A tensor that is entirely zero has no range to work with. In that case the scale
is `1.0` and the zero point is `0`, so every code word is zero and the round trip
is exact.

## The bound

Rounding to the nearest multiple of the scale moves a value by at most half a
scale. So for every element of the tensor the parameters were computed from:

```
abs(x - dequantize(quantize(x))) <= scale / 2
```

Per channel, the scale in that inequality is the scale belonging to that
element's channel. This is the whole guarantee quantization offers, and if it
fails then the scale, the integer range, or the clamp is wrong — there is
nothing else it could be.

## Per channel

`quantize_per_channel` gives each slice along `axis` its own scale and zero
point, computed from the values in that slice — which means reducing over every
axis *except* `axis`. A weight tensor whose channels differ in magnitude by
three orders of magnitude is the reason this exists: a single scale wide enough
for the loudest channel leaves the quiet ones with almost no levels at all.

`axis` may be negative and counts from the end when it is. `scale` comes back as
a one-dimensional `float64` array and `zero_point` as a one-dimensional `int32`
array, both of length `x.shape[axis]`, in channel order. `dequantize` broadcasts,
so a caller reshapes those for the tensor they came from and passes them
straight in.

## Conventions

- `x` is array-like: a numpy array of `float32` or `float64`, or a nested list.
  It has at least one element and contains no NaN and no infinity.
- `q` has the same shape as `x` and dtype `np.int32`, for every bit width.
- For `quantize_per_tensor`, `scale` is a scalar float and `zero_point` a scalar
  integer. The scale is always positive.
- `dequantize` returns `float64` and broadcasts its arguments the way numpy
  does.
- `num_bits` is a parameter, not a constant. It ranges over the widths anyone
  actually quantizes to: from 2 up to 16.
- Neither quantize function modifies its input.
- No `torch`, no `scipy`. numpy is enough, and a quantization library would be
  answering a different question.
