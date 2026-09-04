# The backward pass of a row-wise softmax, as one Metal kernel

The forward softmax at the end of a transformer block is cheap and its backward
is not, because the gradient of every element of a row depends on every other
element of the same row. Written naively that is a materialised Jacobian, or an
extra full-width buffer holding the softmax the forward already computed. On
Apple silicon we would rather pay for neither: one kernel, one threadgroup per
row, the row's statistics computed where the row already is.

Fill in `softmax_backward_kernel.py`. It holds one name, `SOURCE`, a string of
Metal Shading Language. Whoever launches it compiles the string with
`torch.mps.compile_shader` and calls the kernel it declares, so the string is
the whole deliverable and the kernel's signature is the interface:

```metal
kernel void softmax_backward_rows(
    device float* dx         [[buffer(0)]],
    constant float* logits   [[buffer(1)]],
    constant float* dy       [[buffer(2)]],
    constant uint& n_cols    [[buffer(3)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
```

The name, the parameter list and the buffer indices are fixed. The caller binds
those four buffers and nothing else, and it declares no threadgroup memory on
your behalf.

## The arithmetic

`logits` and `dy` are row-major contiguous float32 matrices of the same shape
`(n_rows, n_cols)`; row `r` starts at element `r * n_cols`. `logits` is what the
forward softmax was computed from, and `dy` is the gradient arriving from
whatever consumed that softmax. Write the gradient with respect to the logits:

```
y[r, j]  = exp(logits[r, j]) / sum_k exp(logits[r, k])

dx[r, j] = y[r, j] * (dy[r, j] - sum_k y[r, k] * dy[r, k])
```

The forward's softmax is not handed to you and there is nowhere to put it: `dx`
is the only buffer you may write, it is exactly the shape of `logits`, and the
row it holds is the answer rather than scratch space. Rows do not interact. `dx`
may be a window into a longer buffer, and the elements past `n_rows * n_cols`
belong to somebody else.

Logits arrive from a real model, unnormalised, and can be several hundred in
magnitude in either direction, which is far enough for `exp` to leave float32 in
both directions. The answer has to come back finite and right anyway. `dy` is an
ordinary gradient and is not extreme in that way, but nothing about the answer
may depend on its scale or on where its mean sits.

Accumulate in float32. The result is compared against the same gradient computed
in double precision on the CPU, so the arithmetic has to be the well-conditioned
form rather than merely the algebraically equal one.

## The launch

The caller dispatches **one threadgroup per row**: `n_rows * G` threads with a
threadgroup size of `G`, so `row` is the row index and `tid` runs over the
group. `G` is the caller's choice, anywhere from 1 to 1024, and it has no
relationship to `n_cols` — it may be a fraction of the row, a multiple of it, or
larger than the row entirely, and it is not necessarily a power of two or a
multiple of the simdgroup width. `n_cols` itself has no upper bound worth
assuming; a vocabulary row is tens of thousands of elements.

`tpg` is the group the kernel was actually launched with. The kernel is
launched many times with different values of `G` over the lifetime of one
compiled library, and the same inputs at the same `G` must produce the same
bytes every time.

Nothing here needs a second kernel, a second pass over the grid, or any
host-side work: one launch computes the whole batch.
