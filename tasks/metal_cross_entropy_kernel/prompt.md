# Softmax cross-entropy for a batch of rows, as one Metal kernel

The loss at the end of a language model reads the whole vocabulary row twice —
once through softmax, once to gather the target — and on Apple silicon we pay
for both passes in memory traffic. Fuse it: one kernel, one pass structure, the
row's statistics computed where the row already is.

Fill in `cross_entropy_kernel.py`. It holds one name, `SOURCE`, a string of
Metal Shading Language. Whoever launches it compiles the string with
`torch.mps.compile_shader` and calls the kernel it declares, so the string is
the whole deliverable and the kernel's signature is the interface:

```metal
kernel void cross_entropy_rows(
    device float* out        [[buffer(0)]],
    constant float* logits   [[buffer(1)]],
    constant int* targets    [[buffer(2)]],
    constant uint& n_cols    [[buffer(3)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
```

The name, the parameter list and the buffer indices are fixed. The caller binds
those four buffers and nothing else, and it declares no threadgroup memory on
your behalf.

## The arithmetic

`logits` is a row-major contiguous float32 matrix of shape `(n_rows, n_cols)`;
row `r` starts at element `r * n_cols`. `targets` holds one class index per row,
in `[0, n_cols)`. For row `r` the loss is

```
out[r] = log(sum_j exp(logits[r, j])) - logits[r, targets[r]]
```

which is the negative log probability the row's softmax assigns to its target.
Rows do not interact. `out` has one float32 element per row and may be a window
into a longer buffer: the elements past `n_rows` belong to somebody else.

Logits arrive from a real model, unnormalised, and can be several hundred in
magnitude in either direction, which is far enough for `exp` to leave float32 in
both directions. The loss has to come back finite and right anyway.

Accumulate in float32. The result is compared against the same loss computed in
double precision on the CPU, so the arithmetic has to be the well-conditioned
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

Nothing here needs a second kernel, a backward pass, or any host-side work: one
launch computes the whole batch.
