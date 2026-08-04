# A fused RMSNorm forward pass in Triton

We normalise activations with RMSNorm and the PyTorch version of it costs us
several passes over memory per layer. Write the forward half as a single Triton
kernel instead.

Fill in `rmsnorm_kernel.py`. It holds two things, both already declared:
`rmsnorm_fwd_kernel`, a `@triton.jit` kernel, and `rmsnorm`, the Python wrapper
that launches it. Neither signature may change — the kernel is launched
directly as well as through the wrapper, so its parameter order and its
`BLOCK_SIZE` constant are part of the interface rather than an implementation
detail.

## The arithmetic

For one row `x` of length N and a weight vector `w` of the same length, the
output row is `x / sqrt(mean(x**2) + eps) * w`, elementwise, where the mean is
taken over that row and nothing else. Epsilon goes inside the square root,
added to the mean of the squares. Column `j` of the row is scaled by `w[j]`.
Rows do not interact.

## The kernel

One program handles one row: the launch grid is one-dimensional with M
programs, and `tl.program_id(0)` is the row index.

`x_row_stride` and `out_row_stride` are the distance in elements between
consecutive rows of the input and of the output. Within a row the elements are
adjacent, so column `j` of a row sits one element after column `j - 1`. The two
strides are independent of each other and neither is necessarily N — the output
the kernel is handed may be a window into a wider buffer, and writing outside
the row would be writing into somebody else's memory.

`n_cols` is N. `BLOCK_SIZE` is a `tl.constexpr` power of two chosen by whoever
launches the kernel, and it has no fixed relationship to N: it may be smaller,
in which case the row takes more than one pass, or larger, in which case the
elements past the end of the row must not be read or written.

Accumulate the sum of squares in float32 whatever the input dtype is. The input
arrives as float16 or float32 and the weights are always float32; a float16
input can carry values whose squares are past the range of float16, so squaring
in the input dtype loses the row to an infinity rather than to an error. The
output has the same dtype as the input, so the float32 result is cast on the way
out. The input and the weights are read-only.

## The wrapper

`rmsnorm(x, weight, eps=1e-6)` takes a two-dimensional CUDA tensor `x` of shape
(M, N) and a contiguous one-dimensional float32 CUDA tensor `weight` of length
N. It chooses a block size and a grid, launches the kernel once, and returns a
new contiguous tensor of shape (M, N) with `x`'s dtype on `x`'s device.

`x` may be a view whose rows are strided rather than a contiguous tensor: its
columns are adjacent, but consecutive rows need not be. M may be zero and N is
at least one.

Nothing here needs a backward pass, autotuning, or more than one kernel launch
per call.
