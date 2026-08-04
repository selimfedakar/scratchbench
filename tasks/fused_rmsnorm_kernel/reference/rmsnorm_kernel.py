"""A fused RMSNorm forward pass as a single Triton kernel, one program per row."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Rows longer than this are walked in several passes rather than held in one
#: block. 1024 elements per program is a comfortable size for a reduction that
#: has to run twice over the row.
MAX_BLOCK_SIZE = 1024


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One row of RMSNorm per program."""
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * x_row_stride
    out_row_ptr = out_ptr + row * out_row_stride

    # float32 accumulation regardless of the input dtype. This is the whole
    # reason the kernel is worth writing: a float16 accumulator runs out of
    # range on a long row, and the failure is silent — an infinity in the sum
    # of squares comes back as zeros, not as an error.
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # `other=0.0` keeps the masked tail out of the sum: zeros square to
        # zero and contribute nothing.
        values = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        accumulator += values * values

    mean_square = tl.sum(accumulator, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)

    # Second pass over the row. The alternative is keeping the whole row in
    # registers between the two passes, which stops working the moment the row
    # is longer than one block — and long rows are the case that matters.
    for block in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        values = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        result = values * inv_rms * weight
        tl.store(out_row_ptr + cols, result.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension of a 2-D tensor."""
    n_rows, n_cols = x.shape
    # Not `empty_like`: x may be a strided view, and the contract is that the
    # result comes back contiguous.
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    if n_rows == 0:
        return out

    block_size = min(MAX_BLOCK_SIZE, triton.next_power_of_2(n_cols))
    rmsnorm_fwd_kernel[(n_rows,)](
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
