"""A fused RMSNorm forward pass as a single Triton kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    raise NotImplementedError


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension of a 2-D tensor."""
    raise NotImplementedError
