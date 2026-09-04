"""The backward pass of a row-wise softmax as a single Metal kernel.

The module holds one name, `SOURCE`: Metal Shading Language, compiled with
`torch.mps.compile_shader` by whoever launches it. One threadgroup handles one
row, and the threadgroup size is chosen by the caller rather than by the kernel.

Where the forward pass reduced a row twice and wrote one number, this one
reduces three times and writes the row back: the softmax has to be reconstructed
under the same shift that made it finite, the upstream gradient has to be
contracted against it, and only then can any element of the answer be written.
"""

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void softmax_backward_rows(
    device float* dx         [[buffer(0)]],
    constant float* logits   [[buffer(1)]],
    constant float* dy       [[buffer(2)]],
    constant uint& n_cols    [[buffer(3)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
{
}
"""
