"""A fused softmax cross-entropy forward pass as a single Metal kernel.

The module holds one name, `SOURCE`: Metal Shading Language, compiled with
`torch.mps.compile_shader` by whoever launches it. One threadgroup handles one
row, and the threadgroup size is chosen by the caller rather than by the kernel,
so the kernel has to work with a group that is larger than the row, smaller than
the row, or not a power of two.
"""

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_rows(
    device float* out        [[buffer(0)]],
    constant float* logits   [[buffer(1)]],
    constant int* targets    [[buffer(2)]],
    constant uint& n_cols    [[buffer(3)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
{
}
"""
