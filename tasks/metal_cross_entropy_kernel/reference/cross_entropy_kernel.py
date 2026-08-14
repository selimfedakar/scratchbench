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
    // The library gives no way to set a threadgroup memory length, so the
    // scratch space is declared here at the largest threadgroup the caller is
    // allowed to ask for.
    threadgroup float scratch[1024];
    const uint base = row * n_cols;

    // Pass one: the row maximum. A thread walks its own columns with a stride
    // of the threadgroup size, so a row wider than the group takes several
    // steps and a group wider than the row leaves some threads with nothing.
    // Those threads still hold the identity of the reduction and still reach
    // every barrier below.
    float local_max = -INFINITY;
    for (uint c = tid; c < n_cols; c += tpg) {
        local_max = max(local_max, logits[base + c]);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The tree folds the upper half of the live range onto the lower half.
    // `active` is the number of entries still carrying a partial result, and
    // the stride is its half rounded up, so a group size that is not a power of
    // two loses nothing: the write set [0, active - stride) and the read set
    // [stride, active) never overlap.
    for (uint active = tpg; active > 1; ) {
        const uint stride = (active + 1) >> 1;
        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }
    const float row_max = scratch[0];
    // Everybody has read the maximum before anybody overwrites the scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass two: the sum of the shifted exponentials, over the same columns.
    float local_sum = 0.0f;
    for (uint c = tid; c < n_cols; c += tpg) {
        local_sum += exp(logits[base + c] - row_max);
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint active = tpg; active > 1; ) {
        const uint stride = (active + 1) >> 1;
        if (tid + stride < active) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }

    if (tid == 0) {
        const uint target = uint(targets[row]);
        out[row] = row_max + log(scratch[0]) - logits[base + target];
    }
}
"""
