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
    // The library gives no way to set a threadgroup memory length, so the
    // scratch space is declared here at the largest threadgroup the caller is
    // allowed to ask for. One array serves all three reductions, which is what
    // makes the barrier after each of them load-bearing rather than decorative.
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

    // Pass two walks the row once and accumulates both of the sums the answer
    // needs: the softmax denominator, and the same exponentials weighted by the
    // upstream gradient. Splitting them into two walks would be equally
    // correct; keeping them together is one read of the row instead of two.
    float local_sum = 0.0f;
    float local_weighted = 0.0f;
    for (uint c = tid; c < n_cols; c += tpg) {
        const float e = exp(logits[base + c] - row_max);
        local_sum += e;
        local_weighted += e * dy[base + c];
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
    const float row_sum = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    scratch[tid] = local_weighted;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint active = tpg; active > 1; ) {
        const uint stride = (active + 1) >> 1;
        if (tid + stride < active) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }
    // Dividing the weighted sum by the plain one is the contraction of the
    // upstream gradient with the softmax, without ever materialising the
    // softmax to take it.
    const float dot = scratch[0] / row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass three: the softmax is rebuilt from the same shift and the same
    // denominator every thread agreed on, and the row is written.
    for (uint c = tid; c < n_cols; c += tpg) {
        const float y = exp(logits[base + c] - row_max) / row_sum;
        dx[base + c] = y * (dy[base + c] - dot);
    }
}
"""
