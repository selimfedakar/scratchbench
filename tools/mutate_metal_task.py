"""Mutation check for the Metal tasks.

Passing tests prove the reference is right. Only mutants prove the tests would
catch anything else, and `CONTRIBUTING.md` asks for that evidence in the pull
request rather than on trust.

It is separate from `mutate_v2_tasks.py` for the same reason `mutate_rmsnorm.py`
is: this one needs hardware. Every mutant here is a wrong Metal kernel, so the
machine running it needs a Metal device — any Apple silicon Mac with a PyTorch
that reports `torch.backends.mps.is_available()`.

Usage, from anywhere:

    python tools/mutate_metal_task.py                       # every task below
    python tools/mutate_metal_task.py --repo /path/to/repo --task <slug>

Each mutant carries the verdict it is expected to get, and the script fails in
both directions: a mutant expected to be CAUGHT that escapes is a hole in the
task, and a mutant expected to SURVIVE that is caught is news worth having.
`docs/LESSONS.md` L31 through L33 is one entry per survivor of the forward task,
and each of them is a fact about the task rather than a gap in it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

#: `run_tests` returns this instead of pytest's exit code when the run never
#: answered. The shell convention for a killed-by-timeout command.
TIMED_OUT = 124

SIGNATURE_TAIL = """    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
{"""

# --- metal_cross_entropy_kernel -------------------------------------------

FORWARD_SCRATCH = """    // The library gives no way to set a threadgroup memory length, so the
    // scratch space is declared here at the largest threadgroup the caller is
    // allowed to ask for.
    threadgroup float scratch[1024];
"""

FORWARD_MAX_FOLD = """    for (uint active = tpg; active > 1; ) {
        const uint stride = (active + 1) >> 1;
        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }"""

FORWARD_SUM_FOLD_AND_WRITE = """    scratch[tid] = local_sum;
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
    }"""

FORWARD_SIMDGROUP_SUM = """    const float row_sum = simd_sum(local_sum);

    if (tid == 0) {
        const uint target = uint(targets[row]);
        out[row] = row_max + log(row_sum) - logits[base + target];
    }"""

FORWARD_MUTANTS: list[tuple[str, ...]] = [
    (
        "threadgroup memory as a bound buffer",
        "takes the scratch as a [[threadgroup(0)]] argument, which the caller never binds",
        SIGNATURE_TAIL + "\n" + FORWARD_SCRATCH,
        """    threadgroup float* scratch [[threadgroup(0)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
{
""",
    ),
    (
        "simdgroup reduction only",
        "sums with simd_sum, which never crosses the 32-lane simdgroup",
        FORWARD_SUM_FOLD_AND_WRITE,
        FORWARD_SIMDGROUP_SUM,
    ),
    (
        "power-of-two fold",
        "halves the live range downwards, which drops entries when the group is not a power of two",
        FORWARD_MAX_FOLD,
        """    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }""",
    ),
    (
        "max fold bounded by the group instead of the live range",
        "folds entries that were already folded, which max absorbs: it is idempotent and monotone, "
        "so the extra merges and even their stale reads cannot lose an element",
        "SURVIVES",
        "        if (tid + stride < active) {\n            scratch[tid] = max(scratch[tid], scratch[tid + stride]);",
        "        if (tid + stride < tpg) {\n            scratch[tid] = max(scratch[tid], scratch[tid + stride]);",
    ),
    (
        "sum fold bounded by the group instead of the live range",
        "the same off-by-a-range one step later, where the arithmetic cannot absorb it",
        "        if (tid + stride < active) {\n            scratch[tid] += scratch[tid + stride];",
        "        if (tid + stride < tpg) {\n            scratch[tid] += scratch[tid + stride];",
    ),
    (
        "idle lanes never publish an identity",
        "only threads holding a column write the scratch, so the fold reads uninitialised memory",
        "    scratch[tid] = local_max;",
        "    if (tid < n_cols) { scratch[tid] = local_max; }",
    ),
    (
        "out-of-range lanes return early",
        "returns before the barriers when the group is wider than the row",
        "    const uint base = row * n_cols;\n",
        "    const uint base = row * n_cols;\n    if (tid >= n_cols) { return; }\n",
    ),
    (
        "barrier inside the branch",
        "only the threads that folded reach the barrier",
        """        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;""",
        """        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        active = stride;""",
    ),
    (
        "no barrier after publishing the partial maxima",
        "folds the scratch before every thread has written to it",
        "    scratch[tid] = local_max;\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    scratch[tid] = local_max;",
    ),
    (
        "no shift before the exponential",
        "sums exp(logit) directly, which leaves float32 in both directions",
        "        local_sum += exp(logits[base + c] - row_max);",
        "        local_sum += exp(logits[base + c]);",
    ),
    (
        "shift never added back",
        "returns log(sum) without the maximum it was subtracted from",
        "        out[row] = row_max + log(scratch[0]) - logits[base + target];",
        "        out[row] = log(scratch[0]) - logits[base + target];",
    ),
    (
        "thread 0 does the whole row",
        "a correct kernel that uses one lane of the group and no threadgroup memory at all: "
        "the hidden tests grade the answer, and only a wall-clock assertion could tell the "
        "difference, which this repository does not allow anywhere",
        "SURVIVES",
        FORWARD_SCRATCH + "    const uint base = row * n_cols;",
        """    const uint base = row * n_cols;
    if (tid != 0) { return; }
    float serial_max = -INFINITY;
    for (uint c = 0; c < n_cols; ++c) { serial_max = max(serial_max, logits[base + c]); }
    float serial_sum = 0.0f;
    for (uint c = 0; c < n_cols; ++c) { serial_sum += exp(logits[base + c] - serial_max); }
    out[row] = serial_max + log(serial_sum) - logits[base + uint(targets[row])];
    return;
    threadgroup float scratch[1024];""",
    ),
    (
        "no barrier between reading the maximum and reusing the scratch",
        "thread 0 may overwrite scratch[0] while the rest of the group is still reading it",
        "SURVIVES",
        "    const float row_max = scratch[0];\n    // Everybody has read the maximum before anybody overwrites the scratch.\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    const float row_max = scratch[0];",
    ),
]

# --- metal_softmax_backward_kernel ----------------------------------------

BACKWARD_SCRATCH = """    // The library gives no way to set a threadgroup memory length, so the
    // scratch space is declared here at the largest threadgroup the caller is
    // allowed to ask for. One array serves all three reductions, which is what
    // makes the barrier after each of them load-bearing rather than decorative.
    threadgroup float scratch[1024];
"""

BACKWARD_MAX_FOLD = """    for (uint active = tpg; active > 1; ) {
        const uint stride = (active + 1) >> 1;
        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }"""

#: The two summing folds are textually identical, so every mutant that targets
#: one of them carries the line that follows it. Otherwise `str.replace(..., 1)`
#: silently patches the denominator and calls it a test of the contraction.
BACKWARD_DENOMINATOR_FOLD = """    scratch[tid] = local_sum;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);"""

BACKWARD_DENOMINATOR_TAIL = """        if (tid + stride < active) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }
    const float row_sum = scratch[0];"""

BACKWARD_CONTRACTION_TAIL = """        if (tid + stride < active) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }
    // Dividing the weighted sum by the plain one is the contraction of the"""

BACKWARD_WRITE = """    for (uint c = tid; c < n_cols; c += tpg) {
        const float y = exp(logits[base + c] - row_max) / row_sum;
        dx[base + c] = y * (dy[base + c] - dot);
    }"""

BACKWARD_MUTANTS: list[tuple[str, ...]] = [
    (
        "threadgroup memory as a bound buffer",
        "takes the scratch as a [[threadgroup(0)]] argument, which the caller never binds",
        SIGNATURE_TAIL + "\n" + BACKWARD_SCRATCH,
        """    threadgroup float* scratch [[threadgroup(0)]],
    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
{
""",
    ),
    (
        "simdgroup reduction for the denominator",
        "sums with simd_sum, which never crosses the 32-lane simdgroup",
        BACKWARD_DENOMINATOR_FOLD,
        "    const float row_sum = simd_sum(local_sum);",
    ),
    (
        "power-of-two fold",
        "halves the live range downwards, which drops entries when the group is not a power of two",
        BACKWARD_MAX_FOLD,
        """    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }""",
    ),
    (
        "max fold bounded by the group instead of the live range",
        "folds entries that were already folded, which max absorbs: it is idempotent and monotone, "
        "so the extra merges and even their stale reads cannot lose an element",
        "SURVIVES",
        "        if (tid + stride < active) {\n            scratch[tid] = max(scratch[tid], scratch[tid + stride]);",
        "        if (tid + stride < tpg) {\n            scratch[tid] = max(scratch[tid], scratch[tid + stride]);",
    ),
    (
        "denominator fold bounded by the group instead of the live range",
        "the same off-by-a-range one step later, where the arithmetic cannot absorb it",
        BACKWARD_DENOMINATOR_TAIL,
        BACKWARD_DENOMINATOR_TAIL.replace(
            "if (tid + stride < active) {", "if (tid + stride < tpg) {", 1
        ),
    ),
    (
        "contraction fold bounded by the group instead of the live range",
        "the third reduction gets the same off-by-a-range, and nothing about the softmax hides it",
        BACKWARD_CONTRACTION_TAIL,
        BACKWARD_CONTRACTION_TAIL.replace(
            "if (tid + stride < active) {", "if (tid + stride < tpg) {", 1
        ),
    ),
    (
        "idle lanes never publish an identity",
        "only threads holding a column write the scratch, so the fold reads uninitialised memory",
        "    scratch[tid] = local_max;",
        "    if (tid < n_cols) { scratch[tid] = local_max; }",
    ),
    (
        "out-of-range lanes return early",
        "returns before the barriers when the group is wider than the row",
        "    const uint base = row * n_cols;\n",
        "    const uint base = row * n_cols;\n    if (tid >= n_cols) { return; }\n",
    ),
    (
        "barrier inside the branch",
        "only the threads that folded reach the barrier",
        """        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;""",
        """        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        active = stride;""",
    ),
    (
        "no barrier after publishing the partial maxima",
        "folds the scratch before every thread has written to it",
        "    scratch[tid] = local_max;\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    scratch[tid] = local_max;",
    ),
    (
        "no shift at all",
        "exponentiates the raw logits, which leaves float32 in both directions",
        "    const float row_max = scratch[0];",
        "    const float row_max = 0.0f;",
    ),
    (
        "the contraction is never subtracted",
        "treats the softmax backward as elementwise: dx = y * dy, which is the Jacobian's "
        "diagonal and nothing else",
        "        dx[base + c] = y * (dy[base + c] - dot);",
        "        dx[base + c] = y * dy[base + c];",
    ),
    (
        "the contraction is never normalised",
        "contracts the upstream gradient against the unnormalised exponentials",
        "    const float dot = scratch[0] / row_sum;",
        "    const float dot = scratch[0];",
    ),
    (
        "the contraction forgets the softmax weight",
        "sums the upstream gradient itself instead of weighting it by the exponentials",
        "        local_weighted += e * dy[base + c];",
        "        local_weighted += dy[base + c];",
    ),
    (
        "the softmax is never normalised when the row is written",
        "writes the shifted exponential in place of the probability",
        "        const float y = exp(logits[base + c] - row_max) / row_sum;",
        "        const float y = exp(logits[base + c] - row_max);",
    ),
    (
        "one column per thread instead of a strided walk",
        "writes only the column the thread is named after, so every row wider than the group "
        "keeps the sentinel it was filled with",
        BACKWARD_WRITE,
        """    if (tid < n_cols) {
        const float y = exp(logits[base + tid] - row_max) / row_sum;
        dx[base + tid] = y * (dy[base + tid] - dot);
    }""",
    ),
    (
        "thread 0 does the whole row",
        "a correct kernel that uses one lane of the group and no threadgroup memory at all: "
        "the hidden tests grade the answer, and only a wall-clock assertion could tell the "
        "difference, which this repository does not allow anywhere",
        "SURVIVES",
        BACKWARD_SCRATCH + "    const uint base = row * n_cols;",
        """    const uint base = row * n_cols;
    if (tid != 0) { return; }
    float serial_max = -INFINITY;
    for (uint c = 0; c < n_cols; ++c) { serial_max = max(serial_max, logits[base + c]); }
    float serial_sum = 0.0f;
    float serial_weighted = 0.0f;
    for (uint c = 0; c < n_cols; ++c) {
        const float e = exp(logits[base + c] - serial_max);
        serial_sum += e;
        serial_weighted += e * dy[base + c];
    }
    const float serial_dot = serial_weighted / serial_sum;
    for (uint c = 0; c < n_cols; ++c) {
        const float y = exp(logits[base + c] - serial_max) / serial_sum;
        dx[base + c] = y * (dy[base + c] - serial_dot);
    }
    return;
    threadgroup float scratch[1024];""",
    ),
    (
        "no barrier between reading the maximum and reusing the scratch",
        "thread 0 may overwrite scratch[0] while the rest of the group is still reading it",
        "SURVIVES",
        "    const float row_max = scratch[0];\n    // Everybody has read the maximum before anybody overwrites the scratch.\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    const float row_max = scratch[0];",
    ),
    (
        "no barrier between reading the denominator and reusing the scratch",
        "the same race one reduction later, and this one the hardware does expose: nothing "
        "stands between the read and the reuse, where the maximum's reuse has a whole pass "
        "over the row in the way. Caught in 8 runs of 8, with 21 to 23 of the 81 tests failing "
        "from run to run, so the verdict is stable and the blast radius is not. LESSONS L39",
        "    const float row_sum = scratch[0];\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    const float row_sum = scratch[0];",
    ),
]

# slug -> (module file, [mutant, ...]), where a mutant is either
#
#     (name, what it does wrong, old text, new text)                -> expected CAUGHT
#     (name, what it does wrong, "SURVIVES", old text, new text)    -> expected to pass
TASKS: dict[str, tuple[str, list[tuple[str, ...]]]] = {
    "metal_cross_entropy_kernel": ("cross_entropy_kernel.py", FORWARD_MUTANTS),
    "metal_softmax_backward_kernel": ("softmax_backward_kernel.py", BACKWARD_MUTANTS),
}


def graded_environment() -> dict[str, str]:
    """The same pinned environment the runner grades in."""
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.pop("PYTHONPATH", None)
    return env


def time_limit_of(task_dir: Path) -> int:
    """The task's own `time_limit_s`, read from the one place it is declared.

    A mutant is a solution, so it gets the limit a solution gets. Hard-coding a
    second number here would be a second list, and this file would be the third
    place in the repository to learn why that is a bad idea (L11, L20).
    """
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text(encoding="utf-8"))
    return int(meta["time_limit_s"])


def run_tests(source: str, task_dir: Path, module: str, timeout_s: int) -> tuple[int, str]:
    workdir = Path(tempfile.mkdtemp(prefix="mutate-metal-"))
    try:
        (workdir / module).write_text(source, encoding="utf-8")
        for test_file in sorted((task_dir / "hidden_tests").glob("test_*.py")):
            shutil.copy2(test_file, workdir / test_file.name)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"],
                cwd=workdir,
                env=graded_environment(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            # The graded harness calls this a `timeout`: evidence, and a
            # failure. So does this script. It is not a hypothetical — the
            # mutant that returns out-of-range lanes before a barrier is
            # undefined behaviour in Metal, and on this hardware it can hang the
            # GPU rather than produce a wrong answer. Left unbounded it ate
            # thirty minutes and every measurement taken after it.
            #
            # 124 rather than 1 so the caller can tell "the tests failed it"
            # from "it never answered". A mutant is CAUGHT either way, but the
            # untouched starter timing out proves nothing about the tests and
            # must not be read as the starter failing cleanly.
            return TIMED_OUT, f"timed out after {timeout_s}s"
        return completed.returncode, (completed.stdout + completed.stderr).strip()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def unpack(mutant: tuple[str, ...]) -> tuple[str, str, str, str, str]:
    """Both mutant shapes, normalised. A four-field mutant expects CAUGHT."""
    if len(mutant) == 5:
        name, description, expected, old, new = mutant
        return name, description, expected, old, new
    name, description, old, new = mutant
    return name, description, "CAUGHT", old, new


def last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def check(repo: Path, slug: str) -> int:
    module, mutants = TASKS[slug]
    task_dir = repo / "tasks" / slug
    timeout_s = time_limit_of(task_dir)
    reference = (task_dir / "reference" / module).read_text(encoding="utf-8")
    starter = (task_dir / "starter" / module).read_text(encoding="utf-8")

    print(f"=== {slug} ===")
    print("--- reference ---")
    code, output = run_tests(reference, task_dir, module, timeout_s)
    print(last_line(output))
    if code != 0:
        print("\nthe reference does not pass; nothing below means anything")
        print(output)
        return 1

    print("--- untouched starter ---")
    code, output = run_tests(starter, task_dir, module, timeout_s)
    print(last_line(output))
    if code == 0:
        print("\nthe untouched starter passes; the tests prove nothing")
        return 1
    if code == TIMED_OUT:
        print("\nthe untouched starter never answered; nothing below means anything")
        return 1

    print("--- mutants ---")
    wrong = 0
    for mutant in mutants:
        name, description, expected, old, new = unpack(mutant)
        if old not in reference:
            print(f"SKIPPED   {name}: the reference no longer contains the patched text")
            wrong += 1
            continue
        code, output = run_tests(reference.replace(old, new, 1), task_dir, module, timeout_s)
        verdict = "SURVIVED" if code == 0 else "CAUGHT  "
        agrees = verdict.startswith(expected[:6])
        wrong += int(not agrees)
        note = "" if agrees else f"  <-- expected {expected}"
        print(f"{verdict}  {name} ({description}){note}")
        print(f"          {last_line(output)}")

    print()
    if wrong:
        print(f"{slug}: {wrong} mutant(s) did not match their expected verdict")
        return 1
    print(f"{slug}: all {len(mutants)} mutants behaved as expected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="path to the scratchbench checkout")
    parser.add_argument("--task", action="append", choices=sorted(TASKS), help="one slug")
    arguments = parser.parse_args()

    repo = Path(arguments.repo).resolve()
    failures = 0
    for slug in arguments.task or sorted(TASKS):
        failures += check(repo, slug)
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
