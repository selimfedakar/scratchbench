"""Mutation check for `metal_cross_entropy_kernel`.

Passing tests prove the reference is right. Only mutants prove the tests would
catch anything else, and `CONTRIBUTING.md` asks for that evidence in the pull
request rather than on trust.

It is separate from `mutate_v2_tasks.py` for the same reason `mutate_rmsnorm.py`
is: this one needs hardware. Every mutant here is a wrong Metal kernel, so the
machine running it needs a Metal device — any Apple silicon Mac with a PyTorch
that reports `torch.backends.mps.is_available()`.

Usage, from anywhere:

    python tools/mutate_metal_task.py
    python tools/mutate_metal_task.py --repo /path/to/repo

Each mutant carries the verdict it is expected to get. Nine are expected to be
CAUGHT and one — the missing write-after-read barrier on the scratch — is
expected to SURVIVE, because this hardware does not expose that race through the
kernel's output. A mutant that disagrees with its expectation fails the script,
in both directions: a caught survivor is news worth having.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SLUG = "metal_cross_entropy_kernel"
MODULE = "cross_entropy_kernel.py"

SIGNATURE_TAIL = """    uint tid                 [[thread_position_in_threadgroup]],
    uint row                 [[threadgroup_position_in_grid]],
    uint tpg                 [[threads_per_threadgroup]])
{"""

SCRATCH_DECLARATION = """    // The library gives no way to set a threadgroup memory length, so the
    // scratch space is declared here at the largest threadgroup the caller is
    // allowed to ask for.
    threadgroup float scratch[1024];
"""

MAX_FOLD = """    for (uint active = tpg; active > 1; ) {
        const uint stride = (active + 1) >> 1;
        if (tid + stride < active) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride;
    }"""

SUM_FOLD_AND_WRITE = """    scratch[tid] = local_sum;
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

SIMDGROUP_SUM = """    const float row_sum = simd_sum(local_sum);

    if (tid == 0) {
        const uint target = uint(targets[row]);
        out[row] = row_max + log(row_sum) - logits[base + target];
    }"""

# (name, what it does wrong, expected verdict, old text, new text)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    (
        "threadgroup memory as a bound buffer",
        "takes the scratch as a [[threadgroup(0)]] argument, which the caller never binds",
        "CAUGHT",
        SIGNATURE_TAIL + "\n" + SCRATCH_DECLARATION,
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
        "CAUGHT",
        SUM_FOLD_AND_WRITE,
        SIMDGROUP_SUM,
    ),
    (
        "power-of-two fold",
        "halves the live range downwards, which drops entries when the group is not a power of two",
        "CAUGHT",
        MAX_FOLD,
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
        "CAUGHT",
        "        if (tid + stride < active) {\n            scratch[tid] += scratch[tid + stride];",
        "        if (tid + stride < tpg) {\n            scratch[tid] += scratch[tid + stride];",
    ),
    (
        "idle lanes never publish an identity",
        "only threads holding a column write the scratch, so the fold reads uninitialised memory",
        "CAUGHT",
        "    scratch[tid] = local_max;",
        "    if (tid < n_cols) { scratch[tid] = local_max; }",
    ),
    (
        "out-of-range lanes return early",
        "returns before the barriers when the group is wider than the row",
        "CAUGHT",
        "    const uint base = row * n_cols;\n",
        "    const uint base = row * n_cols;\n    if (tid >= n_cols) { return; }\n",
    ),
    (
        "barrier inside the branch",
        "only the threads that folded reach the barrier",
        "CAUGHT",
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
        "CAUGHT",
        "    scratch[tid] = local_max;\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    scratch[tid] = local_max;",
    ),
    (
        "no shift before the exponential",
        "sums exp(logit) directly, which leaves float32 in both directions",
        "CAUGHT",
        "        local_sum += exp(logits[base + c] - row_max);",
        "        local_sum += exp(logits[base + c]);",
    ),
    (
        "shift never added back",
        "returns log(sum) without the maximum it was subtracted from",
        "CAUGHT",
        "        out[row] = row_max + log(scratch[0]) - logits[base + target];",
        "        out[row] = log(scratch[0]) - logits[base + target];",
    ),
    (
        "thread 0 does the whole row",
        "a correct kernel that uses one lane of the group and no threadgroup memory at all: "
        "the hidden tests grade the answer, and only a wall-clock assertion could tell the "
        "difference, which this repository does not allow anywhere",
        "SURVIVES",
        SCRATCH_DECLARATION + "    const uint base = row * n_cols;",
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


def run_tests(source: str, task_dir: Path) -> tuple[int, str]:
    workdir = Path(tempfile.mkdtemp(prefix="mutate-metal-"))
    try:
        (workdir / MODULE).write_text(source, encoding="utf-8")
        for test_file in sorted((task_dir / "hidden_tests").glob("test_*.py")):
            shutil.copy2(test_file, workdir / test_file.name)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"],
            cwd=workdir,
            env=graded_environment(),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return completed.returncode, (completed.stdout + completed.stderr).strip()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def check(repo: Path) -> int:
    task_dir = repo / "tasks" / SLUG
    reference = (task_dir / "reference" / MODULE).read_text(encoding="utf-8")
    starter = (task_dir / "starter" / MODULE).read_text(encoding="utf-8")

    print(f"=== {SLUG} ===")
    print("--- reference ---")
    code, output = run_tests(reference, task_dir)
    print(last_line(output))
    if code != 0:
        print("\nthe reference does not pass; nothing below means anything")
        print(output)
        return 1

    print("--- untouched starter ---")
    code, output = run_tests(starter, task_dir)
    print(last_line(output))
    if code == 0:
        print("\nthe untouched starter passes; the tests prove nothing")
        return 1

    print("--- mutants ---")
    wrong = 0
    for name, description, expected, old, new in MUTANTS:
        if old not in reference:
            print(f"SKIPPED   {name}: the reference no longer contains the patched text")
            wrong += 1
            continue
        code, output = run_tests(reference.replace(old, new, 1), task_dir)
        verdict = "SURVIVED" if code == 0 else "CAUGHT"
        agrees = verdict.startswith(expected[:6])
        wrong += int(not agrees)
        note = "" if agrees else f"  <-- expected {expected}"
        print(f"{verdict:9s} {name} ({description}){note}")
        print(f"          {last_line(output)}")

    print()
    if wrong:
        print(f"{SLUG}: {wrong} mutant(s) did not match their expected verdict")
        return 1
    print(f"{SLUG}: all {len(MUTANTS)} mutants behaved as expected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="path to the scratchbench checkout")
    arguments = parser.parse_args()
    return check(Path(arguments.repo).resolve())


if __name__ == "__main__":
    sys.exit(main())
