"""Mutation check for `tasks/fused_rmsnorm_kernel`, to be run on a CUDA machine.

Passing tests prove the reference is right. Only mutants prove the tests would
catch anything else, and `CONTRIBUTING.md` asks for that evidence in the pull
request rather than on trust. This applies six wrong implementations a real
attempt would plausibly produce and reports, for each, whether the hidden tests
noticed.

It lives in the repository rather than beside it because a check nobody can
re-run is a claim, not a check — and because the first time it was needed it was
on a rented GPU box that did not have it.

Usage, from anywhere:

    python tools/mutate_rmsnorm.py /path/to/scratchbench

Every mutant must come back CAUGHT. A SURVIVED line is a hole in the task.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SLUG = "fused_rmsnorm_kernel"
MODULE = "rmsnorm_kernel.py"

# (name, what it does wrong, old text, new text)
MUTANTS = [
    (
        "float16 accumulator",
        "sums the squares in the input dtype instead of float32",
        """    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # `other=0.0` keeps the masked tail out of the sum: zeros square to
        # zero and contribute nothing.
        values = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        accumulator += values * values""",
        """    accumulator = tl.zeros([BLOCK_SIZE], dtype=x_ptr.dtype.element_ty)
    for block in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        values = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        accumulator += values * values""",
    ),
    (
        "off-by-one mean",
        "divides the sum of squares by n_cols - 1",
        "mean_square = tl.sum(accumulator, axis=0) / n_cols",
        "mean_square = tl.sum(accumulator, axis=0) / (n_cols - 1)",
    ),
    (
        "mean over the block rather than the row",
        "divides by BLOCK_SIZE, which is only the row length by coincidence",
        "mean_square = tl.sum(accumulator, axis=0) / n_cols",
        "mean_square = tl.sum(accumulator, axis=0) / BLOCK_SIZE",
    ),
    (
        "epsilon outside the root",
        "adds eps to the root-mean-square instead of to the mean square",
        "inv_rms = 1.0 / tl.sqrt(mean_square + eps)",
        "inv_rms = 1.0 / (tl.sqrt(mean_square) + eps)",
    ),
    (
        "one stride for both tensors",
        "walks the input with the output's row stride",
        "x_row_ptr = x_ptr + row * x_row_stride",
        "x_row_ptr = x_ptr + row * out_row_stride",
    ),
    (
        "weight indexed by row",
        "scales the whole row by a single weight instead of one per column",
        "weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)",
        "weight = tl.load(weight_ptr + row)",
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
    workdir = Path(tempfile.mkdtemp(prefix="mutate-rmsnorm-"))
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
            timeout=600,
        )
        return completed.returncode, (completed.stdout + completed.stderr).strip()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    task_dir = repo / "tasks" / SLUG
    reference = (task_dir / "reference" / MODULE).read_text(encoding="utf-8")
    starter = (task_dir / "starter" / MODULE).read_text(encoding="utf-8")

    print("=== reference ===")
    code, output = run_tests(reference, task_dir)
    print(last_line(output))
    if code != 0:
        print("\nthe reference does not pass; nothing below means anything")
        print(output)
        return 1

    print("\n=== untouched starter ===")
    code, output = run_tests(starter, task_dir)
    print(last_line(output))
    if code == 0:
        print("\nthe untouched starter passes; the tests prove nothing")
        return 1

    print("\n=== mutants ===")
    survivors = 0
    for name, description, old, new in MUTANTS:
        if old not in reference:
            print(f"SKIPPED   {name}: the reference no longer contains the patched text")
            survivors += 1
            continue
        code, output = run_tests(reference.replace(old, new, 1), task_dir)
        verdict = "SURVIVED" if code == 0 else "CAUGHT  "
        survivors += int(code == 0)
        print(f"{verdict}  {name} ({description})")
        print(f"          {last_line(output)}")

    print()
    if survivors:
        print(f"{survivors} mutant(s) survived: the task has a hole in it")
        return 1
    print(f"all {len(MUTANTS)} mutants caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
