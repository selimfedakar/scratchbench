"""Grading one task: assemble a workdir, run pytest in it, read the verdict.

The order of operations in `grade` is the part that matters. The solver is
handed a directory containing the starter files and nothing else; the hidden
tests are copied in only after it has finished writing. A solver cannot read
what is not on disk yet, and that is a cheaper guarantee than any amount of
prompt discipline.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tasks import Task, available_accelerators

# Exit codes pytest uses. 2 and 3 mean the tests never really ran, which is a
# different event from a wrong answer and is reported as such.
PYTEST_ALL_PASSED = 0
PYTEST_TESTS_FAILED = 1
PYTEST_NO_TESTS = 5

SUMMARY_PATTERN = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")


@dataclass
class Outcome:
    """What happened when one task was graded."""

    slug: str
    passed: bool
    status: str  # passed | failed | timeout | collection_error | missing_deps | adapter_error
    duration_s: float
    returncode: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    detail: str = ""
    output: str = ""

    def summary(self) -> str:
        if self.counts:
            parts = [f"{count} {name}" for name, count in sorted(self.counts.items())]
            return ", ".join(parts)
        return self.detail or self.status


def parse_summary(stdout: str) -> dict[str, int]:
    """Pull `3 passed, 1 failed` style counts out of pytest's last lines."""
    counts: dict[str, int] = {}
    for count, label in SUMMARY_PATTERN.findall(stdout):
        label = "error" if label == "errors" else label
        counts[label] = counts.get(label, 0) + int(count)
    return counts


def assemble(task: Task, solution_dir: Path, workdir: Path) -> None:
    """Copy the solution files for `task` into a flat `workdir`."""
    workdir.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(solution_dir).glob("*.py")):
        shutil.copy2(source, workdir / source.name)


def add_hidden_tests(task: Task, workdir: Path) -> None:
    """Drop the hidden tests in beside the solution, once it is written."""
    for source in task.hidden_test_files():
        shutil.copy2(source, workdir / source.name)


def pytest_environment() -> dict[str, str]:
    """A deterministic, offline environment for the graded run."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Thread counts change floating-point reduction order in numpy and torch,
    # so a graded run pins them and stays reproducible across machines.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("PYTHONPATH", None)
    return env


def run_pytest(workdir: Path, timeout_s: int) -> tuple[int | None, str, float]:
    """Run the tests in `workdir`. Returns (returncode, output, seconds)."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=pytest_environment(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as expired:
        elapsed = time.perf_counter() - started
        partial = (expired.stdout or b"").decode(errors="replace") if expired.stdout else ""
        return None, partial, elapsed

    elapsed = time.perf_counter() - started
    return completed.returncode, completed.stdout + completed.stderr, elapsed


def grade(task: Task, solve, keep_workdir: Path | None = None) -> Outcome:
    """Grade one task.

    `solve(task, workdir)` is called with a directory holding the starter
    files. Whatever it leaves behind is what gets tested.
    """
    if task.requires_gpu and task.accelerator not in available_accelerators():
        # Neither a pass nor a failure: this machine cannot produce evidence
        # either way, and rounding an absence of evidence down to zero is how a
        # leaderboard starts lying about hardware it never had.
        return Outcome(
            slug=task.slug,
            passed=False,
            status="needs_accelerator",
            duration_s=0.0,
            detail=f"no {task.accelerator} device on this machine",
        )

    missing = task.missing_deps()
    if missing:
        return Outcome(
            slug=task.slug,
            passed=False,
            status="missing_deps",
            duration_s=0.0,
            detail=f"not installed: {', '.join(missing)}",
        )

    started_at = time.perf_counter()
    temporary = tempfile.mkdtemp(prefix=f"scratchbench-{task.slug}-")
    workdir = Path(temporary)
    try:
        assemble(task, task.starter_dir, workdir)

        # A solver that raises — an unwritten adapter, a rate limit, a dropped
        # connection twenty tasks into a sweep — fails this task and only this
        # task. It is not the same event as a wrong answer, so it gets its own
        # status rather than being counted as one.
        try:
            solve(task, workdir)
        except Exception as error:  # noqa: BLE001 — the solver is arbitrary code
            return Outcome(
                slug=task.slug,
                passed=False,
                status="adapter_error",
                duration_s=time.perf_counter() - started_at,
                detail=f"{type(error).__name__}: {error}".strip(),
            )

        add_hidden_tests(task, workdir)

        returncode, output, elapsed = run_pytest(workdir, task.time_limit_s)

        if returncode is None:
            return Outcome(
                slug=task.slug,
                passed=False,
                status="timeout",
                duration_s=elapsed,
                detail=f"exceeded {task.time_limit_s}s",
                output=output,
            )

        counts = parse_summary(output)
        if returncode == PYTEST_ALL_PASSED:
            status = "passed"
        elif returncode == PYTEST_TESTS_FAILED and not counts.get("error"):
            status = "failed"
        elif returncode == PYTEST_NO_TESTS:
            status = "collection_error"
        else:
            # Import errors, syntax errors, fixtures blowing up: the tests
            # never got to judge anything.
            status = "collection_error"

        return Outcome(
            slug=task.slug,
            passed=status == "passed",
            status=status,
            duration_s=elapsed,
            returncode=returncode,
            counts=counts,
            output=output,
        )
    finally:
        if keep_workdir is not None:
            destination = Path(keep_workdir) / task.slug
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(workdir, destination)
        shutil.rmtree(temporary, ignore_errors=True)
