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

# The two exit codes that mean the suite actually ran and reached a verdict.
# Everything else — a collection error, no tests found, an internal error —
# means nothing judged the solution, and which side of the evidence line that
# lands on is decided by `starter_collects_cleanly` rather than by the code.
PYTEST_ALL_PASSED = 0
PYTEST_TESTS_FAILED = 1

SUMMARY_PATTERN = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")


@dataclass(frozen=True)
class StatusMeaning:
    """What one `Outcome.status` means to everything downstream of it."""

    label: str
    #: Did anything actually judge the solution? If not, this outcome is an
    #: absence of evidence and stays out of every rate: a zero here would be a
    #: published zero about somebody else's model.
    evidence: bool
    #: Is the run itself unusable because of it? Those exit non-zero. Missing
    #: hardware and missing optional dependencies are not in this class — they
    #: are expected limits of the machine, not defects.
    harness_failure: bool


#: The single classification of every status a graded task can end in.
#: Everything downstream derives from this table: the printed label, the pass
#: rates, and the exit code.
#:
#: It exists as one table because the same idea used to live in two places with
#: two different memberships, and the gap between them published a wrong number
#: twice — `docs/LESSONS.md` L11 and L20. A status that is not in here cannot be
#: constructed, so the next one added has to be classified on the way in.
STATUSES: dict[str, StatusMeaning] = {
    "passed": StatusMeaning("pass", evidence=True, harness_failure=False),
    "failed": StatusMeaning("FAIL", evidence=True, harness_failure=False),
    # The solution ran and did not finish inside the task's own limit. That is
    # the solution failing, not the harness, and it is the only status in this
    # table where "no answer" still counts as an answer.
    "timeout": StatusMeaning("TIMEOUT", evidence=True, harness_failure=False),
    # The solver's own files do not import: a syntax error, a NameError at
    # module scope, a missing import it never wrote. Nothing judged it, and
    # that is still a fact about the solution rather than about the harness —
    # `validate` proves the untouched starter imports cleanly, so an import
    # error during a graded run was introduced by whatever wrote the file.
    # Scoring it as an absence of evidence would drop it out of the
    # denominator, and the models that emit unparseable Python are exactly the
    # weak ones, so the bias runs in the direction that flatters them.
    "solution_error": StatusMeaning("NOIMPORT", evidence=True, harness_failure=False),
    # pytest could not collect for a reason the solution does not explain: a
    # broken hidden test, a fixture blowing up, a missing dependency of the
    # test module itself. Nothing judged the solution and nothing about the
    # solution caused it.
    "collection_error": StatusMeaning("BROKEN", evidence=False, harness_failure=True),
    # The adapter never produced a gradeable file: an API error, a refusal, a
    # truncated response, a substituted model.
    "adapter_error": StatusMeaning("ADAPTER", evidence=False, harness_failure=True),
    # The task declares dependencies this machine does not have.
    "missing_deps": StatusMeaning("SKIPPED", evidence=False, harness_failure=False),
    # The task needs an accelerator this machine does not have.
    "needs_accelerator": StatusMeaning("NO GPU", evidence=False, harness_failure=False),
}

#: Statuses that produced nothing to score. Kept out of every rate.
NO_EVIDENCE = frozenset(name for name, meaning in STATUSES.items() if not meaning.evidence)

#: Statuses that make the whole run unusable. A run containing one exits
#: non-zero: the number it produced is not a measurement of anything.
HARNESS_FAILURES = frozenset(
    name for name, meaning in STATUSES.items() if meaning.harness_failure
)


@dataclass
class Outcome:
    """What happened when one task was graded."""

    slug: str
    passed: bool
    status: str
    duration_s: float
    returncode: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    detail: str = ""
    output: str = ""

    def __post_init__(self) -> None:
        # The point of failing here is that a new status cannot reach the
        # reporting layer unclassified and be silently scored as a zero.
        if self.status not in STATUSES:
            raise ValueError(
                f"unclassified outcome status {self.status!r}: add it to "
                "runner.sandbox.STATUSES and decide whether it is evidence"
            )

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


def starter_collects_cleanly(task: Task) -> bool:
    """Do the hidden tests still collect and fail properly against the starter?

    Run only when a graded solution produced a collection error, to answer the
    one question that decides which side of the evidence line that outcome
    belongs on. The starter is the baseline the whole task contract is built
    on: `validate` requires it to fail every hidden test with real assertion
    errors. If it still does that here, the tests and the task are intact and
    the collection error came from whatever the solver wrote — a syntax error,
    a renamed function the tests import, a module-scope exception. If the
    starter fails to collect too, the task itself is broken and nothing about
    the solution has been measured.

    Asked as a question about the baseline rather than about the solution
    because there is no reliable way to read blame off pytest's exit code, and
    guessing wrong here either publishes a zero nobody earned or drops a real
    failure out of the denominator.
    """
    temporary = tempfile.mkdtemp(prefix=f"scratchbench-{task.slug}-baseline-")
    baseline = Path(temporary)
    try:
        assemble(task, task.starter_dir, baseline)
        add_hidden_tests(task, baseline)
        returncode, output, _ = run_pytest(baseline, task.time_limit_s)
        return returncode == PYTEST_TESTS_FAILED and not parse_summary(output).get("error")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def first_error_line(output: str) -> str:
    """The last `E   SomeError: ...` line pytest printed, for the results file."""
    lines = [line[1:].strip() for line in output.splitlines() if line.startswith("E ")]
    return lines[-1] if lines else ""


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
        detail = ""
        if returncode == PYTEST_ALL_PASSED:
            status = "passed"
        elif returncode == PYTEST_TESTS_FAILED and not counts.get("error"):
            status = "failed"
        else:
            # Nothing was judged. Two very different reasons look identical in
            # pytest's exit code, and they belong on opposite sides of the
            # evidence line, so the solution is asked to import on its own.
            if starter_collects_cleanly(task):
                status = "solution_error"
                detail = first_error_line(output) or (
                    "the tests could not run against this solution, and they "
                    "collect against the untouched starter"
                )
            else:
                status = "collection_error"
                detail = "the hidden tests do not collect against the starter either"

        return Outcome(
            slug=task.slug,
            passed=status == "passed",
            status=status,
            duration_s=elapsed,
            returncode=returncode,
            counts=counts,
            detail=detail,
            output=output,
        )
    finally:
        if keep_workdir is not None:
            destination = Path(keep_workdir) / task.slug
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(workdir, destination)
        shutil.rmtree(temporary, ignore_errors=True)
