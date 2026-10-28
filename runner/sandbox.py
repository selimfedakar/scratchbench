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

from .mps_stall import GUARD_ENV, MAX_ATTEMPTS, STALL_SENTINEL, STALLED_EXIT
from .tasks import Task, available_accelerators

#: The name the stall guard is copied into a graded workdir under. Distinctive
#: enough that no solution and no hidden test module can collide with it.
STALL_GUARD_MODULE = "_scratchbench_mps_stall"

#: Accelerators whose runs go through the guard. Only Metal so far, because the
#: stall is a measured property of this machine and nothing says the rented CUDA
#: box has anything like it — arming the guard there would be an assumption
#: wearing a measurement's clothes.
GUARDED_ACCELERATORS = frozenset({"metal"})

# The two exit codes that mean the suite actually ran and reached a verdict.
# Everything else — a collection error, no tests found, an internal error —
# means nothing judged the solution, and which side of the evidence line that
# lands on is decided by `starter_collects_cleanly` rather than by the code.
PYTEST_ALL_PASSED = 0
PYTEST_TESTS_FAILED = 1

SUMMARY_PATTERN = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")

#: pytest's short test summary, one line per failure: `FAILED <nodeid> - <why>`.
#: `-q` and `--tb=short` both leave it on, so the names arrive without asking
#: for anything new. `ERROR` lines are deliberately not matched — an error means
#: nothing judged the solution, and a list of names would read as if something
#: had (`STATUSES`, and `docs/LESSONS.md` L20).
FAILED_PATTERN = re.compile(r"^FAILED (\S+)", re.MULTILINE)


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
    # Every attempt landed in a process where MPS is a hundred times slower for
    # the life of that process, so nothing judged the solution. This is the
    # status that exists so the alternative does not: without it a stalled run
    # is recorded `timeout`, which is evidence and a failure, and the failure
    # shape this repository publishes would be describing the machine.
    # `runner/mps_stall.py` and `docs/LESSONS.md` L42.
    "mps_stalled": StatusMeaning("STALLED", evidence=False, harness_failure=True),
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
    #: Node ids of the failing tests. Empty whenever nothing judged the
    #: solution, so an empty list never has to be read as "nothing failed".
    failed_tests: list[str] = field(default_factory=list)
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


def parse_failed_tests(stdout: str) -> list[str]:
    """The node ids of the tests that failed, in the order pytest reported them.

    A count is not a shape. `9/10` says a model lost one draw and says nothing
    about whether it lost it to the mechanism the task is named after or to a
    tensor it forgot to reshape, and those are different findings about
    different models (`docs/LESSONS.md` L27, and L45, which is the session that
    established this by hand). The names are already on stdout; all that was
    missing was keeping them.

    Duplicates are dropped and order is preserved, because a parametrised test
    that fails under several ids is several results and a repeated id is not.
    """
    seen: set[str] = set()
    names: list[str] = []
    for nodeid in FAILED_PATTERN.findall(stdout):
        if nodeid not in seen:
            seen.add(nodeid)
            names.append(nodeid)
    return names


def assemble(task: Task, solution_dir: Path, workdir: Path) -> None:
    """Copy the solution files for `task` into a flat `workdir`."""
    workdir.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(solution_dir).glob("*.py")):
        shutil.copy2(source, workdir / source.name)


def add_hidden_tests(task: Task, workdir: Path) -> None:
    """Drop the hidden tests in beside the solution, once it is written."""
    for source in task.hidden_test_files():
        shutil.copy2(source, workdir / source.name)


def add_stall_guard(task: Task, workdir: Path) -> bool:
    """Copy the MPS stall guard in beside the hidden tests, if this task needs it.

    Copied at the same moment and for the same reason as the hidden tests: a
    solver cannot subvert a file that is not on disk while it is writing.
    """
    if task.accelerator not in GUARDED_ACCELERATORS:
        return False
    source = Path(__file__).with_name("mps_stall.py")
    shutil.copy2(source, workdir / f"{STALL_GUARD_MODULE}.py")
    return True


#: Substrings (case-insensitive) that mark an environment variable as a
#: credential. The graded child has no business reading any of them: the model's
#: answer is written by the adapter in the *parent* process, where the keys live
#: and are needed, and the child only runs pytest over a file that is already on
#: disk. Stripping these keeps a generated solution from reading the key that
#: paid for it. This is the cheap half of F02, and it is not a security boundary
#: — a solution runs in the same interpreter as the grader and a denylist cannot
#: see a credential whose name does not say so. The honest limit is in `grade`'s
#: docstring and in the README; this only removes the one secret the harness
#: itself introduces.
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def is_secret_env(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_ENV_MARKERS)


def pytest_environment(guarded: bool = False) -> dict[str, str]:
    """A deterministic, credential-stripped environment for the graded run.

    Deterministic because the thread and hash seeds are pinned; credential-
    stripped because anything that looks like a key is removed before the child
    sees it. It is not an isolated environment: the child still inherits the
    rest of the parent's variables and the parent's filesystem access, and the
    README says so plainly. Real isolation is a disposable container or VM, which
    is its own work rather than a line here (docs/LESSONS.md F02).
    """
    env = {name: value for name, value in os.environ.items() if not is_secret_env(name)}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Thread counts change floating-point reduction order in numpy and torch,
    # so a graded run pins them and stays reproducible across machines.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("PYTHONPATH", None)
    if guarded:
        env[GUARD_ENV] = "1"
    else:
        env.pop(GUARD_ENV, None)
    return env


def stall_fired(workdir: Path) -> bool:
    """Did the stall guard write its sentinel into `workdir` on the last run?

    This is what separates `mps_stalled` — a stalled process the guard refused
    to grade in — from a solution that merely called `os._exit(125)` on its own.
    Both produce the same exit code; only the guard writes the file, and only on
    a Metal task where the guard is armed (docs/LESSONS.md L50, F06).
    """
    return (workdir / STALL_SENTINEL).exists()


def run_pytest(
    workdir: Path, timeout_s: int, guarded: bool = False
) -> tuple[int | None, str, float]:
    """Run the tests in `workdir`. Returns (returncode, output, seconds)."""
    # Clear any sentinel a previous attempt left, so `stall_fired` reflects this
    # run and not the last one.
    (workdir / STALL_SENTINEL).unlink(missing_ok=True)
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
    if guarded:
        command += ["-p", STALL_GUARD_MODULE]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=pytest_environment(guarded),
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


def run_pytest_until_healthy(
    workdir: Path, timeout_s: int, guarded: bool
) -> tuple[int | None, str, float, int]:
    """Run the tests, relaunching while the child reports itself stalled.

    Returns (returncode, output, seconds, attempts). The seconds are the sum
    across attempts, because that is what the run actually cost; a discarded
    attempt is a few milliseconds, since the guard fires before the first test.

    When every attempt stalls the returncode stays `STALLED_EXIT` and the caller
    turns it into a status. Retrying rather than reporting on the first stall is
    the whole point: launches alternate, so the second attempt is almost always
    healthy, and a run that silently costs one extra process is much cheaper
    than a leaderboard row that describes the machine.
    """
    total = 0.0
    attempts = 0
    returncode: int | None = STALLED_EXIT
    output = ""
    for attempts in range(1, (MAX_ATTEMPTS if guarded else 1) + 1):
        returncode, output, elapsed = run_pytest(workdir, timeout_s, guarded)
        total += elapsed
        # Relaunch only for a stall the guard actually reported. A bare 125 from
        # the solution itself is its own answer, not a reason to spend three more
        # processes looking for a healthy one.
        if not (returncode == STALLED_EXIT and stall_fired(workdir)):
            break
    return returncode, output, total, attempts


def starter_collects_cleanly(task: Task) -> bool | None:
    """Do the hidden tests still collect and fail properly against the starter?

    Three answers, not two. `None` means the baseline itself never ran — every
    attempt landed in a stalled MPS process — so it says nothing about the task
    either way. Collapsing that into `False` published the machine's state as a
    verdict on the task: status `collection_error`, printed label `BROKEN`,
    which is the one alarm here that must not cry wolf, because it means every
    number ever measured against the task is void (docs/LESSONS.md L42, L48).

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
        guarded = add_stall_guard(task, baseline)
        returncode, output, _, attempts = run_pytest_until_healthy(
            baseline, task.time_limit_s, guarded
        )
        if guarded and returncode == STALLED_EXIT and stall_fired(baseline):
            return None
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

    Trust boundary, stated plainly because the comments elsewhere imply more
    than is true. The solution is arbitrary Python and pytest runs it in the
    same interpreter as the tests, in a subprocess that inherits this machine's
    filesystem and most of its environment. Credential-shaped variables are
    stripped (`pytest_environment`), and the hidden tests are only copied in
    after the solver has written, so a solver cannot read a test that is not yet
    on disk — but neither of those is a security sandbox. There is no network
    block, no memory limit, and no filesystem jail: generated code can read
    outside the workdir and reach the network. Run a model's output the way you
    would run any untrusted code — on a machine or in a disposable container you
    are willing to hand it — until the isolated grading path F02 calls for
    exists. The README's "Running generated code" note says the same thing.
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
        guarded = add_stall_guard(task, workdir)

        returncode, output, elapsed, attempts = run_pytest_until_healthy(
            workdir, task.time_limit_s, guarded
        )

        if guarded and returncode == STALLED_EXIT and stall_fired(workdir):
            return Outcome(
                slug=task.slug,
                passed=False,
                status="mps_stalled",
                duration_s=elapsed,
                returncode=returncode,
                detail=(
                    f"every one of {attempts} attempts landed in a stalled MPS "
                    "process; nothing judged the solution"
                ),
                output=output,
            )

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
        # A verdict means the suite ran every hidden test and reached the end.
        # The exit code does not prove that on its own: a solution that calls
        # os._exit(0) while its module is imported kills the process with a
        # success code having judged nothing, and `0 passed` would otherwise be
        # read as a pass. `hidden_tests` is the declared count, re-derived by
        # `validate` on hardware that can run the task, so comparing the graded
        # run against it turns "the process exited 0" into "every test ran and
        # passed" (docs/LESSONS.md L50, F01).
        judged = counts.get("passed", 0) + counts.get("failed", 0)
        complete = (
            returncode in (PYTEST_ALL_PASSED, PYTEST_TESTS_FAILED)
            and not counts.get("error")
            and judged == task.hidden_tests
        )
        if complete and returncode == PYTEST_ALL_PASSED:
            status = "passed"
        elif complete and returncode == PYTEST_TESTS_FAILED:
            status = "failed"
        else:
            # Nothing was judged to completion. Several very different reasons
            # look identical in pytest's exit code — a collection error, an
            # early exit, a suite that ran a fraction of its tests — and they
            # belong on opposite sides of the evidence line, so the solution is
            # asked to import on its own.
            baseline = starter_collects_cleanly(task)
            if baseline is None:
                # The baseline never ran, so it accused nobody. Not the task,
                # and not the solution either.
                status = "mps_stalled"
                detail = (
                    "the solution produced no gradeable run, and every attempt to "
                    "check the baseline landed in a stalled MPS process"
                )
            elif baseline:
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
            # Only when the tests judged the solution. A `solution_error` run
            # can still print `FAILED` lines before the error that stopped it,
            # and publishing those beside a status that means "nothing was
            # judged" would give a shape to an absence of evidence.
            failed_tests=parse_failed_tests(output) if status == "failed" else [],
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
