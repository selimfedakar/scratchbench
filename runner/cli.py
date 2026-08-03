"""`scratchbench` — the command line.

    scratchbench list
    scratchbench validate [--tasks all]
    scratchbench run --model reference [--tasks all]
    scratchbench report [--last]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import report as reporting
from .sandbox import Outcome, grade
from .tasks import TaskError, select_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
RESULTS_ROOT = REPO_ROOT / "results"


def _starter_solve(task, workdir) -> None:
    """Leave the starter exactly as it is. Used by `validate`."""


def command_list(args: argparse.Namespace) -> int:
    tasks = select_tasks(TASKS_ROOT, args.tasks, args.tier)
    rows = [
        [
            task.slug,
            task.category,
            task.tier if not task.accelerator else f"{task.tier} ({task.accelerator})",
            str(task.difficulty),
            f"{task.time_limit_s}s",
            ", ".join(task.deps) or "-",
            reporting._clip(task.probes, 60),
        ]
        for task in tasks
    ]
    print(
        reporting._table(
            ["task", "category", "tier", "diff", "limit", "deps", "probes"], rows
        )
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Both halves of the task contract, for every task.

    The reference must pass the hidden tests, and the untouched starter must
    fail them — with real test failures, not an import error. An import error
    and a wrong answer score the same and mean completely different things, so
    this is checked rather than assumed.
    """
    from adapters.reference import solve as reference_solve

    tasks = select_tasks(TASKS_ROOT, args.tasks, args.tier)
    rows: list[list[str]] = []
    broken = 0
    checked = 0
    unchecked: list[str] = []

    for task in tasks:
        reference = grade(task, reference_solve)
        starter = grade(task, _starter_solve)

        if reference.status == "needs_accelerator":
            # Unvalidatable here, and saying "ok" would be a lie. It is listed
            # so the gap stays visible rather than vanishing from the table.
            rows.append([task.slug, "-", "-", "-", f"UNCHECKED  {reference.detail}"])
            unchecked.append(task.slug)
            continue
        checked += 1

        problems = []
        if not reference.passed:
            problems.append(f"reference does not pass ({reference.summary()})")
        if starter.passed:
            problems.append("untouched starter passes — the tests prove nothing")
        elif starter.status == "collection_error":
            problems.append("starter fails to import rather than fails a test")
        elif starter.status == "timeout":
            problems.append("starter timed out instead of failing")

        if reference.status == "missing_deps":
            verdict = f"SKIPPED  {reference.detail}"
        elif problems:
            verdict = "BROKEN   " + "; ".join(problems)
            broken += 1
        else:
            verdict = "ok"

        rows.append(
            [
                task.slug,
                f"{reference.summary()}",
                f"{starter.summary()}",
                f"{reference.duration_s:.2f}s",
                verdict,
            ]
        )
        if problems and args.verbose:
            print(f"\n--- {task.slug}: reference output ---\n{reference.output}", file=sys.stderr)
            print(f"\n--- {task.slug}: starter output ---\n{starter.output}", file=sys.stderr)

    print(reporting._table(["task", "reference", "untouched starter", "time", "verdict"], rows))
    print()
    if broken:
        print(f"{broken} task(s) broken.")
        return 1
    # Counted, not inferred from the number of rows: a task this machine could
    # not run is in the table but has not been validated, and rolling it into
    # the total would be the report claiming evidence it does not have.
    print(f"{checked} task(s) validated: reference passes, starter fails cleanly.")
    if unchecked:
        print(f"{len(unchecked)} task(s) not checked here: {', '.join(unchecked)}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    from adapters import resolve

    tasks = select_tasks(TASKS_ROOT, args.tasks, args.tier)
    solve = resolve(args.model)

    outcomes: list[Outcome] = []
    started = time.perf_counter()
    for task in tasks:
        print(f"  {task.slug} ... ", end="", flush=True)
        outcome = grade(task, solve, keep_workdir=args.keep)
        outcomes.append(outcome)
        mark = reporting.MARKS.get(outcome.status, outcome.status)
        print(f"{mark}  ({reporting._clip(outcome.summary())})")
    wall_clock = time.perf_counter() - started

    # Both come off the adapter rather than off a flag: a pass on attempt
    # eleven is not the same result as a pass on attempt one, and a sweep that
    # cost forty dollars is not the same tool as one that cost two. The
    # reference solver is a plain function with neither attribute, and its
    # control run is correctly recorded as one attempt at no cost.
    payload = reporting.build_payload(
        args.model,
        tasks,
        outcomes,
        wall_clock,
        attempts=getattr(solve, "max_attempts", 1),
        usd_cost=getattr(solve, "usd_cost", None),
    )
    print()
    print(reporting.format_run(payload))

    if not args.no_write:
        path = reporting.write_payload(payload, args.results)
        print(f"\nwritten to {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")

    harness_failures = [
        outcome for outcome in outcomes if outcome.status in reporting.HARNESS_FAILURES
    ]
    if harness_failures:
        print(f"\n{len(harness_failures)} task(s) did not produce a measurement:")
        for outcome in harness_failures:
            print(f"  {outcome.slug}: {outcome.detail or outcome.status}")
        return 1
    return 0


def command_report(args: argparse.Namespace) -> int:
    payloads = reporting.load_payloads(args.results)
    if args.last:
        if not payloads:
            print("No results yet.")
            return 1
        print(reporting.format_run(payloads[-1]))
        return 0
    print(reporting.format_leaderboard(payloads))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scratchbench",
        description="Can the model actually implement it? ML tasks graded by hidden tests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_tier(parser_: argparse.ArgumentParser) -> None:
        parser_.add_argument(
            "--tier",
            default="laptop",
            choices=["laptop", "accelerated", "all"],
            help=(
                "which tier to include. Default 'laptop': no GPU, reproducible "
                "anywhere, and the only tier the leaderboard rate is computed over. "
                "Naming tasks explicitly with --tasks overrides this."
            ),
        )

    listing = subparsers.add_parser("list", help="show the task set")
    listing.add_argument("--tasks", default="all", help="'all' or a comma-separated list of slugs")
    add_tier(listing)
    listing.set_defaults(func=command_list, tier="all")

    validate = subparsers.add_parser(
        "validate", help="check every task: reference passes, untouched starter fails cleanly"
    )
    validate.add_argument("--tasks", default="all")
    add_tier(validate)
    validate.add_argument("--verbose", action="store_true", help="print pytest output for failures")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="grade a model against the task set")
    run.add_argument("--model", required=True, help="model id, or 'reference' for the control run")
    run.add_argument("--tasks", default="all")
    add_tier(run)
    run.add_argument("--results", type=Path, default=RESULTS_ROOT)
    run.add_argument("--no-write", action="store_true", help="do not write a results file")
    run.add_argument(
        "--keep", type=Path, default=None, help="copy each graded workdir here for inspection"
    )
    run.set_defaults(func=command_run)

    report = subparsers.add_parser("report", help="show results")
    report.add_argument("--results", type=Path, default=RESULTS_ROOT)
    report.add_argument("--last", action="store_true", help="show the most recent run in full")
    report.set_defaults(func=command_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TaskError as error:
        print(f"scratchbench: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"scratchbench: {error.args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
