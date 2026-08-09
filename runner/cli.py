"""`scratchbench` — the command line.

    scratchbench list
    scratchbench validate [--tasks all]
    scratchbench run --model reference [--tasks all] [--repeat N]
    scratchbench report [--last | --variance]
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
    tasks = select_tasks(TASKS_ROOT, args.tasks, args.tier, args.set)
    rows = [
        [
            task.slug,
            task.category,
            task.tier if not task.accelerator else f"{task.tier} ({task.accelerator})",
            str(task.difficulty),
            f"{task.time_limit_s}s",
            ", ".join(task.deps) or "-",
            task.frozen_set,
            # Difficulty is an opinion until models have been asked; this is
            # the column that turns it into a measurement.
            " · ".join(str(entry) for entry in task.calibration) or "-",
            reporting._clip(task.probes, 50),
        ]
        for task in tasks
    ]
    print(
        reporting._table(
            ["task", "category", "tier", "diff", "limit", "deps", "set", "calibration", "probes"],
            rows,
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

    tasks = select_tasks(TASKS_ROOT, args.tasks, args.tier, args.set)
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
            # The starter is the baseline everything else is judged against, so
            # there is nothing to compare it to: this says only that the two
            # halves do not collect together, which is an import error rather
            # than a test failure and breaks the task either way.
            problems.append("starter and hidden tests do not collect: an import error, not a failure")
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


def _one_sweep(args, tasks, repeat: dict | None) -> tuple[dict, list[Outcome]]:
    """One complete pass over the task set, and the document describing it."""
    from adapters import resolve

    # Built per sweep rather than once: an adapter accumulates spend and token
    # counts, so a reused one would write the running total into every repeat's
    # results file and the last draw would look ten times as expensive as the
    # first.
    solve = resolve(args.model)

    # Kept workdirs are named after the task, so several draws of the same task
    # land on the same directory and only the last survives. The failure shape
    # is the thing repeated draws exist to show — 1 test out of 24 and 24 out of
    # 24 are the same verdict and completely different information — so each
    # draw gets its own directory rather than the right to overwrite the others.
    keep = args.keep
    if keep is not None and repeat is not None:
        keep = keep / f"draw{repeat['index']}"

    outcomes: list[Outcome] = []
    started = time.perf_counter()
    for task in tasks:
        print(f"  {task.slug} ... ", end="", flush=True)
        outcome = grade(task, solve, keep_workdir=keep)
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
        tokens=getattr(solve, "token_usage", None),
        repeat=repeat,
    )
    return payload, outcomes


def command_run(args: argparse.Namespace) -> int:
    if args.repeat < 1:
        print("scratchbench: --repeat must be at least 1", file=sys.stderr)
        return 2

    tasks = select_tasks(TASKS_ROOT, args.tasks, args.tier, args.set)

    # One identifier shared by every draw in this invocation, so the files can
    # be grouped afterwards without guessing from timestamps.
    group = f"{args.model}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    payloads: list[dict] = []
    failed = 0

    for index in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\ndraw {index} of {args.repeat}")
        repeat = (
            None
            if args.repeat == 1
            else {"group": group, "index": index, "of": args.repeat}
        )
        payload, outcomes = _one_sweep(args, tasks, repeat)
        payloads.append(payload)

        print()
        print(reporting.format_run(payload))

        if not args.no_write:
            path = reporting.write_payload(payload, args.results)
            written = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            print(f"\nwritten to {written}")

        harness_failures = [
            outcome for outcome in outcomes if outcome.status in reporting.HARNESS_FAILURES
        ]
        if harness_failures:
            failed += 1
            print(f"\n{len(harness_failures)} task(s) did not produce a measurement:")
            for outcome in harness_failures:
                print(f"  {outcome.slug}: {outcome.detail or outcome.status}")

    # The spread is the whole reason to repeat, so it is printed rather than
    # left for a second command to find.
    if args.repeat > 1:
        # `--tier all` has no single tier to spread, so the headline one is
        # what gets an error bar. The others are still in each results file.
        tier = args.tier if args.tier != "all" else reporting.HEADLINE_TIER
        print()
        print(reporting.format_variance(payloads, tier=tier))

    return 1 if failed else 0


def command_report(args: argparse.Namespace) -> int:
    payloads = reporting.load_payloads(args.results)
    if args.last:
        if not payloads:
            print("No results yet.")
            return 1
        print(reporting.format_run(payloads[-1]))
        return 0
    if args.variance:
        print(reporting.format_variance(payloads))
        return 0
    print(reporting.format_leaderboard(payloads))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scratchbench",
        description="Can the model actually implement it? ML tasks graded by hidden tests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_set(parser_: argparse.ArgumentParser) -> None:
        parser_.add_argument(
            "--set",
            default="all",
            help=(
                "which frozen set to include. Default 'all'. The laptop tier "
                "holds more than one published set now — `warmup` is where a "
                "task goes when every frontier model solves it — so a sweep "
                "over everything averages across them, and a leaderboard row "
                "wants one set."
            ),
        )

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
    add_set(listing)
    listing.set_defaults(func=command_list, tier="all")

    validate = subparsers.add_parser(
        "validate", help="check every task: reference passes, untouched starter fails cleanly"
    )
    validate.add_argument("--tasks", default="all")
    add_tier(validate)
    add_set(validate)
    validate.add_argument("--verbose", action="store_true", help="print pytest output for failures")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="grade a model against the task set")
    run.add_argument("--model", required=True, help="model id, or 'reference' for the control run")
    run.add_argument("--tasks", default="all")
    add_tier(run)
    add_set(run)
    run.add_argument("--results", type=Path, default=RESULTS_ROOT)
    run.add_argument("--no-write", action="store_true", help="do not write a results file")
    run.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "how many independent sweeps to run. Each one is a complete run "
            "with its own results file; together they are the error bar a "
            "single sweep cannot have, because the model is sampled and one "
            "run is one draw. This is not a retry: no draw sees another's "
            "results."
        ),
    )
    run.add_argument(
        "--keep", type=Path, default=None, help="copy each graded workdir here for inspection"
    )
    run.set_defaults(func=command_run)

    report = subparsers.add_parser("report", help="show results")
    report.add_argument("--results", type=Path, default=RESULTS_ROOT)
    report.add_argument("--last", action="store_true", help="show the most recent run in full")
    report.add_argument(
        "--variance",
        action="store_true",
        help="how many of N draws each task passed, per model and task set",
    )
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
