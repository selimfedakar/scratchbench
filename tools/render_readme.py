"""Render the README's derived counts from the task directories they describe.

The claims audit in session 18 found eight wrong numbers on the public pages,
and every one of them had been correct on the day it was typed: sixteen tasks
when there were seventeen, thirty-two calibration entries when there were
thirty-five, forty-four lessons when there were forty-six. Nothing was careless.
The README was holding derived values by hand while the values kept moving.

`tools/render_leaderboard.py` already solved this one directory over, and this
is the same mechanism pointed at `README.md`. Only the **derived** parts are
generated — counts and the task table. The prose is not, and must not be: the
prose is the part worth reading and no generator has an opinion about it.

    <!-- rendered: task-counts -->
    Seventeen tasks and 805 hidden tests...
    <!-- /rendered -->

Run it two ways:

    python tools/render_readme.py            # rewrite the file in place
    python tools/render_readme.py --check    # exit 1 if it would change

`--check` is what CI runs, so a stale count fails the build instead of sitting
on the front page for three sessions.

The hidden-test totals come from each task's `meta.yaml` rather than from
collecting the suites here, and the reason is worth stating because the first
version of this file got it wrong. Collecting means importing the task's
dependencies, and `fused_rmsnorm_kernel` imports triton, which has no macOS
wheel and is not installed on a CPU-only CI runner. So a collecting renderer
produced the right answer on some machines and crashed on others — which is
precisely the defect L47 is about, reintroduced by the tool written to prevent
its cousin. The count is checked in as evidence instead, and
`scratchbench validate` re-derives it on any machine that can actually run the
task.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.tasks import discover_tasks  # noqa: E402

PAGE = ROOT / "README.md"
TASKS_ROOT = ROOT / "tasks"

REGION = re.compile(
    r"(?P<open><!-- rendered: (?P<name>[a-z0-9-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close><!-- /rendered -->)",
    re.DOTALL,
)

NUMBER_WORDS = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    13: "Thirteen",
    14: "Fourteen",
    15: "Fifteen",
    16: "Sixteen",
    17: "Seventeen",
    18: "Eighteen",
    19: "Nineteen",
    20: "Twenty",
}


TENS = {
    2: "Twenty",
    3: "Thirty",
    4: "Forty",
    5: "Fifty",
    6: "Sixty",
    7: "Seventy",
    8: "Eighty",
    9: "Ninety",
}


def spell(n: int) -> str:
    """Numbers up to ninety-nine in words, as the rest of the page writes them.

    The page spells its counts out, so the check has to compare against the
    spelling rather than the digits, or it would pass on a sentence no reader
    would ever see.
    """
    if n in NUMBER_WORDS:
        return NUMBER_WORDS[n]
    if 20 <= n <= 99:
        tens, units = divmod(n, 10)
        if units == 0:
            return TENS[tens]
        return f"{TENS[tens]}-{NUMBER_WORDS[units].lower()}"
    return str(n)


def task_counts(tasks: list, counts: dict[str, int]) -> str:
    laptop = [t for t in tasks if t.tier == "laptop"]
    accelerated = [t for t in tasks if t.tier != "laptop"]
    total_tests = sum(counts.values())
    accelerators = sorted({t.accelerator for t in accelerated if t.accelerator})
    hardware = " and ".join(a.upper() if a == "cuda" else a.capitalize() for a in accelerators)
    return (
        f"{spell(len(tasks))} tasks and {total_tests} hidden tests. "
        f"{spell(len(laptop))} of them run on any laptop and need nothing but "
        f"Python; the other {spell(len(accelerated)).lower()} need a GPU "
        f"({hardware}), are reported beside the headline with the hardware each "
        f"one wants, and are never folded into it."
    )


def task_table(tasks: list, counts: dict[str, int]) -> str:
    """Every task, in the order the page reads them: by set, then difficulty."""
    set_order = {"v1": 0, "v2": 1, "warmup": 2, "unvalidated": 3}
    ordered = sorted(
        tasks,
        key=lambda t: (set_order.get(t.frozen_set, 9), t.difficulty, t.slug),
    )
    lines = [
        "| Task | Set | Category | Difficulty | Tests | Calibrated against |",
        "|---|---|---|:---:|---:|---|",
    ]
    for task in ordered:
        mark = " ⚡" if task.tier != "laptop" else ""
        if task.calibration:
            against = " · ".join(f"`{c.model}` {c.passed}/{c.draws}" for c in task.calibration)
        else:
            # A v1 task predates the calibration machinery. Saying "not
            # calibrated" is the honest cell: its difficulty is still the
            # author's opinion, which is what L21 is about.
            against = "not calibrated"
        lines.append(
            f"| `{task.slug}`{mark} | {task.frozen_set} | {task.category} "
            f"| {task.difficulty} | {counts[task.slug]} | {against} |"
        )
    return "\n".join(lines)


RENDERERS = {
    "task-counts": task_counts,
    "task-table": task_table,
}

#: Counts that live inside a sentence, where generating the whole sentence would
#: mean generating prose. They are checked instead of rendered: the tool says
#: what the number is, a human writes the sentence around it, and the build
#: fails if the two stop agreeing. `Forty-four` sat on the front page for three
#: sessions while the file held forty-six.
LESSONS = ROOT / "docs" / "LESSONS.md"
LESSON_HEADING = re.compile(r"^## L(\d+)", re.M)


def prose_counts() -> list[tuple[str, str]]:
    """`(what it counts, the phrase that must appear)` for each checked count."""
    lessons = len(LESSON_HEADING.findall(LESSONS.read_text()))
    return [(f"{LESSONS.name} entries", f"{spell(lessons)} entries")]


def check_prose(page: str) -> list[str]:
    return [
        f"README.md should say '{phrase}' ({what}) and does not"
        for what, phrase in prose_counts()
        if phrase not in page
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the rendered regions differ from the file",
    )
    args = parser.parse_args()

    current = PAGE.read_text()
    if not REGION.search(current):
        print(f"{PAGE}: no <!-- rendered: ... --> regions found", file=sys.stderr)
        return 1

    tasks = sorted(discover_tasks(TASKS_ROOT), key=lambda t: t.slug)
    counts = {task.slug: task.hidden_tests for task in tasks}

    names = [match.group("name") for match in REGION.finditer(current)]
    unknown = [name for name in names if name not in RENDERERS]
    if unknown:
        print(f"{PAGE}: no renderer for region(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    def replace(match: re.Match) -> str:
        body = RENDERERS[match.group("name")](tasks, counts)
        return f"{match.group('open')}{body}\n{match.group('close')}"

    updated = REGION.sub(replace, current)

    prose_problems = check_prose(updated)
    if prose_problems:
        for problem in prose_problems:
            print(problem, file=sys.stderr)
        return 1

    if args.check:
        if updated != current:
            print(
                f"{PAGE} is out of date with the task directories behind it.\n"
                "Run: python tools/render_readme.py",
                file=sys.stderr,
            )
            return 1
        print(f"{len(names)} rendered region(s) match the task directories.")
        return 0

    PAGE.write_text(updated)
    print(f"{len(names)} rendered region(s) written: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
