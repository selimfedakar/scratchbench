"""Re-derive every `calibration:` block from the draws it was computed from.

`meta.yaml` says a model passed 15 of 15 draws of a task. Without the draws
themselves that is a claim, and this repository's argument is that a published
number should not have to be taken on trust — `tools/check_cost.py` exists for
exactly the same reason, one column over.

So the draws are checked in, and this walks them: for every task carrying a
calibration block, count the draws that produced evidence about it and how many
of them passed, and compare against what the metadata claims.

Two directories hold draws. `calibration/` is where a sweep run to calibrate a
task lands. `leaderboard/` is where a published row's draws live, and a task
calibrated out of a row it was already part of has its evidence there already —
copying those files into `calibration/` would put the same evidence in the
repository twice, with two chances to drift.

Draws with no evidence are not counted, on either side. An adapter that never
answered measured nothing about the model, so it belongs in neither the
numerator nor the denominator — the same rule the rates themselves follow, from
the same table (`STATUSES` in `runner/sandbox.py`).

Usage:

    python tools/check_calibration.py                 # every calibrated task
    python tools/check_calibration.py --repo /path

Exits non-zero if any block disagrees with its own evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runner.sandbox import NO_EVIDENCE  # noqa: E402
from runner.tasks import discover_tasks  # noqa: E402


def measured(*directories: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """`(model, slug) -> (passed, draws that produced evidence)`, over every directory."""
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for path in sorted(
        (path for directory in directories for path in directory.glob("*.json")),
        key=lambda path: path.name,
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"unreadable  {path.name}: {error}")
            continue
        model = payload.get("model")
        for entry in payload.get("tasks", []):
            if entry.get("status") in NO_EVIDENCE:
                continue
            counts = tally[(model, entry["slug"])]
            counts[0] += int(entry.get("passed", False))
            counts[1] += 1
    return {key: (value[0], value[1]) for key, value in tally.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO_ROOT), help="path to the checkout")
    arguments = parser.parse_args()

    repo = Path(arguments.repo).resolve()
    directories = [repo / "calibration", repo / "leaderboard"]
    missing = [directory for directory in directories if not directory.is_dir()]
    if missing:
        print(f"{', '.join(str(directory) for directory in missing)}: not a directory")
        return 1

    evidence = measured(*directories)
    tasks = [task for task in discover_tasks(repo / "tasks") if task.calibration]
    if not tasks:
        print("no task carries a calibration block; nothing to check")
        return 0

    failures = 0
    checked = 0
    for task in tasks:
        for entry in task.calibration:
            checked += 1
            passed, draws = evidence.get((entry.model, task.slug), (0, 0))
            claimed = f"{entry.passed}/{entry.draws}"
            found = f"{passed}/{draws}"
            if (passed, draws) == (entry.passed, entry.draws):
                print(f"ok        {task.slug:30s} {entry.model:20s} {found}")
            else:
                failures += 1
                print(
                    f"MISMATCH  {task.slug:30s} {entry.model:20s} "
                    f"claims {claimed}, the draws say {found}"
                )

    print()
    if failures:
        print(f"{failures} of {checked} calibration entries do not match their own draws")
        return 1
    draws = sum(len(list(directory.glob("*.json"))) for directory in directories)
    print(f"{checked} calibration entr{'y' if checked == 1 else 'ies'} re-derived from "
          f"{draws} draw(s) in calibration/ and leaderboard/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
