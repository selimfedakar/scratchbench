"""Results: the JSON that gets written and the table that gets printed."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .sandbox import Outcome
from .tasks import Task

RESULTS_VERSION = 1


def build_payload(
    model: str,
    tasks: list[Task],
    outcomes: list[Outcome],
    wall_clock_s: float,
    attempts: int = 1,
    usd_cost: float | None = None,
) -> dict:
    """Assemble the result document for one run."""
    by_slug = {task.slug: task for task in tasks}
    # A task the machine had no hardware for produced no evidence. It is not a
    # failure and it is not a pass, so it is kept out of every rate rather than
    # rounded down to zero — a 0% next to "not run" is still a 0% to whoever
    # reads the table.
    measured = [outcome for outcome in outcomes if outcome.status != "needs_accelerator"]
    passed = [outcome for outcome in measured if outcome.passed]

    by_category: dict[str, dict[str, int]] = {}
    frozen_sets = sorted({task.frozen_set for task in tasks})
    for outcome in measured:
        category = by_slug[outcome.slug].category
        bucket = by_category.setdefault(category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(outcome.passed)

    return {
        "results_version": RESULTS_VERSION,
        "model": model,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_set": frozen_sets[0] if len(frozen_sets) == 1 else frozen_sets,
        "pass_rate": len(passed) / len(measured) if measured else None,
        "measured": len(measured),
        "not_measured": len(outcomes) - len(measured),
        "pass_rate_by_category": {
            category: bucket["passed"] / bucket["total"]
            for category, bucket in sorted(by_category.items())
        },
        # Reported separately and never folded together: the laptop rate is the
        # one anyone can reproduce, and mixing a GPU tier into it would quietly
        # break the claim the whole repository rests on.
        "pass_rate_by_tier": _by_tier(by_slug, outcomes),
        "attempts": attempts,
        "wall_clock_s": round(wall_clock_s, 2),
        "usd_cost": usd_cost,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "tasks": [
            {
                "slug": outcome.slug,
                "category": by_slug[outcome.slug].category,
                "tier": by_slug[outcome.slug].tier,
                "difficulty": by_slug[outcome.slug].difficulty,
                "task_version": by_slug[outcome.slug].version,
                "passed": outcome.passed,
                "status": outcome.status,
                "duration_s": round(outcome.duration_s, 2),
                "counts": outcome.counts,
                "detail": outcome.detail,
            }
            for outcome in outcomes
        ],
    }


def _by_tier(by_slug: dict, outcomes: list[Outcome]) -> dict:
    """Pass rate per tier, plus how many tasks had no hardware to run on."""
    tiers: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        bucket = tiers.setdefault(
            by_slug[outcome.slug].tier, {"passed": 0, "total": 0, "no_accelerator": 0}
        )
        bucket["total"] += 1
        bucket["passed"] += int(outcome.passed)
        bucket["no_accelerator"] += int(outcome.status == "needs_accelerator")
    rates = {}
    for tier, bucket in sorted(tiers.items()):
        measured = bucket["total"] - bucket["no_accelerator"]
        rates[tier] = {
            # None, not zero: a tier where nothing ran has no pass rate, and
            # writing one down would invent a measurement.
            "pass_rate": bucket["passed"] / measured if measured else None,
            "solved": bucket["passed"],
            "total": bucket["total"],
            "no_accelerator": bucket["no_accelerator"],
        }
    return rates


def write_payload(payload: dict, results_dir: Path) -> Path:
    """Write one run to `results/<model>-<timestamp>.json`."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["run_at"].replace(":", "").replace("-", "")
    safe_model = payload["model"].replace("/", "_")
    path = results_dir / f"{safe_model}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_payloads(results_dir: Path) -> list[dict]:
    """Every result document in `results_dir`, oldest first."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sorted(payloads, key=lambda payload: payload.get("run_at", ""))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [line(headers), "  ".join("-" * width for width in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


MARKS = {
    "passed": "pass",
    "failed": "FAIL",
    "timeout": "TIMEOUT",
    "collection_error": "BROKEN",
    "missing_deps": "SKIPPED",
    "adapter_error": "ADAPTER",
    "needs_accelerator": "NO GPU",
}

#: Statuses that mean the harness failed rather than the model. A run
#: containing any of these exits non-zero: the number it produced is not a
#: measurement of anything.
HARNESS_FAILURES = frozenset({"collection_error", "adapter_error"})


def format_run(payload: dict) -> str:
    """One run, task by task."""
    rows = [
        [
            task["slug"],
            task["category"],
            task.get("tier", "laptop"),
            str(task["difficulty"]),
            MARKS.get(task["status"], task["status"]),
            f"{task['duration_s']:.2f}s",
            _clip(task["detail"] or _counts(task["counts"])),
        ]
        for task in payload["tasks"]
    ]
    header = f"{payload['model']}  ·  {payload['run_at']}  ·  task set {payload['task_set']}"
    table = _table(["task", "category", "tier", "diff", "result", "time", "tests"], rows)

    lines = [header, "", table, ""]

    # The tiers are reported one per line rather than summed. The laptop rate
    # is the reproducible one; presenting a single blended number would hide
    # which half of it anyone can check.
    for tier, stats in payload.get("pass_rate_by_tier", {}).items():
        if stats["pass_rate"] is None:
            lines.append(
                f"{tier:12s} not run    "
                f"({stats['total']} task(s): no accelerator on this machine)"
            )
            continue
        measured = stats["total"] - stats["no_accelerator"]
        line = f"{tier:12s} {stats['pass_rate']:.0%} ({stats['solved']}/{measured})"
        if stats["no_accelerator"]:
            line += f"  ·  {stats['no_accelerator']} not run: no accelerator on this machine"
        lines.append(line)

    # Attempts and cost sit next to the pass rate on purpose: a model that
    # solves the set on its eleventh try, or for eleven dollars, is a different
    # tool from one that solves it on the first for two.
    footer = f"{payload['wall_clock_s']:.1f}s wall clock"
    footer += f"  ·  {payload.get('attempts', 1)} attempt(s) per task"
    if payload.get("usd_cost") is not None:
        footer += f"  ·  ${payload['usd_cost']:.2f}"
    lines.append(footer)
    lines.append(
        "  ".join(
            f"{category} {rate:.0%}"
            for category, rate in payload["pass_rate_by_category"].items()
        )
    )
    return "\n".join(lines)


def _clip(text: str, width: int = 58) -> str:
    """Keep the table readable in an eighty-column terminal."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def _counts(counts: dict) -> str:
    if not counts:
        return ""
    return ", ".join(f"{value} {key}" for key, value in sorted(counts.items()))


def format_leaderboard(payloads: list[dict]) -> str:
    """Every run, most recent first, one row each."""
    if not payloads:
        return "No results yet. Run `scratchbench run --model reference --tasks all` first."

    rows = []
    for payload in reversed(payloads):
        tasks = payload["tasks"]
        passed = sum(1 for task in tasks if task["passed"])
        measured = payload.get("measured", len(tasks))
        rate = payload.get("pass_rate")
        rows.append(
            [
                payload["model"],
                "-" if rate is None else f"{rate:.0%}",
                f"{passed}/{measured}",
                f"{payload['wall_clock_s']:.1f}s",
                "-" if payload.get("usd_cost") is None else f"${payload['usd_cost']:.2f}",
                str(payload.get("task_set", "?")),
                payload["run_at"],
            ]
        )
    return _table(
        ["model", "pass rate", "solved", "wall clock", "cost", "set", "run at"], rows
    )
