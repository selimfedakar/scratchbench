"""Results: the JSON that gets written and the table that gets printed."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__ as HARNESS_VERSION
from .sandbox import HARNESS_FAILURES, NO_EVIDENCE, STATUSES, Outcome
from .tasks import Task

#: Bumped to 2 when the headline `pass_rate` stopped being computed across
#: every tier. A version 1 file is still readable here and still means what it
#: said: every published version 1 run was laptop-only, so its headline rate
#: and its laptop rate are the same number.
RESULTS_VERSION = 2

#: The tier the headline `pass_rate` is computed over, and the only one it is
#: ever computed over. Every document in this repository says the leaderboard
#: rate is the reproducible tier; this is that sentence in code, in the one
#: place the number is produced.
HEADLINE_TIER = "laptop"

__all__ = [
    "HARNESS_FAILURES",
    "HEADLINE_TIER",
    "NO_EVIDENCE",
    "RESULTS_VERSION",
    "build_payload",
]


def build_payload(
    model: str,
    tasks: list[Task],
    outcomes: list[Outcome],
    wall_clock_s: float,
    attempts: int = 1,
    usd_cost: float | None = None,
    tokens: dict | None = None,
    repeat: dict | None = None,
) -> dict:
    """Assemble the result document for one run."""
    by_slug = {task.slug: task for task in tasks}
    # An outcome in NO_EVIDENCE produced nothing to score. It is not a failure
    # and it is not a pass, so it is kept out of every rate rather than rounded
    # down to zero — a 0% next to "not run" is still a 0% to whoever reads the
    # table, and it is a 0% about somebody else's model.
    measured = [outcome for outcome in outcomes if outcome.status not in NO_EVIDENCE]

    # The headline rate is one tier's rate. Averaging a laptop tier and an
    # accelerated tier into a single percentage is the same failure as rounding
    # an absence of evidence to zero: the resulting number is not a measurement
    # of anything anyone can reproduce, and it is printed in the column readers
    # actually look at. It could not fire while no machine here had a GPU,
    # which is exactly the shape of L11 and L20.
    headline = [
        outcome for outcome in measured if by_slug[outcome.slug].tier == HEADLINE_TIER
    ]
    headline_passed = [outcome for outcome in headline if outcome.passed]

    by_category: dict[str, dict[str, int]] = {}
    frozen_sets = sorted({task.frozen_set for task in tasks})
    for outcome in headline:
        category = by_slug[outcome.slug].category
        bucket = by_category.setdefault(category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(outcome.passed)

    return {
        "results_version": RESULTS_VERSION,
        "harness_version": HARNESS_VERSION,
        "model": model,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_set": frozen_sets[0] if len(frozen_sets) == 1 else frozen_sets,
        # Named rather than implied, so a reader never has to work out which
        # tasks are behind the percentage.
        "headline_tier": HEADLINE_TIER,
        "pass_rate": len(headline_passed) / len(headline) if headline else None,
        # These two count the whole run, not the headline tier: they exist to
        # say how much of what was attempted produced evidence at all.
        "measured": len(measured),
        "not_measured": len(outcomes) - len(measured),
        # Per category within the headline tier, for the same reason the
        # headline rate is one tier: `kernels` is accelerated-only today, and a
        # category rate that silently spans tiers is the same blend one level
        # further down.
        "pass_rate_by_category": {
            category: bucket["passed"] / bucket["total"]
            for category, bucket in sorted(by_category.items())
        },
        # Reported separately and never folded together: the laptop rate is the
        # one anyone can reproduce, and mixing a GPU tier into it would quietly
        # break the claim the whole repository rests on.
        "pass_rate_by_tier": _by_tier(by_slug, outcomes),
        "attempts": attempts,
        # Which draw this is, when a sweep was repeated. Each repeat is a
        # complete, independently valid run — the group only says that several
        # of them were asked the same question, which is what an error bar
        # needs and what a single row cannot give (L19).
        "repeat": repeat,
        "wall_clock_s": round(wall_clock_s, 2),
        "usd_cost": usd_cost,
        # The tokens the cost was computed from. Without them `usd_cost` is a
        # number nobody can check, including me: a price table that has drifted
        # and a run that used more tokens than expected produce the same figure
        # and there is no way to tell them apart afterwards.
        "tokens": tokens,
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
    """Pass rate per tier, plus how many tasks produced no evidence and why.

    Every tier the run touched appears here, including the ones the headline
    rate deliberately leaves out. That is the point of the split: the
    accelerated tier is reported in full, beside the laptop rate rather than
    inside it, so dropping it out of the headline hides nothing.
    """
    tiers: dict[str, dict] = {}
    for outcome in outcomes:
        task = by_slug[outcome.slug]
        bucket = tiers.setdefault(
            task.tier,
            {
                "passed": 0,
                "total": 0,
                "no_accelerator": 0,
                "no_evidence": 0,
                "categories": {},
            },
        )
        bucket["total"] += 1
        bucket["passed"] += int(outcome.passed)
        bucket["no_accelerator"] += int(outcome.status == "needs_accelerator")
        bucket["no_evidence"] += int(outcome.status in NO_EVIDENCE)
        if outcome.status not in NO_EVIDENCE:
            category = bucket["categories"].setdefault(
                task.category, {"passed": 0, "total": 0}
            )
            category["total"] += 1
            category["passed"] += int(outcome.passed)
    rates = {}
    for tier, bucket in sorted(tiers.items()):
        measured = bucket["total"] - bucket["no_evidence"]
        rates[tier] = {
            # None, not zero: a tier where nothing was measured has no pass
            # rate, and writing one down would invent a measurement.
            "pass_rate": bucket["passed"] / measured if measured else None,
            "solved": bucket["passed"],
            "measured": measured,
            "total": bucket["total"],
            "no_accelerator": bucket["no_accelerator"],
            "no_evidence": bucket["no_evidence"],
            "categories": {
                name: counts["passed"] / counts["total"]
                for name, counts in sorted(bucket["categories"].items())
            },
        }
    return rates


def write_payload(payload: dict, results_dir: Path) -> Path:
    """Write one run to `results/<model>-<timestamp>.json`.

    The timestamp has one-second resolution and a sweep can finish inside a
    second, so a repeated run would otherwise write every draw to the same
    filename and keep only the last one — a variance measurement silently
    reduced to a single sample. The draw's index disambiguates, and a counter
    catches anything else that lands on an occupied name.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["run_at"].replace(":", "").replace("-", "")
    safe_model = payload["model"].replace("/", "_")
    repeat = payload.get("repeat")
    draw = f"-draw{repeat['index']}" if repeat else ""

    path = results_dir / f"{safe_model}-{stamp}{draw}.json"
    suffix = 2
    while path.exists():
        path = results_dir / f"{safe_model}-{stamp}{draw}-{suffix}.json"
        suffix += 1

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


#: How each status prints. Derived from the one table that classifies them, so
#: a new status cannot appear in a report without having been classified.
MARKS = {name: meaning.label for name, meaning in STATUSES.items()}


def _unmeasured_reason(no_accelerator: int, no_evidence: int) -> str:
    """Why a tier has tasks with no evidence, without guessing at the reason."""
    if no_evidence and no_accelerator == no_evidence:
        return "no accelerator on this machine"
    if no_accelerator:
        return f"{no_accelerator} with no accelerator, the rest produced no result"
    return "no result was produced; see the per-task detail"


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
        # Two different absences, and saying the wrong one is worse than saying
        # nothing: a sweep that died on an API key once printed "no accelerator
        # on this machine" and blamed the hardware for it.
        no_accelerator = stats.get("no_accelerator", 0)
        no_evidence = stats.get("no_evidence", no_accelerator)
        unmeasured = _unmeasured_reason(no_accelerator, no_evidence)
        if stats["pass_rate"] is None:
            lines.append(f"{tier:12s} not run    ({stats['total']} task(s): {unmeasured})")
            continue
        measured = stats.get("measured", stats["total"] - no_evidence)
        line = f"{tier:12s} {stats['pass_rate']:.0%} ({stats['solved']}/{measured})"
        if tier == payload.get("headline_tier", HEADLINE_TIER):
            line += "  ·  headline"
        if no_evidence:
            line += f"  ·  {no_evidence} not measured: {unmeasured}"
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


def group_draws(payloads: list[dict]) -> dict[tuple, list[dict]]:
    """Runs that asked the same question, keyed by (model, task set, coverage).

    Two runs are draws from the same experiment when the model matches, the
    frozen task set matches, **and they covered the same tasks**. That third
    condition is not pedantry. `results/` fills up with one-task probe runs
    while a task or an adapter is being debugged, and folding those in with a
    full sweep produces a spread that is an artefact of what was asked rather
    than of how the model answered: six one-task runs and one eight-task run
    reported as "7 draws, 0% to 100%", which is three different experiments
    wearing one error bar.

    A different task set is a different question and a different model is the
    whole point, so neither is folded either.
    """
    groups: dict[tuple, list[dict]] = {}
    for payload in payloads:
        coverage = tuple(sorted(entry["slug"] for entry in payload["tasks"]))
        key = (payload["model"], str(payload.get("task_set", "?")), coverage)
        groups.setdefault(key, []).append(payload)
    return groups


def _measured_slugs(payload: dict, tier: str) -> set[str]:
    """The tasks on `tier` this draw actually produced evidence about."""
    return {
        entry["slug"]
        for entry in payload["tasks"]
        if entry.get("tier", HEADLINE_TIER) == tier and entry["status"] not in NO_EVIDENCE
    }


def format_variance(payloads: list[dict], tier: str = HEADLINE_TIER) -> str:
    """Per task, how many of N draws passed. The error bar a single row lacks.

    A single sweep is one sample of a sampled process, and this repository has
    already watched the same model pass a task on one run and fail it on the
    next. Printing k out of N per task is the smallest honest answer to that:
    it separates a task a model reliably solves from one it solves half the
    time, and those two look identical in a single run.
    """
    groups = group_draws(payloads)
    if not groups:
        return "No results yet."

    blocks = []
    for (model, task_set, coverage), draws in sorted(groups.items()):
        slugs: list[str] = []
        passes: dict[str, int] = {}
        measured: dict[str, int] = {}
        for payload in draws:
            for entry in payload["tasks"]:
                if entry.get("tier", HEADLINE_TIER) != tier:
                    continue
                if entry["slug"] not in passes:
                    slugs.append(entry["slug"])
                    passes[entry["slug"]] = 0
                    measured[entry["slug"]] = 0
                # A draw that produced no evidence is not a failed draw. It
                # comes out of this task's denominator the same way it comes
                # out of the rate, or the spread would be an artefact of a
                # dropped connection.
                if entry["status"] in NO_EVIDENCE:
                    continue
                measured[entry["slug"]] += 1
                passes[entry["slug"]] += int(entry["passed"])

        if not slugs:
            continue

        # The set rate for the tier being spread, not the headline one: asking
        # for the accelerated tier's variance and being handed the laptop
        # tier's rate would be a table whose two halves describe different runs.
        #
        # And only over draws that measured every task the tier was asked for.
        # A sweep that dies halfway still writes a file, and that file's rate is
        # over the tasks it reached — folding it in next to a complete draw
        # compares two different experiments, which is L25 one level down: there
        # the two draws asked different sets, here they asked the same set and
        # answered different amounts of it.
        complete = [
            payload for payload in draws if _measured_slugs(payload, tier) == set(slugs)
        ]
        rates = [
            payload["pass_rate_by_tier"][tier]["pass_rate"]
            for payload in complete
            if payload.get("pass_rate_by_tier", {}).get(tier, {}).get("pass_rate")
            is not None
        ]
        rows = [
            [
                slug,
                f"{passes[slug]}/{measured[slug]}" if measured[slug] else "0 draws",
                ""
                if not measured[slug]
                else "always" if passes[slug] == measured[slug]
                else "never" if passes[slug] == 0
                else "SPLIT",
            ]
            for slug in slugs
        ]
        partial = len(draws) - len(complete)
        header = (
            f"{model}  ·  task set {task_set}  ·  {len(draws)} draw(s)"
            f"  ·  {tier} tier  ·  {len(coverage)} task(s) asked"
        )
        if partial:
            header += f"  ·  {partial} draw(s) incomplete"
        drawn = "every complete draw" if partial else "every draw"
        if not rates:
            summary = "set pass rate: not measured in any complete draw"
        elif min(rates) == max(rates):
            summary = f"set pass rate: {rates[0]:.0%} in {drawn}"
        else:
            summary = (
                f"set pass rate: {min(rates):.0%} to {max(rates):.0%}"
                f"  ·  mean {sum(rates) / len(rates):.0%}"
                f"  ·  over {len(rates)} complete draw(s)"
            )
        blocks.append(
            "\n".join([header, "", _table(["task", "passed", "spread"], rows), "", summary])
        )
    return "\n\n".join(blocks)


def headline_of(payload: dict) -> tuple[float | None, int, int]:
    """The (rate, solved, measured) triple a leaderboard row is allowed to show.

    Read off the headline tier's own bucket rather than off the whole run, so a
    row can never be a blend of a tier anyone can reproduce and a tier that
    needs hardware. Results files written before the split carry no
    `headline_tier`; every one of them was laptop-only, so defaulting to the
    laptop bucket reads them as what they were.
    """
    tier = payload.get("headline_tier", HEADLINE_TIER)
    bucket = payload.get("pass_rate_by_tier", {}).get(tier)
    if bucket is None:
        return payload.get("pass_rate"), 0, payload.get("measured", 0)
    measured = bucket.get("measured", bucket["total"] - bucket.get("no_evidence", 0))
    return bucket["pass_rate"], bucket["solved"], measured


def format_leaderboard(payloads: list[dict]) -> str:
    """Every run, most recent first, one row each."""
    if not payloads:
        return "No results yet. Run `scratchbench run --model reference --tasks all` first."

    rows = []
    for payload in reversed(payloads):
        rate, solved, measured = headline_of(payload)
        rows.append(
            [
                payload["model"],
                "-" if rate is None else f"{rate:.0%}",
                f"{solved}/{measured}",
                f"{payload['wall_clock_s']:.1f}s",
                "-" if payload.get("usd_cost") is None else f"${payload['usd_cost']:.2f}",
                str(payload.get("task_set", "?")),
                payload["run_at"],
            ]
        )
    return _table(
        ["model", "pass rate", "solved", "wall clock", "cost", "set", "run at"], rows
    )
