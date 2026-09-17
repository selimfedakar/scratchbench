"""Render the leaderboard's tables from the results files they describe.

`leaderboard/README.md` was written by hand. That was right when it held two
rows; it is a drift risk the moment a sweep adds a dozen, and a leaderboard that
drifts is this repository's worst failure mode — a wrong number about somebody
else's model, published under the owner's name, looking exactly like a right one.

Only the **tables** are generated. The prose is not, and must not be: the prose
is the part worth reading, it explains what each number means and what it does
not, and no generator has an opinion about that. A rendered region is delimited
by a pair of HTML comments and everything outside them is copied through
untouched:

    <!-- rendered: laptop-v1 -->
    | Model | ... |
    <!-- /rendered -->

Run it two ways:

    python tools/render_leaderboard.py            # rewrite the file in place
    python tools/render_leaderboard.py --check    # exit 1 if it would change

`--check` is what CI runs, so a hand-edited number fails the build the same way
a hand-edited cost already does (`tools/check_cost.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "leaderboard"
PAGE = LEADERBOARD / "README.md"

REGION = re.compile(
    r"(?P<open><!-- rendered: (?P<name>[a-z0-9-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close><!-- /rendered -->)",
    re.DOTALL,
)

#: Which results files feed which region, keyed by the tasks the run covered.
#: Keyed by content rather than by filename because a filename is a naming
#: convention and this is a claim about what was measured.
GROUPS: dict[str, frozenset[str]] = {
    "laptop-v1": frozenset(
        {
            "attention_causal_mask",
            "bpe_merge_order",
            "grad_accumulation",
            "kv_cache_equivalence",
            "online_softmax_attention",
            "quantization_error_bounds",
            "sharded_dataloader",
            "softmax_stability",
        }
    ),
    "metal-cross-entropy": frozenset({"metal_cross_entropy_kernel"}),
    "cuda-rmsnorm": frozenset({"fused_rmsnorm_kernel"}),
    # The first `v2` sweep, and the only group here whose draws come off two
    # different machines. `v2` spans three kinds of hardware, so one sweep
    # cannot run on one box: this is the Mac half, where the CUDA member comes
    # back `needs_accelerator` and is dropped by `graded()` rather than scored
    # as a loss. The Triton member's own draws are the `cuda-rmsnorm` region,
    # taken on a rented box, and the page says so where the rows are.
    #
    # ⚠ These four slugs are `v2` **as it stood on 2026-09-17**, which is what
    # the draws in this group actually measured. `flash_attention_backward` left
    # for `warmup` on the strength of this very sweep (ROADMAP §11), so a later
    # `--set v2` run writes three slugs and will not match here. That is correct
    # rather than a bug: the key is a claim about what was measured, and these
    # files measured four tasks. A future sweep is a new region, not an edit to
    # this one.
    "v2-mac": frozenset(
        {
            "flash_attention_backward",
            "fused_rmsnorm_kernel",
            "metal_cross_entropy_kernel",
            "metal_softmax_backward_kernel",
        }
    ),
}

#: The order models appear in, strongest first. Written down rather than sorted,
#: because sorting by pass rate would reorder the table every time a sweep moves
#: a number and make every diff unreadable.
MODEL_ORDER = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]


def load(group: str) -> list[dict]:
    """Every draw belonging to one region.

    Runs with no `repeat` block are excluded. The two files from 2026-08-03
    predate repeated sweeps and are single draws; folding a run of one into a
    column that reports a spread over five would state a range that was never
    measured (`docs/LESSONS.md` L19). They are still published, still checked by
    `check_cost.py`, and the prose is where they are discussed.
    """
    wanted = GROUPS[group]
    payloads = []
    for path in sorted(LEADERBOARD.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if frozenset(entry["slug"] for entry in payload["tasks"]) != wanted:
            continue
        if not payload.get("repeat"):
            continue
        payloads.append(payload)
    return payloads


def by_model(payloads: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for payload in payloads:
        grouped.setdefault(payload["model"], []).append(payload)
    return {
        model: grouped[model]
        for model in MODEL_ORDER + sorted(set(grouped) - set(MODEL_ORDER))
        if model in grouped
    }


def _span(values: list[float], fmt: str) -> str:
    """`a to b`, or a single number when they round to the same thing."""
    low, high = format(min(values), fmt), format(max(values), fmt)
    return low if low == high else f"{low} to {high}"


def graded(payload: dict) -> list[dict]:
    """The task entries in one draw that actually judged the solution.

    Everything in `NO_EVIDENCE` is dropped, not just `needs_accelerator`: a
    timeout, an adapter error and a missing GPU are all absences, and an absence
    scored as a loss is a published zero about somebody else's model
    (`docs/LESSONS.md` L20). The first version of this function filtered only
    `needs_accelerator` and disagreed with the hand-written table it was meant
    to reproduce, which is how the omission was found.
    """
    from runner.sandbox import NO_EVIDENCE

    return [entry for entry in payload["tasks"] if entry["status"] not in NO_EVIDENCE]


def model_table(payloads: list[dict]) -> str:
    """One row per model: how often it solved the set, and what that cost.

    The rate is solved over measured across every draw, so the two shapes of
    section share one definition: a single-task group's rate is the fraction of
    draws it solved, and a multi-task group's is the fraction of task-draws. The
    per-draw spread is appended only when the draws disagree, because that is
    the one thing a pooled rate cannot show and the reason repeats exist (L19).
    """
    from runner.report import cost_per_solved

    lines = [
        "| Model | Pass rate | Draws | Solved | Cost per draw | $ per solved | "
        "Wall clock | Attempts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, all_draws in by_model(payloads).items():
        # A draw that graded nothing is not a draw. It cost money and produced
        # no evidence, and counting it would shrink every rate in the row.
        draws = [d for d in all_draws if graded(d)]
        solved = sum(1 for d in draws for t in graded(d) if t["passed"])
        measured = sum(len(graded(d)) for d in draws)
        per_draw = [
            sum(1 for t in graded(d) if t["passed"]) / len(graded(d)) for d in draws
        ]
        pooled = solved / measured if measured else None
        # With one task per draw the spread is always 0% to 100% and says
        # nothing: it is a restatement of "some draws passed and some did not",
        # which the pooled rate already carries.
        one_task = all(len(graded(d)) == 1 for d in draws)
        if pooled is None:
            rate = "-"
        elif one_task or min(per_draw) == max(per_draw):
            suffix = "" if one_task else ", every draw"
            rate = f"**{pooled:.0%}**{suffix}"
        else:
            rate = f"**{pooled:.0%}**, {min(per_draw):.0%} to {max(per_draw):.0%} by draw"
        costs = [d["usd_cost"] for d in draws if d.get("usd_cost") is not None]
        per_solved = [c for c in (cost_per_solved(d) for d in draws) if c is not None]
        lines.append(
            "| `{model}` | {rate} | {n} | {solved}/{measured} | {cost} | {each} | "
            "{wall} | {attempts} |".format(
                model=model,
                rate=rate,
                n=len(draws),
                solved=solved,
                measured=measured,
                cost=_span(costs, ".3f").replace("0.", "$0.") if costs else "-",
                each=_span(per_solved, ".3f").replace("0.", "$0.") if per_solved else "-",
                wall=_span([d["wall_clock_s"] for d in draws], ".0f") + "s",
                # Kept even though it has read `1` in every published run: it is
                # the column that says a rate was not bought with retries, and a
                # column that has never varied is exactly the one a reader needs
                # to see rather than infer.
                attempts=_span([float(d.get("attempts", 1)) for d in draws], ".0f"),
            )
        )
    return "\n".join(lines)


def task_table(payloads: list[dict]) -> str:
    """`k` out of `n` draws, per task and per model.

    The denominator is per cell rather than per column, because a task that
    produced no evidence in one draw shortens that cell alone. A shared
    denominator would silently score the absence as a loss.
    """
    grouped = by_model(payloads)
    models = list(grouped)
    slugs: list[tuple[int, str]] = []
    for payload in payloads:
        for entry in payload["tasks"]:
            key = (entry.get("difficulty", 0), entry["slug"])
            if key not in slugs:
                slugs.append(key)
    slugs.sort()

    lines = [
        "| Task | Difficulty | " + " | ".join(f"`{m}`" for m in models) + " |",
        "|---|:---:|" + ":---:|" * len(models),
    ]
    for difficulty, slug in slugs:
        cells = []
        for model in models:
            entries = [
                entry
                for payload in grouped[model]
                for entry in graded(payload)
                if entry["slug"] == slug
            ]
            if not entries:
                # No graded draw at all — every one was an absence, which on
                # this page means the machine could not run the task. `0/0` is
                # arithmetically true and reads as "failed everything", so the
                # cell says nothing instead of saying zero (docs/LESSONS.md L20).
                cells.append("–")
                continue
            k = sum(1 for entry in entries if entry["passed"])
            cell = f"{k}/{len(entries)}"
            # Bold marks a mixed result: the only cells where the draw-to-draw
            # spread is visible, and the reason repeats exist at all.
            cells.append(f"**{cell}**" if 0 < k < len(entries) else cell)
        lines.append(f"| `{slug}` | {difficulty} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


RENDERERS = {
    "laptop-v1": model_table,
    "laptop-v1-by-task": task_table,
    "metal-cross-entropy": model_table,
    "cuda-rmsnorm": model_table,
    "v2-mac": model_table,
    "v2-mac-by-task": task_table,
}


def render(name: str) -> str:
    group = name.removesuffix("-by-task")
    return RENDERERS[name](load(group))


def rewrite(text: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group("name")
        if name not in RENDERERS:
            raise SystemExit(
                f"{PAGE}: no renderer for region {name!r}; "
                f"known regions are {sorted(RENDERERS)}"
            )
        return match.group("open") + render(name) + "\n" + match.group("close")

    return REGION.sub(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the rendered tables differ from the file",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    current = PAGE.read_text(encoding="utf-8")
    if not REGION.search(current):
        print(f"{PAGE}: no <!-- rendered: ... --> regions found", file=sys.stderr)
        return 1
    updated = rewrite(current)

    regions = [match.group("name") for match in REGION.finditer(current)]
    if args.check:
        if updated != current:
            print(
                f"{PAGE} is out of date: re-run `python tools/render_leaderboard.py`.\n"
                "A table in this file was edited by hand, or a results file changed "
                "under it. Either way the published numbers and their evidence "
                "disagree, and the evidence is right.",
                file=sys.stderr,
            )
            return 1
        print(f"{len(regions)} rendered region(s) match their results files.")
        return 0

    PAGE.write_text(updated, encoding="utf-8")
    print(f"{len(regions)} rendered region(s) written: {', '.join(regions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
