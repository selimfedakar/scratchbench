"""Recompute every published cost from the tokens beside it.

    python tools/check_cost.py leaderboard/*.json

A `usd_cost` on a public page is a claim about somebody's bill, and until now
the only check on it was arithmetic done by hand in prose, in a README, once.
That is a check that runs the day it is written and never again: the price
table drifts, a results file is edited, a new model is added with a typo in its
rate, and nothing notices. This is the same sum as a command anyone can run.

It re-derives the cost from the four token counts in each file and the price
table the adapter uses, and exits non-zero on any disagreement. Files with no
`tokens` block are reported as unverifiable rather than passed: a cost with
nothing behind it is exactly the state this repository decided was not good
enough (`docs/LESSONS.md` L18).

The tolerance is half a millionth of a dollar, which is below the smallest
figure the arithmetic can produce and far below anything that would be reported
as money.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adapters.anthropic_api import price_from_counts  # noqa: E402

TOLERANCE = 5e-7


def check(path: Path) -> tuple[str, str]:
    """Returns (verdict, detail) for one results file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", "?")
    claimed = payload.get("usd_cost")
    tokens = payload.get("tokens")

    if claimed is None and tokens is None:
        # The reference solver makes no calls. Four zeros here would read as a
        # measurement, so it reports neither, and that is the correct state.
        return "n/a", "no calls were made"
    if tokens is None:
        return "UNCHECKABLE", f"claims ${claimed} with no token counts behind it"

    recomputed = price_from_counts(
        model,
        tokens.get("input", 0),
        tokens.get("output", 0),
        cache_read=tokens.get("cache_read", 0),
        cache_write=tokens.get("cache_write", 0),
    )
    if recomputed is None:
        return "UNCHECKABLE", f"no published price for {model}"
    if claimed is None:
        return "MISMATCH", f"file says null, the tokens say ${recomputed:.6f}"
    if abs(recomputed - claimed) > TOLERANCE:
        return "MISMATCH", f"file says ${claimed:.6f}, the tokens say ${recomputed:.6f}"

    sum_of_parts = (
        f"{tokens.get('input', 0)} in + {tokens.get('output', 0)} out"
        f" = ${recomputed:.6f}"
    )
    return "ok", sum_of_parts


def main(argv: list[str]) -> int:
    paths = [Path(argument) for argument in argv]
    if not paths:
        paths = sorted((REPO_ROOT / "leaderboard").glob("*.json"))
    if not paths:
        print("no results files given and none in leaderboard/")
        return 2

    bad = 0
    for path in paths:
        verdict, detail = check(path)
        if verdict not in ("ok", "n/a"):
            bad += 1
        print(f"{verdict:12s} {path.name}  {detail}")

    print()
    if bad:
        print(f"{bad} file(s) whose cost does not reproduce.")
        return 1
    print(f"{len(paths)} file(s) checked: every cost reproduces from its own tokens.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
