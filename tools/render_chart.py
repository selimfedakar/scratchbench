"""Render the pass-rate chart from the calibration blocks it describes.

A picture is a claim like any other, so it is generated from the same numbers
`tools/check_calibration.py` re-derives from the checked-in draws — never drawn
by hand, never a screenshot of a notebook nobody can re-run. The output is a
self-contained SVG with no plotting dependency: pure text this script writes, so
CI can regenerate and diff it on a machine with no matplotlib, the same argument
`render_readme.py` makes about counting tests by collecting them.

The chart is the repository's thesis in one view. Each calibrated task gets a row
of per-model bars; a task whose bars are all full has stopped measuring
(`warmup`), and a task whose bars spread is one that still separates the field.
The eye sees the saturation the headline rate hides.

    python tools/render_chart.py            # rewrite docs/pass-rates.svg in place
    python tools/render_chart.py --check    # exit 1 if it would change

`--check` is what CI runs, so a stale figure fails the build instead of sitting
on the front page past the sweep that moved it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.tasks import discover_tasks  # noqa: E402

TASKS_ROOT = ROOT / "tasks"
OUT = ROOT / "docs" / "pass-rates.svg"

# A fixed model order and palette, so the figure is deterministic and the colours
# mean the same thing from one render to the next.
MODEL_ORDER = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]
COLOUR = {
    "claude-opus-5": "#c2410c",
    "claude-sonnet-5": "#ea580c",
    "claude-haiku-4-5": "#f59e0b",
    "gpt-5.6-sol": "#0f766e",
    "gpt-5.6-terra": "#0d9488",
    "gpt-5.6-luna": "#5eead4",
}
MIN_DRAWS = 5  # the admission rule's floor: fewer draws is not read, so not drawn

# Layout constants (px).
LEFT = 232          # task-label gutter
TRACK = 300         # full-width (100%) bar length
BAR_H = 13          # one model bar
BAR_GAP = 3         # between model bars
ROW_PAD = 16        # between task groups
TOP = 54            # header height
RIGHT_PAD = 92      # room for the "p/d" annotation
FONT = "ui-monospace, SFMono-Regular, Menlo, monospace"


def _rows(tasks):
    """`[(slug, frozen_set, [(model, passed, draws)...])]`, tasks that have ≥2
    readable calibration entries, hardest-separating first."""
    out = []
    for task in tasks:
        entries = [c for c in (task.calibration or []) if c.draws >= MIN_DRAWS]
        if len(entries) < 2:
            continue
        entries.sort(key=lambda c: MODEL_ORDER.index(c.model) if c.model in MODEL_ORDER else 99)
        bars = [(c.model, c.passed, c.draws) for c in entries]
        out.append((task.slug, task.frozen_set, bars))
    # Spread first: a task the field splits on is the one worth seeing at the top.
    def spread(row):
        rates = [p / d for _, p, d in row[2]]
        return max(rates) - min(rates)
    out.sort(key=lambda r: (-spread(r), r[0]))
    return out


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(tasks) -> str:
    rows = _rows(tasks)
    height = TOP
    for _, _, bars in rows:
        height += len(bars) * (BAR_H + BAR_GAP) + ROW_PAD
    width = LEFT + TRACK + RIGHT_PAD

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">'
    )
    out.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    out.append(
        f'<text x="16" y="24" font-size="15" font-weight="700" fill="#111827">'
        f'scratchbench — pass rate by model</text>'
    )
    out.append(
        f'<text x="16" y="42" font-size="11" fill="#6b7280">'
        f'one attempt per task, each model at its own defaults; '
        f'full bars = stopped measuring</text>'
    )

    y = TOP
    for slug, frozen_set, bars in rows:
        group_h = len(bars) * (BAR_H + BAR_GAP) - BAR_GAP
        label_y = y + group_h / 2 + 4
        out.append(
            f'<text x="{LEFT - 10}" y="{label_y:.0f}" font-size="11" text-anchor="end" '
            f'fill="#111827">{_esc(slug)}</text>'
        )
        out.append(
            f'<text x="{LEFT - 10}" y="{label_y + 13:.0f}" font-size="9" text-anchor="end" '
            f'fill="#9ca3af">{_esc(frozen_set)}</text>'
        )
        for model, passed, draws in bars:
            rate = passed / draws
            fill_w = TRACK * rate
            colour = COLOUR.get(model, "#9ca3af")
            out.append(
                f'<rect x="{LEFT}" y="{y}" width="{TRACK}" height="{BAR_H}" '
                f'rx="2" fill="#f3f4f6"/>'
            )
            out.append(
                f'<rect x="{LEFT}" y="{y}" width="{fill_w:.1f}" height="{BAR_H}" '
                f'rx="2" fill="{colour}"/>'
            )
            out.append(
                f'<text x="{LEFT + 6}" y="{y + BAR_H - 3}" font-size="9" '
                f'fill="#ffffff">{_esc(model)}</text>'
            )
            out.append(
                f'<text x="{LEFT + TRACK + 8}" y="{y + BAR_H - 3}" font-size="9" '
                f'fill="#374151">{passed}/{draws} · {rate * 100:.0f}%</text>'
            )
            y += BAR_H + BAR_GAP
        y += ROW_PAD

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the figure is stale")
    args = parser.parse_args()

    svg = render(discover_tasks(TASKS_ROOT))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != svg:
            print(f"{OUT} is out of date with the calibration blocks behind it.")
            print("Run: python tools/render_chart.py")
            return 1
        print(f"{OUT.name} matches the calibration blocks.")
        return 0

    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
