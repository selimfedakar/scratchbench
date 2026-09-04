"""Re-derive the MPS stall this machine produces, and show what the guard does.

`docs/ROADMAP.md` section 9 item 3 recorded a task that could not be published
because roughly half its runs were a hundred times slower. That reading was
wrong in an important way: the stall is not the task's. On this machine
(macOS 26.5.1, M1 Pro) a process that uses `torch.mps` is either healthy, in
which case a host to device to host round trip costs about half a millisecond,
or stalled, in which case the same round trip costs hundreds of milliseconds for
the whole life of that process. No task, kernel, tolerance or assertion is
needed to produce it, and the two populations do not overlap. `docs/LESSONS.md`
L42 has the seven suspects this killed.

The measurement itself lives in `runner/mps_stall.py`, next to the pytest hook
that acts on it, and is imported here rather than copied. Two versions of one
threshold is how this repository published a wrong number twice already —
`docs/LESSONS.md` L11 and L20.

    python tools/check_mps_stall.py --launches 14

starts that many fresh processes and prints each one's probe median beside the
wall time of a workload that follows it, so the separation can be seen rather
than believed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runner.mps_stall import STALL_THRESHOLD_S, probe  # noqa: E402


def _workload(iterations: int, rows: int = 6, cols: int = 513) -> float:
    """A plain torch loop, deliberately with no `compile_shader` anywhere.

    The point of the shape is that it is ordinary: the stall does not need a
    custom Metal pipeline, and a survey that used one would leave that open.
    """
    import torch

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(rows, cols, generator=generator)
    started = time.perf_counter()
    for _ in range(iterations):
        device = logits.to("mps")
        out = torch.softmax(device, dim=-1)
        torch.mps.synchronize()
        _ = out.cpu()
    return time.perf_counter() - started


def _child(iterations: int) -> int:
    print(f"{probe():.6f} {_workload(iterations):.6f}")
    return 0


def _survey(launches: int, iterations: int) -> int:
    print(f"{'launch':>6} {'probe_ms':>10} {'workload_s':>11}  verdict")
    stalled = 0
    healthy_probes = []
    stalled_probes = []
    for index in range(1, launches + 1):
        done = subprocess.run(
            [sys.executable, __file__, "--child", "--iterations", str(iterations)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if done.returncode != 0:
            print(done.stdout + done.stderr)
            return done.returncode
        median_s, total_s = (float(value) for value in done.stdout.split())
        if median_s > STALL_THRESHOLD_S:
            stalled += 1
            stalled_probes.append(median_s)
            verdict = "STALLED"
        else:
            healthy_probes.append(median_s)
            verdict = "healthy"
        print(
            f"{index:>6} {median_s * 1000:>10.2f} {total_s:>11.3f}  {verdict}",
            flush=True,
        )

    print(f"\n{stalled} of {launches} launches stalled")
    if healthy_probes and stalled_probes:
        print(
            f"healthy probes {min(healthy_probes) * 1000:.2f}-"
            f"{max(healthy_probes) * 1000:.2f} ms, "
            f"stalled probes {min(stalled_probes) * 1000:.2f}-"
            f"{max(stalled_probes) * 1000:.2f} ms, "
            f"threshold {STALL_THRESHOLD_S * 1000:.0f} ms"
        )
        print(
            "separation "
            f"{min(stalled_probes) / max(healthy_probes):.0f}x between the two "
            "populations"
        )
    elif not stalled_probes:
        print("no launch stalled in this survey, which is itself worth repeating")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launches", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return _child(args.iterations)
    return _survey(args.launches, args.iterations)


if __name__ == "__main__":
    raise SystemExit(main())
