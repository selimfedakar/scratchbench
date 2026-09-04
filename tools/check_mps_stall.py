"""Measure the per-process MPS stall this machine produces, and say whether the
current process is in it.

`docs/ROADMAP.md` section 9 item 3 recorded a task that could not be published
because roughly half its runs were a hundred times slower. That reading was
wrong in an important way: the stall is not the task's. On this machine
(macOS 26.5.1, M1 Pro) a process that uses `torch.mps` is either healthy, in
which case a host to device to host round trip costs about half a millisecond,
or stalled, in which case the same round trip costs hundreds of milliseconds
for the whole life of the process. No task, kernel, tolerance or assertion is
needed to produce it, and the two populations do not overlap.

Two entry points:

`probe()` is what a graded run should call before it grades anything. It is a
fixed micro-benchmark, it costs a few milliseconds when the process is healthy,
and it uses a real copy in each direction because a device-side allocation is
fast even in a stalled process.

`python tools/check_mps_stall.py --launches 14` re-derives the claim: it starts
that many fresh processes and prints each one's probe median beside the wall
time of a workload that follows it, so the separation can be seen rather than
believed.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time

#: A healthy round trip on this machine measures 0.47-0.72 ms over fourteen
#: launches; a stalled one measures 232-776 ms. Ten milliseconds sits two
#: orders of magnitude above the healthy population and one below the stalled
#: one, so it separates them without pretending to be a tuned threshold.
STALL_THRESHOLD_S = 0.010


def probe(rounds: int = 10, width: int = 256) -> float:
    """Median seconds for one host -> device -> host round trip.

    Imports torch lazily so that a caller on a machine without MPS pays nothing
    for asking.
    """
    import torch

    host = torch.randn(1, width)
    timings = []
    for _ in range(rounds):
        started = time.perf_counter()
        device = host.to("mps")
        device = device + 1.0
        torch.mps.synchronize()
        _ = device.cpu()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def is_stalled(threshold_s: float = STALL_THRESHOLD_S) -> bool:
    """True when this process is in the slow state and should not be graded."""
    return probe() > threshold_s


def _workload(iterations: int, rows: int = 6, cols: int = 513) -> float:
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
    median = probe()
    total = _workload(iterations)
    print(f"{median:.6f} {total:.6f}")
    return 0


def _survey(launches: int, iterations: int) -> int:
    print(f"{'launch':>6} {'probe_ms':>10} {'workload_s':>11}  verdict")
    stalled = 0
    for index in range(1, launches + 1):
        done = subprocess.run(
            [sys.executable, __file__, "--child", "--iterations", str(iterations)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        median_s, total_s = (float(value) for value in done.stdout.split())
        verdict = "STALLED" if median_s > STALL_THRESHOLD_S else "healthy"
        stalled += verdict == "STALLED"
        print(
            f"{index:>6} {median_s * 1000:>10.2f} {total_s:>11.3f}  {verdict}",
            flush=True,
        )
    print(f"\n{stalled} of {launches} launches stalled")
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
