"""The MPS stall, and the gate that keeps it out of a graded run.

On this machine — macOS 26.5.1, M1 Pro — a process that touches `torch.mps` is
either healthy or stalled, for the whole of its life, and which one it is has
nothing to do with what it is about to run. A host to device to host round trip
costs 0.47-0.72 ms healthy and 232-776 ms stalled, with no overlap over fourteen
consecutive launches. `docs/LESSONS.md` L42 has the measurements and the seven
suspects they killed; `ROADMAP.md` section 9.2 has the choice this file is the
answer to.

Why it needs answering at all: `STATUSES` calls a `timeout` evidence *and* a
failure, which is right when a solution genuinely did not finish. Under the
stall a wrong answer and a healthy process wearing a hundredfold tax produce the
same word, and the failure shape is what this repository publishes instead of
partial credit.

Two things live here, and they are one file because they must not drift apart:

`probe()` is the measurement. `pytest_configure` is the gate, and it is a pytest
plugin hook — `runner/sandbox.py` copies this module into the graded workdir
beside the hidden tests, after the solver has finished writing, and loads it
with `-p`. A solver cannot subvert a file that is not on disk while it works,
which is the same guarantee the hidden tests get and for the same reason.

The gate only fires when `SCRATCHBENCH_STALL_GUARD` is set, so importing this
module anywhere else costs nothing and changes nothing.
"""

from __future__ import annotations

import os
import statistics
import time

#: Seconds for one round trip, above which this process is stalled. Two orders
#: of magnitude above the healthy population and one below the stalled one, so
#: it separates the two measured clusters without pretending to be tuned.
STALL_THRESHOLD_S = 0.010

#: The exit code a stalled child reports. Outside pytest's own 0-5, and distinct
#: from the 124 that `tools/mutate_metal_task.py` uses for a timeout, so a
#: caller never has to guess which of the two it is looking at.
STALLED_EXIT = 125

#: The environment variable that arms the gate.
GUARD_ENV = "SCRATCHBENCH_STALL_GUARD"

#: How many fresh processes a graded run may spend looking for a healthy one.
#: Launches observed so far alternate almost perfectly, so two attempts would
#: usually be enough; four leaves room for the day that stops being true and
#: still bounds the cost at a few seconds.
MAX_ATTEMPTS = 4


def probe(rounds: int = 10, width: int = 256) -> float:
    """Median seconds for one host -> device -> host round trip.

    A copy in each direction, not a device-side allocation: the first MPS
    operation in a process measures about 17 ms whether that process is healthy
    or about to be a hundred times slower, so allocating proves nothing.

    torch is imported lazily so that asking the question on a machine that has
    no MPS costs nothing.
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
    """True when this process is in the slow state and must not grade anything."""
    return probe() > threshold_s


def pytest_configure(config) -> None:  # noqa: ANN001 — pytest's own hook signature
    """Refuse to grade in a stalled process, before a single test has run.

    Exiting here rather than failing the tests is deliberate: a stalled process
    has measured nothing about the solution, and the one thing this repository
    cannot do is let an absence of evidence look like a verdict.
    """
    if not os.environ.get(GUARD_ENV):
        return

    import pytest

    median = probe()
    if median > STALL_THRESHOLD_S:
        pytest.exit(
            f"mps stall guard: round trip median {median * 1000:.1f}ms exceeds "
            f"{STALL_THRESHOLD_S * 1000:.0f}ms, this process cannot grade "
            "anything (docs/LESSONS.md L42)",
            returncode=STALLED_EXIT,
        )
