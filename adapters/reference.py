"""The reference adapter: solve a task by copying its known-correct solution.

This is not a model and it is not cheating. It is the control: every hidden
test suite has to pass against it and no starter may pass without it, or the
benchmark is measuring its own bugs. CI runs exactly this on every commit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from runner.tasks import Task


def solve(task: Task, workdir: Path) -> None:
    """Overwrite the starter files with the reference solution."""
    for source in task.reference_files():
        shutil.copy2(source, Path(workdir) / source.name)
