"""Task discovery and validation.

A task is a directory that follows the contract in TASK_FORMAT.md. Everything
in this module exists to fail loudly when it does not, because a malformed task
does not produce a lower score — it produces a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = {
    "tokenization",
    "attention",
    "training",
    "data",
    "numerics",
    "kernels",
}

# The founding constraint, enforced rather than documented: every task is
# gradeable in under five minutes. A GPU is not a licence to need an hour.
MAX_TIME_LIMIT_S = 300

#: `laptop` is the headline tier: no GPU, reproducible by anyone who clones the
#: repository, and the only tier the leaderboard pass rate is computed over.
#: `accelerated` exists because writing a CUDA kernel is a real skill and
#: leaving it out would cut the most interesting question in ML engineering out
#: of the benchmark. The two are reported separately and never mixed.
TIERS = {"laptop", "accelerated"}

ACCELERATORS = {"cuda", "metal"}

REQUIRED_META_KEYS = {
    "slug",
    "version",
    "category",
    "difficulty",
    "probes",
    "time_limit_s",
    "requires_gpu",
    "deps",
    "added",
    "frozen_set",
}


class TaskError(Exception):
    """A task directory does not satisfy the contract."""


@dataclass(frozen=True)
class Task:
    """One benchmark task, loaded from disk and checked."""

    slug: str
    version: int
    category: str
    difficulty: int
    probes: str
    time_limit_s: int
    requires_gpu: bool
    deps: tuple[str, ...]
    added: str
    frozen_set: str
    path: Path
    tier: str = "laptop"
    accelerator: str | None = None

    @property
    def starter_dir(self) -> Path:
        return self.path / "starter"

    @property
    def hidden_tests_dir(self) -> Path:
        return self.path / "hidden_tests"

    @property
    def reference_dir(self) -> Path:
        return self.path / "reference"

    @property
    def prompt_path(self) -> Path:
        return self.path / "prompt.md"

    def prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

    def starter_files(self) -> list[Path]:
        return sorted(self.starter_dir.glob("*.py"))

    def hidden_test_files(self) -> list[Path]:
        return sorted(self.hidden_tests_dir.glob("test_*.py"))

    def reference_files(self) -> list[Path]:
        return sorted(self.reference_dir.glob("*.py"))

    def missing_deps(self) -> list[str]:
        """Declared dependencies that cannot be imported in this interpreter."""
        from importlib.util import find_spec

        missing = []
        for dependency in self.deps:
            module = {"pyyaml": "yaml", "pillow": "PIL"}.get(dependency.lower(), dependency)
            try:
                found = find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing.append(dependency)
        return missing


def load_task(path: Path) -> Task:
    """Load one task directory, raising TaskError on any contract violation."""
    path = Path(path)
    meta_path = path / "meta.yaml"
    if not meta_path.is_file():
        raise TaskError(f"{path}: no meta.yaml")

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise TaskError(f"{meta_path}: expected a mapping")

    missing = REQUIRED_META_KEYS - meta.keys()
    if missing:
        raise TaskError(f"{meta_path}: missing keys {sorted(missing)}")

    slug = meta["slug"]
    if slug != path.name:
        raise TaskError(f"{meta_path}: slug '{slug}' does not match directory '{path.name}'")
    if meta["category"] not in CATEGORIES:
        raise TaskError(f"{meta_path}: unknown category '{meta['category']}'")
    if not 1 <= meta["difficulty"] <= 5:
        raise TaskError(f"{meta_path}: difficulty must be 1..5, got {meta['difficulty']}")
    tier = meta.get("tier", "laptop")
    if tier not in TIERS:
        raise TaskError(f"{meta_path}: unknown tier '{tier}', expected one of {sorted(TIERS)}")
    if tier == "laptop" and meta["requires_gpu"]:
        raise TaskError(
            f"{meta_path}: requires_gpu must be false on the laptop tier — the constraint is "
            "the point. Move the task to tier: accelerated if it genuinely needs hardware."
        )

    accelerator = meta.get("accelerator")
    if accelerator is not None and accelerator not in ACCELERATORS:
        raise TaskError(
            f"{meta_path}: unknown accelerator '{accelerator}', expected one of "
            f"{sorted(ACCELERATORS)}"
        )
    if meta["requires_gpu"] and accelerator is None:
        raise TaskError(f"{meta_path}: requires_gpu is true but no accelerator is declared")
    if accelerator is not None and tier != "accelerated":
        raise TaskError(f"{meta_path}: accelerator is only meaningful on tier: accelerated")

    if not 0 < meta["time_limit_s"] <= MAX_TIME_LIMIT_S:
        raise TaskError(
            f"{meta_path}: time_limit_s must be 1..{MAX_TIME_LIMIT_S}, got {meta['time_limit_s']}"
        )

    task = Task(
        slug=slug,
        version=int(meta["version"]),
        category=meta["category"],
        difficulty=int(meta["difficulty"]),
        probes=str(meta["probes"]).strip(),
        time_limit_s=int(meta["time_limit_s"]),
        requires_gpu=bool(meta["requires_gpu"]),
        deps=tuple(meta["deps"] or ()),
        added=str(meta["added"]),
        frozen_set=str(meta["frozen_set"]),
        path=path,
        tier=tier,
        accelerator=accelerator,
    )
    _check_layout(task)
    return task


def _check_layout(task: Task) -> None:
    """The directory has all four parts, and they line up with each other."""
    if not task.prompt_path.is_file():
        raise TaskError(f"{task.slug}: no prompt.md")

    starter = task.starter_files()
    reference = task.reference_files()
    tests = task.hidden_test_files()

    if not starter:
        raise TaskError(f"{task.slug}: starter/ has no Python files")
    if not reference:
        raise TaskError(f"{task.slug}: reference/ has no Python files")
    if not tests:
        raise TaskError(f"{task.slug}: hidden_tests/ has no test_*.py files")

    # The workdir is flat, so the reference has to be a drop-in replacement for
    # the starter: same file names, or the tests import something that is not
    # there.
    starter_names = {p.name for p in starter}
    reference_names = {p.name for p in reference}
    if starter_names != reference_names:
        raise TaskError(
            f"{task.slug}: starter/ and reference/ hold different files: "
            f"{sorted(starter_names)} vs {sorted(reference_names)}"
        )
    overlap = starter_names & {p.name for p in tests}
    if overlap:
        raise TaskError(f"{task.slug}: solution and test files collide: {sorted(overlap)}")


def discover_tasks(tasks_root: Path) -> list[Task]:
    """Every task under `tasks_root`, in slug order."""
    tasks_root = Path(tasks_root)
    if not tasks_root.is_dir():
        raise TaskError(f"{tasks_root}: not a directory")
    return [
        load_task(child)
        for child in sorted(tasks_root.iterdir())
        if child.is_dir() and (child / "meta.yaml").is_file()
    ]


def select_tasks(tasks_root: Path, spec: str, tier: str = "all") -> list[Task]:
    """Resolve an `--tasks` argument: `all`, or a comma-separated list of slugs.

    `tier` filters the result to one tier. Naming a task explicitly overrides
    the filter — asking for a slug and being handed nothing is worse than
    running it.
    """
    tasks = discover_tasks(tasks_root)

    if spec.strip() == "all":
        if tier == "all":
            return tasks
        if tier not in TIERS:
            raise TaskError(f"unknown tier '{tier}', expected one of {sorted(TIERS)} or 'all'")
        return [task for task in tasks if task.tier == tier]

    wanted = [slug.strip() for slug in spec.split(",") if slug.strip()]
    by_slug = {task.slug: task for task in tasks}
    unknown = [slug for slug in wanted if slug not in by_slug]
    if unknown:
        raise TaskError(f"unknown task(s): {', '.join(unknown)}. Known: {', '.join(by_slug)}")
    return [by_slug[slug] for slug in wanted]


def available_accelerators() -> set[str]:
    """Which accelerators this machine can actually run a kernel on."""
    found: set[str] = set()
    try:
        import torch
    except ImportError:
        return found

    if torch.cuda.is_available():
        found.add("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        found.add("metal")
    return found
