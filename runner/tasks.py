"""Task discovery and validation.

A task is a directory that follows the contract in TASK_FORMAT.md. Everything
in this module exists to fail loudly when it does not, because a malformed task
does not produce a lower score — it produces a wrong one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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

#: A numbered frozen set from v2 onward is calibrated against models before
#: anything enters it. `v1` is exempt because it predates the rule and is
#: published as what it was: difficulty numbers assigned by how hard each task
#: felt to write, which is a measurement of the author rather than of the task
#: (docs/LESSONS.md L21). Anything else is a task that is not in a headline set:
#: `unvalidated` for one that is not finished, `warmup` for one that is finished
#: and solved by everything at the top, and whatever the harness's own tests use.
CALIBRATED_SET = re.compile(r"^v(\d+)$")
FIRST_CALIBRATED_SET = 2

#: Fewer draws than this cannot say much about a sampled process: the same
#: model on the same task has passed once and failed once (L19).
CALIBRATION_MIN_DRAWS = 5

#: How many measured entries the admission rule reads, and therefore the
#: minimum a numbered set requires. One model is not the frontier: a task the
#: single strongest model clears can still be the only task in the repository
#: that separates the model below it from the model above (docs/LESSONS.md L35).
CALIBRATION_TOP_N = 2

REQUIRED_CALIBRATION_KEYS = {"model", "draws", "passed", "date"}


class TaskError(Exception):
    """A task directory does not satisfy the contract."""


@dataclass(frozen=True)
class Calibration:
    """What one model scored on this task, over several independent draws."""

    model: str
    draws: int
    passed: int
    date: str

    @property
    def rate(self) -> float:
        return self.passed / self.draws

    def __str__(self) -> str:
        return f"{self.model} {self.passed}/{self.draws}"


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
    calibration: tuple[Calibration, ...] = field(default_factory=tuple)

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

    calibration = _load_calibration(meta_path, meta.get("calibration"))
    _check_admission(meta_path, str(meta["frozen_set"]), calibration)

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
        calibration=calibration,
    )
    _check_layout(task)
    return task


def _load_calibration(meta_path: Path, raw) -> tuple[Calibration, ...]:
    """Parse the optional `calibration:` block, refusing anything unreadable."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TaskError(f"{meta_path}: calibration must be a list of entries")

    entries = []
    for index, item in enumerate(raw):
        where = f"{meta_path}: calibration[{index}]"
        if not isinstance(item, dict):
            raise TaskError(f"{where}: expected a mapping")
        missing = REQUIRED_CALIBRATION_KEYS - item.keys()
        if missing:
            raise TaskError(f"{where}: missing keys {sorted(missing)}")

        try:
            draws, passed = int(item["draws"]), int(item["passed"])
        except (TypeError, ValueError) as error:
            raise TaskError(f"{where}: draws and passed must be integers") from error
        if draws < 1:
            raise TaskError(f"{where}: draws must be at least 1, got {draws}")
        if not 0 <= passed <= draws:
            raise TaskError(f"{where}: passed must be 0..{draws}, got {passed}")

        entries.append(
            Calibration(
                model=str(item["model"]),
                draws=draws,
                passed=passed,
                date=str(item["date"]),
            )
        )
    return tuple(entries)


def _check_admission(meta_path: Path, frozen_set: str, calibration: tuple[Calibration, ...]) -> None:
    """The rule that stops a set being calibrated against the author's intuition.

    v1's difficulty numbers were assigned by how hard each task felt to write,
    and then a frontier model solved every one of them on the first attempt
    (docs/LESSONS.md L21). So from v2 onward a task earns its place by having
    been asked of models, and a task the frontier clears is refused: it costs a
    sweep and tells nobody anything. It can live in a warm-up set. It cannot be
    in the headline.

    The frontier is read as the top two measured entries rather than the single
    best one. A rule that reads one entry treats the strongest model as the
    field, and it refused the one laptop task in this repository that separated
    `claude-sonnet-5` from `claude-opus-5` (docs/LESSONS.md L35). Entries with
    fewer than CALIBRATION_MIN_DRAWS draws are not read at all: a measurement
    too small to admit a task is too small to refuse one either.
    """
    numbered = CALIBRATED_SET.match(frozen_set)
    if not numbered or int(numbered.group(1)) < FIRST_CALIBRATED_SET:
        return

    if not calibration:
        raise TaskError(
            f"{meta_path}: frozen_set '{frozen_set}' needs a calibration block — no task "
            "enters a calibrated set on the strength of a difficulty number somebody "
            "assigned by feel"
        )

    measured = sorted(
        (entry for entry in calibration if entry.draws >= CALIBRATION_MIN_DRAWS),
        key=lambda entry: (entry.rate, entry.draws),
        reverse=True,
    )
    if len(measured) < CALIBRATION_TOP_N:
        raise TaskError(
            f"{meta_path}: {len(measured)} calibration entr"
            f"{'y' if len(measured) == 1 else 'ies'} with {CALIBRATION_MIN_DRAWS} draws or "
            f"more, and '{frozen_set}' needs {CALIBRATION_TOP_N} — fewer draws than that "
            "cannot tell a hard task from an unlucky one, and one model is not a field"
        )

    top = measured[:CALIBRATION_TOP_N]
    if all(entry.passed == entry.draws for entry in top):
        cleared = ", ".join(f"{entry.model} {entry.passed}/{entry.draws}" for entry in top)
        raise TaskError(
            f"{meta_path}: the top of the field cleared this task ({cleared}), so it cannot "
            f"be in '{frozen_set}' — a task the frontier never fails adds cost and no "
            "information"
        )


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


def select_tasks(
    tasks_root: Path, spec: str, tier: str = "all", frozen_set: str = "all"
) -> list[Task]:
    """Resolve an `--tasks` argument: `all`, or a comma-separated list of slugs.

    `tier` filters the result to one tier and `frozen_set` to one published set.
    Naming a task explicitly overrides both — asking for a slug and being handed
    nothing is worse than running it.

    The frozen-set filter exists because the laptop tier stopped being one set.
    `warmup` holds tasks that every frontier model solves, so a sweep over every
    laptop task now spans two sets and its `pass_rate` is an average across them,
    which is the same mistake as averaging two tiers (L23) one axis over. There
    is no default filtering here: a run that asks for everything gets everything
    and its results file records `task_set` as the list it actually covered, so
    the blend is visible rather than implied.
    """
    tasks = discover_tasks(tasks_root)

    if spec.strip() == "all":
        selected = tasks
        if tier != "all":
            if tier not in TIERS:
                raise TaskError(
                    f"unknown tier '{tier}', expected one of {sorted(TIERS)} or 'all'"
                )
            selected = [task for task in selected if task.tier == tier]
        if frozen_set != "all":
            # Checked against every task rather than the tier-filtered ones, so
            # `--set v1 --tier accelerated` says "no accelerated task is in v1"
            # by returning nothing, rather than "there is no such set".
            known = sorted({task.frozen_set for task in tasks})
            if frozen_set not in known:
                raise TaskError(
                    f"unknown frozen set '{frozen_set}', expected one of {known} or 'all'"
                )
            selected = [task for task in selected if task.frozen_set == frozen_set]
        return selected

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
