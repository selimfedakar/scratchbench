"""Tests for the harness itself.

The harness is the thing that turns a model into a number. If it is wrong, the
number is wrong in a way nobody can see by reading it, so it gets the same
treatment the tasks get.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from adapters.reference import solve as reference_solve
from runner import report as reporting
from runner.sandbox import grade, parse_summary
from runner.tasks import TaskError, discover_tasks, load_task, select_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"


# -- the real task set -----------------------------------------------------


def test_every_task_in_the_repository_loads():
    tasks = discover_tasks(TASKS_ROOT)
    assert tasks, "no tasks found"
    assert [task.slug for task in tasks] == sorted(task.slug for task in tasks)


def test_no_task_takes_over_five_minutes():
    for task in discover_tasks(TASKS_ROOT):
        assert 0 < task.time_limit_s <= 300


def test_the_laptop_tier_never_asks_for_a_gpu():
    # The claim the README makes, as a test: the headline tier is reproducible
    # by anyone who clones the repository.
    for task in discover_tasks(TASKS_ROOT):
        if task.tier == "laptop":
            assert task.requires_gpu is False
            assert task.accelerator is None


def test_selecting_a_tier_filters_the_set():
    every = select_tasks(TASKS_ROOT, "all", tier="all")
    laptop = select_tasks(TASKS_ROOT, "all", tier="laptop")
    accelerated = select_tasks(TASKS_ROOT, "all", tier="accelerated")
    assert len(laptop) + len(accelerated) == len(every)
    assert all(task.tier == "laptop" for task in laptop)


def test_naming_a_task_overrides_the_tier_filter():
    # Asking for a slug and being handed nothing is worse than running it.
    named = select_tasks(TASKS_ROOT, "softmax_stability", tier="accelerated")
    assert [task.slug for task in named] == ["softmax_stability"]


def test_selecting_a_frozen_set_filters_the_set():
    # The laptop tier stopped being one published set the moment `warmup`
    # existed, so a sweep over every laptop task averages a frozen set with a
    # warm-up set. That is L23 one axis over, and this is the flag that lets a
    # leaderboard row ask about one of them.
    everything = select_tasks(TASKS_ROOT, "all", "laptop")
    v1_only = select_tasks(TASKS_ROOT, "all", "laptop", "v1")

    assert {task.frozen_set for task in v1_only} == {"v1"}
    assert len(v1_only) < len(everything)
    assert {task.slug for task in v1_only} < {task.slug for task in everything}


def test_naming_a_task_overrides_the_frozen_set_filter():
    # Same rule the tier filter follows: being handed nothing when a slug was
    # asked for is worse than running it.
    picked = select_tasks(TASKS_ROOT, "speculative_decoding_verify", "laptop", "v1")
    assert [task.slug for task in picked] == ["speculative_decoding_verify"]


def test_an_unknown_frozen_set_is_refused():
    with pytest.raises(TaskError, match="unknown frozen set"):
        select_tasks(TASKS_ROOT, "all", "all", "v9")


def test_an_unknown_tier_is_refused():
    with pytest.raises(TaskError, match="unknown tier"):
        select_tasks(TASKS_ROOT, "all", tier="quantum")


def test_selecting_all_matches_discovery():
    assert select_tasks(TASKS_ROOT, "all") == discover_tasks(TASKS_ROOT)


def test_selecting_a_subset_keeps_the_order_asked_for():
    tasks = select_tasks(TASKS_ROOT, "kv_cache_equivalence,softmax_stability")
    assert [task.slug for task in tasks] == ["kv_cache_equivalence", "softmax_stability"]


def test_an_unknown_slug_is_refused():
    with pytest.raises(TaskError, match="unknown task"):
        select_tasks(TASKS_ROOT, "no_such_task")


# -- a synthetic task, so the failure paths can be exercised --------------

META = """\
slug: {slug}
version: 1
category: numerics
difficulty: 1
probes: A synthetic task used by the harness tests.
time_limit_s: {time_limit}
requires_gpu: false
deps: []
added: 2026-07-26
frozen_set: test
"""


def make_task(
    root: Path,
    slug: str = "synthetic",
    time_limit: int = 30,
    test_body: str = "def test_it():\n    assert answer() == 42\n",
    meta_overrides: str = "",
) -> Path:
    path = root / slug
    for part in ("starter", "hidden_tests", "reference"):
        (path / part).mkdir(parents=True)

    meta = META.format(slug=slug, time_limit=time_limit) + meta_overrides
    (path / "meta.yaml").write_text(meta)
    (path / "prompt.md").write_text("Return 42.\n")
    (path / "starter" / "solution.py").write_text(
        "def answer():\n    raise NotImplementedError\n"
    )
    (path / "reference" / "solution.py").write_text("def answer():\n    return 42\n")
    (path / "hidden_tests" / "test_solution.py").write_text(
        "from solution import answer\n\n" + test_body
    )
    return path


def test_a_synthetic_task_loads(tmp_path):
    task = load_task(make_task(tmp_path))
    assert task.slug == "synthetic"
    assert task.starter_files()[0].name == "solution.py"


@pytest.mark.parametrize(
    "override,message",
    [
        ("requires_gpu: true\n", "requires_gpu must be false on the laptop tier"),
        ("category: astrology\n", "unknown category"),
        ("difficulty: 9\n", "difficulty"),
        ("time_limit_s: 3600\n", "time_limit_s"),
        ("slug: something_else\n", "does not match directory"),
        ("tier: cloud\n", "unknown tier"),
        ("accelerator: cuda\n", "only meaningful on tier: accelerated"),
        ("tier: accelerated\nrequires_gpu: true\n", "no accelerator is declared"),
        ("tier: accelerated\naccelerator: tpu\n", "unknown accelerator"),
    ],
)
def test_meta_violations_are_refused(tmp_path, override, message):
    # A later key wins in YAML, so appending is enough to override.
    path = make_task(tmp_path, meta_overrides=override)
    with pytest.raises(TaskError, match=message):
        load_task(path)


def test_a_reference_that_does_not_match_the_starter_is_refused(tmp_path):
    path = make_task(tmp_path)
    (path / "reference" / "solution.py").unlink()
    (path / "reference" / "different_name.py").write_text("def answer():\n    return 42\n")
    with pytest.raises(TaskError, match="different files"):
        load_task(path)


def test_a_task_without_hidden_tests_is_refused(tmp_path):
    path = make_task(tmp_path)
    (path / "hidden_tests" / "test_solution.py").unlink()
    with pytest.raises(TaskError, match="hidden_tests"):
        load_task(path)


def test_a_missing_prompt_is_refused(tmp_path):
    path = make_task(tmp_path)
    (path / "prompt.md").unlink()
    with pytest.raises(TaskError, match="prompt.md"):
        load_task(path)


# -- calibration, and the rule it exists to enforce ------------------------
#
# v1 was calibrated against how hard each task felt to write, and a frontier
# model then solved every one of them first try (docs/LESSONS.md L21). From v2
# onward a task carries what models actually scored on it, and a task nothing
# ever fails cannot be in the set.

CALIBRATED = """\
calibration:
  - model: claude-opus-5
    draws: 5
    passed: 2
    date: 2026-08-10
  - model: claude-haiku-4-5
    draws: 5
    passed: 0
    date: 2026-08-10
"""


def test_a_task_without_calibration_has_an_empty_block(tmp_path):
    assert load_task(make_task(tmp_path)).calibration == ()


def test_calibration_is_read_off_the_metadata(tmp_path):
    task = load_task(make_task(tmp_path, meta_overrides=CALIBRATED))
    assert [entry.model for entry in task.calibration] == [
        "claude-opus-5",
        "claude-haiku-4-5",
    ]
    assert task.calibration[0].draws == 5
    assert task.calibration[0].passed == 2
    assert task.calibration[0].rate == 0.4
    assert str(task.calibration[0]) == "claude-opus-5 2/5"


@pytest.mark.parametrize(
    "block,message",
    [
        ("calibration: nope\n", "must be a list"),
        ("calibration:\n  - just a string\n", "expected a mapping"),
        ("calibration:\n  - model: m\n    draws: 5\n    date: 2026-08-10\n", "missing keys"),
        (
            "calibration:\n  - model: m\n    draws: 0\n    passed: 0\n    date: 2026-08-10\n",
            "draws must be at least 1",
        ),
        (
            "calibration:\n  - model: m\n    draws: 5\n    passed: 6\n    date: 2026-08-10\n",
            "passed must be 0..5",
        ),
        (
            "calibration:\n  - model: m\n    draws: five\n    passed: 1\n    date: 2026-08-10\n",
            "must be integers",
        ),
    ],
)
def test_an_unreadable_calibration_block_is_refused(tmp_path, block, message):
    with pytest.raises(TaskError, match=message):
        load_task(make_task(tmp_path, meta_overrides=block))


def test_a_calibrated_set_needs_a_calibration_block(tmp_path):
    path = make_task(tmp_path, meta_overrides="frozen_set: v2\n")
    with pytest.raises(TaskError, match="needs a calibration block"):
        load_task(path)


def test_a_task_the_best_model_never_fails_cannot_enter_a_calibrated_set(tmp_path):
    # The rule that v1 did not have: a set the top of the field clears
    # completely has stopped measuring the top of the field.
    override = (
        "frozen_set: v2\n"
        "calibration:\n"
        "  - model: claude-opus-5\n"
        "    draws: 5\n"
        "    passed: 5\n"
        "    date: 2026-08-10\n"
        "  - model: claude-haiku-4-5\n"
        "    draws: 5\n"
        "    passed: 1\n"
        "    date: 2026-08-10\n"
    )
    with pytest.raises(TaskError, match="passed all 5 draws"):
        load_task(make_task(tmp_path, meta_overrides=override))


def test_one_draw_is_not_a_calibration(tmp_path):
    override = (
        "frozen_set: v2\n"
        "calibration:\n"
        "  - model: claude-opus-5\n"
        "    draws: 1\n"
        "    passed: 0\n"
        "    date: 2026-08-10\n"
    )
    with pytest.raises(TaskError, match="draws or more"):
        load_task(make_task(tmp_path, meta_overrides=override))


def test_a_calibrated_task_that_clears_the_bar_loads(tmp_path):
    task = load_task(make_task(tmp_path, meta_overrides="frozen_set: v2\n" + CALIBRATED))
    assert task.frozen_set == "v2"
    assert len(task.calibration) == 2


def test_v1_predates_the_rule_and_is_left_alone(tmp_path):
    # Every v1 task would fail the admission check, and that is the point:
    # they are published as what they were, not retrofitted.
    assert load_task(make_task(tmp_path, meta_overrides="frozen_set: v1\n")).frozen_set == "v1"


def test_a_task_waiting_for_calibration_is_not_held_to_the_rule(tmp_path):
    path = make_task(tmp_path, meta_overrides="frozen_set: unvalidated\n")
    assert load_task(path).calibration == ()


def test_the_warm_up_set_is_where_a_task_everything_solves_goes(tmp_path):
    # The other half of the admission rule. A task the top of the field never
    # fails is not broken and not useless — it still separates a small model
    # from a large one — so it keeps its calibration block and its place, just
    # not in the headline.
    override = (
        "frozen_set: warmup\n"
        "calibration:\n"
        "  - model: claude-opus-5\n"
        "    draws: 15\n"
        "    passed: 15\n"
        "    date: 2026-08-09\n"
    )
    task = load_task(make_task(tmp_path, meta_overrides=override))
    assert task.frozen_set == "warmup"
    assert task.calibration[0].rate == 1.0


def test_a_warm_up_task_carries_the_measurement_that_put_it_there():
    # The admission rule is the loader's job and `test_every_task_in_the_
    # repository_loads` is what enforces it, so re-asserting it here would be a
    # second copy of a rule that already lives in one place. This is the part
    # the loader deliberately does not police: `warmup` is a claim that models
    # were asked and cleared the task, and a claim with nothing behind it is
    # what this repository is about not publishing.
    for task in discover_tasks(TASKS_ROOT):
        if task.frozen_set == "warmup":
            assert task.calibration, f"{task.slug}: in the warm-up set with nothing to show"


# -- grading ---------------------------------------------------------------


def leave_alone(task, workdir):
    """A solver that writes nothing."""


def test_the_reference_passes_and_the_untouched_starter_does_not(tmp_path):
    task = load_task(make_task(tmp_path))
    assert grade(task, reference_solve).passed
    starter = grade(task, leave_alone)
    assert not starter.passed
    assert starter.status == "failed"


def test_the_solver_never_sees_the_hidden_tests(tmp_path):
    seen = []

    def peek(task, workdir):
        seen.extend(sorted(p.name for p in Path(workdir).iterdir()))
        reference_solve(task, workdir)

    task = load_task(make_task(tmp_path))
    assert grade(task, peek).passed
    assert seen == ["solution.py"]


def test_a_solver_that_raises_is_reported_and_does_not_escape(tmp_path):
    def explode(task, workdir):
        raise RuntimeError("rate limited")

    outcome = grade(load_task(make_task(tmp_path)), explode)
    assert outcome.status == "adapter_error"
    assert not outcome.passed
    assert "rate limited" in outcome.detail


def test_a_solution_that_does_not_import_is_the_solutions_failure(tmp_path):
    # It is tempting to file this under "nothing judged the solution", and that
    # was the old behaviour. It is wrong: the starter imports cleanly by
    # contract, so a syntax error in the graded directory was written by the
    # solver. Calling it an absence of evidence takes the task out of the
    # denominator, and the models that emit unparseable Python are the weak
    # ones, so the mistake runs in the direction that flatters them.
    from runner.sandbox import HARNESS_FAILURES, NO_EVIDENCE

    def write_garbage(task, workdir):
        (Path(workdir) / "solution.py").write_text("def answer(:\n")

    outcome = grade(load_task(make_task(tmp_path)), write_garbage)
    assert outcome.status == "solution_error"
    assert not outcome.passed
    assert outcome.status not in NO_EVIDENCE
    assert outcome.status not in HARNESS_FAILURES


def test_a_solution_missing_a_name_the_tests_import_is_also_its_failure(tmp_path):
    # The commoner version of the same event, and the one an import check on
    # the solution alone would have missed: the file imports perfectly well and
    # simply does not contain what the task asked for.
    def rename_everything(task, workdir):
        (Path(workdir) / "solution.py").write_text("def reply():\n    return 42\n")

    outcome = grade(load_task(make_task(tmp_path)), rename_everything)
    assert outcome.status == "solution_error"
    assert not outcome.passed


def test_hidden_tests_that_cannot_collect_at_all_are_not_the_models_fault(tmp_path):
    # The other side of the line. Nothing about the solution explains this one,
    # so it is an absence of evidence and it breaks the run.
    from runner.sandbox import HARNESS_FAILURES, NO_EVIDENCE

    path = make_task(tmp_path)
    (path / "hidden_tests" / "test_solution.py").write_text(
        "import a_module_that_does_not_exist\n\n\ndef test_it():\n    assert True\n"
    )
    outcome = grade(load_task(path), reference_solve)
    assert outcome.status == "collection_error"
    assert outcome.status in NO_EVIDENCE
    assert outcome.status in HARNESS_FAILURES


def test_a_solution_that_hangs_is_stopped_at_the_time_limit(tmp_path):
    path = make_task(
        tmp_path,
        slug="slowpoke",
        time_limit=2,
        test_body="def test_it():\n    assert answer() == 42\n",
    )
    (path / "reference" / "solution.py").write_text(
        "import time\n\n\ndef answer():\n    time.sleep(60)\n    return 42\n"
    )
    outcome = grade(load_task(path), reference_solve)
    assert outcome.status == "timeout"
    assert not outcome.passed
    assert outcome.duration_s < 30


def test_the_workdir_can_be_kept_for_inspection(tmp_path):
    task = load_task(make_task(tmp_path / "task"))
    kept = tmp_path / "kept"
    grade(task, reference_solve, keep_workdir=kept)
    assert (kept / "synthetic" / "solution.py").read_text() == "def answer():\n    return 42\n"
    assert (kept / "synthetic" / "test_solution.py").is_file()


def test_a_task_needing_hardware_this_machine_lacks_is_labelled_not_failed(tmp_path):
    # An absence of evidence is not a zero. Rounding it down is how a
    # leaderboard starts reporting on hardware it never had.
    from runner.tasks import available_accelerators

    absent = next(iter({"cuda", "metal"} - available_accelerators()), None)
    if absent is None:
        pytest.skip("this machine has every accelerator the schema knows about")

    path = make_task(
        tmp_path,
        meta_overrides=f"tier: accelerated\nrequires_gpu: true\naccelerator: {absent}\n",
    )
    outcome = grade(load_task(path), reference_solve)
    assert outcome.status == "needs_accelerator"
    assert not outcome.passed
    assert absent in outcome.detail


def test_the_tier_breakdown_keeps_the_laptop_rate_separate(tmp_path):
    tasks = select_tasks(TASKS_ROOT, "softmax_stability,kv_cache_equivalence")
    outcomes = [grade(tasks[0], reference_solve), grade(tasks[1], leave_alone)]
    payload = reporting.build_payload("dummy", tasks, outcomes, wall_clock_s=1.0)

    assert payload["pass_rate_by_tier"]["laptop"] == {
        "pass_rate": 0.5,
        "solved": 1,
        "measured": 2,
        "total": 2,
        "no_accelerator": 0,
        "no_evidence": 0,
        "categories": {"attention": 0.0, "numerics": 1.0},
    }
    assert all(entry["tier"] == "laptop" for entry in payload["tasks"])


def test_a_tier_with_no_hardware_has_no_pass_rate_at_all(tmp_path):
    # The failure this guards against is subtle: a task that never ran was
    # being counted as a task that failed, so a machine without a GPU reported
    # "accelerated 0%" and "kernels 0%". Both read as a measurement.
    from runner.tasks import available_accelerators

    absent = next(iter({"cuda", "metal"} - available_accelerators()), None)
    if absent is None:
        pytest.skip("this machine has every accelerator the schema knows about")

    gpu_task = load_task(
        make_task(
            tmp_path,
            slug="gpu_only",
            meta_overrides=(
                f"category: kernels\ntier: accelerated\n"
                f"requires_gpu: true\naccelerator: {absent}\n"
            ),
        )
    )
    laptop_task = select_tasks(TASKS_ROOT, "softmax_stability")[0]
    outcomes = [grade(laptop_task, reference_solve), grade(gpu_task, reference_solve)]
    payload = reporting.build_payload(
        "dummy", [laptop_task, gpu_task], outcomes, wall_clock_s=1.0
    )

    assert payload["pass_rate"] == 1.0  # one measured task, and it passed
    assert payload["measured"] == 1
    assert payload["not_measured"] == 1
    assert payload["pass_rate_by_tier"]["accelerated"]["pass_rate"] is None
    assert "kernels" not in payload["pass_rate_by_category"]
    assert "accelerated  not run" in reporting.format_run(payload)


def test_a_task_whose_dependency_is_missing_is_skipped_not_failed_silently(tmp_path):
    path = make_task(tmp_path, meta_overrides="deps: [definitely_not_installed]\n")
    outcome = grade(load_task(path), reference_solve)
    assert outcome.status == "missing_deps"
    assert not outcome.passed
    assert "definitely_not_installed" in outcome.detail


# -- absence of evidence, in every place a rate is computed ----------------


def test_a_run_where_the_adapter_never_answered_has_no_pass_rate():
    # The whole sweep failing on a missing API key produced `pass_rate: 0.0`
    # and `measured: 8` — a published zero about a model the harness never
    # reached. Missing hardware was handled and this was not, in the same
    # arithmetic, which is why the classification is one table now.
    def explodes(task, workdir):
        raise RuntimeError("no API key")

    tasks = select_tasks(TASKS_ROOT, "softmax_stability,kv_cache_equivalence")
    outcomes = [grade(task, explodes) for task in tasks]
    assert {outcome.status for outcome in outcomes} == {"adapter_error"}

    payload = reporting.build_payload("dummy", tasks, outcomes, wall_clock_s=1.0)

    assert payload["pass_rate"] is None
    assert payload["measured"] == 0
    assert payload["not_measured"] == 2
    assert payload["pass_rate_by_category"] == {}
    assert payload["pass_rate_by_tier"]["laptop"]["pass_rate"] is None
    assert payload["pass_rate_by_tier"]["laptop"]["no_evidence"] == 2
    # And nothing about it reads as a hardware problem, which is the other
    # absence of evidence and a completely different conversation.
    assert payload["pass_rate_by_tier"]["laptop"]["no_accelerator"] == 0


def test_the_headline_rate_is_one_tier_even_when_both_were_measured():
    # The failure this guards against could not happen on the machine that
    # wrote the harness, because an accelerated task there always comes back
    # `needs_accelerator` and drops out on the no-evidence rule. On a box with
    # a GPU both tiers are measured, and the headline number — the one the
    # leaderboard column prints — quietly became their average. Every document
    # in this repository says that number is the laptop tier.
    from runner.sandbox import Outcome

    tasks = select_tasks(TASKS_ROOT, "all", tier="all")
    accelerated = [task for task in tasks if task.tier == "accelerated"]
    assert accelerated, "this test needs an accelerated task in the set"

    outcomes = [
        Outcome(
            slug=task.slug,
            passed=task.tier == "laptop",
            status="passed" if task.tier == "laptop" else "failed",
            duration_s=0.1,
        )
        for task in tasks
    ]
    payload = reporting.build_payload("dummy", tasks, outcomes, wall_clock_s=1.0)

    assert payload["headline_tier"] == "laptop"
    assert payload["pass_rate"] == 1.0
    assert payload["pass_rate_by_tier"]["accelerated"]["pass_rate"] == 0.0
    # And the category rates do not blend either: `kernels` is an accelerated
    # category, so it belongs in that tier's block rather than in the headline.
    assert "kernels" not in payload["pass_rate_by_category"]
    assert payload["pass_rate_by_tier"]["accelerated"]["categories"] == {"kernels": 0.0}

    # The leaderboard row reads the headline tier's own numbers rather than
    # counting every task in the file.
    rate, solved, measured = reporting.headline_of(payload)
    assert (rate, solved, measured) == (1.0, len(tasks) - len(accelerated), len(tasks) - len(accelerated))


def test_a_results_file_written_before_the_split_still_reads_as_laptop_only():
    # Both published rows are version 1 documents with no `headline_tier`, and
    # every one of them was a laptop run. Reading them as anything else would
    # rewrite a published number.
    v1 = {
        "results_version": 1,
        "model": "claude-opus-5",
        "run_at": "2026-08-03T10:29:32+00:00",
        "pass_rate": 1.0,
        "measured": 8,
        "pass_rate_by_tier": {
            "laptop": {
                "pass_rate": 1.0,
                "solved": 8,
                "total": 8,
                "no_accelerator": 0,
                "no_evidence": 0,
            }
        },
        "tasks": [],
    }
    assert reporting.headline_of(v1) == (1.0, 8, 8)


def test_the_results_file_says_which_harness_produced_it():
    # CONTRIBUTING asks for the harness version with every submitted result,
    # and until now nothing in the file carried one.
    from runner import __version__

    tasks = select_tasks(TASKS_ROOT, "softmax_stability")
    payload = reporting.build_payload(
        "dummy", tasks, [grade(tasks[0], reference_solve)], wall_clock_s=1.0
    )
    assert payload["harness_version"] == __version__
    assert payload["results_version"] == reporting.RESULTS_VERSION


def test_every_status_is_classified_and_an_unknown_one_cannot_be_built():
    from runner.sandbox import HARNESS_FAILURES, NO_EVIDENCE, STATUSES, Outcome

    # The two sets are derived from the table rather than written out twice.
    assert NO_EVIDENCE | {name for name in STATUSES if name not in NO_EVIDENCE} == set(
        STATUSES
    )
    assert HARNESS_FAILURES <= NO_EVIDENCE

    # The classifications that decide whether a zero gets published.
    assert "adapter_error" in NO_EVIDENCE
    assert "collection_error" in NO_EVIDENCE
    assert "needs_accelerator" in NO_EVIDENCE
    assert "missing_deps" in NO_EVIDENCE
    # A solution that will not import is the other non-obvious case, and it
    # goes the other way: the starter imports by contract, so the breakage came
    # from whatever wrote the file, and that is a measurement of it.
    assert "solution_error" not in NO_EVIDENCE
    assert "solution_error" not in HARNESS_FAILURES
    # A solution that ran and did not finish is the one non-obvious case: that
    # is the solution failing, not the harness, so it stays a measurement.
    assert "timeout" not in NO_EVIDENCE
    assert "timeout" not in HARNESS_FAILURES

    with pytest.raises(ValueError, match="unclassified outcome status"):
        Outcome(slug="x", passed=False, status="something_new", duration_s=0.0)


# -- reading pytest's summary line ----------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("29 passed in 0.22s", {"passed": 29}),
        ("1 failed, 28 passed in 0.31s", {"failed": 1, "passed": 28}),
        ("2 errors in 0.10s", {"error": 2}),
        ("3 passed, 1 skipped in 0.05s", {"passed": 3, "skipped": 1}),
        ("no tests ran in 0.01s", {}),
    ],
)
def test_summary_counts_are_read_off_the_last_line(line, expected):
    assert parse_summary(line) == expected


# -- results ---------------------------------------------------------------


def test_the_payload_reports_pass_rates_per_category(tmp_path):
    tasks = select_tasks(TASKS_ROOT, "softmax_stability,kv_cache_equivalence")
    outcomes = [
        grade(tasks[0], reference_solve),
        grade(tasks[1], leave_alone),
    ]
    payload = reporting.build_payload("dummy", tasks, outcomes, wall_clock_s=1.0)

    assert payload["pass_rate"] == 0.5
    assert payload["pass_rate_by_category"] == {"attention": 0.0, "numerics": 1.0}
    assert payload["task_set"] == "v1"
    assert [entry["slug"] for entry in payload["tasks"]] == [
        "softmax_stability",
        "kv_cache_equivalence",
    ]


def test_a_written_result_can_be_read_back(tmp_path):
    tasks = select_tasks(TASKS_ROOT, "softmax_stability")
    payload = reporting.build_payload(
        "dummy", tasks, [grade(tasks[0], reference_solve)], wall_clock_s=2.0
    )
    path = reporting.write_payload(payload, tmp_path)

    assert json.loads(path.read_text()) == payload
    assert reporting.load_payloads(tmp_path) == [payload]


def test_a_corrupt_result_file_is_ignored_rather_than_fatal(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    assert reporting.load_payloads(tmp_path) == []


def test_an_empty_results_directory_says_so(tmp_path):
    assert "No results yet" in reporting.format_leaderboard([])


def test_the_run_table_renders(tmp_path):
    tasks = select_tasks(TASKS_ROOT, "softmax_stability")
    payload = reporting.build_payload(
        "dummy", tasks, [grade(tasks[0], reference_solve)], wall_clock_s=2.0
    )
    rendered = reporting.format_run(payload)
    assert "softmax_stability" in rendered
    assert "laptop       100% (1/1)" in rendered


# -- repeated draws --------------------------------------------------------
#
# The same model passed `softmax_stability` on one run and failed it on the
# next, four hours apart, with the same prompt and the same settings. A single
# sweep is one draw of a sampled process, so the only honest error bar is
# several of them.


def _draw(model: str, statuses: dict[str, bool], task_set: str = "v1") -> dict:
    """A results document with a chosen outcome per task, for the aggregation."""
    return {
        "model": model,
        "task_set": task_set,
        "headline_tier": "laptop",
        "pass_rate": sum(statuses.values()) / len(statuses),
        "pass_rate_by_tier": {
            "laptop": {
                "pass_rate": sum(statuses.values()) / len(statuses),
                "solved": sum(statuses.values()),
                "measured": len(statuses),
                "total": len(statuses),
                "no_accelerator": 0,
                "no_evidence": 0,
            }
        },
        "tasks": [
            {
                "slug": slug,
                "tier": "laptop",
                "passed": passed,
                "status": "passed" if passed else "failed",
            }
            for slug, passed in statuses.items()
        ],
    }


def test_repeated_draws_are_grouped_by_model_and_task_set():
    draws = [
        _draw("claude-opus-5", {"a": True}),
        _draw("claude-opus-5", {"a": False}),
        _draw("claude-opus-5", {"a": True}, task_set="v2"),
        _draw("claude-haiku-4-5", {"a": True}),
    ]
    groups = reporting.group_draws(draws)
    assert sorted(groups) == [
        ("claude-haiku-4-5", "v1", ("a",)),
        ("claude-opus-5", "v1", ("a",)),
        ("claude-opus-5", "v2", ("a",)),
    ]
    assert len(groups[("claude-opus-5", "v1", ("a",))]) == 2


def test_a_one_task_probe_run_is_not_a_draw_of_the_full_sweep():
    # `results/` fills up with single-task runs while an adapter is being
    # debugged. Folded in with a real sweep they produce a spread that measures
    # what was asked rather than how the model answered: six one-task runs and
    # one eight-task run came out as "7 draws, 0% to 100%".
    probe = _draw("claude-haiku-4-5", {"softmax_stability": False})
    sweep = _draw("claude-haiku-4-5", {"softmax_stability": True, "bpe_merge_order": True})

    groups = reporting.group_draws([probe, sweep])
    assert len(groups) == 2
    assert all(len(draws) == 1 for draws in groups.values())

    printed = reporting.format_variance([probe, sweep])
    assert "1 task(s) asked" in printed
    assert "2 task(s) asked" in printed
    assert "0% to 100%" not in printed


def test_a_task_that_passes_some_draws_and_not_others_is_marked_as_split():
    draws = [
        _draw("claude-opus-5", {"softmax_stability": True, "bpe_merge_order": True}),
        _draw("claude-opus-5", {"softmax_stability": False, "bpe_merge_order": True}),
        _draw("claude-opus-5", {"softmax_stability": True, "bpe_merge_order": True}),
    ]
    printed = reporting.format_variance(draws)

    assert "3 draw(s)" in printed
    assert "softmax_stability" in printed and "2/3" in printed and "SPLIT" in printed
    assert "bpe_merge_order" in printed and "3/3" in printed and "always" in printed
    # The set rate is a range, not a point, and saying so is the entire purpose.
    assert "50% to 100%" in printed


def test_a_draw_that_produced_no_evidence_is_not_counted_as_a_failed_draw():
    # Otherwise a dropped connection on the third draw would show up as the
    # model being less reliable than it is, which is a fabricated error bar.
    good = _draw("claude-opus-5", {"a": True})
    dropped = _draw("claude-opus-5", {"a": False})
    dropped["tasks"][0]["status"] = "adapter_error"

    printed = reporting.format_variance([good, dropped])
    assert "1/1" in printed


def test_a_sweep_that_died_halfway_is_not_a_draw_of_the_set_rate():
    # A billing failure emptied four of five draws mid-sweep. The per-task
    # column was right — it counts only what was measured — and the set rate
    # underneath it read "100% in every draw" over five draws when one draw had
    # measured both tasks, one had measured one of them at 100%, and three had
    # measured nothing at all.
    whole = _draw("claude-opus-5", {"a": True, "b": True})
    half = _draw("claude-opus-5", {"a": True, "b": False})
    half["tasks"][1]["status"] = "adapter_error"
    half["pass_rate_by_tier"]["laptop"]["pass_rate"] = 1.0
    empty = _draw("claude-opus-5", {"a": False, "b": False})
    for entry in empty["tasks"]:
        entry["status"] = "adapter_error"
    empty["pass_rate_by_tier"]["laptop"]["pass_rate"] = None

    printed = reporting.format_variance([whole, half, empty])
    assert "3 draw(s)" in printed
    assert "2 draw(s) incomplete" in printed
    assert "in every complete draw" in printed
    assert "a" in printed and "2/2" in printed
    assert "1/1" in printed


def test_a_group_with_no_complete_draw_reports_no_set_rate():
    partial = _draw("claude-opus-5", {"a": True, "b": False})
    partial["tasks"][1]["status"] = "adapter_error"
    printed = reporting.format_variance([partial])
    assert "not measured in any complete draw" in printed
    # The evidence that does exist is still shown rather than thrown away.
    assert "1/1" in printed


def test_a_long_detail_is_clipped_to_keep_the_table_readable():
    clipped = reporting._clip("x" * 300)
    assert len(clipped) <= 58


# -- the CLI ---------------------------------------------------------------


def test_validate_passes_on_the_real_task_set(capsys):
    from runner.cli import main

    assert main(["validate", "--tasks", "bpe_merge_order"]) == 0
    assert "validated" in capsys.readouterr().out


def test_validate_does_not_count_what_it_could_not_run(capsys):
    # A task this machine has no hardware for appears in the table as
    # UNCHECKED. Counting it as validated would be the summary line claiming
    # evidence that was never produced.
    from runner.cli import main
    from runner.tasks import available_accelerators

    accelerated = [
        task for task in discover_tasks(TASKS_ROOT) if task.tier == "accelerated"
    ]
    if not accelerated or all(
        task.accelerator in available_accelerators() for task in accelerated
    ):
        pytest.skip("no accelerated task this machine is missing hardware for")

    assert main(["validate", "--tier", "all"]) == 0
    printed = capsys.readouterr().out
    checked = len(discover_tasks(TASKS_ROOT)) - len(accelerated)
    assert f"{checked} task(s) validated" in printed
    assert "not checked here" in printed


def test_list_prints_every_task(capsys):
    from runner.cli import main

    assert main(["list"]) == 0
    printed = capsys.readouterr().out
    for task in discover_tasks(TASKS_ROOT):
        assert task.slug in printed


def test_an_unknown_model_is_a_clean_error(capsys):
    from runner.cli import main

    assert main(["run", "--model", "llama-9000", "--tasks", "bpe_merge_order"]) == 2
    assert "no adapter" in capsys.readouterr().err


def test_a_run_records_the_adapters_attempts_and_cost(tmp_path, monkeypatch):
    # A pass on attempt eleven is a different result from a pass on attempt
    # one, so the results file has to carry both numbers off the adapter that
    # produced them rather than off a default.
    import adapters
    from runner.cli import main

    class FakeAdapter:
        max_attempts = 3
        usd_cost = 1.25
        token_usage = {"input": 1000, "output": 200, "cache_read": 0, "cache_write": 0}

        def __call__(self, task, workdir):
            reference_solve(task, workdir)

    monkeypatch.setitem(adapters.ADAPTERS, "fake", lambda model: FakeAdapter())
    assert (
        main(
            [
                "run",
                "--model",
                "fake",
                "--tasks",
                "softmax_stability",
                "--results",
                str(tmp_path),
            ]
        )
        == 0
    )

    payload = reporting.load_payloads(tmp_path)[-1]
    assert payload["attempts"] == 3
    assert payload["usd_cost"] == 1.25
    # The cost has to be checkable against something. A figure with no usage
    # behind it cannot be told apart from a figure computed off a stale price.
    assert payload["tokens"] == {
        "input": 1000,
        "output": 200,
        "cache_read": 0,
        "cache_write": 0,
    }


def test_repeating_a_run_writes_one_independent_result_per_draw(tmp_path):
    from runner.cli import main

    assert (
        main(
            [
                "run",
                "--model",
                "reference",
                "--tasks",
                "softmax_stability",
                "--repeat",
                "2",
                "--results",
                str(tmp_path),
            ]
        )
        == 0
    )
    payloads = reporting.load_payloads(tmp_path)
    assert len(payloads) == 2
    # Each file is a complete run in its own right; the group only records that
    # they were asked the same question.
    groups = {payload["repeat"]["group"] for payload in payloads}
    assert len(groups) == 1
    assert sorted(payload["repeat"]["index"] for payload in payloads) == [1, 2]
    assert all(payload["repeat"]["of"] == 2 for payload in payloads)
    assert all(payload["pass_rate"] == 1.0 for payload in payloads)


def test_each_draw_keeps_its_own_workdir(tmp_path):
    # `--keep` names the directory after the task, so five draws of one task
    # used to leave one directory holding the fifth. What a repeated sweep is
    # for is the shape of each failure, and that is exactly what the overwrite
    # threw away.
    from runner.cli import main

    kept = tmp_path / "workdirs"
    assert (
        main(
            [
                "run",
                "--model",
                "reference",
                "--tasks",
                "softmax_stability",
                "--repeat",
                "3",
                "--keep",
                str(kept),
                "--results",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )
    assert sorted(path.name for path in kept.iterdir()) == ["draw1", "draw2", "draw3"]
    for draw in kept.iterdir():
        assert (draw / "softmax_stability").is_dir()


def test_a_single_run_keeps_its_workdir_where_it_was_asked_to(tmp_path):
    # No draw index when there is only one draw: the path the user typed is the
    # path they get.
    from runner.cli import main

    kept = tmp_path / "workdirs"
    assert (
        main(
            [
                "run",
                "--model",
                "reference",
                "--tasks",
                "softmax_stability",
                "--keep",
                str(kept),
                "--results",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )
    assert (kept / "softmax_stability").is_dir()


def test_a_single_run_carries_no_repeat_block(tmp_path):
    from runner.cli import main

    main(
        [
            "run",
            "--model",
            "reference",
            "--tasks",
            "softmax_stability",
            "--results",
            str(tmp_path),
        ]
    )
    assert reporting.load_payloads(tmp_path)[-1]["repeat"] is None


def test_each_repeat_gets_its_own_adapter_so_costs_do_not_accumulate(tmp_path, monkeypatch):
    # An adapter accumulates spend across the tasks it solves. Reused across
    # draws it would write a running total into every file, and the last draw
    # would look several times as expensive as the first for the same work.
    import adapters
    from runner.cli import main

    class CountingAdapter:
        max_attempts = 1
        token_usage = None

        def __init__(self):
            self.usd_cost = 0.0

        def __call__(self, task, workdir):
            self.usd_cost += 1.0
            reference_solve(task, workdir)

    monkeypatch.setitem(adapters.ADAPTERS, "counting", lambda model: CountingAdapter())
    assert (
        main(
            [
                "run",
                "--model",
                "counting",
                "--tasks",
                "softmax_stability",
                "--repeat",
                "3",
                "--results",
                str(tmp_path),
            ]
        )
        == 0
    )
    costs = [payload["usd_cost"] for payload in reporting.load_payloads(tmp_path)]
    assert costs == [1.0, 1.0, 1.0]


def test_repeat_must_be_at_least_one(capsys):
    from runner.cli import main

    assert main(["run", "--model", "reference", "--tasks", "softmax_stability", "--repeat", "0"]) == 2
    assert "--repeat" in capsys.readouterr().err


def test_the_control_run_reports_one_attempt_and_no_cost(tmp_path):
    from runner.cli import main

    assert (
        main(
            [
                "run",
                "--model",
                "reference",
                "--tasks",
                "softmax_stability",
                "--results",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = reporting.load_payloads(tmp_path)[-1]
    assert payload["attempts"] == 1
    assert payload["usd_cost"] is None
    # The reference solver is a plain function: no calls, no tokens, and no
    # zeros pretending to be a measurement.
    assert payload["tokens"] is None


# -- the Anthropic adapter -------------------------------------------------
#
# Everything here runs offline. The network path is not covered by these
# tests and is marked as such in the session report; what they do cover is the
# part that would silently corrupt a score: what the model is shown, and what
# is allowed to land in the graded directory.


def test_an_adapter_is_built_with_the_model_it_was_asked_for():
    from adapters import resolve
    from adapters.anthropic_api import DEFAULT_MODEL, AnthropicAdapter

    haiku = resolve("claude-haiku-4-5")
    assert isinstance(haiku, AnthropicAdapter)
    assert haiku.model == "claude-haiku-4-5"
    # The bare adapter alias means "this adapter's default model".
    assert resolve("anthropic").model == DEFAULT_MODEL
    assert resolve("reference") is reference_solve


def test_the_model_is_shown_the_prompt_and_the_starter_but_never_the_tests(tmp_path):
    from adapters.anthropic_api import build_prompt

    task = select_tasks(TASKS_ROOT, "softmax_stability")[0]
    workdir = tmp_path / "work"
    workdir.mkdir()
    for source in task.starter_files():
        (workdir / source.name).write_text(source.read_text())

    prompt = build_prompt(task, workdir)

    assert task.prompt().strip()[:200] in prompt
    assert "NotImplementedError" in prompt  # the starter body, verbatim
    for test_file in task.hidden_test_files():
        for line in test_file.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("def test_"):
                assert stripped not in prompt


def test_the_response_schema_only_admits_the_files_the_task_declared():
    from adapters.anthropic_api import response_schema

    schema = response_schema(["stable_softmax.py"])
    item = schema["properties"]["files"]["items"]
    assert item["properties"]["filename"]["enum"] == ["stable_softmax.py"]
    assert item["additionalProperties"] is False


def test_only_the_declared_files_are_written(tmp_path):
    from adapters.anthropic_api import write_files

    payload = {
        "files": [
            {"filename": "solution.py", "content": "answer = 42\n"},
            # A model that could drop a conftest.py into the graded directory
            # could rig its own score, and one that could write outside it
            # could do worse than that.
            {"filename": "conftest.py", "content": "collect_ignore_glob = ['*']\n"},
            {"filename": "../escaped.py", "content": "print('outside')\n"},
        ]
    }
    written = write_files(tmp_path, payload, {"solution.py"})

    assert written == ["solution.py"]
    assert (tmp_path / "solution.py").read_text() == "answer = 42\n"
    assert not (tmp_path / "conftest.py").exists()
    assert not (tmp_path.parent / "escaped.py").exists()


def test_a_response_with_no_usable_file_is_refused(tmp_path):
    from adapters.anthropic_api import AnthropicResponseError, write_files

    with pytest.raises(AnthropicResponseError, match="none of the files"):
        write_files(tmp_path, {"files": []}, {"solution.py"})


class FakeUsage:
    def __init__(self, input_tokens, output_tokens, cache_read=0, cache_write=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


def test_cost_is_computed_from_the_published_price():
    from adapters.anthropic_api import price_of

    # 1M input at $5 and 1M output at $25.
    assert price_of("claude-opus-5", FakeUsage(1_000_000, 1_000_000)) == 30.0
    # Cache reads are a tenth of fresh input; writes are a quarter more.
    assert price_of(
        "claude-opus-5", FakeUsage(0, 0, cache_read=1_000_000, cache_write=1_000_000)
    ) == pytest.approx(5.0 * 1.35)


def test_an_unpriced_model_reports_no_cost_rather_than_a_guess():
    from adapters.anthropic_api import price_of

    assert price_of("claude-not-a-model", FakeUsage(1000, 1000)) is None


def test_the_dated_snapshot_an_alias_resolves_to_is_the_same_model():
    from adapters.anthropic_api import same_model

    # This is what the API actually answers with. Reading it as a substitution
    # turns every run into an adapter_error and measures nothing.
    assert same_model("claude-haiku-4-5", "claude-haiku-4-5-20251001")
    assert same_model("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001")


def test_a_different_model_is_still_a_substitution():
    from adapters.anthropic_api import same_model

    assert not same_model("claude-haiku-4-5", "claude-opus-5")
    # A prefix is not a snapshot. Only a trailing date is.
    assert not same_model("claude-opus-4", "claude-opus-4-5")
    assert not same_model("claude-opus-5", "claude-opus-5-fast")


def test_a_dated_snapshot_is_priced_as_the_alias_it_resolves_to():
    from adapters.anthropic_api import price_of

    # Otherwise every live run reports usd_cost: null, which reads as "this
    # model has no published price" rather than "the lookup missed".
    assert price_of("claude-haiku-4-5-20251001", FakeUsage(1_000_000, 1_000_000)) == 6.0


# -- the published costs ---------------------------------------------------


def test_every_published_cost_reproduces_from_its_own_tokens():
    # The arithmetic behind the leaderboard was checked once, by hand, in
    # prose. That check stops running the moment it is written: a price table
    # drifts, a file is edited, and nothing notices. This is the same sum as a
    # command, over the files that are actually published.
    sys.path.insert(0, str(REPO_ROOT))
    from tools.check_cost import check

    published = sorted((REPO_ROOT / "leaderboard").glob("*.json"))
    assert published, "no published results to check"
    for path in published:
        verdict, detail = check(path)
        assert verdict == "ok", f"{path.name}: {detail}"


def test_every_calibration_block_reproduces_from_its_own_draws():
    # A task is refused from a numbered frozen set on the strength of a number
    # in `meta.yaml`. The draws that number came from are checked in under
    # `calibration/`, so the claim is evidence rather than a claim.
    sys.path.insert(0, str(REPO_ROOT))
    from tools.check_calibration import measured

    evidence = measured(REPO_ROOT / "calibration")
    calibrated = [task for task in discover_tasks(TASKS_ROOT) if task.calibration]
    assert calibrated, "no task carries a calibration block"
    for task in calibrated:
        for entry in task.calibration:
            assert evidence.get((entry.model, task.slug)) == (entry.passed, entry.draws), (
                f"{task.slug} / {entry.model}: meta.yaml claims {entry.passed}/{entry.draws}, "
                f"the published draws say {evidence.get((entry.model, task.slug))}"
            )


def test_a_calibration_block_that_outruns_its_draws_is_caught(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from tools.check_calibration import measured

    (tmp_path / "one.json").write_text(
        json.dumps(
            {
                "model": "claude-opus-5",
                "tasks": [
                    {"slug": "a", "status": "passed", "passed": True},
                    {"slug": "b", "status": "adapter_error", "passed": False},
                ],
            }
        )
    )
    # The adapter error measured nothing, so it is in neither the numerator nor
    # the denominator, exactly as it is in every rate.
    assert measured(tmp_path) == {("claude-opus-5", "a"): (1, 1)}


def test_a_cost_that_does_not_match_its_tokens_is_caught(tmp_path):
    # A checker that has never rejected anything is not a checker.
    sys.path.insert(0, str(REPO_ROOT))
    from tools.check_cost import check

    path = tmp_path / "tampered.json"
    path.write_text(
        json.dumps(
            {
                "model": "claude-opus-5",
                "usd_cost": 0.99,
                "tokens": {
                    "input": 15858,
                    "output": 18453,
                    "cache_read": 0,
                    "cache_write": 0,
                },
            }
        )
    )
    verdict, detail = check(path)
    assert verdict == "MISMATCH"
    assert "0.540615" in detail


def test_a_cost_with_no_tokens_behind_it_is_unverifiable_rather_than_fine(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from tools.check_cost import check

    path = tmp_path / "trust_me.json"
    path.write_text(json.dumps({"model": "claude-opus-5", "usd_cost": 0.54}))
    verdict, _ = check(path)
    assert verdict == "UNCHECKABLE"


def test_nothing_published_was_billed_at_a_cache_rate():
    # The cache multipliers are the one part of the price arithmetic no live
    # call has ever exercised: this adapter sends each task once, as its own
    # prompt, so there is no prefix to reuse and the two cache counts are zero
    # by construction. Saying that is only worth anything if the published
    # files agree, so this is where the claim is checked rather than asserted.
    for path in sorted((REPO_ROOT / "leaderboard").glob("*.json")):
        tokens = json.loads(path.read_text())["tokens"]
        assert tokens["cache_read"] == 0, path.name
        assert tokens["cache_write"] == 0, path.name
