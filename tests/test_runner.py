"""Tests for the harness itself.

The harness is the thing that turns a model into a number. If it is wrong, the
number is wrong in a way nobody can see by reading it, so it gets the same
treatment the tasks get.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import json
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


def test_a_solution_that_does_not_import_is_broken_not_merely_wrong(tmp_path):
    def write_garbage(task, workdir):
        (Path(workdir) / "solution.py").write_text("def answer(:\n")

    outcome = grade(load_task(make_task(tmp_path)), write_garbage)
    assert outcome.status == "collection_error"
    assert not outcome.passed


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
        "total": 2,
        "no_accelerator": 0,
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
