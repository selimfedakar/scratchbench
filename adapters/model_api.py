"""The adapter contract, and the skeletons that have not been filled in yet.

The reference adapter came first, and the reason is worth stating rather than
apologising for: a harness that has never been proven against a known-correct
solution will happily report that a model failed when what actually failed was
the harness. CI runs the reference on every commit, and only then does pointing
a model at this benchmark produce a number anyone should believe.

`adapters/anthropic_api.py` and `adapters/openai_api.py` are both written
against the contract below, and both use the shared prompt and schema in this
module so the two vendors are asked the same question.

What an adapter has to do:

1. Read `task.prompt()` and the starter files in `workdir`.
2. Ask the model for the finished files. The prompt names them; the model is
   not shown, and must not be shown, anything from `hidden_tests/`.
3. Write the files back into `workdir`, replacing the starters.
4. Optionally iterate: run the *starter's* own sanity checks if a task has
   them, or simply re-prompt on a syntax error. Every edit-and-run cycle counts
   as an attempt and `TASK_FORMAT.md` says attempts are reported, so they have
   to be counted here.

What an adapter must not do:

- read, glob, or otherwise touch `task.hidden_tests_dir`;
- read `task.reference_dir`;
- write outside `workdir`;
- retry so many times that the run stops measuring the model and starts
  measuring the scaffolding. Whatever limit is chosen goes in the results file,
  because a pass at attempt eleven is not the same result as a pass at
  attempt one.
"""

from __future__ import annotations

from pathlib import Path

from runner.tasks import Task


# ---------------------------------------------------------------------------
# Shared by every hosted adapter.
#
# These four pieces live here rather than in one vendor's module because two
# leaderboard columns are only comparable if the models behind them were asked
# the same question. A second copy of the prompt would drift, and nothing in a
# results file would say that GPT had been asked something Claude was not. The
# schema and the writer are here for a second reason: they are what stops a
# model writing a file the task never declared, and that guarantee should not
# be re-implemented per vendor.
# ---------------------------------------------------------------------------


class ResponseError(RuntimeError):
    """The response cannot be graded, and is not evidence about the model.

    `grade` turns this into `adapter_error`: no pass, no fail, and out of every
    rate. A truncated answer, a refusal, and an answer served by a different
    model than the one asked are all this, and scoring them as failures would
    file a harness problem under the model's name.
    """



SYSTEM_PROMPT = (
    "You are implementing machine learning machinery from a written "
    "specification. Return complete, runnable file contents. Follow the "
    "specification's stated conventions exactly — shapes, dtypes, and edge "
    "case behaviour are part of the requirement, not suggestions."
)

INSTRUCTIONS = (
    "Return the full contents of every file listed below, with the stubs "
    "implemented. Return whole files rather than diffs or fragments, and keep "
    "the given function and class signatures unchanged."
)


def build_prompt(task: Task, workdir: Path) -> str:
    """The user turn: the task's specification, then the files to fill in.

    Everything here comes from `prompt.md` and the starter files sitting in
    `workdir`. Nothing reads `hidden_tests/` or `reference/`.
    """
    parts = [task.prompt().strip(), "", INSTRUCTIONS, ""]
    for path in sorted(Path(workdir).glob("*.py")):
        parts.append(f"--- {path.name} ---")
        parts.append(path.read_text(encoding="utf-8").rstrip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def response_schema(filenames: list[str]) -> dict:
    """A JSON schema that can only describe the files this task declared."""
    return {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # An enum rather than a free string: the filename is
                        # not something the model gets to invent, because the
                        # hidden tests import one specific module name.
                        "filename": {"type": "string", "enum": sorted(filenames)},
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["files"],
        "additionalProperties": False,
    }


def write_files(workdir: Path, payload: dict, allowed: set[str]) -> list[str]:
    """Write the returned files into `workdir`. Returns the names written.

    `allowed` is the authority. The schema already constrains the filename, so
    a name outside the set means either a schema that was not enforced or a
    model working around it; either way the file does not land.
    """
    workdir = Path(workdir)
    written: list[str] = []
    for entry in payload.get("files") or []:
        name = entry.get("filename")
        content = entry.get("content")
        if name not in allowed or not isinstance(content, str):
            continue
        (workdir / name).write_text(content, encoding="utf-8")
        written.append(name)

    if not written:
        raise ResponseError(
            "the response contained none of the files the task asked for "
            f"({', '.join(sorted(allowed))})"
        )
    return written


class ModelAdapter:
    """Base class for adapters that call a hosted model."""

    name = "model"
    #: How many edit-and-run cycles an attempt may use. Recorded in results.
    max_attempts = 1

    def __init__(self, model: str | None = None) -> None:
        #: The model id this adapter was asked for. Adapters are constructed
        #: per run, so the id travels with the instance rather than through a
        #: lookup table of every model that will ever exist.
        self.model = model or self.name

    def __call__(self, task: Task, workdir: Path) -> None:
        self.solve(task, workdir)

    def solve(self, task: Task, workdir: Path) -> None:
        raise NotImplementedError(
            f"the {self.name} adapter is a skeleton. "
            "Run `scratchbench run --model reference` until it is written; "
            "see adapters/model_api.py for the contract."
        )
