"""The adapter contract, and the skeletons that have not been filled in yet.

The reference adapter came first, and the reason is worth stating rather than
apologising for: a harness that has never been proven against a known-correct
solution will happily report that a model failed when what actually failed was
the harness. CI runs the reference on every commit, and only then does pointing
a model at this benchmark produce a number anyone should believe.

`adapters/anthropic_api.py` is the first real one and is written against the
contract below; `OpenAIAdapter` still raises.

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


class OpenAIAdapter(ModelAdapter):
    name = "openai"
