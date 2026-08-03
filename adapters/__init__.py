"""Adapters: the thing that turns a starter directory into an attempt.

An adapter is any callable with the signature

    solve(task: Task, workdir: Path) -> None

It is given a directory holding the task's starter files, and it may edit them
however it likes. It is called *before* the hidden tests are copied in, so it
cannot read them. What it leaves on disk is what gets graded.

`reference` came first, for a reason that outranks convenience: the runner has
to be proven against a known correct solution before any model's score is worth
writing down. `anthropic` is the first real model adapter; `openai` is still a
skeleton — see `adapters/model_api.py`.

Adapters are built per run rather than kept as module-level singletons, because
the model id is an argument to the adapter: `--model claude-opus-5` and
`--model claude-haiku-4-5` are the same code and different measurements.
"""

from __future__ import annotations

from .anthropic_api import AnthropicAdapter
from .model_api import OpenAIAdapter
from .reference import solve as reference_solve


def _reference(model: str):
    """The control solver takes no model id — it copies a known-good answer."""
    return reference_solve


ADAPTERS = {
    "reference": _reference,
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
}

# Model names are routed to an adapter by prefix, so `--model claude-opus-5`
# reaches the Anthropic adapter without a lookup table of every model id.
PREFIXES = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
)


def resolve(model: str):
    """Return the solver for a `--model` argument, configured with that model."""
    if model in ADAPTERS:
        return ADAPTERS[model](model)
    for prefix, adapter in PREFIXES:
        if model.startswith(prefix):
            return ADAPTERS[adapter](model)
    known = ", ".join(sorted(ADAPTERS))
    raise KeyError(f"no adapter for model '{model}'. Known adapters: {known}")
