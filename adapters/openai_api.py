"""The OpenAI adapter: ask a GPT model to fill in a task's starter files.

Written against the same contract as `anthropic_api.py`, and deliberately as
close to it as the two APIs allow. Every rule in that module's docstring applies
here for the same reasons, so they are not repeated: one call per task, the
tests are never on disk when `solve` runs, only declared filenames are written,
an ungradeable response raises rather than scoring zero, and no sampling or
effort knob is set.

The prompt, the response schema and the writer are imported from `model_api`
rather than copied. That is the point of them living there: a leaderboard whose
Claude column and GPT column were asked different questions is not a comparison,
and nothing in a results file would reveal it.

Two things differ from the Anthropic adapter, and both are properties of the
API rather than choices:

- **The Responses API, not Chat Completions.** It is where structured output
  and the reasoning models are, and `response.output_text` gives the assembled
  text without walking content blocks.
- **Cached input is a price, not a multiplier.** Anthropic bills cache reads as
  a fraction of the input rate; OpenAI publishes a third per-million figure. So
  `PRICES` here carries three numbers per model and `price_from_counts` takes
  the cached count out of the input count rather than scaling it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from runner.tasks import Task

from .model_api import (
    INSTRUCTIONS,
    SYSTEM_PROMPT,
    ModelAdapter,
    ResponseError,
    build_prompt,
    response_schema,
    write_files,
)

#: Used when the model is named by its adapter alias (`--model openai`) rather
#: than by a model id.
DEFAULT_MODEL = "gpt-5.6-sol"

#: `gpt-5.4` resolves to `gpt-5.4-2026-03-05`. Same model under its full name,
#: so the identity check and the price lookup both have to allow it, and allow
#: nothing else — anything else is a substitution.
SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

#: USD per million tokens, (input, cached input, output), at the published list
#: price on 2026-09-08. Only models this table knows can be costed; anything
#: else reports `usd_cost: null` rather than a made-up number. Promotional and
#: batch rates are deliberately not modelled: a cost that depends on the day it
#: was run is not a property of the model, and a leaderboard row outlives the
#: promotion.
#:
#: The long-context surcharge some models carry above 272K tokens is not
#: modelled either, and it cannot bite here: a task is one prompt file and a few
#: hundred lines of starter, and the largest prompt this repository has ever
#: sent is three orders of magnitude below that threshold.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-astra": (10.0, 1.0, 50.0),
    "gpt-5.6-sol": (4.0, 0.4, 20.0),
    "gpt-5.6-terra": (2.0, 0.2, 12.0),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.5": (5.0, 0.5, 30.0),
    "gpt-5.5-pro": (30.0, 30.0, 180.0),
    "gpt-5.4": (2.50, 0.25, 15.0),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-pro": (30.0, 30.0, 180.0),
    "gpt-5.2": (1.75, 0.175, 14.0),
    "gpt-5.1": (1.25, 0.125, 10.0),
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-5-mini": (0.25, 0.025, 2.0),
    "gpt-5-nano": (0.05, 0.005, 0.40),
}

#: Generous, for the same reason as the Anthropic adapter's: a task's answer is
#: one or more complete source files, and a response cut off mid-file is not a
#: wrong answer, it is a missing one. On a reasoning model this ceiling covers
#: reasoning tokens as well as the visible answer, which is why it is larger
#: here than the text alone would need.
MAX_OUTPUT_TOKENS = 100000


class OpenAIResponseError(ResponseError):
    """The response cannot be graded, and is not evidence about the model."""


def same_model(asked: str, answered: str) -> bool:
    """True when `answered` is `asked`, or the dated snapshot `asked` names."""
    return answered == asked or SNAPSHOT_SUFFIX.sub("", answered) == asked


def price_from_counts(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float | None:
    """USD for these counts, or None if this model has no price here.

    Same signature as the Anthropic function so `tools/check_cost.py` can hold
    one dispatch and no special cases. `cache_write` is accepted and ignored:
    OpenAI does not bill for writing to the prompt cache, and a parameter that
    silently did something different per vendor would be worse than one that
    documents doing nothing.

    `cache_read` is **subtracted** from `input_tokens` rather than added
    alongside it, because the API reports cached tokens as a subset of the
    input count. Adding them would bill the same tokens twice.
    """
    price = PRICES.get(model) or PRICES.get(SNAPSHOT_SUFFIX.sub("", model))
    if price is None:
        return None
    input_price, cached_price, output_price = price
    fresh_input = max(input_tokens - cache_read, 0)
    return (
        fresh_input * input_price
        + cache_read * cached_price
        + output_tokens * output_price
    ) / 1e6


def price_of(model: str, usage) -> float | None:
    """USD for one call, or None if this model has no published price here."""
    details = getattr(usage, "input_tokens_details", None)
    return price_from_counts(
        model,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        cache_read=getattr(details, "cached_tokens", 0) or 0,
    )


class OpenAIAdapter(ModelAdapter):
    """Solve tasks with a GPT model, one call each."""

    name = "openai"
    max_attempts = 1

    def __init__(self, model: str | None = None) -> None:
        self.model = DEFAULT_MODEL if model in (None, self.name) else model
        self._client = None
        self._cost: float = 0.0
        self._priced = True
        self.calls: list[dict] = []

    @property
    def usd_cost(self) -> float | None:
        """Total spend, or None if any call was served by an unpriced model."""
        if not self.calls:
            return None
        return round(self._cost, 6) if self._priced else None

    @property
    def token_usage(self) -> dict[str, int] | None:
        """The tokens `usd_cost` was computed from, so it can be checked.

        Four keys, the same four the Anthropic adapter reports, so one results
        schema covers both vendors. `cache_write` is always zero here and is
        present rather than omitted: a missing key and a zero would be
        indistinguishable to a reader, and only one of them is a measurement.
        """
        if not self.calls:
            return None
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        for call in self.calls:
            for key in totals:
                totals[key] += call[key]
        return totals

    def client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as error:  # pragma: no cover - environment
                raise OpenAIResponseError(
                    "the openai SDK is not installed: pip install 'scratchbench[openai]'"
                ) from error
            # Ten minutes, matching the Anthropic adapter: a task's answer is a
            # whole source file and these models think before they write.
            self._client = openai.OpenAI(timeout=600.0)
        return self._client

    def _record(self, model: str, usage) -> None:
        details = getattr(usage, "input_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        self.calls.append(
            {
                "model": model,
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
                "cache_read": cache_read,
                "cache_write": 0,
            }
        )
        cost = price_of(self.model, usage)
        if cost is None:
            self._priced = False
        else:
            self._cost += cost

    def solve(self, task: Task, workdir: Path) -> None:
        workdir = Path(workdir)
        allowed = {path.name for path in workdir.glob("*.py")}
        if not allowed:
            raise OpenAIResponseError(f"{task.slug}: no starter files in the workdir")

        response = self.client().responses.create(
            model=self.model,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            instructions=SYSTEM_PROMPT,
            input=build_prompt(task, workdir),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "task_files",
                    "strict": True,
                    "schema": response_schema(sorted(allowed)),
                }
            },
        )

        self._record(response.model, response.usage)

        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            raise OpenAIResponseError(
                f"the response is incomplete ({reason}) and the files it "
                "contains may be cut off mid-line"
            )
        if response.status != "completed":
            raise OpenAIResponseError(
                f"the response ended with status {response.status!r}; no evidence "
                "either way"
            )
        # A refusal is not a wrong answer. It arrives as its own content type
        # rather than as text, so looking only at `output_text` would turn one
        # into an empty response and then into a parse error.
        for item in response.output:
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) == "refusal":
                    raise OpenAIResponseError(
                        f"{self.model} declined the request "
                        f"({content.refusal}); no evidence either way"
                    )
        if not same_model(self.model, response.model):
            # Nothing here asks for a substitution, so a different model means
            # the run would publish its answer under the requested model's name.
            raise OpenAIResponseError(
                f"asked {self.model} and was answered by {response.model}"
            )

        text = response.output_text
        if not text:
            raise OpenAIResponseError("the response carried no text to parse")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise OpenAIResponseError(
                f"the response is not the JSON the schema asked for: {error}"
            ) from error

        write_files(workdir, payload, allowed)
