"""The Anthropic adapter: ask a Claude model to fill in a task's starter files.

One model call per task, and that is a measurement decision rather than a
budget one. A harness that re-prompts until something passes reports how good
its own retry loop is; the number this benchmark publishes should be about the
model. `max_attempts` is written into every results file so a reader never has
to take that on trust.

What this adapter is careful about:

- **It never sees the tests.** It reads `task.prompt()` and the starter files
  the runner already copied into `workdir`. `hidden_tests/` is not on disk yet
  when `solve` runs, and `reference/` is never touched.
- **It only writes files the task declared.** The response schema restricts
  filenames to the starter's own names and the writer checks again. A model
  that could drop a `conftest.py` into the graded directory could rig its own
  score, which would be a hole in the benchmark rather than a bug in a task.
- **It refuses to guess when the response is not a measurement.** A truncated
  response, a refusal, or an answer served by a different model than the one
  asked all raise, and `grade` turns that into `adapter_error` — no pass, no
  fail, and a non-zero exit for the whole run. Scoring those as failures would
  file a harness problem under the model's name.
- **It does not enable server-side fallbacks.** They are the right default for
  an application, which wants an answer; they are wrong here, because a
  silently substituted model would publish one model's score under another's
  name. A refusal is reported as an absence of evidence instead.
- **It sets no sampling, thinking or effort knobs.** Every model is asked at
  its own defaults. The alternative is a per-model table of what each one
  accepts, which is a table that goes stale silently and, worse, makes two
  columns of the leaderboard mean different things: a score measured at
  `effort: high` and a score measured at whatever a cheaper model happens to
  support are not comparable, and nothing in the results file would say so.
  What this benchmark reports is the model as the API ships it.

Transport-level retries the SDK performs on 429s and 5xx are not attempts in
the sense above: the model gets no feedback from them, so they cannot teach it
anything about the task. Only edit-and-run cycles count.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from runner.tasks import Task

from .model_api import ModelAdapter

#: Used when the model is named by its adapter alias (`--model anthropic`)
#: rather than by a model id.
DEFAULT_MODEL = "claude-opus-5"

#: An alias resolves to a dated snapshot: ask for `claude-haiku-4-5` and
#: `claude-haiku-4-5-20251001` answers. That is the same model under its full
#: name, so both the identity check and the price lookup have to allow it — and
#: allow nothing else, because anything else is a substitution.
SNAPSHOT_SUFFIX = re.compile(r"-\d{8}$")

#: USD per million tokens, (input, output), at the published list price. Only
#: models this table knows can be costed; anything else reports `usd_cost:
#: null` rather than a made-up number. Promotional and introductory rates are
#: deliberately not modelled: a cost that depends on the date it was run is not
#: a property of the model, and a leaderboard row outlives the promotion.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

#: Cache writes cost more than fresh input and cache reads cost far less. This
#: adapter does not cache — every task is a different prompt asked once — but
#: the accounting covers it so the cost stays right if that ever changes.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

#: Generous, because a task's answer is one or more complete source files and a
#: response cut off mid-file is not a wrong answer, it is a missing one. The
#: call streams, so a large ceiling costs nothing until it is used.
MAX_TOKENS = 64000

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


class AnthropicResponseError(RuntimeError):
    """The response cannot be graded, and is not evidence about the model."""


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
        raise AnthropicResponseError(
            "the response contained none of the files the task asked for "
            f"({', '.join(sorted(allowed))})"
        )
    return written


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
    """USD for these four counts, or None if this model has no price here.

    The arithmetic lives here, in one function, because the same sum is done in
    two places that must never disagree: the adapter costing a live call, and
    `tools/check_cost.py` recomputing a published figure from the counts in a
    results file. A second copy would make the check pass by construction.
    """
    price = PRICES.get(model) or PRICES.get(SNAPSHOT_SUFFIX.sub("", model))
    if price is None:
        return None
    input_price, output_price = price
    billed_input = (
        input_tokens
        + cache_write * CACHE_WRITE_MULTIPLIER
        + cache_read * CACHE_READ_MULTIPLIER
    )
    return (billed_input * input_price + output_tokens * output_price) / 1e6


def price_of(model: str, usage) -> float | None:
    """USD for one call, or None if this model has no published price here."""
    return price_from_counts(
        model,
        usage.input_tokens or 0,
        usage.output_tokens or 0,
        cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class AnthropicAdapter(ModelAdapter):
    """Solve tasks with a Claude model, one call each."""

    name = "anthropic"
    max_attempts = 1

    def __init__(self, model: str | None = None) -> None:
        # `--model anthropic` means "this adapter's default model"; anything
        # else is passed through as a model id, so a new model needs no code.
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

        A published cost with no usage behind it is a number the reader has to
        take on trust, and so do I: a stale price and a run that used more
        tokens than expected produce the same figure. These four counts and
        `PRICES` reproduce it exactly.
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
                import anthropic
            except ImportError as error:  # pragma: no cover - environment
                raise AnthropicResponseError(
                    "the anthropic SDK is not installed: pip install 'scratchbench[anthropic]'"
                ) from error
            # Ten minutes: a task's answer is a whole source file and this
            # model thinks before it writes.
            self._client = anthropic.Anthropic(timeout=600.0)
        return self._client

    def solve(self, task: Task, workdir: Path) -> None:
        workdir = Path(workdir)
        allowed = {path.name for path in workdir.glob("*.py")}
        if not allowed:
            raise AnthropicResponseError(f"{task.slug}: no starter files in the workdir")

        # Streamed because `max_tokens` is large and a non-streaming request
        # that size runs into the SDK's own timeout guard. `output_config` is
        # the only knob set, and it constrains the shape of the answer rather
        # than the model's behaviour.
        with self.client().messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": response_schema(sorted(allowed))},
            },
            messages=[{"role": "user", "content": build_prompt(task, workdir)}],
        ) as stream:
            message = stream.get_final_message()

        self._record(task, message)

        if message.stop_reason == "refusal":
            raise AnthropicResponseError(
                f"{self.model} declined the request; no evidence either way"
            )
        if message.stop_reason == "max_tokens":
            raise AnthropicResponseError(
                f"the response hit the {MAX_TOKENS} token ceiling and is incomplete"
            )
        if not same_model(self.model, message.model):
            # Nothing in this adapter asks for a substitution, so a different
            # model here means the run would publish its answer under the
            # requested model's name.
            raise AnthropicResponseError(
                f"asked {self.model} and was answered by {message.model}"
            )

        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise AnthropicResponseError(f"response was not JSON: {error}") from error

        write_files(workdir, payload, allowed)

    def _record(self, task: Task, message) -> None:
        cost = price_of(message.model, message.usage)
        if cost is None:
            self._priced = False
        else:
            self._cost += cost
        usage = message.usage
        self.calls.append(
            {
                "slug": task.slug,
                "model": message.model,
                "stop_reason": message.stop_reason,
                "input": usage.input_tokens or 0,
                "output": usage.output_tokens or 0,
                "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                "usd_cost": cost,
            }
        )
