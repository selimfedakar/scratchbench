# CLAUDE.md — scratchbench

A benchmark that asks one question: **can the model actually implement it?** ML
machinery — attention masks, RoPE, KV-caches, gradient accumulation,
quantization — written from a specification and graded by hidden pytest suites.
Public repository, and the owner's public evidence of ML engineering
competence: code and prose quality here are the product.

## Thinking protocols (mandatory)

Read `~/claude_efficiency/THINKING_PROTOCOLS.md` at session start
(public mirror: https://github.com/selimfedakar/my_claude_efficiency).
Non-negotiable triggers:
any bug → T2 (3 hypotheses + evidence before fix) and search `docs/PATTERNS.md` first;
any edit to existing code → T1 (read + call sites);
any non-trivial feature → T4 altitude check + T7 edge grid;
any fix → T3 root-cause gate; any design → T5 least-mechanism order;
multi-step task → T6 assumption ledger; before any commit plan/report → T8 rule refresh.
Before presenting substantial work, self-grade with `~/claude_efficiency/FABLE_BAR_RUBRIC.md`.

## Verification levels

Every completion claim carries a level from
`~/claude_efficiency/VERIFICATION_PROTOCOL.md`:
L0 compiled · L1 tests pass · L2 exercised and observed · L3 + adversarial + sweep.

**In this repository L2 means a task was actually run and its output pasted. A
passing test on its own is L1.** For a task, L2 is specifically *both* halves:

1. the reference solution passes the hidden tests, and
2. the untouched starter fails them with real assertion errors — not import
   errors, not collection errors.

Both outputs get pasted. A task with only half of that pair is not ready, and
saying it is ready is the one failure this repository cannot survive: a broken
task produces a wrong number about a model, published under the owner's name.

## Commands

```bash
pip install -e .                                   # editable install, gives the CLI

scratchbench validate                              # every task: reference passes, starter fails cleanly
scratchbench validate --tasks kv_cache_equivalence # one task
scratchbench run --model reference --tasks all     # full harness path, reference as the solver
scratchbench report                                # table over results/*.json

python -m pytest tasks/<slug>/hidden_tests -q      # do NOT: tests need the module beside them
```

Python is Anaconda 3.10 at `/Users/selimfedakar/anaconda3/bin/python3`.

## Map

| Path | Holds |
|---|---|
| `README.md` | the pitch. Design is locked — implement it, do not redesign it. |
| `TASK_FORMAT.md` | the task contract. Also locked. Read before writing any task. |
| `tasks/<slug>/` | one task: `meta.yaml`, `prompt.md`, `starter/`, `hidden_tests/`, `reference/` |
| `runner/tasks.py` | task discovery and `meta.yaml` validation |
| `runner/sandbox.py` | temp workdir assembly, seeding, timeout, pytest invocation |
| `runner/cli.py` | `scratchbench run` / `report` / `validate` |
| `runner/report.py` | results JSON schema, aggregation, the printed table |
| `adapters/` | model adapters. `reference.py` is real; the rest are skeletons. |
| `results/` | one JSON per run, gitignored except `.gitkeep` |
| `leaderboard/` | published results, checked in by hand |

## How a run works

`scratchbench run` discovers tasks under `tasks/`, and for each one: makes a
fresh temp directory, copies `starter/` into it, asks the adapter to edit those
files (the `reference` adapter simply overwrites them with `reference/`), copies
`hidden_tests/` in beside them, and runs pytest there with a fixed seed
environment and the task's `time_limit_s` as a hard timeout. Exit code 0 with
zero failures is a pass; anything else is a fail. Scoring is binary per task —
`TASK_FORMAT.md` is explicit that partial credit hides the failures worth
knowing about. Results land in `results/<model>-<timestamp>.json`.

The flat temp directory is deliberate: `hidden_tests/` and the solution files
sit side by side, so a test does `from kv_cache import CachedAttention` with no
path juggling. Each task therefore has exactly **one** module file, named the
same in `starter/` and `reference/`.

## Writing a task (the order is not optional)

`reference/` → `hidden_tests/` → `prompt.md` → `meta.yaml`. Writing the
reference first surfaces the conventions you were about to leave unstated;
writing the prompt last means you are describing something that exists.

## Landmines

- **A hidden test that can fail for a reason `prompt.md` never states is a
  broken task.** This is the single most common task-authoring mistake and it
  is the one that silently invalidates a benchmark number. After writing the
  tests, walk each assertion and point at the sentence in the prompt that
  licenses it. No sentence → add it to the prompt or delete the assertion.
- **Then walk it the other way: every promise in the prompt needs a test that
  enforces it.** An unlicensed assertion fails a model for guessing a different
  convention; an unenforced promise lets two disagreeing implementations both
  pass, which means the score no longer measures one thing. A promise that
  cannot be tested does not belong in the prompt (see `docs/LESSONS.md` L8).
- **Mutate the reference before calling a task ready.** Write the three or four
  wrong versions a real implementation would produce and confirm the tests
  catch each one. Passing tests prove the reference is right; only mutants
  prove the tests would catch anything else. An equivalence property is blind
  to any mistake that is symmetric across the two things it compares (L3).
- **Do NOT copy anything from any curriculum** — not CS336, not any course:
  no code, no tests, no assignment prose, no problem numbering. The README
  claims this in public. Every task is written from scratch. Inspiration is
  fine; redistribution is not.
- **Do NOT enumerate edge cases as a checklist in `prompt.md`.** State
  conventions as a colleague handing over work would, in prose. A prompt that
  lists its own edge cases measures reading comprehension instead of
  engineering.
- **Never leak the tests.** `prompt.md` does not name test files, test
  functions, tolerances chosen for tests, or the number of tests.
- **Starter files must import and collect cleanly.** Signatures correct,
  imports present, bodies `raise NotImplementedError`. An import error and a
  wrong answer score the same and mean completely different things.
- **No GPU, ever, and no task over five minutes.** That constraint is the
  entire reason this benchmark exists; a task that violates it is not a hard
  task, it is a different project.
- **Tests are deterministic.** Seeded RNG (`np.random.default_rng(seed)`,
  `torch.manual_seed`), no network, no wall-clock assertions, no dependence on
  dict/set iteration order.
- **Numeric tolerances carry a comment justifying the number.** `atol=1e-9`
  with no reason is a future flake.
- **New dependencies need a written justification** in the task's `meta.yaml`
  `deps` and in the session report. Today: `numpy` everywhere, `torch` for
  `grad_accumulation` only — gradient equivalence cannot be measured without
  autograd, and hand-rolling a backward pass would measure a different skill.

## Workflow rules (non-negotiable)

- **Never run `git commit` or `git push`.** Selim runs them. Append the exact
  commands to `COMMITS.md`, one file per commit, and stop there.
- **Atomic commits: one file per commit.** No exceptions.
- **No AI attribution anywhere** — commits, comments, docs, none of it.
- **English only** in every source file and document in this repository.
- **A task is not "done" until both halves of L2 are pasted** (see above).

## Session journal and lessons

Every step of the build gets a short entry in `docs/sessions/NN-title.md`, in
English, first person: what was done, why, which technology carries it, and
what was verified with pasted output. Written as the step finishes, not
retrospectively. 00 through 03 exist; the next one is 04.

`docs/LESSONS.md` is the other half and it is **mandatory, every session**:
what I got wrong, in my own voice, newest first. Not a changelog — the entries
are the mistakes, the near-misses, and the fixes I almost made instead. An
entry that makes me look good is usually the wrong entry. The format is fixed:
what I expected · what happened · why it survived or where I nearly went wrong ·
what changed · what it cost.

## Reporting to Selim (every report, no exceptions)

Reports follow this order, and none of the five sections is optional:

1. **Flow** — the steps in the order they happened, each with what it produced.
2. **Evidence** — a table of claim · level · pasted observation. Numbers come
   from this session's output, never from memory.
3. **Not verified** — every open claim, each one *checked with a command before
   it is written down*. "Not verified" is itself a claim: run the command that
   proves the gap is real, do not assert it from memory.
4. **What I need from you** — numbered, concrete, each with a recommended
   default so a one-word answer is enough. If there is genuinely nothing, say
   "nothing" and mean it.
5. **Self-review** — the D1–D8 line from `~/claude_efficiency/FABLE_BAR_RUBRIC.md`
   with the weakest dimension named.

**The rule that outranks the rest: never end a report with a dangling
concern.** Raising a problem and leaving it — "I did X but it should have been
your call", "this is a gap, flagging it" — is half a job. Either resolve it and
say what was decided, or turn it into a numbered item in section 4 with a
default and a one-line undo. A concern that is only mentioned is a concern that
will be forgotten by both of us.

## Tiers

`laptop` is the default and the headline: no GPU, reproducible by anyone,
`requires_gpu` refused by the loader, and the only tier the leaderboard rate is
computed over. `accelerated` may set `requires_gpu: true` and must declare
`accelerator: cuda | metal`; it is reported beside the laptop rate and never
folded into it. Missing hardware returns `needs_accelerator` — not a pass, not
a failure, an absence of evidence. An accelerated task stays out of the frozen
set until its reference has actually run on hardware. See `TASK_FORMAT.md`.

## State as of 2026-07-28 (verify before trusting)

- **Not a git repository yet.** `git init` is the first line of `COMMITS.md`,
  which now holds six blocks, one file per commit, waiting for Selim. The
  public repository `selimfedakar/scratchbench` does not exist yet.
- Tasks: **nine — eight L2 verified on the laptop tier, one unvalidated.**
  `fused_rmsnorm_kernel` (kernels, difficulty 4, tier `accelerated`,
  `accelerator: cuda`, Triton, 25 hidden tests) is written and has **never
  run**: no CUDA device here. It is `frozen_set: unvalidated`, out of the
  leaderboard, and stays there until its reference passes on hardware, its
  untouched starter fails cleanly, and its five mutants are caught. The
  mutation script lives outside the repository; the three open questions are
  listed in `docs/sessions/05`.
- Adapters: `anthropic` is **real** (`adapters/anthropic_api.py`) —
  one call per task, streamed, JSON-schema-constrained to the task's own
  filenames, no server-side fallbacks, and anything that is not a measurement
  (refusal, truncation, model substitution) raises into `adapter_error`.
  Verified **offline only**: eight tests cover the prompt, the write
  allowlist and the cost arithmetic. **No live API request has been made.**
  The SDK is an extra: `pip install 'scratchbench[anthropic]'`. `openai` is
  still a skeleton.
- Results now carry `attempts` and `usd_cost` off the adapter, plus `measured`
  and `not_measured`. Outcomes with no evidence are kept out of every rate; a
  tier with nothing measured reports `pass_rate: null`, not `0`.
- Runner: **L2** — `validate --tier all` reports 8 ok and 1 not checked here;
  `run --model reference --tasks all --tier all` reports laptop 100% (8/8) and
  accelerated not run, in 4.5s. Its own suite: **L1** — 56 passed in 16.64s.
- CI: the laptop job is unchanged and still **L0** (no remote yet). A second,
  manual `accelerated` job targets a `[self-hosted, gpu]` runner and asserts
  `torch.cuda.is_available()` before validating anything — no such runner
  exists, so it has never run either.
- Next: run `fused_rmsnorm_kernel` on hardware (Selim is renting a GPU), then
  the first real model sweep and the first leaderboard entry.

- `README.md` and `TASK_FORMAT.md`: design locked. The laptop task set is
  unchanged and still L2: `softmax_stability` 29 tests · `bpe_merge_order` 30 ·
  `attention_causal_mask` 26 · `kv_cache_equivalence` 21 · `grad_accumulation`
  27 · `sharded_dataloader` 53 · `quantization_error_bounds` 65 ·
  `online_softmax_attention` 57, five of them mutation-tested.
- Environment: numpy 1.23.5, pytest 7.1.2, torch 2.8.0, PyYAML 6.0, all present
  under Anaconda 3.10. M1 Pro, 16 GB, **no GPU** — which is why the accelerated
  tier exists and why nothing on it can be verified here.
- Standing decisions: the leaderboard rate is the laptop tier only; the
  Anthropic adapter gets **one attempt per task**, because a model that passes
  on the eleventh try measures the scaffolding; an accelerated task stays out
  of the frozen set until its reference has run on real hardware.
