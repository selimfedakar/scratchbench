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
scratchbench run --model <m> --tasks all --set v1  # one published set: a leaderboard row wants one
scratchbench report                                # table over results/*.json

python -m pytest tasks/<slug>/hidden_tests -q      # do NOT: tests need the module beside them
```

Python is Anaconda 3.10 at `/Users/selimfedakar/anaconda3/bin/python3`.

## Map

| Path | Holds |
|---|---|
| `README.md` | the pitch. Design is locked — implement it, do not redesign it. |
| `TASK_FORMAT.md` | the task contract. Also locked. Read before writing any task. |
| `docs/ROADMAP.md` | **every remaining session from here to launch, one section each.** Open the section for the session you are in and work it; its *Preconditions* are a stop condition, not a formality. |
| `tasks/<slug>/` | one task: `meta.yaml`, `prompt.md`, `starter/`, `hidden_tests/`, `reference/` |
| `runner/tasks.py` | task discovery and `meta.yaml` validation |
| `runner/sandbox.py` | temp workdir assembly, seeding, timeout, pytest invocation, **and `STATUSES`: the one table saying which outcomes count as evidence** |
| `runner/cli.py` | `scratchbench run` / `report` / `validate` |
| `runner/report.py` | results JSON schema, aggregation, the printed table |
| `adapters/` | model adapters. `reference.py` and `anthropic_api.py` are real; the OpenAI one is a skeleton class in `model_api.py`, not its own file. |
| `tools/` | `mutate_rmsnorm.py` and `verify_accelerated.sh` (the CUDA task's evidence, on a rented box), `mutate_metal_task.py` (the Metal task's, on any Apple silicon Mac), `mutate_v2_tasks.py` (the same pass for the laptop-tier v2 candidates), `check_cost.py` (every published cost, re-derived from its tokens), `check_calibration.py` (every `calibration:` block, re-derived from its draws) |
| `results/` | one JSON per run, gitignored except `.gitkeep` |
| `leaderboard/` | published results, checked in by hand |
| `calibration/` | the draws every `calibration:` block was computed from. Checked in, and re-derived in CI, because a task is refused from a frozen set on the strength of those numbers |
| `notes/` | the author's Turkish study notes. Gitignored, never published, never copied into a tracked file. |

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
writing the prompt last means you are describing something that exists. It also
surfaces the design claims you were about to leave unchecked, which is one level
up from what the order was introduced for (L28).

Then the task is calibrated before it can enter a numbered frozen set from `v2`
onward: several independent draws per model, recorded in `meta.yaml`, and a task
that the best model tried never fails is refused by the loader. See
`TASK_FORMAT.md`. `difficulty` is an opinion until that block exists beside it.

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
- **An outcome that produced no evidence is never a zero.** Missing hardware, a
  missing optional dependency, an adapter that never returned a gradeable file,
  tests that could not be collected: none of them scored the solution, so none
  of them enters a rate. The classification lives in exactly one place,
  `STATUSES` in `runner/sandbox.py`, and everything derives from it. This has
  now been got wrong twice (L11, L20), both times because a second list existed
  somewhere. Do not write a second list.
- **A published number is one draw, not a converged value.** The same model on
  the same task passed once and failed once. One attempt per task is still the
  rule, because a retry with feedback measures the scaffolding — but a single
  sweep is a sample, and any leaderboard row has to be labelled as one (L19).
- **New dependencies need a written justification** in the task's `meta.yaml`
  `deps` and in the session report. Today: `numpy` everywhere, `torch` for
  `grad_accumulation` only — gradient equivalence cannot be measured without
  autograd, and hand-rolling a backward pass would measure a different skill.

## Workflow rules (non-negotiable)

- **Never run `git commit` or `git push`.** Selim runs them. Append the exact
  commands to `COMMITS.md`, one file per commit, and stop there.
- **Atomic commits: one file per commit.** No exceptions.
- **No AI attribution anywhere** — commits, comments, docs, none of it.
- **English only** in every source file and document in this repository. The
  single exception is `notes/`, which holds the author's Turkish study and
  speaking notes, is gitignored, and must stay that way. It is kept beside the
  project so it is not lost, never published. Never move its content into a
  tracked file.
- **A task is not "done" until both halves of L2 are pasted** (see above).

## Session journal and lessons

Every step of the build gets a short entry in `docs/sessions/NN-title.md`, in
English, first person: what was done, why, which technology carries it, and
what was verified with pasted output. Written as the step finishes, not
retrospectively. 00 through 12 exist; the next one is 13.

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

## State as of 2026-08-17, session 12 (verify before trusting)

- **The admission rule reads the top two calibration entries, not the best one.**
  A numbered set from `v2` onward now needs two entries of five draws or more and
  refuses the task only if the highest two pass rates are **both 100%**. Entries
  under five draws are read on neither side, which is the absence-of-evidence
  rule one column over. `_check_admission` in `runner/tasks.py`;
  `CALIBRATION_TOP_N = 2` beside `CALIBRATION_MIN_DRAWS = 5`.
- **It moves exactly one task.** `flash_attention_backward` goes `warmup` → `v2`
  on Opus 15/15 with Sonnet 8/10. `speculative_decoding_verify` and
  `activation_checkpointing_rng` stay in `warmup` — Opus *and* Sonnet clear both.
- **`fused_rmsnorm_kernel` moved `v1` → `v2` in the same session, and it is not
  a consequence of the rule change** — it clears the old rule too. Opus 1/5,
  Haiku 3/5, from the 2026-08-09 RTX 4000 Ada sweep already in `leaderboard/`;
  the only thing missing was a `calibration:` block. Sonnet has never been asked
  it (rented box). Lowest frontier rate in the repository.
- **`v2` has three of its four intended members:** `flash_attention_backward`
  (laptop), `fused_rmsnorm_kernel` (accelerated, cuda),
  `metal_cross_entropy_kernel` (accelerated, metal). The fourth is a second Metal
  task, not written. Set counts today: `v1` **five** (all laptop, no accelerated
  member), `v2` **three**, `warmup` **five**.
- **The rule changed after it refused a task I liked, and journal 10 had
  already decided the other way.** That entry argued *for* the strict rule and
  set a revisit condition — a model between Sonnet and Opus — which has **not**
  been met. The overturn rests on the other half of its own argument: the loader
  cannot tell convention noise from an absent mechanism, only the author can, and
  admitting on the author's reading is the channel L21 closed. `docs/LESSONS.md`
  **L35** is the honest version; journal 12 carries the tables.
- **Rejected:** "admit a task if any two models differ". `grad_accumulation`
  separates Haiku (0 of 6) from Opus (6 of 6) and Sonnet (5 of 5), and so does
  most of v1 — that rule readmits everything L21 was written about.
- Numbers from this session's runs: harness suite **105 passed** (two tests
  added, one rewritten); `validate --tier all` **12 task(s) validated, 1 not
  checked here**; control run **laptop 100% (11/11) · accelerated 100% (1/1)** in
  25.3s; `check_cost.py` **57 file(s) checked**; `check_calibration.py` **23
  calibration entries re-derived from 122 draw(s)**;
  `tools/mutate_metal_task.py` **all 13 mutants behaved as expected**.
- **No model was asked anything.** Every rate above is a published draw from
  2026-08-09 or 2026-08-13. Spend this session: **$0**.
- **Still open, and `docs/ROADMAP.md` is now the plan of record for all of it.**
  `v2` is three tasks and still being assembled rather than published, so there
  is no v2 leaderboard row and **no `--set v2` sweep has ever been run** — the
  three members' draws come from three different days against three different
  task sets.
- **The gating problem, written down for the first time in `ROADMAP.md` §0.1:
  `v2` has exactly one laptop task**, and the leaderboard headline is the laptop
  tier by design (`HEADLINE_TIER` in `runner/report.py`). A headline over one
  task is not a benchmark. The next session's first job is the decision in
  `ROADMAP.md` §1.2 — recommended: keep the tiers and hold `v2` back until its
  laptop half is real — and that decision determines which tier the next task is
  written for, so it comes before any task writing.

## State as of 2026-08-13, session 11 (verify before trusting)

- **The Metal task exists and it is the first task a frontier model fails.**
  `metal_cross_entropy_kernel` (kernels, `accelerated`, `accelerator: metal`,
  torch only, 68 hidden tests, `frozen_set: v2`). Ten draws per model:
  `claude-opus-5` **8/10**, `claude-sonnet-5` **4/10**, `claude-haiku-4-5`
  **1/9** — three models, three rates, in order, which no other single task here
  does. Haiku's tenth draw was an API `overloaded` error, measured nothing, and
  is out of both sides of the rate.
- **No rented hardware.** `torch.mps.compile_shader(source)` compiles a Metal
  string at runtime and returns callable kernels, so the model writes `SOURCE`
  and the harness compiles and dispatches it. The launch geometry is therefore
  not the solution's to choose, which is the `flash_attention_backward` move one
  tier over (L28). Four measured API facts carry the task: threadgroup memory
  bound as a `[[threadgroup(0)]]` argument is **zero length and silently
  reads zeros**, unwritten threadgroup memory is garbage rather than zeros,
  `simd_sum` reduces a simdgroup and never a threadgroup, and the dispatch is
  non-uniform so `threads_per_threadgroup` is not constant.
- **The failures have a name: `half`.** Both of Opus's failures and five of
  Sonnet's six are the same compile error — `half` is a type in Metal Shading
  Language and cannot name a variable — with a correct reduction underneath. I
  made the same mistake in the first draft of the reference (L34). Haiku fails
  differently: six draws call Metal functions that do not exist, and one wrote a
  fold whose stride never reaches zero, hung the GPU, and was killed by the
  300 second limit, the first `timeout` this repository has recorded.
- **Thirteen mutants, three of them expected to survive, and the expectation is
  checked in both directions.** `tools/mutate_metal_task.py` fails if a survivor
  is caught. The three: over-folding the maximum is idempotent so it is not a
  mutant at all (L31); the missing write-after-read barrier is a real race this
  hardware will not expose, hunted over thirty configurations (L32); and a
  correct kernel that does the whole row on one lane passes all 68 tests,
  because only a wall-clock assertion could see it and those are banned
  everywhere here (L33).
- **`v1` is five laptop tasks and was eight when its rows were measured.**
  `attention_causal_mask`, `sharded_dataloader` and `quantization_error_bounds`
  are `frozen_set: warmup` with blocks derived from draws that were already
  published: six each for Opus and Haiku, five for Sonnet. The leaderboard page
  says so where the rows are.
- **`tools/check_calibration.py` now walks `calibration/` and `leaderboard/`.**
  Copying seventeen published files into `calibration/` would have put the same
  evidence in the repository twice; the tool reads both and the suite re-derives
  from both. Two published-file tests were stricter than the tool and demanded
  costs from a draw whose adapter never answered — both now accept `n/a`,
  which is the same absence-of-evidence rule one column over.
- Numbers from this session's runs: harness suite **103 passed**;
  `validate --tier all` **12 task(s) validated, 1 not checked here**
  (`fused_rmsnorm_kernel`, no CUDA on this machine); control run
  **laptop 100% (11/11) · accelerated 100% (1/1)** in 16.2s;
  `check_cost.py` **57 file(s) checked**; `check_calibration.py` **21
  calibration entries re-derived from 122 draw(s)**. Thirteen tasks, **507**
  hidden tests. `docs/LESSONS.md` runs to **L34**, journals to **11**.
- Spend: **$1.71** for thirty draws.
- **Still open:** `v2` has exactly one member and is being assembled rather than
  published, so there is no v2 leaderboard row. The laptop tier is still
  saturated at the top and every remaining `V2_DESIGN.md` §3 candidate is a
  reasoning task, which §2.0 says is the wrong axis. The obvious next piece of
  work is a second Metal task — a backward kernel — because the tier that
  discriminates is now the tier this laptop can run.

## State as of 2026-08-09, session 10 (verify before trusting)

- **Three v2 candidates written end to end, and all three refused by the
  admission rule.** `speculative_decoding_verify` (attention, numpy, 24 tests),
  `flash_attention_backward` (attention, torch, 49 tests) and
  `activation_checkpointing_rng` (training, torch, 34 tests). Every one is L2 on
  both halves and mutation-tested: 8, 10 and 10 mutants, all caught.
  `tools/mutate_v2_tasks.py` re-runs the whole pass on a laptop.
- **Three models asked.** `claude-opus-5` went 40 draws for 40 (15/15, 15/15,
  10/10). `claude-sonnet-5` went 10/10, **8/10**, 10/10. `claude-haiku-4-5` went
  10/20, 3/19, 1/10. All three tasks are `frozen_set: warmup`, which is the rule
  working rather than the tasks failing.
- **`flash_attention_backward` separates Sonnet from Opus** and is still refused,
  because the rule looks at the best entry. Sonnet's two failures are the same
  pair of tests both times and match a mutant's signature exactly (*queries not
  put at the end of the key range*), so the failure is a convention missed at the
  edge rather than a mechanism absent — which is what `warmup` is for. The
  tension between the rule and its stated intent is written up in journal 10.
- **`L30` is the finding and it is uncomfortable.** The four properties in
  `V2_DESIGN.md` §2 were copied off `fused_rmsnorm_kernel` and they are not what
  makes it hard: its difficulty is an obscure fact about a tool, not a step in an
  argument. §2.0 now says so. The laptop tier's remaining difficulty looks like
  unfamiliarity rather than depth.
- **The calibration machinery exists.** Optional `calibration:` in `meta.yaml`,
  validated in `runner/tasks.py`, and a numbered set from `v2` onward is refused
  by the loader if the best entry passed every draw or if no entry has five
  draws. `frozen_set: warmup` is where a refused task goes. `TASK_FORMAT.md` has
  the table; `runner/cli.py list` prints the block.
- **Two harness defects, both found by a billing failure mid-sweep.**
  `report --variance` folded a half-finished draw into the set rate and printed
  "100% in every draw" over one complete draw (L29, fixed); `--keep` named
  workdirs after the task so nine of ten draws were overwritten and the failure
  shapes were lost (fixed, and it is how the failure tables in journal 10 exist).
- **Credit ran out twice**, both times mid-sweep, and both times the machinery
  reported the hole rather than averaging over it. Sonnet's fourth v1 draw
  measured seven tasks of eight for the same reason and is published as such.
- **The calibration blocks are evidence, not claims.** The 65 draws they were
  computed from are checked in under `calibration/`, `tools/check_calibration.py`
  re-derives all nine entries from them, and CI runs it.
- **`--set` filters a sweep to one frozen set**, on `run`, `validate` and `list`.
  The laptop tier stopped being one published set the moment `warmup` existed, so
  `--tasks all --tier laptop` now averages `v1` with `warmup` and a leaderboard
  row wants one of them. Same shape as L23, one axis over.
- Harness suite **100 passed**; `validate` reports **11 task(s) validated** here
  (the CUDA task is not checkable on this machine); the control run is
  **laptop 100% (11/11)** in 11.3s. Twelve task directories, 439 laptop hidden
  tests. `docs/LESSONS.md` runs to **L30**, journals to **10**.
- Spend: **$7.43**. `tools/check_cost.py` reports **27 file(s) checked**: five
  Sonnet draws were published, and the v1 laptop tier now has a third row at
  100%, every draw.
- **Still open:** the accelerated tier has one task and is still the only tier
  discriminating at the top. `torch.backends.mps.is_available()` is **True** on
  this machine, so a `metal` task is verifiable here without renting anything —
  that is the next piece of work and it is written up in the restart file.

## State as of 2026-08-09, sessions 08 and 09 (verify before trusting)

- **The repository is clean and CI is green.** `main` is ninety commits after a
  full replay; the fifteen red crosses are gone because their commits are.
- **Laptop tier, five draws each.** `claude-opus-5` 40/40, 100% in every draw,
  $0.49 to $0.56 per draw. `claude-haiku-4-5` 18/40, 38% to 50%. Opus's input is
  15858 tokens every draw and its output ranges 16476 to 19070: the model is
  visibly sampling and the score does not move. **The laptop tier is saturated
  at the top, measured rather than suspected.**
- **Accelerated tier, first model runs ever.** `fused_rmsnorm_kernel` puts Opus
  at **1 of 5** and Haiku at 3 of 5, on an RTX 4000 Ada. The rates are not
  distinguishable at n=5; **the failure shapes are the finding**: every captured
  Opus failure (4 for 4) is the same test, `test_a_single_column`, while Haiku's
  failures are 24 of 24 and 23 of 24 tests. A working kernel missing an edge and
  a kernel that does not run are the same word under binary scoring.
  Cause: Triton specializes an integer argument whose value is 1 into a
  compile-time constant, so `n_cols.to(tl.float32)` fails only at `N = 1`.
  `prompt.md` licenses that case explicitly, which was checked before the result
  was written down. See L27.
- **`solution_error` earned itself in its first real use.** Haiku emitted
  unparseable Python in three of five laptop draws. Under the pre-L22
  classification each of those would have published 4/7 = 57% instead of
  4/8 = 50%, and exited non-zero.
- **The tier split verified in the wild:** `run --tier all` on the GPU box
  printed `accelerated 100% (1/1)` and `laptop 100% (8/8) · headline` as two
  lines, with `kernels` absent from the headline categories, and every
  accelerated-only results file carries `"pass_rate": null` rather than
  promoting the accelerated number into the headline slot.
- Leaderboard: **22 results files**, every cost re-derives from its own tokens.
  Everything on the page cost $3.30 total.
- Harness suite **78 passed** on macOS, **77 passed 1 skipped** on Linux+CUDA
  (the skip is the missing-hardware test, correctly inapplicable there).
  `docs/LESSONS.md` runs to **L27**.
- `docs/V2_DESIGN.md` updated: the accelerated tier is the only one still
  discriminating, so it grows rather than shrinks, and §1b states the v2
  specification using a task that actually separates frontier models instead of
  using my intuition.
- Still open: **no v2 task is written**, `kernels` has one task and the README
  promises Metal too, and only two models have been asked anything.

## State as of 2026-08-03, session 07 (verify before trusting)

Everything in the older block below still holds except where this one contradicts
it. Session 07 changed no tasks and audited the arithmetic instead.

- **The headline `pass_rate` is one tier, and it is named in the file.**
  `HEADLINE_TIER` in `runner/report.py`, `headline_tier` in the payload,
  `headline_of()` for anything that renders a row. It used to average whatever
  tiers a run measured, which could not fire on a machine with no GPU and would
  have fired on the first `--tier all` sweep on the rented box (L23).
  `RESULTS_VERSION` is **2**; version 1 files were all laptop-only and read
  correctly without a `headline_tier`.
- **`solution_error` is a new status: evidence, and a failure.** A graded run
  that will not collect is attributed by re-running the hidden tests against the
  untouched starter. Starter collects and fails properly → the solution broke it
  → `solution_error`. Starter cannot collect either → the task is broken →
  `collection_error`, unchanged. A model that emitted unparseable Python used to
  drop out of the denominator (L22).
- **`--repeat N` and `report --variance`.** N independent sweeps, one complete
  results file each, `repeat: {group, index, of}` inside. Not retries: no draw
  sees another's results and `max_attempts` is still 1. Adapters are built per
  sweep or the cost accumulates across draws; the draw index is in the filename
  or draws inside the same second overwrite each other.
- **`tools/check_cost.py`** re-derives every published `usd_cost` from the
  tokens beside it, shares `price_from_counts` with the adapter, and runs in CI.
  Both published files reproduce exactly. Cache counts are zero in both, by
  construction and now by test: the adapter sends each task as its own prompt,
  so there is no prefix to reuse.
- **`docs/V2_DESIGN.md`** is the design for the set that replaces v1 as the
  headline: what discriminates at the top, eight scored candidates, the two to
  write first (`speculative_decoding_verify`, `flash_attention_backward`), and
  the admission rule that fixes L21 — a task is calibrated against models before
  it enters a frozen set. **The calibration machinery is deliberately not built
  yet** (L11: infrastructure before its first user).
- Harness suite: **77 passed** here, up from 61.
- Still open, and neither is blocked by code: no model has been asked an
  accelerated task, and v1 does not discriminate at the top.

## State as of 2026-08-02 (verify before trusting)

- **The repository is public and live**, at `selimfedakar/scratchbench`.
  Blocks 0 through 5 of `COMMITS.md` have run; **Blocks 7 through 14 are
  pending** and hold everything below that is not yet committed.
- **CI on `main` is red, and every red cross is correct.**
  `tests/test_runner.py` is committed and imports `adapters.anthropic_api`,
  which is still in the commit queue, so `pytest` exits 2 at collection. The
  cause is the push cadence, not the history: one file per commit means most
  tips are incomplete, and the commits were pushed one at a time instead of
  once at the end. Do not rewrite history over this. Run the pending blocks,
  then **push once** (`docs/LESSONS.md` L17).
- Tasks: **nine, all nine L2 verified and all nine in `frozen_set: v1`.**
  Eight on the laptop tier; `fused_rmsnorm_kernel` (kernels, difficulty 4,
  `accelerated`, `accelerator: cuda`, Triton, **24** hidden tests) passed on an
  RTX A4500 on 2026-08-02 — reference 24 passed, untouched starter 24 failed,
  six mutants out of six caught. Transcript in `docs/sessions/06`.
- Its first hardware run found two real holes, both fixed and both worth
  reading before writing another task: the mutant for the claim the task is
  named after survived (L14), and two tests could not fail against the
  untouched starter (L15).
- Adapters: `anthropic` is **real and exercised against the live API**. One
  call per task, streamed, JSON-schema-constrained to the task's own filenames,
  no server-side fallbacks, and anything that is not a measurement raises into
  `adapter_error`. **It sets no sampling, thinking or effort knobs** — every
  model is asked at its own defaults, deliberately, so two rows of the same
  leaderboard mean the same thing. An alias answered by its dated snapshot is
  the same model, in both the identity check and the price lookup (L16). The
  SDK is an extra: `pip install 'scratchbench[anthropic]'`. `openai` is still a
  skeleton.
- Results carry `attempts`, `usd_cost` and **`tokens`** (input, output, cache
  read, cache write) off the adapter, plus `measured` and `not_measured`. The
  tokens are there so the cost can be reproduced rather than trusted (L18), and
  it checks out: `(1182 × 1 + 1003 × 5) / 1e6 = 0.006197`, exactly what the file
  says.
- **`runner/sandbox.py` holds one table, `STATUSES`, that classifies every
  outcome status: is it evidence, and does it break the run.** `NO_EVIDENCE`,
  `HARNESS_FAILURES` and the printed labels all derive from it, and
  `Outcome.__post_init__` refuses an unclassified status. This exists because
  the same idea used to live in two files with two different memberships, and
  the gap published `pass_rate: 0.0` for a sweep that never reached the model
  (L20). **Do not re-introduce a second list.**
- Runner: **L2** — `validate --tier all` reports 8 ok and 1 not checked here;
  `run --model reference --tasks all --tier all` reports laptop 100% (8/8) and
  accelerated not run, in 5.6s. Its own suite: **L1** — 61 passed here,
  58 passed and 1 skipped on Linux with a CUDA torch (before the last two tests).
- First graded model runs: `claude-haiku-4-5` on `softmax_stability` twice.
  Run one passed all 29 tests; run two failed three of them. **Same model, same
  task, same settings.** The harness is deterministic and the model is not, so a
  sweep is one draw and the leaderboard has to say so (L19). One attempt per
  task stays: retries measure the scaffolding, repeated runs measure variance,
  and they are different things.
- `tools/` holds the two scripts the accelerated tier's evidence comes from:
  `mutate_rmsnorm.py` and `verify_accelerated.sh`. They are in the repository
  because a check nobody can re-run is a claim.
- CI: the laptop job is real. **The accelerated job is deleted**, deliberately: a
  self-hosted GPU runner on a public repository can be reached by a pull request
  that edits `runs-on`. `tools/verify_accelerated.sh` replaces it and
  `CONTRIBUTING.md` says so.
- Next: the first full eight-task sweep and the first leaderboard entry.

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
