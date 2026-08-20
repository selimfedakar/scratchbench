# 12 — The rule that read one model

The admission rule in `V2_DESIGN.md` §4 exists so that a task earns its place in
a headline set by being asked of models rather than by how hard it felt to write
(L21). As written on 2026-08-09 it read one number: the highest pass rate in the
calibration block, refused if it was 100%. This session widened it to read the
top two, and moved one task.

Nothing was measured here. Every number below was already published and checked
in; the work was the rule, the three documents that state it, and an entry in
`docs/LESSONS.md` that says plainly what changing a rule after it refuses
something costs.

## What the old rule did to `flash_attention_backward`

The task was written on 2026-08-09, is L2 on both halves, and its ten mutants
are all caught. Its calibration block, re-derived from `leaderboard/` by
`tools/check_calibration.py`:

| model | draws | passed |
|---|---:|---:|
| `claude-opus-5` | 15 | 15 |
| `claude-sonnet-5` | 10 | 8 |
| `claude-haiku-4-5` | 19 | 3 |

Opus at 15 of 15 refused it outright, so the one task on the laptop tier that
separates two frontier models sat in `warmup` beside two tasks that Opus and
Sonnet both clear completely. Those are not the same object, and a set that
cannot tell them apart is not measuring what the set is for.

Sonnet's two failures are the same pair of tests both times and match a mutant's
signature exactly — *queries not put at the end of the key range* — which is
what journal 10 called a convention missed at the edge rather than an absent
mechanism.

## The rule now

> A numbered set from `v2` onward needs **two calibration entries of five draws
> or more**, and the **highest two pass rates among them must not both be 100%**.

Entries below five draws are read on neither side of the comparison: a
measurement too small to admit a task is too small to refuse one. That is the
same absence-of-evidence rule the runner applies one column over, and it is why
`_check_admission` ranks a filtered list rather than the whole block.

Applied to the twenty-one calibration entries in the repository, it changes
exactly one verdict:

| task | old | new | why |
|---|---|---|---|
| `flash_attention_backward` | `warmup` | **`v2`** | Opus 15/15, Sonnet 8/10 — the top two are not both at the ceiling |
| `speculative_decoding_verify` | `warmup` | `warmup` | Opus 15/15 **and** Sonnet 10/10 |
| `activation_checkpointing_rng` | `warmup` | `warmup` | Opus 10/10 **and** Sonnet 10/10 |
| `metal_cross_entropy_kernel` | `v2` | `v2` | Opus 8/10, Sonnet 4/10 |

A rule rewritten to reach a result usually reaches several. This one moves one
task, and that is the strongest evidence available that it is a correction
rather than a loophole — which is a claim about the rule, not a defence of the
timing. The timing is in L35.

## The task that was already qualified and nobody had asked

`fused_rmsnorm_kernel` moved from `v1` to `v2` in the same session, and it is
not a consequence of the rule change: it clears the old rule too. `claude-opus-5`
solves 1 draw of 5 and `claude-haiku-4-5` 3 of 5, measured on an RTX 4000 Ada on
2026-08-09 and published under `leaderboard/` since. It is the lowest frontier
pass rate in the repository, it is the task `V2_DESIGN.md` §1b was written
about, and the only thing standing between it and the set was a `calibration:`
block nobody had written. The block is a re-derivation of files that were
already checked in, not a new measurement, and `tools/check_calibration.py`
re-derives it in CI like every other.

Two things about it are worth stating rather than leaving for a reader to catch.
`claude-sonnet-5` has never been asked this task, because asking costs a rented
CUDA box; the rule reads the top two entries and both exist, so the task
qualifies on the evidence there is. And five draws cannot distinguish 20% from
60%, so the fact that the smaller model scores higher here is not a result —
what is a result is the failure shape: four of Opus's five failures are the same
single test, `test_a_single_column`, and Haiku's are 24 of 24 and 23 of 24
(L27).

`v1` is therefore five laptop tasks with no accelerated member, and the
leaderboard's CUDA section says so above the rows it did not change.

## What was rejected

"Admit a task if any two models differ on it." It sounds like the same idea and
it is the opposite one. `grad_accumulation` separates `claude-haiku-4-5` (0 of
6) from `claude-opus-5` (6 of 6) and `claude-sonnet-5` (5 of 5), and so does
most of v1. That rule readmits the set the calibration machinery was built to
keep out, and separating a small model from a large one is the definition of
`warmup`. The frontier is the top of the field.

## The honesty debt, and why it is bigger than I first wrote

My first draft of L35 mitigated the change by pointing at journal 10: the
tension between the rule and its stated intent was written down there, on
2026-08-09, before there was a second task to lose by it. That is true and it is
not enough. Journal 10 does not merely name the tension — it argues for the
strict rule, keeps the task in `warmup` deliberately, and sets a revisit
condition: *when there is a model between Sonnet and Opus to lose*. No such
model has appeared. The condition is unmet and the decision is being overturned
anyway.

What justifies overturning it is one line of that same old argument, turned
around. Journal 10 kept the task out because Sonnet's failure is convention
noise rather than an absent mechanism. The loader cannot make that distinction;
only I can. So invoking it is my reading of a task's difficulty deciding its
admission, which is the exact channel the calibration block was built to close.
The rule's own sentence says a headline set ranks the top of the *field*, and
with three models measured as a matter of course, one entry is not a field.

`docs/LESSONS.md` L35 is the long version, including the version of the fix that
sounded better and would have undone the whole exercise.

## Verified

Everything below is this session's output.

| Claim | Level | Evidence |
|---|---|---|
| Harness suite passes with the new rule | L1 | `python -m pytest -q` → `105 passed` (was 103; two tests added, one rewritten) |
| A task the top two clear is refused | L1 | `test_a_task_the_top_two_models_never_fail_cannot_enter_a_calibrated_set` |
| One model at the ceiling is not the frontier | L1 | `test_the_strongest_model_alone_is_not_the_frontier` |
| A sub-five-draw entry neither admits nor refuses | L1 | `test_a_small_entry_can_neither_admit_a_task_nor_refuse_it` |
| Every task in the repository still loads | L1 | `test_every_task_in_the_repository_loads`, inside the suite above |
| `v2` has three members | L2 | `python -m runner.cli list --set v2` → `flash_attention_backward`, `fused_rmsnorm_kernel`, `metal_cross_entropy_kernel` |
| Tasks still validate | L2 | `python -m runner.cli validate --tier all` → `12 task(s) validated`, 1 not checked here |
| Calibration blocks still re-derive | L2 | `python tools/check_calibration.py` → `23 calibration entries re-derived from 122 draw(s)` |
| Published costs unchanged | L2 | `python tools/check_cost.py` → `57 file(s) checked` |

## Not verified

- **No model was asked anything this session.** Every rate quoted above is a
  published draw from 2026-08-09 or 2026-08-13, re-derived from
  `calibration/` and `leaderboard/` rather than measured again.
- **`v2` is still not published.** Three of the four intended members exist and
  there is no v2 leaderboard row: a set is announced when it is complete, and
  the fourth member is a second Metal task that has not been written.
- **No v2 sweep has ever been run.** Each of the three members has draws of its
  own, taken on three different days against three different task sets, and
  nobody has run `--set v2` end to end. The published row, when it exists, will
  be a fresh sweep and not an assembly of these numbers.
- **`fused_rmsnorm_kernel` has two calibrated models, not three.**
  `claude-sonnet-5` has never been asked it; that needs a rented CUDA box.
