# Roadmap to launch

`V2_DESIGN.md` says what the second task set should measure. This file says what
is left to build before any of it can be announced, in the order it has to
happen, with the reasoning attached to each step rather than left for the
session that picks it up.

**How to use it.** Each numbered section is one session's brief. Open the
section, read *Preconditions* first and stop if one is unmet, then work the
*Steps*. A session is finished when every line under *Done when* has pasted
evidence behind it, not when the code exists. The standing rules in
`CLAUDE.md` apply to all of them and are not repeated here.

Written 2026-08-19, when the repository was at the state in section 0.

---

## 0. Where this is today

Verified by running the commands, not from memory:

| | |
|---|---|
| Tasks | 13. `v1` five (all laptop), `v2` three, `warmup` five |
| `v2` | `flash_attention_backward` (laptop) · `fused_rmsnorm_kernel` (accelerated/cuda) · `metal_cross_entropy_kernel` (accelerated/metal) |
| Harness suite | 105 passed |
| Validation | 12 validated on this machine, 1 unchecked (no CUDA) |
| Calibration | 23 entries re-derived from 122 draws |
| Published costs | 57 files, every one reproducing from its own tokens |
| Adapters | `reference` and `anthropic` real; `openai` a skeleton class |
| Leaderboard | `v1` laptop rows for three models, plus a Metal section and a CUDA section. **No `v2` row, and no `--set v2` sweep has ever been run** |
| CI | green, two Python versions, six steps |

Total spend to date on model draws: roughly $9.

## 0.1 The gap nobody has written down yet

`v2` has **one laptop task**. The leaderboard's headline rate is the laptop
tier by design — that is the founding claim, and `runner/report.py` enforces it
with `HEADLINE_TIER`. So as things stand, announcing `v2` would announce a
headline computed over a single task, with the other two members reported beside
it as hardware-gated footnotes.

A benchmark whose headline is one task is not a benchmark. **This is the gating
problem for everything below**, it is a consequence of the accelerated tier
turning out to be the interesting one (journal 09 through 12), and it has to be
decided before another task is written, because the decision determines what
tier the next task is written for.

Section 1 is that decision. Sections 2 onward assume it went the way section 1
recommends and say what changes if it did not.

---

## 1. Session 13 — Decide what the headline is, then start closing it

**Goal.** Resolve how `v2` is reported, record the decision, and write the first
of the laptop-tier tasks that decision requires.

**Why now.** Every later session is downstream. Writing another accelerated task
first would make the imbalance worse and would have to be partly redone.

**Preconditions.** Session 12's commits pushed and CI green.

### 1.1 Put the numbers in front of the decision

```bash
cd ~/scratchbench
source ~/.zshrc >/dev/null 2>&1
python -m runner.cli list --set v2                 # three tasks
python -m runner.cli list --set v2 --tier laptop   # one task
python -m runner.cli list --set v1 --tier laptop   # five, for contrast
```

### 1.2 The three options, and the recommendation

**Option A — keep the tiers, grow the laptop half.** `v2` does not ship until it
has at least three laptop tasks. The accelerated members stay where they are and
are reported beside the headline exactly as today. Nothing in `runner/` changes.
Cost: two or three more laptop tasks that clear the admission rule, which is the
hardest kind of work in this repository (`V2_DESIGN.md` §2.0, `LESSONS.md` L30).

**Option B — re-cut the tiers.** Replace `laptop | accelerated` with something
like `portable | apple | cuda`, and let the headline span portable and apple on
the grounds that a Mac is a laptop. Cheap in tasks, expensive in credibility:
the reason anyone can check the headline number today is that it needs no
hardware at all, and widening it to "any Apple silicon Mac" quietly narrows who
can reproduce the claim while the README still says "reproducible by anyone who
clones this".

**Option C — headline the whole set, report per tier underneath.** One `v2`
number over all three tasks, with the tier breakdown and the hardware each task
needs stated beside it. Honest, and it abandons the one property that
distinguishes this from every GPU-gated benchmark.

**Recommended: A, with C's reporting shape.** Keep the tier code untouched, hold
`v2` back until its laptop half is a real set, and present the accelerated tasks
beside it with their hardware named. The argument is the same one L21 is about:
the fastest way to make the current inventory look like a set is to redefine
what a set is, and that is the failure this repository is built to avoid, in a
new costume. B is worth revisiting only if two independent attempts at a
discriminating laptop task both fail admission — see section 2.6.

**Record it.** A new subsection in `V2_DESIGN.md` §5 stating the decision, the
two rejected options, and the condition that would reopen it. One paragraph in
`docs/LESSONS.md` only if something was learned; a decision on its own is not a
lesson.

### 1.3 Write laptop candidate 1

`V2_DESIGN.md` §2.0 is the criterion and it is narrow: **the difficulty is a
fact about a tool, and an obscure one.** Reasoning tasks are saturated at the
top and three of them are already in `warmup` proving it. The Metal task worked
because Metal Shading Language has facts that no amount of reasoning about
cross-entropy reaches. The laptop tier needs the same shape without a GPU.

Candidates, each named by the fact it turns on rather than by its subject:

1. **`custom_autograd_double_backward`** *(recommended first)*. A
   `torch.autograd.Function` whose backward must itself be differentiable, so
   the graded artifact survives `torch.autograd.gradgradcheck`. The facts:
   `ctx.needs_input_grad` decides which gradients may be `None`; a backward
   written with `.data` or `.detach()` produces correct first gradients and
   silently wrong second ones; `@once_differentiable` makes the failure explicit
   and is the thing a model reaches for to make the error go away;
   `save_for_backward` versus attributes on `ctx` differ under in-place
   modification because of the version counter. Exactly checkable against
   autograd in float64, cheap, CPU-only, and the wrong implementation passes
   every first-order test.
2. **`byte_level_bpe_roundtrip`**. Byte-level BPE where the difficulty is the
   byte-to-printable-codepoint mapping and what happens to input that is not
   valid UTF-8. The fact: the encoder operates on bytes, so a multi-byte
   codepoint can be split across merges, and only the decoder's byte buffer
   makes the round trip exact. Distinct from `bpe_merge_order`, which is v1 and
   about merge ranks.
3. **`pairwise_sum_error_bound`**. A numerics task whose assertion is an error
   bound a naive accumulation cannot meet. The fact: `numpy.sum` is pairwise
   and a Python loop over the same array is not, so "matches `np.sum`" and
   "sums correctly" are different claims at float32.
4. **`strided_view_aliasing`**. Sliding windows built with
   `numpy.lib.stride_tricks`, where the fact is that the result aliases its
   input, is not writeable by default, and any implementation that materialises
   a copy passes correctness while failing a memory-sharing assertion the prompt
   licenses explicitly.

Write it in the mandated order — `reference/` → `hidden_tests/` → `prompt.md` →
`meta.yaml` — then the three gates and the licensing walk in both directions
(`CLAUDE.md` *Landmines*, `LESSONS.md` L8). Mutants go in
`tools/mutate_v2_tasks.py` beside the existing ones, each with its expected
verdict.

### 1.4 Calibrate it

```bash
mkdir -p ~/Desktop/scratchbench-eski-results && mv results/*.json ~/Desktop/scratchbench-eski-results/
source ~/.zshrc >/dev/null 2>&1
for m in claude-opus-5 claude-sonnet-5 claude-haiku-4-5; do
  python -m runner.cli run --model "$m" --tasks <slug> --repeat 10 --keep "/tmp/draws-$m"
done
cp results/*.json calibration/
python tools/check_calibration.py
```

Ten draws, not five: two five-draw sweeps of the same model on the same task
have come back 2/5 and 5/5 here. Read the failures **by name** out of the
`--keep` directories (L27) — a count is not a shape — and put the names in the
journal.

**Done when.**

- The decision is written into `V2_DESIGN.md` with its rejected alternatives.
- The new task is L2 on both halves, with both outputs pasted.
- Every mutant returns its expected verdict, script checked in.
- The calibration block is in `meta.yaml`, the draws are in `calibration/`, and
  `tools/check_calibration.py` re-derives it.
- `python -m pytest -q` and `validate --tier all` both green, counts pasted.

**Cost.** About $3 for thirty draws, less if the task is short.

**Risk.** The task clears the admission rule and lands in `warmup`. That is a
result, not a failure; it goes in the journal with its numbers and section 2
picks up the next candidate.

---

## 2. Session 14 — The second Metal task, and laptop candidate 2

**Goal.** `v2`'s accelerated half reaches the three tasks `V2_DESIGN.md` §5
promised, and the laptop half gets its second attempt.

**Preconditions.** Section 1 done; the tier decision recorded.

### 2.1 The backward kernel

Cross-entropy or softmax **backward** in Metal Shading Language, with the launch
geometry chosen by the caller exactly as in `metal_cross_entropy_kernel`. The
forward task's four measured API facts still hold and are in journal 11; do not
re-derive them, but do re-measure anything the new kernel depends on that is not
already on that list.

What makes the backward version a different task rather than the same one again:
the gradient of the loss with respect to the logits needs the softmax the
forward computed, so the kernel either recomputes it under the same numerical
shift or reads it back, and the upstream gradient scales a row that has already
been reduced. Both put a second reduction and a second barrier discipline in the
same kernel.

Follow the established pattern:

```bash
python tools/mutate_metal_task.py       # extend it, do not fork it
```

Every mutant carries an **expected verdict** and the script fails when a
survivor is caught, not only when a mutant escapes — that two-way expectation is
what made L31 through L33 findings instead of guesses.

### 2.2 Laptop candidate 2

The next name off the section 1.3 list. Same order, same gates, same
calibration protocol.

### 2.3 The branch

If both laptop candidates cleared the admission rule, stop and reopen section
1.2. Two independent attempts at a discriminating laptop task failing is
evidence about the tier, not about the tasks, and it is the condition under
which option B stops being a shortcut and becomes the honest description of what
this benchmark now measures. Write it up in `LESSONS.md` before changing
anything.

**Done when.** Both tasks L2 on both halves, mutants checked in with expected
verdicts, both calibrated with the draws committed, journal written.

**Cost.** About $5 for two calibrations of thirty draws.

---

## 3. Session 15 — The OpenAI adapter

**Goal.** `adapters/model_api.py`'s `OpenAIAdapter` stops being a skeleton, and
the leaderboard stops being single-provider.

**Why it matters more than it looks.** A leaderboard with one vendor's models on
it reads as a fan project regardless of how good the tasks are. This is the
highest-signal single addition before launch and it is cheap.

**Preconditions.** `OPENAI_API_KEY` in `~/.zshrc` beside the Anthropic one, and
about $10 of credit. Claude's shell does not read `.zshrc`, so every command
that needs the key is prefixed `source ~/.zshrc >/dev/null 2>&1 &&`.

### 3.1 Read the provider's current docs first

Do not write this adapter from memory of the API. Structured output, the token
accounting fields and the parameter names have all moved in the last year. Open
the current reference, then write. The repository's own rule applies: a number
that cannot be reproduced is a number nobody should trust, and the cost
arithmetic here is exactly that kind of number.

### 3.2 What the adapter must do, and what it must refuse to do

Mirror `adapters/anthropic_api.py` — read it first (T1), because the invariants
below are already implemented there and the second implementation is where they
drift:

- **One call per task, one attempt.** A retry with feedback measures the
  scaffolding.
- **Output constrained to the task's own filenames**, the same way the Anthropic
  adapter constrains its JSON schema. A response that is prose rather than files
  is an `adapter_error`, not a zero.
- **No server-side fallback of any kind.** If the responding model is not the
  requested model, raise. An alias answered by its dated snapshot is the same
  model; anything else is a different measurement wearing the same row.
- **No sampling, reasoning or effort knobs.** Every model is asked at its own
  defaults so that two rows mean the same thing. This is a standing decision and
  the OpenAI side is where it will be tempting to break it, because those knobs
  are more prominent there.
- **Tokens recorded in the same four fields** the results schema already has, so
  `tools/check_cost.py` re-derives the cost with the same `price_from_counts`
  arithmetic. Watch for provider-side token categories that are billed at output
  rates but reported separately: if a category exists and is billed, it is part
  of the cost and the checker has to see it, or the published number silently
  understates.
- **Anything that is not a measurement raises into `adapter_error`** — which is
  an absence of evidence, never a zero. `STATUSES` in `runner/sandbox.py` stays
  the only place that classification lives.

### 3.3 Tests

Add to `tests/test_runner.py` beside the Anthropic ones: identity mismatch
raises; a malformed response raises rather than scoring; the price table
round-trips through `price_from_counts`; and a smoke test that the adapter is
constructible without network. Then one real run against one cheap task, output
pasted.

### 3.4 First rows

```bash
source ~/.zshrc >/dev/null 2>&1
python -m runner.cli run --model <openai-model> --tasks all --set v1 --tier laptop --repeat 5
python tools/check_cost.py
```

`--set v1` because a leaderboard row is one frozen set (L23, one axis over).

**Done when.** A real sweep is published under `leaderboard/`, its cost
re-derives, and the page has a second provider on it with the same columns.

**Cost.** $1 to $3. The $10 credit is ample.

---

## 4. Session 16 — The measurement changes `V2_DESIGN.md` §6 promised

**Goal.** Ship the three reporting changes that were specified to go with v2 and
have not been built. Two of them are what makes the leaderboard readable by
someone who did not write it.

### 4.1 Cost per solved task

§6-2. Two models can both score 100% and be an order of magnitude apart in what
that cost, and that is the axis frontier models genuinely differ on.

In `runner/report.py`: a column derived from the same payload the table already
carries. Divide total spend by tasks **solved**, guard the zero-solved case
explicitly rather than letting it produce `inf`, and compute it only over tasks
that produced evidence — a `needs_accelerator` task cost nothing and solved
nothing, and folding it in either direction is the L20 mistake in a new column.

### 4.2 Failure shape, by name

§6-3, and `LESSONS.md` L27 says the same thing louder: `1 of 24` and `24 of 24`
are the same verdict and completely different results, and every failure table
in this repository so far was assembled **by hand** out of `--keep` directories.

The results file already carries counts. It needs the **names**. The mechanism:
pytest already prints the failing node ids in its short summary, so the sandbox
can capture them without a new dependency. Parse them where the counts are
parsed today, add a `failed_tests` list to the results payload, bump
`RESULTS_VERSION`, and keep the reader tolerant of older files — every published
results file in `leaderboard/` and `calibration/` is an older file, they are
checked in CI, and breaking them breaks the evidence.

### 4.3 Generate the leaderboard tables

The page is written by hand today. That was right when it had two rows; it is a
drift risk the moment a v2 sweep adds twelve. Write `tools/render_leaderboard.py`
that emits **only the tables** between explicit markers in
`leaderboard/README.md`, leaving the prose alone — the prose is the part worth
reading and it is not generated. Add a CI step that re-renders and diffs, so a
hand-edited number fails the build the way a hand-edited cost already does.

**Done when.** All three land with tests; CI re-renders the leaderboard and the
diff is empty; one existing published file is read back through the new reader
to prove the older schema still loads.

**Cost.** $0. No model is asked anything.

---

## 5. Session 17 — The first v2 sweep and the row that gets announced

**Goal.** The number the launch is about.

**Preconditions.** `v2` has its full membership from sections 1 and 2, section 4
has shipped, and a CUDA machine is available for the Triton member.

### 5.1 The hardware problem, stated plainly

`v2` spans three kinds of machine: no accelerator, Apple silicon, and CUDA. One
sweep cannot run on one box. The run is therefore three runs merged by set, and
that has to be said on the page rather than papered over:

```bash
# on the Mac: laptop tier and the Metal task
python -m runner.cli run --model <m> --set v2 --tier all --repeat 5

# on a rented CUDA box: the Triton task only
tools/verify_accelerated.sh          # reference and starter first, always
python -m runner.cli run --model <m> --tasks fused_rmsnorm_kernel --repeat 5
```

Rent by the hour, verify the reference before spending anything on model draws,
and copy the results files back rather than re-running anything at home.

### 5.2 Draws and models

Five draws minimum per §6-1, ten if the budget allows — the variance measured in
this repository has been large enough to matter at five. Four models if the
OpenAI adapter is in by then, three if not.

### 5.3 Publishing

Every published draw's cost re-derives (`tools/check_cost.py`), every
calibration block re-derives (`tools/check_calibration.py`), the tables are
generated rather than typed (section 4.3), and the page states: the number of
draws, that a row is a sample rather than a converged value (L19), which machine
each tier ran on, and the date. A row without those is a claim.

**Done when.** A `v2` section exists on the leaderboard with per-task `k/N`, a
range on the set rate, the cost-per-solved column, and the failure shapes by
name.

**Cost.** $15 to $40 depending on model count and draws, plus a few dollars of
GPU rental.

---

## 6. Session 18 — Launch hardening

**Goal.** Everything a stranger touches in the first ten minutes works, and
every claim in the README is one they can check.

### 6.1 The clean-clone test, which has never been run

Everything to date has been verified in the working directory, with an editable
install and a shell that already had the dependencies. That is not what a reader
gets.

```bash
cd $(mktemp -d)
git clone https://github.com/selimfedakar/scratchbench
cd scratchbench
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
scratchbench validate
scratchbench run --model reference --tasks all
```

Every line of that transcript goes in the journal. Anything that needs a step
the README does not mention is a README bug, and it is the most common reason a
repository gets one look and no second one.

### 6.2 The README rewrite

The page currently tells the story chronologically, because it grew that way. A
first-time reader needs, in order: what the benchmark asks, the headline number,
how to reproduce it in three commands, what makes a task hard here, and only
then the archaeology. The findings are the best material in the repository —
the `half` compile error, the mutant a real model reproduced exactly, the
degenerate kernel that passes every test — and they are currently below the
fold.

### 6.3 The claims audit

Walk every factual claim in `README.md`, `CONTRIBUTING.md` and
`leaderboard/README.md` and put a command beside it that proves it. Task counts,
test counts, set membership, spend, "no curriculum code", the reproducibility
promise. This session's own history is the argument for doing it: a stale
"nine entries" and a superseded admission rule both sat in the public README for
days, and both were found by grep rather than by intent.

### 6.4 The contamination question, answered before it is asked

The prompts are public and so are the hidden tests — "hidden" means hidden from
the model at solve time, not secret. Every number here therefore has a shelf
life, and journal 11 already says so about the `half` finding.

Decide and write down which of these the project does:

- **Say it plainly and publish the date on every row.** Cheapest, honest, and
  what the repository already half does.
- **Keep a small private holdout**, calibrated the same way, never published,
  used to check whether a public score has drifted. Strongest answer, and it
  costs a task-writing session plus the discipline never to leak it.
- **Version and re-measure**: `v3` exists partly to be a fresh set.

Recommended: the first now, the second stated as intent with a date. Claiming a
holdout that does not exist is worse than having none.

### 6.5 Housekeeping

`LICENSE` present and named in `pyproject.toml`; no key material anywhere in
history; `results/` still ignored; issue templates only if they will be read;
a one-screen `docs/README.md` index so the eleven session journals are
navigable.

**Done when.** The clean-clone transcript is in the journal, every README claim
has a command behind it, and the contamination policy is a paragraph on the page
rather than a plan in a file.

**Cost.** $0.

---

## 7. Session 19 — Show HN

**Preconditions.** Sections 1 through 6 done. `handmade-llm`'s launch has
already happened — two launches in one week split the attention and that
repository is further along.

### 7.1 What goes out

A Show HN with a title that is the claim, not the category. The first comment is
the interesting part and it should be the findings, not the pitch: a frontier
model failing on a reserved word, a mutant a model reproduced exactly, a correct
kernel that uses one lane and passes every test that can be written. Those are
the things a reader forwards.

### 7.2 Timing

Tuesday or Wednesday, early Pacific morning. Be at a keyboard for the following
four hours; an unanswered thread dies regardless of the work behind it.

### 7.3 The questions that will come, and where the answers already live

| Question | Answer |
|---|---|
| The tests are in the repo, so this is contaminated | Section 6.4's paragraph, plus the dated rows and L30 |
| Thirteen tasks is not a benchmark | True, and the page says the sample size next to every number. The claim is that the tasks are *verified*, not that they are many |
| Why binary scoring | `TASK_FORMAT.md`: partial credit hides the failures worth knowing about, and the failure shapes are published instead |
| Why one attempt | A model that passes on the eleventh try measures the scaffolding. Variance is measured with repeated independent draws instead |
| Why only Anthropic models | Fixed in section 3, and if it is not, say so before someone else does |
| How do I know the numbers are real | Every draw is checked in, and two CI tools re-derive every cost and every calibration block from them |

### 7.4 After

Answer everything for a day. File the issues people raise rather than arguing
them into the thread. Write the last journal entry with what the launch actually
taught, which is the only part of this file that cannot be planned.

---

## 8. Dependencies at a glance

```
13 tier decision ──► 14 second Metal task ──┐
   └─ laptop task 1 ──► laptop task 2 ──────┼──► 17 v2 sweep ──► 18 hardening ──► 19 Show HN
15 OpenAI adapter ──────────────────────────┤
16 measurement changes ─────────────────────┘
```

15 and 16 are independent of 13 and 14 and can move earlier if a task-writing
session stalls. 17 cannot start before 13, 14 and 16. 19 cannot start before 18,
and 18 finds things, so do not schedule them on the same day.

Estimated model spend from here to launch: **$25 to $50**, plus a few dollars of
GPU rental. The founding constraint is unchanged and that number is the point of
it.
