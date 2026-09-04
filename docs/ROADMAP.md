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

**Candidate 1 was written on 2026-08-20 and refused on 2026-08-21** as
`custom_autograd_double_backward`: Opus 10/10, Sonnet 10/10, Haiku 0/10, so it
is `warmup` and it is **attempt one of the two** that reopen §1.2. Before
writing candidate 2, apply the question §2.0 gained from it — *does the
framework have a page about exactly this mistake?* — because candidates 2 and 3
below have not been checked against it and at least one of them may not
survive the check. Candidates 2, 3 and 4 stay on this list and
are the standing task bank: the next session that needs a laptop task takes the
top one rather than inventing a fresh idea, and a candidate is only struck off
this list when it has been written or when a measurement kills it. Each was
chosen for the fact it turns on, and those facts do not go stale just because a
different task got written first.

#### The gate was applied on 2026-08-29, and all three failed it

The question costs nothing to answer, so it was answered before anything was
written, against the installed libraries rather than from memory. Every quote
below is from the docstring of the function the candidate turns on, on this
machine, numpy 1.23.5.

| Candidate | The fact it turns on | Does the framework have a page about exactly this mistake? |
|---|---|---|
| `byte_level_bpe_roundtrip` | the byte-to-printable-codepoint table, and what a decoder does with bytes that are not valid UTF-8 | **Yes, and worse: it ships as code.** The table is a twenty-line function published in OpenAI's GPT-2 release and copied verbatim into every byte-level BPE implementation since. `tiktoken` 0.13.0 and `tokenizers` 0.23.1 are installed on this machine. A task about it measures recall of a specific published function. |
| `pairwise_sum_error_bound` | `numpy.sum` is pairwise and a Python loop is not | **Yes, in `np.sum`'s own Notes:** "the numerical precision of sum (and `np.add.reduce`) is in general limited by directly adding each number individually to the result causing rounding errors in every step. However, often numpy will use a numerically better approach (partial pairwise summation)". |
| `strided_view_aliasing` | the view aliases its input and is not writeable by default | **Yes, twice.** `sliding_window_view`: "The default is false, as this should be used with caution: the returned view contains the same memory location multiple times, so writing to one location will cause others to change." `as_strided` carries a `.. warning::` reading "This function has to be used with extreme care, see notes", and the Notes explain that vectorised writes to such arrays are unpredictable. |

**The structural reason, which is the part worth keeping.** The gate selects for
a tool whose manual is thin: `half` is a type name in Metal Shading Language and
there is no page about not using it as a variable, because MSL's documentation is
a language specification rather than a tutorial ecosystem. The laptop tier's
entire toolset is numpy and torch, which are two of the most exhaustively
documented libraries in existence and whose docstrings *are* the pages. Every
fact nameable in them is documented somewhere in them. That is not a proof that
no laptop candidate can pass the gate, and it must not be treated as one — see
§2.3, where the reopening condition is deliberately a measurement and not an
argument — but it is the reason three consecutive candidates failed.

**Consequence for §2.3, and it is a logical one rather than a budget one.**
§1.2's reopening condition is two independent attempts *written against §2.0*.
A candidate that fails §2.0's own gate before it is written is not an attempt
against §2.0, so calibrating one would not advance the branch: it would buy a
fifth `warmup` task and leave the count of qualifying attempts at one. Spending
the thirty draws is therefore not the cheap way to make progress here; finding a
candidate that passes the gate is.

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

**Not written on 2026-08-29, and the reason is a result rather than a stall.**
The §2.0 gate was applied to all three remaining bank entries first, because it
is free, and all three failed it with the evidence recorded in §1.3. Writing any
of them would produce a `warmup` task that does not count toward §1.2's
reopening condition, which is the one thing this candidate was supposed to buy.
So the open item is not "write the next task", it is **"find a laptop candidate
that passes the gate"**, and that is a design question with a decision attached
to it. It is item 1 of the debt ledger in section 9.

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

---

## 9. Debt ledger

Things found while working a section, not fixed there, and too small or too
sideways to be a section of their own. Each carries where it belongs, so it is
scheduled rather than remembered. Add to it whenever a session walks past
something; delete an entry only when it is done or when it is decided against in
writing.

| # | Found | What | Where it belongs | Cost |
|---|---|---|---|---|
| 1 | 2026-08-29, §2.2 | **No laptop candidate passes the §2.0 gate**, and §9.1 below argues the gate may be the wrong instrument for this tier. The bank is exhausted (§1.3) and the replacement is a design question, not a writing task. Until it is answered, §1.2's reopening condition cannot advance and §5's `v2` sweep cannot start. This is the critical path. | §9.1, then a session of its own before §5 | $0 to decide, $3 to calibrate what it produces |
| 2 | 2026-08-29, §2.1 | **`docs/PATTERNS.md` does not exist.** `CLAUDE.md`'s mandatory loading paragraph says to search it first for any bug. A rule pointing at a missing file trains the next session to skip the rule. Either write the file out of the L-entries that are really debugging patterns, or cut the clause. | §6.3, the claims audit | $0 |
| 3 | 2026-08-30, §2.1. **Diagnosed 2026-09-04, §9.2** | 🔴 **Found, and it is not the task.** On this machine every other process that touches `torch.mps` is stalled for its whole life: a host-to-device-to-host round trip costs 0.47–0.72 ms in a healthy process and 232–776 ms in a stalled one, with no overlap over fourteen consecutive launches, and the workload behind it takes 0.070–0.089 s against 8.055–10.899 s. It needs no custom shader (plain `torch.softmax` reproduces it), no failing test, and no particular task — `metal_cross_entropy_kernel`, which is **already published**, measured 1.89 s, 48.76 s, 1.50 s on three consecutive reference runs. The GPU is not busy while it happens: a second process launched into a 75-second stall did 50 MPS softmaxes in 0.687 s. `recoveryCount` is 0, a ten-second gap does not help, a keeper process holding an MPS context open does not help, and 2599 stack samples of a stalled process are all in `-[_MTLCommandBuffer waitUntilCompleted]`. Full account and the reasoning error that hid it for two sessions: `LESSONS.md` **L42**. Re-derive with `python tools/check_mps_stall.py --launches 14`. **What is still open is not the diagnosis but the response** — §9.2 states the three choices and the recommendation. | §9.2 decides it; the chosen fix lands before §5's `v2` sweep | $0 spent, $0 to decide |
| 6 | 2026-09-04, §9.2 | 🟡 **`metal_cross_entropy_kernel`'s calibration was drawn on this machine while the stall was live and unknown.** Its published block therefore rests on draws that may contain `timeout` where the model actually produced a wrong answer. The rate survives that and the failure shape does not, which is the same argument that kept the backward task unpublished — applied, this time, to something already public. **Done when** the task has been re-drawn under whatever §9.2 chooses, or the leaderboard says in writing which draws predate the guard. | Immediately after §9.2 is decided, and before §6.3's claims audit signs anything | ~$2 to re-draw |
| 3b | 2026-08-30, §2.1 | **Fixed on the way past, keep the reasoning.** `mutate_metal_task.py` waited 1800 s per mutant, and the mutant that returns out-of-range lanes before a barrier — undefined behaviour in MSL — spent every second of it. It now reads the task's own `time_limit_s` from `meta.yaml` and calls a timeout `CAUGHT`, which is what `STATUSES` does. Separately its starter gate read a timeout as "fails cleanly", so a starter that never answered would have certified the tests; One consequence to keep in view: a mutant that is expected to SURVIVE and happens to be killed by the limit is reported CAUGHT, so while item 3 is open a `did not match` on a SURVIVES line has two possible readings and the printed `timed out after 300s` is what tells them apart. `TIMED_OUT = 124` is now distinct from a real failure and stops the run with its own message. | done | $0 |
| 4 | 2026-08-29, §2.1 | **L32's title is one notch more general than its evidence.** "The race this hardware refuses to show me" reads as a fact about the device; the backward kernel shows the same race at its second scratch reuse, so the fact is about the distance between the read and the write instead. The entry's body already scoped itself to "on this machine, on this driver, today" and predicted the failure would be news, so this is a title and a cross-reference to L39, not a retraction. `leaderboard/README.md` does not publish the claim and needs no change. | §6.3, the claims audit | $0 |
| 5 | 2026-09-02, §2.1 | 🟡 **The thirty draws for `metal_softmax_backward_kernel` are owed.** Decided by Selim on 2026-09-02: fix item 3 first, then calibrate — because a leaderboard reader who sees `timeout` concludes the task is broken, and would be right. **Trigger, restated 2026-09-04:** the old one — twenty consecutive starter runs under 5 s — is unreachable, because half of all launches stall whatever the task is (item 3, `LESSONS.md` L42). The trigger is now §9.2 decided and its fix landed, checked by `python tools/check_mps_stall.py --launches 14` reporting every launch healthy under the guard. **Then:** `--repeat 10` for `claude-opus-5`, `claude-sonnet-5` and `claude-haiku-4-5` with `--tier accelerated` and `--keep`, copy the draws into `calibration/`, re-derive with `tools/check_calibration.py`, and let the admission rule decide between `v2` and `warmup` — do not decide it by hand. **Done when** `meta.yaml` carries the block, `frozen_set` is no longer `unvalidated`, and the draws are checked in. | Immediately after item 3 | ~$2 |

### 9.1 The counterexample already in `v2`, and what it suggests for debt item 1

Written 2026-08-29, after the §2.0 gate refused every remaining laptop candidate.
It is an argument rather than a decision, and the decision is Selim's.

Three tasks in this repository discriminate at the top. Two of them satisfy
§2.0's hazard clause and **both are accelerated**:

| Task | Tier | Frontier rate | What makes it hard |
|---|---|---|---|
| `fused_rmsnorm_kernel` | accelerated/cuda | Opus 1/5 | Triton specialises an integer argument whose value is 1 |
| `metal_cross_entropy_kernel` | accelerated/metal | Opus 8/10, Sonnet 4/10 | `half` is a type name in MSL, plus a group size the kernel does not choose |
| `flash_attention_backward` | **laptop** | Opus 15/15, Sonnet 8/10 | the graded function is handed one block of keys and never the rest |

The third one is the only laptop task in the repository that separates two
frontier models, it is in `v2`, and **it does not satisfy §2.0 at all.** Nothing
about it is an undocumented fact; flash attention has papers, tutorials and
reference implementations. What makes it hard is L28's mechanism: the harness
owns a decomposition the solution cannot choose and cannot see around, so `D`
is a property of a whole query row that the graded function is structurally
unable to compute, and an implementation that accumulates it from the columns in
hand is wrong in exactly the way real blocked implementations are wrong.

§2.0 was derived from the two accelerated tasks and generalised to the laptop
tier without a laptop example. The laptop example exists and it works
differently. So the honest reading of three consecutive refusals is not
necessarily "the laptop tier is saturated"; it may be "candidates 1 through 4
were all chosen against the wrong criterion for this tier".

**The choice for the next task-writing session, stated so it can be answered in
one word:**

- **A — write candidate 2 against L28's mechanism instead of §2.0's hazard.**
  A laptop task whose graded unit is handed one piece of a decomposition the
  caller chose, where the correct answer needs state that piece does not have,
  and where materialising the whole thing is unavailable rather than merely
  discouraged. This is the only recipe with a laptop-tier track record here.
  *Recommended.*
- **B — keep hunting for a laptop hazard.** The gate stays as written and the
  next session's job is to find a fact in numpy or torch with no page about it.
  Honest, and §1.3's evidence is that the search space is thin.
- **C — declare the tier saturated and reopen §1.2.** Cheapest, and it is the
  one L35 warns about: an argument standing in for the measurement the rule
  asks for. §1.2's condition is two *measured* attempts and there has been one.

If A is chosen, §2.0 needs a sentence saying it is the accelerated tier's
criterion and L28's mechanism is the laptop tier's, rather than one criterion
pretending to cover both. That sentence is the actual output of this debt item.

### 9.2 What to do about the stall, now that it is measured

Written 2026-09-04, after debt item 3 turned out to be a property of the machine
rather than of `metal_softmax_backward_kernel`. The measurement is settled and
`tools/check_mps_stall.py` re-derives it; what is not settled is what the
harness should do about it, and that is a change to the grading contract, so it
is Selim's.

The constraint that rules out doing nothing: `STATUSES` treats a `timeout` as
evidence *and* as a failure, which is correct when a solution really did not
finish. Under the stall a correct-shaped wrong answer and a stalled healthy run
produce the same word. The pass rate tolerates that; the failure shape does not,
and the failure shape is what this repository publishes instead of partial
credit. Both Metal tasks are exposed, one of them already public (debt item 6).

- **A — a start-up guard in `runner/sandbox.py`.** Before a graded run is
  scored, the child calls `check_mps_stall.probe()`; if it exceeds the
  threshold the run is discarded and relaunched, up to a small fixed number of
  attempts, and a run that never gets a healthy process is reported as a harness
  failure rather than as a `timeout`. Costs one new status meaning and a few
  milliseconds per run. It is the only option that makes the recorded failure
  shape mean what it says. *Recommended.*
- **B — raise `time_limit_s` for the Metal tasks.** Cheapest to write. A stalled
  run is roughly a hundred times slower, so honouring it means a limit near
  30000 s, which is not a limit. It also silently makes a genuinely slow wrong
  answer indistinguishable from a stalled one, which is the current problem with
  a bigger number in front of it.
- **C — publish the tasks with the exposure documented and no code change.**
  Honest, and it moves the cost onto every reader of the leaderboard forever. It
  is also the option that cannot be applied retroactively to draws already
  taken.

Whichever is chosen, `docs/LESSONS.md` L42 and this section are the record of
why the numbers before the change and after it are not the same measurement.
