# What this cost me to learn

Every entry below is something I got wrong first, while building this
repository. The fix is in the code; the reason is not, and the reason is the
part worth having.

Writing a benchmark has a failure mode that writing a library does not. A
library that is wrong breaks for its users. A benchmark that is wrong publishes
a number about somebody else's work, under my name, and looks exactly like a
benchmark that is right. Most of what is below is a version of that.

Newest first.

---

## L56 — a rate cleared the bar on draws that never touched the mechanism

**What I expected.** `chunked_batchnorm_reduction` went to version 2 in the
review — the reference under `no_grad` (F04) and the prompt stating the numeric
domain the single reduction can keep (F05), two hidden tests added. I expected
the re-sweep to look like v1 did: the frontier clearing it, Haiku failing, a
`warmup` by the admission rule.

**What happened.** The two new tests cost Opus one draw and Sonnet one: Opus
8/10, Sonnet 9/10, Haiku 0/10. The top two are no longer both 100%, so
`_check_admission` would now *admit* it — `v2` would finally have a laptop
member. For about a minute that was the headline of the whole session.

**Why it survived — nearly.** The rule reads rates, and the rates said
discriminate. The shapes said the opposite, and I only saw it because L45 made
me look. Every one of Opus's two losses and Sonnet's one fails *all six* chunk
sizes, `chunk_sizes0` included — and `chunk_sizes0` is the single-chunk split,
where the chunked-reduction mistake this task is named after is correct by
construction. A draw that fails there is not getting the reduction wrong; it is
wholesale broken, failing even the case the task cannot distinguish. So the
admission-clearing losses are not the mechanism the task exists to measure. This
is L45 exactly, one version later: a rate that clears the bar on draws that
never exercised the thing the bar is about.

**What changed.** It stays `warmup` by judgement, against the rule, and the
meta.yaml comment and `ROADMAP.md` §9.4 say so in those words. Admitting it would
publish a laptop number the field did not earn — L21 in yet another costume. The
rule is a floor, never a finding (it cannot see a shape), and the second time
that sentence has been true is worth more than the task.

**What it cost.** Nearly a `v2` laptop member announced on a number that does not
mean what the column says. The shape analysis was ten minutes against a day of
wanting the task to count.

## L55 — finishing a measurement can delete the task

**What I expected.** To rent a CUDA box, finish `fused_rmsnorm_kernel` at ten
draws per model across both vendors, and ship `v2` with its three accelerated
members — the CUDA task and the two Metal ones. The sweep was the last thing
between the repository and a launch.

**What happened.** The sweep removed the task. Opus came back 5/10 (all five
losses the single-column edge, the same signature as the August five-draw run),
Haiku 5/10 on the wrapper contract, and `claude-sonnet-5`, `gpt-5.6-sol`,
`gpt-5.6-terra` and `gpt-5.6-luna` all 10/10. Two entries at 100% is what
`_check_admission` refuses, so the task I rented hardware to finish is `warmup`
now and `v2` lost its CUDA member — down to two Metal tasks, no laptop and no
CUDA.

**Why it survived.** It did not survive, and that is the point: this is
flash-attention's 2026-09-17 ejection a second time (the L43 session), and I
should have expected it. More draws and a second vendor saturate the top of a
task that five draws made look hard. The uncomfortable half is L46 wearing a new
hat — `gpt-5.6-luna`, the cheapest model in the field, going 10/10 is half the
reason the top two read 100%, and the rule calls a task Opus still fails half the
time `warmup` on the strength of the cheapest model clearing it. I considered
overriding the rule on that argument. I did not, because the last time a rule was
bent after it refused a task its author liked, the bend was the lesson (L35).

**What changed.** The task moved to `warmup`, its calibration block carries all
six models cumulatively (Opus 6/15 over August and October, a Triton kernel's
correctness not being a property of which CUDA card graded it), and the README
says in its own voice that `v2`'s discrimination is Apple-silicon-only until a
harder task exists. The thing I rented the box to prove is the thing that unmade
the plan the box was rented for.

**What it cost.** The launch's headline. `v2` was going to be "three accelerated
tasks"; it is two Metal tasks, and the honest version of that is less of a pitch.
About $4.44 of API spend and a rented 4090 to learn it, which is cheap for a
number that would otherwise have been wrong.

---

## L54 — the rule was in the prose and not in the renderer

**What I expected.** The leaderboard to keep its own rule. It says, in several
places and in its own code, that the laptop and accelerated tiers are never
folded into one number — the whole reproducibility claim rests on it. I believed
that was enforced.

**What happened.** The `v2` Mac-half table reported one pass rate per model over
a sweep that covered a laptop task and two Metal tasks on the same box. `90%,
27/30` for Opus is a laptop result and a GPU result averaged together under one
heading. The by-task grid beneath it was honest; the headline rate was the exact
blend the page forbids, three lines above where it forbids it. The claim counts
("every set is frozen", "no cloud bill", a stale `251 files / 35 entries`) were
in the same condition — true-sounding sentences nothing checked.

**Why it survived.** `render_leaderboard` pooled `solved / measured` over every
graded task in a draw, and nothing in it knew a tier from another. The rule
lived in the prose, where a reader believes it, and not in the renderer, where it
would be true. The same shape as "offline" (L52): a sentence doing the work the
mechanism was not. A benchmark's claims are not decoration — someone reads "tiers
are never folded" and trusts the number above it — so a claim without a check
behind it is a liability the moment it stops being true, and this one had.

**What changed.** A tier-filtered rate table renders the two Metal tasks on their
own — Opus 85% over 17/20, not 90% over 27/30 — beside the mixed table, which is
relabelled as the blend it is, with `flash_attention_backward` named as the
laptop member set aside. The README claims are corrected to what is true: the CPU
tier is free and the GPU tier and model calls are not, the originality claim is
narrowed to "what I could not find", the frozen-set claim points at the manifest
that now backs it, a row is the summary of however many draws it says, and the
counts are regenerated. The editorial fixes are small; the lesson is that the
rate was wrong in the direction the whole page promises it never is.

**What it cost.** The blended numbers were real draws, correctly graded per task
— nothing was miscounted at the task level, and the by-task grid always told the
truth. What was wrong was the one number a skim reads, pointed the one way the
page swears off. That is the expensive kind of wrong for a benchmark, because it
is exactly the number a casual reader quotes.

---

## L53 — an experiment's name is not its identity

**What I expected.** `group_draws` to already answer "which runs are draws of the
same thing". It keys on model, task set, and the exact list of task slugs, and
it was written carefully — the slug list is there to keep one-task debug probes
out of a full sweep's error bar, which is a real bug it fixed.

**What happened.** The review bumped a task's version under the same slug and the
grouping folded the two together: a draw graded against version 1's tests and a
draw graded against version 2's tests, reported as one experiment with one
spread. The slug matched, so the key matched, and a contract change hid inside
what reads as model sampling. Separately, `frozen_set` in meta.yaml was being
treated as a set's membership, and tasks move between frozen sets as the frontier
clears them — so "v1" had quietly stopped naming the eight tasks its leaderboard
row was measured over.

**Why it survived.** The key was built from what a run is *called* — its model
name, its set name, its slugs — and not from what it *is*. A task's identity
includes the version of its tests; a set's identity is the list it was measured
on, frozen; a model's identity is the dated snapshot behind the alias. Every one
of those can change while the name stays put, and a key made of names cannot see
it. This is L50 and L51 again at the level of the whole experiment: a name is a
claim of sameness, and sameness has to be checked, not assumed from the label.

**What changed.** Grouping now keys coverage on `(slug, task_version)`, so draws
across a version bump are two experiments, not one — and an unversioned legacy
file is kept as `None` rather than assumed to be version 1, because incomplete
provenance is not the same as a known one. Each run now records the git commit it
came from and whether the tree was dirty (`results_version` 4), so a number can
be tied back to the code that made it. And set membership is frozen in
`leaderboard/SET_MANIFESTS.yaml`, re-derived from the published files by a test,
with `frozen_set` demoted to the mutable evaluation label it always was.

**What it cost, and what is still open.** Nothing published moved — every
existing draw is version 1 and still groups as before. What this did not finish
is the rest of a run's identity: the resolved model snapshot behind the alias,
the hashes of the prompt and tests and solution, the dependency lock, and a
stored solution that could actually be replayed. Those are recorded as debt in
`ROADMAP.md` rather than quietly skipped, because the honest version of "I added
provenance" has to say which provenance it did not add.

---

## L52 — "offline" was a word, not a mechanism

**What I expected.** The graded run to be about as contained as it needed to be.
`pytest_environment` is documented as "a deterministic, offline environment",
the hidden tests are copied in only after the solver writes, and the sandbox
module's own docstring talks about guarantees. I had read my own code as
describing a sandbox.

**What happened.** The review set one fake variable and read a sentinel file one
directory up from the workdir, from inside a graded solution:

```
{"probe": "environment", "dummy_secret_inherited": true}
{"probe": "outside_workdir_read", "sentinel_read": true}
```

The child inherited the whole parent environment — including, on a real run, the
API key the adapter was called with — and could read and write anywhere the user
could. "Offline" described nothing: there is no network block, no filesystem
jail, no resource limit. The word was doing the work the code was not.

**Why it survived.** The same shape as the two lessons under this one. "Offline"
is a claim, and I had written it without building the thing it claims, exactly
as exit 0 was read as "passed" and "we never read cache" was read as a fact
about every vendor. A solver cannot read a test that is not on disk yet — that
part is real and load-bearing — and I had let a true, narrow guarantee stand in
for a broad one I had never implemented. The tests copied-in-late defence is
about *information* (the solver must not see the answer); it was quietly
promoted to a defence about *safety* (the solver cannot harm the host), which is
a different problem with none of the same machinery behind it.

**What changed.** Two things, and the line between them is the point.
`pytest_environment` now strips credential-shaped variables, so the adapter's
key — which lives in the parent, where the API call is actually made — does not
reach the child that only runs pytest. That closes the one secret the harness
itself introduces. Everything else is documentation, because env-cleaning is not
isolation and pretending otherwise would be the original mistake again: `grade`'s
docstring and a README "Running generated code" note now say plainly that the
solution runs in the same interpreter with the host's filesystem and network,
and that a real model's output should be run like any untrusted code until the
disposable-container grading path exists. F02 is not closed; it is told the
truth about.

**What it cost.** Nothing leaked that I know of — the published runs were mine
and the reviewer's, on my own machine. The cost is that "offline" sat in a
docstring for the life of the repository telling every reader, me included, that
a thing was safe which was only quiet. A benchmark's whole pitch is that you can
run it; the honest version of that pitch has to include what running it does.

---

## L51 — a contract is only tested where the input makes it bite

**What I expected.** The two chunked-batchnorm references to be done. Both are
L2 on both halves, both are mutation-tested, both have a test named for every
promise in the prompt — including a `test_the_gradients_carry_no_autograd_history`
and a swept tolerance against autograd. The landmine note in CLAUDE.md says walk
every promise to a test; I had.

**What happened.** The same review handed each reference an input the tests never
did. `test_the_gradients_carry_no_autograd_history` builds its case with
detached tensors, so it asks "does the reference *add* history" and never "does
it *pass history through*". Hand in an `x` that requires a gradient and both
references returned `dx`, `dgamma`, `dbeta` carrying its `grad_fn` — the no-grad
promise, broken, under a test written to protect it. And the variance tolerance
was swept over inputs with a variance of order one, where
`sum(x*x)/M - mean*mean` loses three digits to cancellation and the tolerance is
set four orders above that. Cluster the inputs at `1 ± 1e-9` and the raw variance
comes out *negative* — `-2.2e-16` against a true `1.5e-18` — and with an `eps`
small enough not to rescue it, `sqrt` returns NaN. The prompt had only ever said
"`eps` is positive".

**Why it survived.** Both tests exercised the contract in the regime where the
contract is slack. A promise is not tested by a test that names it; it is tested
by an input that would break it if the promise were false. Detached inputs cannot
exhibit a history leak no matter how badly the reference leaks, and an order-one
variance cannot exhibit a cancellation the eps floor is four orders above. This
is L8 — an unenforced promise lets two disagreeing implementations both pass —
but one level meaner, because the enforcing test *existed* and pointed at the
right line while testing the one input that could not fail it.

**What changed.** Both references run every function under `no_grad`, which is
the whole of the contract rather than a line of it, and a new test hands in
tensors that require a gradient and checks the result and the inter-pass buffer
come back clean — verified to fail if the decorator is removed. The reduction
prompt now states the numerical domain the single-reduction interface can
actually keep: inputs of order one and an `eps` of conventional size, over which
`var + eps` stays positive, and explicitly *not* a variance driven below the
arithmetic's floor by a smaller `eps`. A new test confirms the finite result on
near-constant inputs, and claims only finiteness, because `1/sqrt(var + eps)`
amplifies the cancellation and accuracy there is ill-conditioning the prompt
already scopes out. Both tasks are **version 2**; their version-1 calibration is
kept labelled as such and a re-sweep is pending (it would have graded against the
old tests).

**What it cost.** Nothing published — these are laptop `warmup` tasks whose rows
are not the frontier story, and no draw was re-scored dishonestly. What it cost
is the claim that "every promise has a test" meant what it sounded like. It meant
every promise has a test; it did not mean every test has an input with teeth. The
clamp I did not add is the tell: making `var.clamp_min(0)` the fix would have
turned the NaN into a quiet wrong number, which is the trade a benchmark exists
not to make.

---

## L50 — an exit code is a claim the process makes about itself

**What I expected.** The grader to be sound. A reference passes, a wrong answer
fails, a broken task is caught; the whole of `STATUSES` exists so no outcome
reaches the leaderboard unclassified. I believed the classification was the
careful part and the reading of pytest's result was mechanical.

**What happened.** An outside review handed a solver one file —
`import os; os._exit(0)` — and it was recorded **passed**. The hidden tests
import the solution at collection; the solution killed the process before a
single test ran; pytest exited 0 having judged nothing, and the grader read
exit 0 as "everything passed". Two more of the same shape sat beside it. A CPU
solution that returned 125 — the stall guard's exit code, on a task that has no
guard — was recorded `mps_stalled`, machine noise dropped from the rate. And
`validate` counted a task that declared a missing dependency, and so never ran
at all, inside its "14 task(s) validated" total.

**Why it survived.** Each one trusted a number the process produced about
itself as if it were the thing the number stands for. Exit 0 does not mean the
suite passed; it means a process ended with status zero, and anything in that
process can end it. Exit 125 does not mean a stall; it means something returned
125, and the guard is only one of the things that can. "The call returned" is
not "the call was answered" — the same shape as L46, one layer down, and the
`solution_error` docstring had already written the principle for the
import-error case without my noticing it generalised: the starter imports
cleanly by contract, so anything wrong in the graded directory was put there by
whatever wrote the file.

**What changed.** A pass now requires the run to have *reached the end*: the
number of judged tests must equal `hidden_tests`, the count `validate` already
re-derives on hardware that can run the task, so `0 passed` and `1 of 2 passed`
are no longer passes. The stall guard writes a sentinel file beside the tests
the instant it fires, and a `mps_stalled` verdict now requires that file to be
present — an exit code a weak solution can hand back by accident no longer
names a machine fault. `validate` keeps a task it could not run — missing
dependency or absent accelerator — out of the validated total and reports it on
its own line, with `--strict` turning "could not check" into a non-zero exit
for CI. Five regression tests, one per door.

**What it cost.** Nothing published — the reviewer produced these with crafted
solutions, not real draws, and no leaderboard row came from a sabotaged run.
But every number on the page rested on a grader that would have called a
no-op a pass, and the thing a benchmark sells is that it would not. The gap
was open for the life of the repository and closed in the session after it was
shown to me.

---

## L49 — the invariant was true, and it was a fact about one vendor

**What I expected.** `test_nothing_published_was_billed_at_a_cache_rate` to keep
passing. It asserts every published draw read zero cached tokens, and the reason
was solid: each task is sent once, as its own prompt, so there is no prefix to
reuse.

**What happened.** CI went red on the `v2` sweep's own evidence:

```
>           assert tokens["cache_read"] == 0, path.name
E           AssertionError: gpt-5.6-luna-20260917T042822+0000-draw2.json
E           assert 3862 == 0
```

Anthropic caching is **opt-in** — it happens when a request sets
`cache_control`, and this adapter never does. OpenAI caches long prompt prefixes
**automatically**, with nothing to opt into and nothing in the request that asks
for it. Nine of every ten OpenAI draws read about 90% of their prompt from a
cache nobody requested. Every one of the 86 Claude draws still reads zero.

**Why it survived.** Because it was *true*, provably, for two years of this
repository's life, and it was checked rather than assumed — which is the good
version of the habit. What I had not noticed is that the check's justification
lived one layer below the check: not "we do not reuse prefixes" but "this
vendor does not reuse prefixes unless asked". The sentence in the test even says
"this adapter sends each task once", and that is still accurate. The adapter was
never the thing deciding.

**The consequence is on the page, not just in the suite.** The pricing itself
was right — `price_from_counts` subtracts the cached count from the input count
because the API reports it as a subset, and all 311 published costs re-derive.
But the *published* OpenAI cost is a warm-cache price: 5.4% to 6.1% below what
the same draws cost cold, because output dominates the bill. On a page whose
argument is that you can re-run this and get the same numbers, a cost that is
only reachable on your tenth attempt needs saying out loud, so the leaderboard
now carries the cold-cache column beside it.

**What changed.** The test split into the three invariants that are actually
true: nobody is billed for a cache *write* (either vendor, different reasons), a
cached read is always a *subset* of the input it came from — which is the
assertion protecting the subtraction from silently under-billing every row if a
vendor ever reported them alongside instead — and Anthropic draws specifically
read zero, which is what makes two Claude rows comparable to each other.

**What it cost.** One red CI run, and it was the sweep's own data that found it
rather than a reader. The rule worth keeping: **an invariant that holds by
construction is only as portable as the construction.** When a second
implementation arrives, the tests that encode "this cannot happen" are the ones
to re-read first, because they will keep passing right up until they are wrong.

---

## L48 — a bool with three answers, and a measurement that could not have settled it


**What I expected.** The `v2` sweep's own draws to be either passes or failures,
with the no-evidence statuses reserved for the things they are named after.

**What happened.** `claude-haiku-4-5` produced two `collection_error` draws on
`metal_softmax_backward_kernel` and one `timeout` on
`metal_cross_entropy_kernel`. `collection_error` does not mean the model failed.
It means **the task itself is broken** — the hidden tests do not collect against
the untouched starter, so every number ever measured against that task is void.
It prints as `BROKEN`. And `scratchbench validate` says both tasks are fine.

**Where it comes from.** A graded run that produced no gradeable result asks one
question to decide which side of the evidence line it falls on: does the
baseline still collect? The answer is a bool:

```python
        return returncode == PYTEST_TESTS_FAILED and not parse_summary(output).get("error")
```

`run_pytest_until_healthy` returns `STALLED_EXIT` when every attempt landed in a
stalled MPS process. `STALLED_EXIT` is not `PYTEST_TESTS_FAILED`, so a baseline
that **never ran** returns exactly the same `False` as a baseline that ran and
could not collect. The question has three answers and the return type has two,
so the machine's state gets published as a verdict on the task.

This is L42's own failure mode leaking through a seam L42 built. The guard
exists so a stalled process is never recorded as a `timeout`; it does its job
for the graded run and then the attribution path collapses its result into a
different wrong label. Both are `NO_EVIDENCE`, so no rate moves — what moves is
the meaning, and "this task is broken" is the one alarm here that must not cry
wolf.

**The part that is actually about me.** I had the mechanism within a minute and
then tried to confirm it with the wrong measurement. I compared `duration_s` on
the two `collection_error` draws against their neighbours, found them at 1.47s
and 2.06s, and started writing that the durations were too fast for four stalled
attempts. They say nothing at all: `duration_s` times the **graded** run, and
the baseline run happens inside `starter_collects_cleanly`, downstream of it and
unrecorded. I had produced a number, and its only real property was that it
existed.

L42 is *a control that has not been repeated is not a control*. This is one
below that: **a measurement that could not have come out the other way is not a
measurement.** Before believing a number, say out loud what value would have
refuted the hypothesis. I could not have said it, because there wasn't one.

**What changed.** `starter_collects_cleanly` returns `bool | None`, `None`
meaning the baseline never ran and accused nobody, and `grade` maps that to
`mps_stalled` instead of `collection_error`. Two tests hold both sides of the
seam: a stalled baseline must not say the task is broken, and a genuinely
uncollectable task still must.

**What is still open, and it is open rather than resolved.** Whether those two
published draws were stalls is **not established**. The fix is provable by
construction; the diagnosis of the historical draws is not, and confirming it
means re-running Metal work on a machine that had a sweep in flight. The draws
stay exactly as they were recorded either way — relabelling evidence after
seeing which way it went is L43, and it does not become acceptable because I
would be relabelling it in the direction that flatters my own diagnosis.

**What it cost.** Nothing published, because the sweep's rates exclude both
statuses. It came within one paragraph of costing a written claim that a
measurement supported, when it could not have.

---

## L47 — a test that reads the machine reports the machine, and CI told me so three times

**What I expected.** The harness suite was green on this laptop at the end of
every session, so the suite was passing. CI is a second machine running the same
suite, which I thought of as redundancy rather than as a second measurement.

**What happened.** CI had been **red on every push since 2026-09-08** — three
consecutive pushes, both Python versions, always the same single test — and I
opened three sessions on top of it without looking. Session 17's restart file
even says "check whether CI is green, because a new CI step is being added this
round". I wrote that sentence and did not run the command.

The test is `test_every_attempt_stalling_is_reported_as_no_evidence`. It exists
to guard the thing L42 is about: a stalled MPS process recorded as a `timeout`
would be evidence and a failure, and therefore a published claim about a model
that actually describes my laptop. It fakes a permanently stalling `run_pytest`
and asserts the outcome is `mps_stalled`. On a GPU-less runner the assertion
never got that far:

```
>       assert outcome.status == "mps_stalled"
E       AssertionError: assert 'needs_accelerator' == 'mps_stalled'
```

`grade` checks the hardware **before** it reaches the mechanism:

```python
    if task.requires_gpu and task.accelerator not in available_accelerators():
```

The task is `metal_cross_entropy_kernel`, so on my Mac the gate opens and the
test measures the guard; on Linux the gate closes and the test measures the
gate. One assertion, two different meanings, decided by whoever ran it.

**Why it survived.** Because it passed where I was looking. The suite is the
thing I run to find out whether I have broken something, and it answered "no"
every time — correctly, for this machine. The failure needed a machine unlike
mine to appear, which is exactly the class of bug CI is for, and I was treating
CI as a copy of my laptop instead of as the other half of the measurement. Note
the symmetry with what this repository publishes: `needs_accelerator` exists
precisely so that a missing GPU is never silently rounded into a result. The
same confusion I built a status table to prevent in the product, I shipped in
the test suite.

**What changed.** The test now pins the hardware it needs instead of inheriting
it, so the assertion means the same thing everywhere:

```python
    monkeypatch.setattr(sandbox, "available_accelerators", lambda: {"metal"})
```

That is legitimate rather than a test dodging its environment: `add_stall_guard`
keys on `task.accelerator` and never on the host, so the mechanism under test is
genuinely machine-independent and the fixture now says so. I then ran the whole
suite against a plugin that forces `torch.cuda.is_available()` and
`torch.backends.mps.is_available()` to `False`, which is the CI environment
reproduced locally — **131 passed** — so the claim is that the suite is
hardware-independent, measured rather than assumed.

**What it cost.** Three sessions of a red badge on a public repository whose
whole argument is that you should not have to trust its numbers, plus the
particular embarrassment of the badge sitting in the README's first line. The
rule I actually needed is one line: **a green suite on one machine is one
draw.** The same sentence this repository publishes about models applies to its
own tests, and the cheap defence is `gh run list --limit 3` at the start of a
session rather than the end.

---

## L46 — the second vendor did not add a column, it moved the goalposts

**What I expected.** §3 was scoped as plumbing: write the OpenAI adapter, take a
few draws, get a second column on the leaderboard so the page stops reading as
one lab's fan project. The interesting work was supposed to be §5.

**What happened.** `gpt-5.6-luna`, the cheapest model in OpenAI's current
generation at $0.20 and $1.20 per million tokens, solved every task it was
asked. All five of `v1`, on the first attempt, for **$0.01**. Then
`chunked_batchnorm_reduction`, which `claude-haiku-4-5` went 0 for 10 on.
Then `flash_attention_backward`, the **only** laptop task in this repository
that separates two frontier Claude models. Then `metal_softmax_backward_kernel`,
where `claude-sonnet-5` scores 6 of 10. Nine tasks asked, nine solved, about
three cents in total, and not one graded failure to look at.

`gpt-5.6-terra` and `gpt-5.6-sol` also went 5 of 5 on `v1`, at $0.11 and $0.20.

**Why this is a lesson and not just a number.** Every difficulty judgement in
this repository was calibrated against one vendor. `warmup` means "the frontier
clears this" and the frontier meant Claude; L21 is about difficulty assigned by
feel and refuted by a sweep, and this is the same shape one level up —
**difficulty assigned against one lab's models and refuted by another's.** A
task is not hard, it is hard *for the models you asked*, and until today the set
of models asked was three Claudes.

**What the nine draws support, stated as arithmetic rather than as a mood.**
Nine tasks of mixed difficulty, nine solved. If `gpt-5.6-luna`'s true per-task
rate on this set were `p`, the chance of seeing 9 of 9 is `p**9`:

```
p=0.50 -> 0.0020    p=0.70 -> 0.0404    p=0.90 -> 0.3874
p=0.60 -> 0.0101    p=0.80 -> 0.1342    p=0.95 -> 0.6302
```

So the one-sided 95% bound is **p >= 0.72**. That is a real measurement and it
is not weak: a rate of 0.5 would produce this result twice in a thousand tries.
The set as it stands does not challenge this model, and that conclusion is
supported.

**What they do not support, and the distinction is sharp rather than cautious.**
They cannot rank `luna` against Opus or Sonnet on any single task, because one
draw has almost no resolving power against a model that already passes often.
`claude-sonnet-5` scores 8 of 10 on `flash_attention_backward`; a single Sonnet
draw passes 80% of the time, so `luna` passing it once is exactly what a Sonnet
would most likely have done too. The same draw is compatible with `luna` being
better, equal, or somewhat worse. No per-task rate is published from any of this
and none should be.

The one place a single draw carries weight is where the other side is measured
and near zero: `claude-haiku-4-5` is 0 of 10 on `chunked_batchnorm_reduction`,
which bounds it at **p <= 0.26**, and `luna` — a model at the same end of the
price range — solved it. That is a contrast between a measured bound and a
single observation, which is worth stating and is not worth a number.

**Cost, which is measured rather than inferred.** Every figure below is
recomputed from the token counts checked in beside it:

```
gpt-5.6-luna      5/5 on v1        $0.0110   $0.0022/solved
gpt-5.6-terra     5/5 on v1        $0.1050   $0.0210/solved
gpt-5.6-sol       5/5 on v1        $0.1964   $0.0393/solved

claude-haiku-4-5  4/8 laptop set   $0.0766   $0.0191/solved
claude-sonnet-5   8/8 laptop set   $0.3881   $0.0485/solved
claude-opus-5     8/8 laptop set   $0.5560   $0.0695/solved
```

⚠ The two halves are different task sets — five `v1` tasks against the older
eight-task laptop grouping — so the ratio is indicative and not a like-for-like
comparison. What survives that caveat is the order of magnitude: the cheapest
model measured here solves a laptop task for about a fifth of a cent, and the
most expensive one for about seven cents.

**What changed.** `ROADMAP.md` §10.1 shipped**What changed.** `ROADMAP.md` §10.1 shipped `v2` on the argument that a fourth
laptop candidate has unbounded holding cost, and that argument survives — it
gets stronger, because "write a harder task" now means harder for two fields
rather than one. What changes is §5: the first `v2` sweep has to include the
OpenAI models, and the announcement cannot describe the laptop tier as
discriminating without saying for whom.

**What it cost.** $0.32, and a framing I would have published without it.

**A second thing I got wrong, in the other direction.** The first version of
this entry hedged the result into meaninglessness: nine draws, one model, no
repeats, so "the honest statement is the narrow one." That was not honesty, it
was an unwillingness to be wrong in public. 9 of 9 across nine tasks bounds the
rate at 0.72 and that bound was available from the same data I was refusing to
read. **Understating what a measurement supports is not a safer error than
overstating it** — it is the same failure to report what was measured, and here
it would have buried the finding that most affects what this repository claims.
The rule that produces the right answer in both directions is the one already in
L27: go and look at the shape, then say exactly what it shows, no more and no
less.

---

## L45 — I built the mechanism correctly and it still measured something else

**What I expected.** L44 said candidate 2 failed because the whole-batch state
its graded unit was missing arrived in that unit's own signature. Candidate 3,
`chunked_batchnorm_reduction`, was written to remove exactly that: the mean, the
variance, the count and both parameter gradients appear in no signature, the
caller performs one reduction whose contents the solution designs, and the
identity that has to be recovered needs a mean that does not exist yet when the
reduction runs. Every clause of `ROADMAP.md` §9.3's two-halved criterion is
satisfied, and this time by construction rather than by analogy. I expected the
frontier to lose draws to the reduction.

**What happened.** Thirty draws: `claude-opus-5` 9/10, `claude-sonnet-5` 10/10,
`claude-haiku-4-5` 0/10. The count looks like a task that discriminates. The
shapes say otherwise, and the discriminator is one parameter id. `chunk_sizes0`
is the split `(6,)` — one chunk holding the whole batch — where the mistake this
task is named after is correct by construction. Eleven draws were lost and **ten
of them fail `chunk_sizes0` as well**, so ten of the eleven are not the chunked
mistake. Opus's single loss is one of the ten:

```
FAILED test_matches_autograd[chunk_sizes0]
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 2
```

It left `count` at shape `(channels,)` and divided it into a three-dimensional
tensor while the four quantities beside it were reshaped. The reduction it
designed was right in all ten draws.

**The one draw that is the task.** Haiku's tenth, and it is the only loss in
thirty that passes `chunk_sizes0` and fails every other split. It designs a
correct four-row buffer — `sum(dy)`, `sum(x)`, `sum(dy*x)`, `count` — and then
leaves `sum(x*x)` out of it, so the variance has to come from somewhere:

```python
sum_x_sq = (x_masked ** 2).sum(dim=(0, 2))   # this chunk only
variance = (sum_x_sq / N) - mean_sq          # N is the whole batch's count
```

A chunk-local numerator over a whole-batch denominator. With one chunk the two
agree and the answer is exact; with any split it is wrong, which is precisely
the failure the task exists to catch. One draw in thirty.

**What I actually learned, and it is not L44 again.** L44 was a design defect: I
gave the answer away and the measurement was invalid. Here the design is sound
and the measurement is valid, and it reports that **the frontier is 20 for 20 on
the mechanism**. Those are different results and only the second one is
informative about the field. The lesson is that satisfying a discriminating
criterion is not the same as discriminating: the criterion tells me the task
*can* separate models, and only a sweep tells me whether the models I am asking
are in fact separated by it. I had been treating §9.3's check as sufficient
because it was written in response to a failure, and a rule written in response
to a failure feels like a guarantee. It is a filter.

**What it changed.** The task ships as `warmup`. Note that the admission rule
would have taken it: it refuses a task the top two measured entries both clear,
and Sonnet 10/10 with Opus 9/10 does not clear. Reading the rule and stopping
there would have put a task in `v2` whose headline reduction nothing at the top
of the field got wrong, on the strength of a broadcasting slip. **A rule that
counts cannot see a shape, so passing it is a floor and never a finding** — and
the number 9/10, read alone, would have said the opposite of what the thirty
draws say. That is L27 for the third time, and the third time is the one worth
writing down: I keep re-deriving it because the rate is what the tooling hands
me and the shape is what I have to go and look for.

---

## L44 — I copied the shape of the mechanism and handed over its content

**What I expected.** Selim chose option A in `ROADMAP.md` §9.1: laptop candidate
2 gets written against L28's mechanism rather than §2.0's hazard. I wrote the
mechanism down before I designed anything, in the restart file's own words — the
graded unit is handed one piece of a decomposition the *caller* chose, the
correct answer needs state that piece does not have, and materialising the whole
is unavailable rather than discouraged. `chunked_batchnorm_backward` satisfies
every clause of that sentence. Batch normalisation's backward pass needs two
sums over the whole batch, a micro-batch holds neither, and no function in the
task ever sees a second chunk. I expected it to behave like
`flash_attention_backward`.

**What happened.** Thirty draws, and the top of the field never noticed there
was a problem. `claude-opus-5` 10/10, `claude-sonnet-5` 10/10, `claude-haiku-4-5`
2/10. The admission rule refused it and it is `warmup`. Opus wrote the whole of
the supposed difficulty as one line:

```python
dx = gamma_b * inv_std * (dy_masked - (dbeta_b + x_hat * dgamma_b) / n)
```

`dgamma_b`, `dbeta_b` and `n` are `dgamma`, `dbeta` and `total_count`: three of
the function's own arguments. **The state the chunk was missing arrives in the
signature.** There is nothing to reconstruct, so what is left is an algebra step,
and an algebra step is not a discriminator.

**Where I went wrong, and I had the tell in front of me.** While choosing the
API I rejected a variant on the grounds that it "gives away half of the L28
insight", and then justified the one I kept by an analogy that does not hold:
flash attention hands the block function `o` and `do`, so handing this one
`dgamma` and `dbeta` looked like the same move. It is not the same move. `o` and
`do` are forward artefacts, and the whole difficulty of that task is
*recognising* that the row correction it needs is `rowsum(dO * O)` — a quantity
nobody passes it, that has to be derived from the two things it happens to
already hold. I passed the correction itself. The distance between those two
designs is the entire task.

So the mechanism has two halves and I implemented one. A decomposition the
caller chose is **necessary**: it makes materialisation unavailable. It is not
**sufficient**: what discriminates is that a whole-row quantity has to be
recovered from artefacts that were kept for another reason. Handing it over as a
parameter converts a derivation into an exercise in reading a signature, which
is the thing frontier models are furthest ahead at.

**What the failure shapes say, and they say it independently.** All eight of
Haiku's losses fail `test_matches_autograd[chunk_sizes0]` — the case where the
batch is a *single* chunk. The mistake this task was built to catch is by
construction correct there. So not one of the eight failures is the chunked
mistake: one draw drops `gamma` from the scale correction, another sums a
pointwise term over the chunk, and the rest are ordinary algebra. Only one of the
eight also fails split invariance. **The task measured arithmetic care and never
once measured the thing it was named after** — and reading the count, 2/10, would
have told me none of that. This is L27 again, one project over: a rate is not a
shape.

**What changed.** The task ships as it is: `warmup`, calibrated, its thirty
draws checked in. It is a real measurement of a real task and it separates Haiku
cleanly; deleting it after seeing which way it went is the move L43 exists about.
`V2_DESIGN.md` §2.0 gains the scoping sentence §9.1 asked for, and `ROADMAP.md`
§9.3 carries the criterion with **both** halves stated, plus candidate 3 designed
against it: the chunk returns a reduction buffer whose contents are its own
choice, the caller sums it, and nothing in any signature says which two sums the
answer needs.

**What it cost.** $1.71 and a session, for a `warmup` task and a criterion that
is now written down correctly. The cheap check I skipped is a sentence long, and
it is now the first question §9.3 asks of any candidate: **name the quantity the
graded unit cannot compute, then find it in the argument list. If it is there,
there is no task.**

## L43 — I wrote the exposure into the ledger and did not check whether it had already happened

**What I expected.** L42 ended with a consequence I recorded as a risk:
`metal_cross_entropy_kernel` is published, its calibration draws were taken on
this machine before the stall was known, so a `timeout` among them might be a
stalled process rather than a model that could not answer. I wrote that as debt
item 6, scheduled it, and moved on to building the guard. It read like diligence.

**What happened.** Item 6 says "re-draw", so before spending the two dollars I
counted what was already checked in. Thirty seconds of tallying:

```
metal_cross_entropy_kernel  claude-haiku-4-5
  {'solution_error': 6, 'adapter_error': 1, 'passed': 1, 'timeout': 1, 'failed': 1}
```

One `timeout`. In the published evidence. In the denominator of the rate
`meta.yaml` claims and `README.md` prints, since 2026-08-13. The risk I had
written down as something that *could* have happened had already happened, and
it had been sitting in a checked-in JSON file the whole time, countable by
anything that could read a dictionary.

**Where I went wrong.** Writing a risk down feels like handling it. It is not
the same act, and the gap between them has a tell: **a risk stated about data
you already have is not a risk, it is an unread measurement.** The debt ledger
is for work that has to be scheduled. "Does this defect appear in my published
evidence" is not work, it is a query, and it should have run in the same minute
the defect was understood — before the guard, before the roadmap edit, before
the journal.

The near-miss is that I nearly re-drew the task without ever counting the old
draws. I would have got clean new numbers, replaced the block, and never
learned that the published one was wrong, which is the same outcome as not
looking.

**What changed.** The draw stays counted as a failure. Removing it would improve
Haiku's rate on the strength of a fact discovered after seeing which way the
draw went, and a benchmark that edits its own evidence in the direction it just
learned about is not measuring anything. It is also the conservative direction:
a `timeout` scored as a failure can only understate a model, never flatter one.
`meta.yaml` says all of this in the file, `README.md` says it on the page, and
the block is now twenty draws per model: the ten from 2026-08-13 plus ten taken
under the guard.

**What it cost.** Nothing, this time, and only because item 6 happened to say
"re-draw" rather than "document". Had it said the second thing I would have
written a paragraph about an exposure and never opened the file that proved it.

## L42 — It was never the task. Every other process on this machine is stalled.

**What I expected.** L41 left seven suspects dead and no cause, and every one of
those seven was something inside `metal_softmax_backward_kernel`: its shapes,
its sentinel, its expected values, its kernel, the size of what it compares. The
control that justified looking there was two measurements — the task's reference
ran 1.2-3.7 s every time, and the published Metal task's starter ran under 1.4 s
ten times out of ten — so I expected to find the cause inside the suite, and I
started by cutting the suite in half.

**What happened.** The first cut was not a half of the suite, it was the one
factor the seven measurements had all held fixed: *the starter fails all 81
tests*. Every variant in L41 changed the task and kept the failures, so "empty
kernel" and "81 failing assertions" had never been told apart. Four cells, same
sandbox assembly, same environment:

| Cell | Kernel | Tests | Result |
|---|---|---|---|
| A | starter | 81 fail | 2.62 s, 2.10 s, 2.09 s |
| B | reference | 81 pass | killed at 100 s, 120 s, 120 s |
| C | reference, tolerance 0 | 73 fail | 2.05 s, 2.17 s, 1.93 s |
| D | starter, assertions removed | 81 pass | 97.58 s, killed at 120 s, 120 s |

Three rounds, interleaved, and it looked decisive in the wrong direction: the
*reference* was slow every time, which contradicts L41's own control outright,
and the split fell exactly on whether the tests passed. Two more cells agreed —
the passing suite with every MPS tensor deliberately kept alive ran 2.46 s and
1.72 s, and the failing suite with `gc.collect()` and `empty_cache()` after each
test was killed at 90 s twice. A tidy mechanism, both directions confirmed.

It was wrong. Running the identical command six times in a row:

```
release #1  0.00s   release #2  43s   release #3  0.00s
release #4  46s     release #5  0.00s release #6  37s
```

Same flag, same file, same everything. Then with the flags reversed, and the
slow one was still whichever ran second. **The pass/fail split was the ordering,
and every cell table above is a picture of alternation, not of a mechanism.**

What is actually true, measured over fourteen consecutive launches of one
script:

```
launch  probe_ms  workload_s
     1      0.56       0.086
     2    776.01      10.201
     3      0.50       0.071
     4    420.51       8.500
     ...   alternating, 7 pairs, no overlap
```

Every other process that touches `torch.mps` is stalled for its whole life, and
in a stalled process a host-to-device-to-host round trip costs hundreds of
milliseconds instead of half of one. `tools/check_mps_stall.py` re-derives it.

**What I ruled out, each by a measurement rather than by argument.**

| Suspect | How it died |
|---|---|
| The task | `metal_cross_entropy_kernel`, already published, stalls the same way: its reference measured 1.89 s, 48.76 s, 1.50 s |
| The custom shader | A loop of plain `torch.softmax` with no `compile_shader` anywhere stalls too |
| Pass versus fail, and tensor retention | Both reproduced by running one unchanged command twice |
| The GPU being busy | While a suite sat stalled for 75 s, a second process launched *into* that window did 50 MPS softmaxes in 0.687 s |
| A driver reset | `AGXAccelerator` `recoveryCount` is 0 throughout |
| The previous process still tearing down | A ten second gap between launches changes nothing |
| The last Metal client exiting | A keeper process holding an MPS context open the whole time changes nothing except the phase |
| CPU starvation | 2599 samples of a stalled process, all of them in `-[_MTLCommandBuffer waitUntilCompleted]` under `__psynch_cvwait` |

**Where I went wrong, and it is not L41's seven measurements.** Those were good.
The error is one level up: L41 fixed the frame. It asked "what about this task
makes it slow", and every experiment after that inherited the assumption in the
question. The control that licensed the frame — "the reference is fast, the
other Metal task is fast" — was two samples of a fair coin, read as a constant.
L40 already says a cause that explains one observation is not the cause. This is
its neighbour: **a control that has not been repeated is not a control.** Both
of L41's controls were single-digit samples of something that alternates.

And the near miss is worth keeping. Cells A through F produced a clean, falsifiable,
both-directions-confirmed mechanism — retention keeps buffers out of the
allocator — that was pure artefact. It survived six rounds of interleaving
because interleaving defends against drift, not against alternation. The cheap
check that would have killed it in ninety seconds is the one that eventually
did: **run the same cell twice in a row before believing any difference between
two cells.**

**What changed.** `tools/check_mps_stall.py`, which both measures the population
and exposes `probe()` and `is_stalled()` so a graded run can refuse to grade in
that state. `ROADMAP.md` section 9 item 3 is rewritten around the real fault and
section 9.2 carries the choice it forces. L41 keeps its seven measurements and
gains a header saying its frame was wrong, because deleting it would delete the
evidence for this entry.

**What it cost.** Session 14's second half and most of session 15, to arrive at a
fault that had nothing to do with either task — and a published task,
`metal_cross_entropy_kernel`, whose calibration draws were taken on this machine
and are now known to have been exposed to it.

## L41 — Half the runs are a hundred times slower and I cannot say why

⚠ **Read L42 first.** The seven measurements below stand and the conclusion
above them does not: the stall is not this task's, and the control quoted in
this entry — "the reference measured 1.2-3.7 s every time" — does not survive
repetition. What follows is kept because L42's evidence is built on it.


**What I expected.** The new Metal task validated cleanly the first time: 81
tests passed against the reference, 81 failed against the untouched starter, and
the whole thing took 23 seconds. I expected that to keep being true.

**What happened.** It is true about half the time. The same command, the same
machine, nothing else running, the untouched starter:

```
1.0s   1.4s   33s   124s   278s   312s   344s   2920s
```

The reference, in those same runs, measured between 1.2 s and 3.7 s every single
time. And `metal_cross_entropy_kernel` — the Metal task already published, whose
starter also fails every one of its tests — ran ten times out of ten under 1.4 s.

Inside a slow run the time is not in one place. `--durations` shows every test
uniformly slower: 7.17 s, 7.01 s, 6.02 s, 5.86 s where the fast run's slowest
test is 0.35 s. So the process enters a state, for its whole life, where every
MPS operation costs about a hundred times what it should.

**What I ruled out, each by a measurement rather than by argument.**

| Suspect | How it died |
|---|---|
| CPU starvation from the status line | Real, and fixed — see below. The stall came back with the machine idle at load 1.8 |
| The largest test, 128×4096 | Deselected it; a run still took 343.71 s |
| `torch.autograd.grad` in the expected value | Replaced with a closed form; 1.29 s, 311.87 s, 1.07 s |
| Memory pressure or swap | The **slow** run peaked at 261 MB against the fast runs' 326 and 330, with 14 page faults against 88 |
| The NaN sentinel in the output buffer | Made it a finite 1e30; 1.36 s, 301.53 s, 1.33 s |
| A kernel that writes nothing | Made the starter write zeros; the second run took over half an hour |
| The size of the compared tensors | Restricted to shapes no bigger than 2×1000; 1.75 s, 1.02 s, 124.63 s, 1.12 s |

Seven suspects, seven measurements, no cause.

**What it costs, precisely.** `time_limit_s` is 300 s and `STATUSES` calls a
killed run `timeout`: evidence, and a failure. So a pass rate computed over
draws affected by this is still *sound* — a solution that would have failed is
recorded as failing. What is not sound is the failure *shape*, and the failure
shape is the thing this repository publishes instead of partial credit. It is
most of what journals 09 through 13 are made of.

**What changed.** The task ships `frozen_set: unvalidated` and is not
calibrated. Thirty draws would have cost about two dollars and produced pass
rates I could defend beside failure shapes I could not, and publishing the
second alongside the first is how a benchmark stops being worth reading. The
`meta.yaml` says so in the file rather than in my head, and `ROADMAP.md`
section 9 item 3 carries it as the thing to fix before the `v2` sweep — which is
the run it would corrupt.

**What it cost.** The session's second half, a task that is finished and cannot
be published yet, and the reminder that "I verified it once" and "it is
reproducible" are different claims. The first `validate` of this task was green.
So was the second. The third was `BROKEN`.

## L40 — I found a cause, and stopped looking

**What I expected.** `validate` returned `BROKEN  starter timed out instead of
failing` on a task whose starter fails in under two seconds. I went looking for
the cause and I found one, and it was real: `ccusage`, the process that draws my
terminal's status line, was at 748 percent CPU on a ten-core machine. Sampled
once a second for six seconds it was above 98 percent in five of them. I killed
it and the suite that would not finish in ten minutes ran in 1.98 s.

That is a clean experiment with a clean result, and I wrote the finding up,
disabled the status line, updated the note that had predicted this exact
regression, and moved on.

**What happened.** The symptom came back with the machine idle at load 1.8 and
nothing above ten percent in `ps`. Twice. Then five more times.

**Where I went wrong, and it is not the debugging.** Finding `ccusage` was
correct work. The tell that took me too long — `real 17,42` against `user 2,01`,
a program that is starved rather than slow — is a good tell and it is now a card
in the note. The error was what I did *after* the fix worked: I had one
observation, "kill it and the slow run becomes fast", and I treated it as an
explanation of every slow run I had seen. It explained one.

The give-away was available and I did not look at it: the slow runs I already
had in hand were **1.98 s, 33 s, 278 s, 2920 s**. A single cause that switches
on and off does not produce a spread like that. Four points, three orders of
magnitude, and I read them as one phenomenon because I had a story.

**What changed.** L41 is the honest state of the actual problem, with seven
suspects killed by measurement and no cause. And the rule I broke, written where
I will hit it: **a cause that explains one observation is not the cause.** Before
closing an investigation, list the observations the fix has to account for and
check it against all of them, not against the one that prompted it.

**What it cost.** An hour recovered and then several more spent, because the
second investigation started from a conclusion I had already banked. The status
line stays off — that part was a genuine win, it is still eating cores when it
runs, and the measurement stands on its own.

## L39 — The race was never about the hardware. It is about the distance.

**What I expected.** L32 is titled "The race this hardware refuses to show me".
The forward Metal kernel reads the row maximum out of `scratch[0]` and every
thread then overwrites the scratch; dropping the barrier between those two is a
textbook write-after-read race, and I hunted it over thirty configurations
without ever making it fail. The mutant is checked in expecting `SURVIVES`, and
I wrote at the time that a machine which caught it would be news.

The backward kernel reuses one scratch array three times instead of once, so I
wrote the same mutant at the second reuse and gave it the same expectation. It
seemed like bookkeeping.

**What happened.** It is caught, and reliably:

```
trial 1: CAUGHT    23 failed, 58 passed
trial 2: CAUGHT    22 failed, 59 passed
trial 3: CAUGHT    22 failed, 59 passed
trial 4: CAUGHT    22 failed, 59 passed
trial 5: CAUGHT    22 failed, 59 passed
trial 6: CAUGHT    22 failed, 59 passed
```

Eight runs of eight across two sessions of the same machine, with the same
mutant one reduction earlier surviving all 81 tests in the same pass.

**Why, and it is not the device.** The two sites differ in what sits between the
read and the reuse. At the maximum, every thread reads `scratch[0]` and then
walks its whole strided slice of the row before it writes anything back — a
hundred-odd iterations of loads and an exponential each. The read is finished
everywhere long before the first write lands, so the window never opens. At the
denominator, the value the threads are about to publish was computed in that
same pass and is already sitting in a register: the read is followed
*immediately* by the write, and the fastest lane clobbers `scratch[0]` while
slower lanes are still reading it.

So L32's explanation — Apple's scheduler keeps simdgroups close enough in
lockstep that the window never opens — is the right shape and the wrong
variable. The window is not a property of the scheduler alone. It is the
scheduler's spread measured against the work the program puts between the read
and the write, and the forward kernel happened to put a whole pass of the row
there.

**Where I nearly went wrong.** My first move was to change the expectation to
`SURVIVES` and get on with the session, because that is what the sibling mutant
does and the difference looked like noise. What stopped it was the script
refusing to agree with me: the expectation is checked in both directions, so a
survivor that is caught fails the run and has to be explained rather than
adjusted. That is the third time the two-way expectation has produced a finding
instead of a shrug (L31, L36, this).

The second near-miss is quieter. The failure count moves between 21 and 23 from
run to run, so my instinct was to call the mutant flaky and delete it. The
*verdict* is stable across eight runs; only the blast radius wobbles, which is
exactly what a real race looks like and is worth recording as such.

**What changed.** The mutant is checked in expecting `CAUGHT`, with the count
range in its description so nobody later reads a 21 as a regression from a 22.
L32 keeps its evidence and loses its generality: its title claims a fact about
the device and its body already scoped itself to "on this machine, on this
driver, today", which is the sentence that turned out to be doing the work.

**What it cost.** Nothing but the runs, and it bought the backward task a real
barrier: the forward kernel's write-after-read discipline is unenforceable by
any test I could write, and this one's is enforced.

## L38 — A survivor that really was a hole, and the reason it hid

**What I expected.** The backward kernel's mutants are the forward kernel's
mutants with three more reductions to aim at, so I gave each one the verdict its
sibling got. The power-of-two fold — `for (stride = tpg >> 1; stride > 0;
stride >>= 1)`, the halving that drops entries whenever the group size is not a
power of two — is caught by the forward task, and a real model wrote it
unprompted in one of Sonnet's draws. I expected it caught here too.

**What happened.** It passed all 76 tests.

**Why it hid, and this is the part I would not have guessed.** The broken fold
computes the row maximum over only the lanes its binary tree reaches. At
`tpg = 7` that is lanes 0, 1, 3 and 4; lanes 2, 5 and 6 are dropped. So the shift
is not the row's maximum — it is *a* value from the row.

And the softmax does not care. Shifting a row by any constant leaves the answer
algebraically identical; the maximum is chosen for conditioning, not for
correctness. On a row of ordinary logits, a shift that is merely close enough
keeps every exponential inside float32 and the answer is right to the last bit
the tolerance asks for. The forward task caught this mutant on exactly one test
of its sixty-eight. My suite had no test where a dropped lane held a value the
shift actually had to see.

**Where I nearly went wrong.** L31 is the entry about mistaking a correct
program for a hole in my tests, and its lesson — the reflex to add more
parametrisations is usually the wrong one — is strong enough that I spent a
while constructing the argument for `SURVIVES`. The argument was even true:
every input the suite had really is insensitive to which lane is dropped. It was
also the wrong question. L31's survivors are programs no test *can* separate;
this one is separated by any row whose maximum is large enough to matter and
sits in a lane the tree misses.

**What changed.** One test, `test_the_largest_logit_in_every_position`: a seven
by seven matrix whose maximum walks the diagonal, at five group sizes that do
not divide it. Three of the five catch the mutant — 3, 5 and 7, where the
dropped lanes own real columns — and 9 and 33 do not, because there the dropped
lanes hold no column at all and carry the identity. It is licensed by the
sentence the prompt already had about logits reaching several hundred, and the
mutant now reads `3 failed, 78 passed`.

**What it cost.** Nothing, and it is the cheapest lesson of the session, because
the mutation script asked the question for me. The version of this task I would
have shipped without it has a reduction whose correctness no test enforces at
the one group size a model is most likely to get wrong — and `metal_cross_entropy_kernel`
is on the leaderboard today partly because a real model wrote that exact fold.

## L37 — "A fact about a tool" was never the criterion. "A fact with no tutorial" is.

**What I expected.** L30 told me the laptop tier's reasoning tasks were
saturated and that the one task separating frontier models was hard because of
an obscure fact about a tool. So I wrote a laptop task whose entire difficulty
is a tool contract: an `autograd.Function` whose backward has to be
differentiable in turn. No derivation worth the name — the derivative of
`w * x * sigmoid(alpha * x)` is undergraduate calculus — and four separate
places to get the framework's contract wrong. I expected it to separate
`claude-sonnet-5` from `claude-opus-5` the way the Metal task does.

**What happened.** Ten draws each, $1.37:

```
claude-opus-5     10/10
claude-sonnet-5   10/10
claude-haiku-4-5   0/10
```

Refused from `v2` by the admission rule, and correctly.

**Why, and this is the part I could have worked out in advance.** The fact I
built on is *documented*. PyTorch has a page about double backward in custom
Functions; `save_for_backward`, the version counter and `once_differentiable`
are in the reference docs with the failure mode spelled out. A frontier model
has read all of it.

The Metal task's fact is not like that. `half` is a type name in Metal Shading
Language, so it cannot be a variable — there is no tutorial about that, because
it is not a topic. It is the kind of thing you learn by having the compiler
reject your code. Two failures out of ten came from it, in a model that got the
reduction right.

So §2.0's clause is too generous and I wrote a task against the generous
reading. The sharper version, and the one that is actually checkable before
paying for a calibration: **does the framework have a page about exactly this
mistake?** If yes, it is knowledge, and every model at the top has it. If the
only way to meet the fact is to write the code and watch it fail, it is a
hazard, and hazards are what separate models that are all equally well read.

`activation_checkpointing_rng` lost for precisely this reason in session 10 and
L30 says so in its own last paragraph. I had the lesson, quoted it while
planning this task, and still could not apply it, because "obscure fact about a
tool" is a phrase you can talk yourself into. The question above is not.

**Where I nearly went wrong.** Haiku failing all ten draws looked, for a minute,
like discrimination. It is discrimination one rung down, which is the definition
of `warmup` and exactly what L35's rejected alternative would have readmitted.
The rule refused the task without my having to argue myself out of anything,
which is what an admission rule is for and is the strongest evidence yet that
widening it in L35 did not soften it.

**What changed.** §2.0 gains the question. The task goes to `warmup` with its
block and its draws checked in — and it is the first of the two attempts §4b
names as the condition for reopening the headline decision, so the next laptop
task carries more weight than this one did.

**What it cost.** $1.37, a session, and the uncomfortable observation that
having the lesson written down in my own words two sessions earlier was not
enough to stop me repeating it. What made the difference this time was not
insight, it was that the rule is machinery: it refused the task on numbers while
I was still admiring the failure shape.

## L36 — A mechanism I built the task around, the framework does for free

**What I expected.** The new laptop task hands its Function two tensors that
broadcast against each other, so a gradient computed at the broadcast shape has
to be summed back down to the shape of the input it belongs to. I wrote
`_reduce_to` for it, wrote the tests over seven shape pairs broadcasting in both
directions, and wrote two mutants that drop the reduction on each side. I
expected both caught. Reducing a broadcast gradient is a thing people get wrong
in custom Functions, and it was one of the three mechanisms I thought the task
measured.

**What happened.** Both mutants passed all 116 tests.

```
SURVIVED  weight gradient not reduced   116 passed in 0.93s
SURVIVED  input gradient not reduced    116 passed in 0.93s
```

**Where I nearly went wrong.** My first reading was the same one L31 warns
about: the tests have a hole in the broadcast coverage, add more shape pairs.
The reflex is strong because it is *usually* right. What stopped it was that the
mutation script prints the expected verdict beside the observed one, so a
survivor arrives as a question, and the question has a cheap answer — twenty
lines outside the repository, a Function that deliberately returns unreduced
gradients:

```
torch 2.8.0
x.grad shape (1, 5) expected (1, 5)
warnings: []
matches autograd: True True
scalar-for-vector rejected: Function BadBackward returned an invalid gradient
at index 0 - got [] but expected shape compatible with [3]
```

The engine sums it down itself. Silently, with no warning, bit-identically to
doing it by hand. It rejects a shape that is not broadcast-compatible with the
input and absorbs everything else. There is no test I can write that separates
the two implementations, because there is no observable difference.

**What changed.** Both mutants are checked in with `SURVIVES` as their expected
verdict, which meant teaching `tools/mutate_v2_tasks.py` the two-way
expectation that `mutate_metal_task.py` already had. The reference keeps
`_reduce_to` — explicit is worth something, and it does not depend on the
version of a library — but its docstring now says the reduction is explicit
rather than necessary, with the measurement beside it.

The part that cost nothing is the part worth noticing: **`prompt.md` had not
been written yet**, so the promise I would have made about reducing gradients
never entered it. That is the writing order (L28) paying for itself a second
time, and by luck rather than by planning, because I ran the mutants before the
prompt only out of habit.

**What it cost.** Twenty lines and one run. What it would have cost is the
version where I add four more shape pairs, watch them pass, ship a prompt
promising a property nothing enforces (L8), and then describe the task in a
journal as measuring three mechanisms when it measures two.

## L35 — I changed the admission rule after it refused a task I liked

**What I expected.** The rule I wrote in section 4 of `V2_DESIGN.md` was meant
to stop me admitting tasks on my own estimate of difficulty, and it reads one
number: the highest pass rate in the calibration block. If the strongest model
tried never failed, the task is out. I expected that to be the whole of it,
because the failure it was written against — v1's difficulty numbers being a
measurement of me — has exactly that shape.

**What happened.** It refused `flash_attention_backward`. `claude-opus-5` passes
it 15 times out of 15, and `claude-sonnet-5` loses two draws of ten, both times
to the same pair of tests, matching a mutant's signature exactly. That is the
only task on the laptop tier in this repository that separates one frontier model
from another, and it is precisely the discrimination v2 exists to buy. The rule
threw it into the warm-up set beside two tasks that Opus and Sonnet both clear
completely, which are not the same kind of object at all.

**Where I nearly went wrong, twice.** The first is obvious and it is the reason
this entry exists: *a rule that refuses a result I like is the one moment where
changing the rule is most likely to be motivated reasoning, and it feels exactly
like noticing a flaw.*

The mitigation I reached for first does not hold up, and I am writing it down
because I believed it for an hour. I told myself the tension was already on
record in `docs/sessions/10`, written on 2026-08-09 before there was any second
task to lose by it. It is on record — and reading it again, that journal does not
merely name the tension, it **argues for the strict rule and keeps it**: Sonnet's
failure there is called a convention missed at the edge rather than a mechanism
absent, the task is left in `warmup` deliberately, and the entry sets a revisit
condition — *when there is a model between Sonnet and Opus to lose*. That
condition has not been met. No new model has appeared. So the prior writing is
not permission; it is a decision I am now overturning without the trigger I gave
myself, and this entry has to say so in those words or it is a changelog wearing
a lesson's clothes.

What actually justifies overturning it is one line of that old argument, and it
is the line I wrote myself: *convention noise is not the axis a headline set
should be scaling.* The admission rule cannot see the difference between
convention noise and an absent mechanism. I can. So "Sonnet only missed a
convention, keep the task out" is my judgement of a task's difficulty deciding
its admission — which is the exact channel L21 closed and the entire reason the
calibration block exists. The old rule's stated intent is that the headline set
ranks the top of the field; with three models measured as a matter of course,
reading one entry does not implement that intent, it implements "the strongest
model is the field". The revision is narrower than my first reading of it: not
"the rule was too strict" but "the rule read one number where its own sentence
says field".

The second near-miss is the version of the fix that sounds better and is worse.
My first draft of the new rule was "admit a task if any two models differ on
it". That reads well, and it readmits nearly all of v1: `grad_accumulation`
separates `claude-haiku-4-5` (0 of 6) from `claude-opus-5` (6 of 6) and
`claude-sonnet-5` (5 of 5), which is the definition of a warm-up task. A rule
that admits everything I already had is L21 again with extra steps, and it took
running the numbers on the existing blocks to see that the appealing rule was
the one that undid the whole exercise.

**What changed.** The rule now reads the **top two** measured entries and
refuses a task only when both are at 100%, which is the smallest change that
distinguishes "the frontier cleared this" from "the strongest model cleared
this". Entries with fewer than five draws are read on neither side — a
measurement too small to admit a task is too small to refuse one — and a
numbered set now needs two entries of five draws or more rather than one.
Applied to the twenty-one calibration entries that existed the day it changed,
it moves exactly one task, `flash_attention_backward` from `warmup` to `v2`,
which is the strongest evidence I have that it is a correction and not a
loophole: a rule rewritten to get a result usually gets several.

**What it cost.** No draws and no dollars — every number it turns on was already
published. What it cost is that this entry has to exist; that a "revisit if"
condition I set for myself turned out to be worth less than the argument
underneath it, which means the next one needs to name the argument rather than a
future model; and that "the rule is too narrow" is now a sentence I have used
once and have to earn again.

## L34 — The compile error I fixed in thirty seconds was the task

**What I expected.** Writing the reference for the Metal task, my first draft
named a loop variable `half`, because the fold halves a range. The Metal
compiler rejected it: `half` is a type in Metal Shading Language, the 16-bit
float. I renamed it to `stride`, moved on, and did not think about it again.

**What happened.** Ten draws of `claude-opus-5` later, two failures, both
`solution_error`, both this:

```
E   SyntaxError: program_source:39:14: error: cannot combine with previous
    'type-name' declaration specifier
E           uint half = (n + 1u) / 2u;
E                ^
```

Draw 4 and draw 9, the same identifier, and the reduction logic in both is
correct — the ceil-half fold with the live range carried properly, which is the
part I thought the task was about. A frontier model made the exact mistake I had
made, in the same place, for the same reason, and it is the only thing standing
between it and 10 out of 10.

**Where I nearly went wrong.** My own compile error read like noise: a typo, a
minute lost, nothing to record. It was the clearest possible instance of the
property L30 says is the only one that survives — *the difficulty is a fact
about a tool, and an obscure one* — and I had it in my hands an hour before I
had a task. If I had noticed, I would have gone looking for the other reserved
words rather than stumbling into the one that mattered.

**What changed.** Nothing in the code, which is the point: the task was already
shaped to expose this. What changes is where I look for the next task's
difficulty. The candidate list in `V2_DESIGN.md` §3 is still five reasoning
tasks; the thing that actually separated Opus from itself here was a
language-level fact that no amount of reasoning about cross-entropy reaches.

**What it cost.** Thirty seconds at the time and the entry is free, but the
cheapest evidence in this repository was sitting in my terminal unread, and it
took a $0.94 sweep to tell me what my own compiler already had.

## L33 — A kernel that uses one lane of the group passes every test I can write

**What I expected.** The task hands the model a threadgroup per row and a group
size chosen by the caller. I wrote the tests to enforce correctness under every
geometry — group narrower than the row, wider than the row, not a power of two,
wider than a simdgroup — and took it for granted that a solution which ignores
the geometry would fail one of them.

**What happened.** As a mutant, I wrote the degenerate kernel: `if (tid != 0)
return;` and a serial loop over the row, no threadgroup memory, no barriers, no
reduction at all. It passed all sixty-eight tests in 1.57 seconds, against the
reference's 1.77.

**Why it survives, and why it stays.** There is no black-box test for
parallelism. The only assertion that separates the two kernels is a wall-clock
one, and this repository forbids those everywhere — timing on unknown hardware
is not reproducible, and reproducibility is the founding claim. Not on the
laptop tier, not on the accelerated tier, not for this.

So the honest statement of what the task measures is narrower than the one I
would have written before running that mutant: **it measures whether the model
can write a correct Metal kernel under a launch geometry it does not control.**
It does not measure whether the kernel is fast, and the `prompt.md` does not
claim it does — a promise no test can keep is exactly L8, and the prompt was
written before the mutant was run, which is luck rather than discipline.

**What changed.** The mutant is checked in with an expected verdict of
`SURVIVES`, so the hole is a re-runnable fact rather than a paragraph. And ten
Opus draws are checked in beside it: all ten used a real threadgroup reduction
with a static scratch array, none took the shortcut. The hole is theoretical so
far, and I would rather publish it than discover it in a pull request.

**What it cost.** Nothing yet. It costs the day a model works out that the
cheapest way to pass a kernel benchmark is to not write a kernel.

## L32 — The race this hardware refuses to show me

**What I expected.** The reference reads the row maximum out of `scratch[0]`,
then every thread overwrites the scratch with its partial sum. Between those two
there is a barrier, and dropping it is a textbook write-after-read race: thread 0
can reach its write before another thread has read. I wrote it as a mutant
expecting it to be caught.

**What happened.** It passed. Sixty-eight tests, then thirty more configurations
hunting for it by hand: 1024 rows against 2 columns so most lanes read
immediately, group sizes of 96, 512 and 1024 so the read spans many simdgroups,
4096-column rows so the write is far away in time, ten repeats each. The mutant
and the reference agreed to the last bit in every one.

**Why it survives.** Apple's scheduler runs the threadgroup's simdgroups closely
enough in lockstep that the window never opens — on this machine, on this
driver, today. The race is real: the Metal specification does not order those
two accesses, and a future device that schedules simdgroups further apart is
free to break it. What I cannot do is write a test that fails.

**What changed.** The mutant is checked in with `SURVIVES` as its expected
verdict, and the reason is in the script rather than in my head. If a future
machine catches it, `tools/mutate_metal_task.py` fails — the expectation is
checked in both directions on purpose — and that failure is news worth having.

**What it cost.** An hour of hunting, and a sharper version of a thing I already
half knew: on a GPU, "the tests pass" and "the program is correct" are further
apart than they are anywhere else in this repository. Undefined behaviour that
happens to work is the normal state of a kernel, and a benchmark that grades
kernels by output alone is measuring something narrower than correctness.

## L31 — I called a correct program a hole in my tests

**What I expected.** The fold in the reference walks a live range: at each step
the stride is half the *remaining* count rounded up, and a thread folds only if
`tid + stride < active`. Bounding it by the threadgroup size instead —
`tid + stride < tpg` — lets threads keep folding entries that were already
folded, and lets a writer and a reader touch the same slot inside one barrier
interval. I wrote it as a mutant for both reductions and expected both caught.

**What happened.** The sum side was caught, nineteen tests. The max side passed
everything, including a row of ±360 logits at four awkward group sizes, which is
the test I added specifically to catch it.

**Why.** Because on the max side it is not a mutant. `max` is idempotent and
monotone: an extra merge can only move a value up toward the true maximum, never
away from it, and a stale read returns a value that is still a maximum of some
subset. Every element still reaches slot zero through the same chain the correct
algorithm uses; the extra work is extra, not wrong. The mutated kernel computes
the right answer, always, and a test that failed it would be a broken test.

The sum side has none of that. Addition is not idempotent, an extra merge
double-counts, and the tests see it immediately.

**Where I nearly went wrong.** My first reading of the survivor was "the tests
have a hole in the max reduction", and my hand was already on a new test. Two
things stopped it. The mutation script prints the *expected* verdict beside the
observed one, so a survivor is a question rather than a verdict. And the
question — what would that test assert? — has no answer, because the mutant's
output is correct.

**What changed.** The mutant stays, with `SURVIVES` as its expectation and the
algebra as its description. A mutation pass is a measurement, and a measurement
that disagrees with you is sometimes telling you the reference has a property
you had not noticed rather than that the tests are thin.

**What it cost.** One test I did not need to write, and one I did: hunting the
max-side survivor produced `test_a_wide_row_of_large_logits`, which catches four
other things, so the wrong hypothesis paid for itself.

## L30 — I copied the wrong property off the one task that discriminates

**What I expected.** `fused_rmsnorm_kernel` puts Opus 5 at 1 out of 5 while the
entire laptop tier puts it at 40 out of 40, so I read that task closely and wrote
down what made it hard. `V2_DESIGN.md` §2 is the list: the obvious correct answer
is forbidden, mechanisms compose so errors interact, a backward pass is derived
rather than recalled, and the correctness property is provable rather than
plausible. Three tasks were then built to satisfy it.

**What happened.** Forty draws, and Opus 5 did not fail once.

```
speculative_decoding_verify   15/15
flash_attention_backward      15/15
activation_checkpointing_rng  10/10
```

**Why the list was right and beside the point.** Every property on it is real and
every one is present in the kernel task. But the thing actually doing the work
there is not on the list, because I described the task in the vocabulary I had
rather than in the vocabulary of why it is hard: **`fused_rmsnorm_kernel` is not
hard as an algorithm.** RMSNorm is four lines. It is hard because Triton
specialises an integer kernel argument whose value is `1` into a compile-time
constant, so `n_cols.to(tl.float32)` compiles at every row width except one. No
amount of reasoning about root-mean-square normalisation reaches that. You find
it by running the thing.

All three of my tasks are reasoning, and hard reasoning: the accept-and-correct
step has to be derived from the guarantee it provides, the row correction has to
be recognised as not blockwise, the recomputation has to put the generator back
where the forward left it. A model that can carry the derivation gets all of it,
and Opus can carry the derivation.

**The third task is the part that stings**, because it was written *after* I had
noticed this, against the corrected reading. Its difficulty was supposed to be a
fact about a tool rather than a step in an argument: PyTorch's global generator
is state, and recomputation has to bracket it. Opus went 10 for 10. The
correction was not enough, and the distinction that survives is narrower and less
comfortable than "a fact about a tool" — it is a fact about a tool that is
*obscure*. Restoring an RNG state around a checkpointed block is written down in
every framework that implements checkpointing. Triton's specialisation of `1` is
a footnote.

Which means the laptop tier's remaining difficulty is not depth. It is
unfamiliarity, and unfamiliarity is a moving target: it is contamination wearing
a different coat, and designing against it is designing against next year's
training set.

**What changed.** `V2_DESIGN.md` §2 gains the property it was missing and says
plainly that satisfying the other four is not sufficient, on the evidence of
three tasks that satisfy all four. The three go to `frozen_set: warmup` by the
rule written in the same session, which is at least the machinery working. And
the honest read of the accelerated tier changes from "the tier that also
discriminates" to "the only tier where the tooling is unfamiliar enough to."

**What it cost.** $7.43 and a session, and the entry is here because none of the
three tasks is bad. They separate Opus from Haiku by a mile, they catch every
one of the twenty-eight wrong implementations written against them, and their
failure shapes say more than their verdicts do. A criterion can be correct,
satisfied in full, and measuring the wrong axis.

## L29 — A sweep that died halfway went into the error bar as a draw

**What I expected.** The absence-of-evidence rule has been fixed three times
now: L11 kept an unmeasured task out of the rates, L20 put the classification in
one place so a second list could not disagree with it, and L25 stopped two runs
that asked different questions being averaged. I considered the class closed.
Again.

**What happened.** The calibration sweep's API key ran out of credit partway
through the second of five draws. Every task in draws three, four and five came
back `adapter_error`, and `report --variance` said:

```
claude-opus-5  ·  task set unvalidated  ·  5 draw(s)  ·  laptop tier  ·  2 task(s) asked

task                         passed  spread
speculative_decoding_verify  2/2     always
flash_attention_backward     1/1     always

set pass rate: 100% in every draw
```

Five draws, a hundred percent every time. What actually happened is that one
draw measured both tasks, one measured one of them, and three measured nothing
at all.

**Why it survived.** Because every number in it is correct. `measured[slug]`
already excludes outcomes with no evidence, which is exactly why the per-task
column reads `2/2` and `1/1` instead of `2/5` and `1/5` — that part is the L11
fix working. The set rate is read out of each draw's own file, and those are
right too: draw two asked for two tasks, measured one, solved it, and recorded
100%. Three correct components, and the sentence they assemble into is false.
The phrase "in every draw" was written at a time when a draw could only be
whole, and nothing re-examined it when partial draws became possible.

L20 wrote this failure down in advance, in as many words: *"had it been present
and the sweep merely rate-limited halfway, the file would have carried a
plausible-looking rate computed over a denominator that included the tasks that
never ran."* I fixed the denominator and left the sentence.

**What changed.** The set rate is computed over draws that measured every task
the tier was asked for; the header says how many draws were incomplete; the
summary reads "in every complete draw" whenever any draw was not. Two tests, one
of them the exact three-draw shape above.

**What it cost.** Nothing, and it was free for the wrong reason. The outage was
total and loud — three draws of solid red `ADAPTER` — which is why I read the
block underneath the table at all. A rate limit that emptied one draw out of
five prints the same false summary under a table that looks entirely normal.

## L28 — The design document promised a test that cannot exist

**What I expected.** `docs/V2_DESIGN.md` was written before any of the tasks it
describes, which is the point of it: the tasks get written against a stated
criterion instead of against my sense of hard (L21). For
`flash_attention_backward` it stated how the hard constraint would be enforced —
"the function takes a block size, and the tests require identical results across
several block sizes, which a version that quietly builds the full N×N matrix and
slices it cannot fake in a way that also gets `D` right."

**What happened.** I sat down to write the tests and the sentence is simply
false. Identical results across block sizes is a property of *every* correct
implementation. A version that materialises the whole score matrix, computes the
row correction correctly over the whole row, and slices it per block returns
bit-identical answers at every block size — that is what being correct means.
The test I had promised would catch materialisation catches nothing of the kind.
It catches wrong answers, which is a different job it does well.

**Where I nearly went wrong.** The false sentence sits in the same register as
the true ones around it, and it is not careless: it is a plausible argument
about a task that did not exist yet, and there was nothing to check it against.
The next step after writing the tests is writing the prompt, and the prompt would
have said non-materialisation was required, in a task where nothing required it.
That is L8 — a rule stated and unenforced — except it was decided a week before
the prompt existed, in a document that reads like a specification.

**What changed.** The API rather than the tests. `key_block_gradients` is handed
exactly one block of keys and never the rest of them, so inside the function that
carries the hardest part of the task, materialisation is not something to be
caught: it is not available. The top-level driver can still cheat and the prompt
no longer claims otherwise. `V2_DESIGN.md` says what is enforced and by what
mechanism.

**What it cost.** Nothing, because the reference gets written first. L1 says
that order surfaces the conventions I was about to leave unstated; this is the
same effect one level up, where it surfaced a design claim I was about to leave
unchecked. The order is worth more than the reason it was introduced for.

## L27 — I read a rate and nearly published the opposite of what happened

**What I expected.** The accelerated tier had never been asked of a model. I
expected it to behave like the laptop tier: Opus comfortably ahead, Haiku
behind, and the interesting question being whether a kernel task is harder in
some measurable way.

**What happened.** Two single draws, in this order:

```
claude-opus-5      fused_rmsnorm_kernel ... FAIL  (1 failed, 23 passed)
claude-haiku-4-5   fused_rmsnorm_kernel ... pass  (24 passed)
```

The sentence that assembles itself out of that is "Haiku beats Opus at CUDA
kernels", and it is the kind of sentence that gets a benchmark repository
attention. It is also two coin flips.

Five draws each: Opus 1 out of 5, Haiku 3 out of 5. Which does not rescue the
sentence, because five draws of one task cannot distinguish 20% from 60%
either. What the draws actually contain is in a column the leaderboard was not
printing:

| | draws that failed | tests failed, out of 24 |
|---|---:|---|
| `claude-opus-5` | 4 of 5 | **1** every time |
| `claude-haiku-4-5` | 2 of 5 | 24, and 23 |

Opus writes a kernel that works and misses one edge. Haiku, when it fails,
produces something that does not run at all. Binary scoring collapses both into
`FAIL`, the rate collapses them further into a percentage, and the percentage
puts them in the wrong order.

**Where I nearly went wrong, twice.** The first was reporting the two single
draws at all. The second was subtler and I caught it while writing the page: I
had typed that Opus "consistently" fails the same test, on the evidence of four
identical `1 failed, 23 passed` count lines. Identical counts are not an
identical test. Four more draws, each keeping its workdir and re-running pytest
against it by name, said `test_a_single_column` four times out of four. Now it
is a measurement. It cost twenty cents to stop being a guess.

**What the failure turned out to be**, and it is the reason the task earns its
place: Triton specializes an integer kernel argument whose value is `1` into a
compile-time constant, so `n_cols.to(tl.float32)` compiles at every row width
except `N = 1`, where `n_cols` is a Python `int` with no `.to`. The prompt
licenses the case explicitly — "M may be zero and N is at least one" — so the
test is allowed to ask, which I checked before writing a word of the result.

**What changed.** The leaderboard reports the failure shape beside the rate for
this tier, and says in the same breath that the two rates are not
distinguishable. The finding is not the ordering. The finding is that one task
on the tier that was nearly cut puts a frontier model at 1 out of 5, while the
entire laptop tier puts it at 40 out of 40.

**What it cost.** $0.32 and an hour, and the entry is here because the wrong
version of it was more interesting than the right one. That is the condition
under which a benchmark publishes something false: not carelessness, but a
result that is more fun to report than the truth underneath it.

## L26 — I put the error bar on the column that did not need one

**What I expected.** L19 watched a task flip between two runs of the same model
and concluded that a published rate is one draw of a sampled process. So I built
`--repeat`, `report --variance` and a `k/N` column, and wrote the leaderboard
page around the sentence "no row here has an error bar yet". Everything I built
was aimed at the pass rate.

**What happened.** Five draws of Opus 5 over the laptop tier:

```
draw  rate    cost    wall      in     out
   1  1.00  0.5560   215.4   15858   19070
   4  1.00  0.4912   186.5   15858   16476
```

The rate has no spread at all — forty tasks asked, forty passed. The *cost* has
a 14% spread, and the output token count a 16% one, from the same five files.

**Why that is the wrong way round from what I designed for.** The input is
byte-identical every draw, because the prompt is fixed and nothing is cached. So
every bit of the model's variance lands in the output, and cost is a linear
function of output. The score is not: it is a threshold over 308 tests, and a
model that clears the threshold clears it regardless of how it phrased the
answer. I had assumed one source of randomness meant one uncertain number, and
it does not — the sampling shows up undiminished in the column I was about to
print as a single figure and vanishes in the column I built machinery to
qualify.

**Where I nearly went wrong.** The page would have read "100% ± something" and
"$0.54", and both halves would have been backwards. The second is the one that
matters: a cost is the figure a reader is most likely to reuse, quoting it to
the cent from one draw is a 7% claim dressed as a measurement, and nothing in
the file would have said so.

**What changed.** The rate reports as "100%, in all five" and the cost as a
range, `$0.49 to $0.56`. Both say how many draws they came from.

**What it cost.** Nothing beyond the $2.64 the sweep was going to cost anyway,
and the entry is here because the near-miss is not a bug. Every mechanism worked
exactly as designed. I had simply decided in advance which number was the
uncertain one, and then built a tool that could only ever confirm it.

## L25 — The error bar I built to fix L19 had the L19 shape in it

**What I expected.** `--repeat N` and `report --variance` are the answer to a
single sweep being one draw. I wrote them, tested them against the reference
solver, and moved on. The next step was a five-draw sweep of Opus 5, which costs
$2.70 and would have been the first real use.

**What happened.** Running `report --variance` over the `results/` directory
that already exists, before spending anything:

```
claude-haiku-4-5  ·  task set v1  ·  7 draw(s)  ·  laptop tier
softmax_stability          1/3     SPLIT
attention_causal_mask      1/1     always
...
set pass rate: 0% to 100%  ·  mean 20%
```

Seven draws, and they are not seven of anything. Six are one-task probe runs
from an afternoon of debugging the adapter, and one is the eight-task sweep that
is on the leaderboard. "0% to 100%" is the spread between *a run that asked one
task* and *a run that asked eight*. It measures what I typed, not what the model
did.

**Why it survived.** `group_draws` keyed on model and task set, which is the
obvious pair and is where I stopped. Both are properties of the *question*, and
I had assumed the question was fully described by them. It is not: a run over
one task and a run over eight can carry the same model and the same
`task_set: v1`, because `task_set` names the frozen set a task belongs to and
not the subset that was asked. My own tests passed because every draw I
constructed for them covered the same tasks, which is the assumption under test.

**What it would have cost.** The first thing the leaderboard was going to get is
an error bar. This one would have been wider than the real one, in a repository
whose entire argument is that a published number should be checkable, and the
$2.70 sweep would have gone straight into it.

**What changed.** The key includes the exact set of slugs the run covered, and
the header prints how many tasks were asked, so two groups that look alike are
visibly different experiments. The failing case is in the suite.

**What it cost.** Twenty minutes, and it is the fourth entry that is the same
sentence: two things that are not the same measurement were being averaged. L11
rounded an absence of evidence to zero, L20 did it again in a second place, L23
averaged two tiers, and this one averages two experiments. The tool changes
every time and the shape does not.

## L24 — The cleanup script would have deleted `PATH` in the middle of the job

**What I expected.** Fifteen red crosses on `main`, all correct, all from the
same missing file, and an owner who does not want them in the history of a
public portfolio repository. The fix is a reset to the last commit CI never ran
on and a replay of forty-five commits. Forty-five lines of `git add && git
commit` is a lot to paste, so I wrote a loop over a `path|message` list.

**What happened.** I ran it against a throwaway repository first, out of habit
rather than suspicion. Every command after the first iteration:

```
(eval):17: command not found: git
(eval):18: command not found: git
```

**Why.** In zsh, `$path` is the array form of `$PATH`; they are tied. `read -r
path message` assigns to it, and the first iteration empties `PATH`. In bash the
loop is fine, which is exactly what makes it dangerous: it is a shell-specific
trap in a script written for the one shell that has it.

**What it would have cost.** The loop had already made its first commit when
`PATH` went. The rest silently do nothing, `set -e` never fires because the
shell cannot find `git` to fail, and the visible result is a branch that has
been reset fifteen commits back with one commit replayed and forty-four files
loose in the working tree. Recoverable — nothing is ever lost by
`reset --mixed` — but recoverable while looking exactly like a destroyed
repository, on the one operation where that is the fear.

**What changed.** The variable is `file`. The reason is written next to it in
`COMMITS.md`, because the next person to shorten that loop will reach for
`path` again.

**The second half, which only appeared when the plan changed.** The replay was
going to be one push at the end, so I ordered the queue for readability:
`ci.yml` at position 29, the leaderboard files at 32 and 33. Then the owner
asked to split it across two days and push per commit, and that ordering became
a defect. `tools/check_cost.py` exits non-zero when `leaderboard/` is empty, so
CI would have gone red on arrival at commit 29 for a third distinct reason, in a
queue written specifically to stop it going red.

What makes per-commit pushing safe is not the loop, it is where the workflow
file sits: **GitHub runs the workflow that exists in the pushed tree, so every
commit before `ci.yml` starts no run at all.** Moving it to position 34, after
everything its four steps read, turns thirty-three pushes into silence and the
remaining twelve into green ticks. I checked that by assembling the tree as it
will exist at commit 34 and running all four commands against it, rather than by
reasoning about it, which is how the leaderboard dependency turned up.

The general form: an ordering constraint that is invisible under one push
cadence is load-bearing under another. The queue was correct and the cadence was
a parameter I had not written down.

**The other half of this entry** is that I rewrote history after L17 says not
to. That still stands and it is not in tension: L17 is about reaching for the
biggest tool *before reading a log line*. The log line here says
`ModuleNotFoundError: No module named 'adapters.anthropic_api'`, the cause is
one file that was never committed, and the fix for the redness is in the queue
either way. The rewrite is a separate, cosmetic decision about a public
repository, made with the diagnosis already in hand and by the person whose
repository it is. Reaching for a tool is not the same act as reaching for it
blind.

## L23 — The rule lived in seven documents and none of them was the code

**What I expected.** "The leaderboard rate is the laptop tier only" is the most
repeated sentence in this repository. It is in the README, in `TASK_FORMAT.md`,
in `CONTRIBUTING.md`, in `CLAUDE.md`, in the leaderboard page, twice in the
docstrings, and it is the reason the tier split exists at all (L10). I treated
it as settled.

**What happened.** Auditing before the first accelerated model run, I built a
payload by hand with both tiers measured — which is what a run on the rented
GPU box would produce — and asked for the headline number:

```
headline pass_rate : 0.888…  measured: 9
by tier            : {'accelerated': 0.0, 'laptop': 1.0}
```

Eight laptop tasks passed, one accelerated task failed, and the field the
leaderboard column prints was their average. `pass_rate_by_category` was
blended too: `kernels` sitting in the same dictionary as `numerics`.

**Why it survived seven documents and a test suite.** Every machine that has
ever run this harness lacked a GPU, so an accelerated task always came back
`needs_accelerator`, dropped out on the no-evidence rule, and never reached the
arithmetic. The bug needed *both* tiers measured, which is a state that has
never existed here. The test that looks closest — a laptop task and a GPU task
in one payload — asserts the tier breakdown and the null rate, and passes for
the reason the bug is invisible.

That is the same sentence as L11: infrastructure written before its first user
has tests that share the user's blind spot. The difference is that this time the
first user was two commands away. The next thing on the list was to ask a model
the accelerated task, on a box with a GPU, and the first `--tier all` sweep
there would have written a blended number into a results file.

**What changed.** `HEADLINE_TIER` is a constant in `report.py`, the headline
rate and the category rates are computed over that tier alone, the payload
carries `headline_tier` so a reader never has to work it out, and the
leaderboard row reads the tier's own bucket instead of counting every task in
the file. Version 1 files have no `headline_tier` and every one of them was
laptop-only, so they still read as what they were.

**What it cost.** An hour, and the entry is really about where I looked. I went
looking for this by asking what the *next* session would run, not by reading
what the current one produces. Writing the sentence down seven times is what
made it feel checked.

## L22 — A model that writes unparseable Python was getting a smaller denominator

**What I expected.** `collection_error` means pytest never judged the solution,
so nothing was measured, so it stays out of every rate. That is the rule this
repository has been fixing since L11 and I applied it without re-asking who
caused the collection error. There is a test named
`test_a_solution_that_does_not_import_is_broken_not_merely_wrong`, so it was a
decision, not an oversight, which is worse.

**What happened.** A probe: hand the grader a solver that writes
`def softmax(x)` with the colon missing, and grade it beside one passing task.

```
status   : collection_error
evidence?: False
=> published pass_rate: 1.0 over measured: 1 of 2 tasks
```

A model emitted a syntax error and the results file reported it at 100%.

**Why it is the wrong side of the line.** The task contract already guarantees
the other half: `validate` requires the untouched starter to import cleanly and
fail every hidden test with real assertion errors. So the baseline collects, by
construction, and a collection error in a graded directory was introduced by
whatever wrote the file. That is not an absence of evidence about the model, it
is some of the strongest evidence there is. And the bias has a direction:
unparseable output comes from weak models, so the mistake systematically
flatters exactly the models it should be catching.

The commoner version is worse than the syntax error, and it is what killed my
first fix. I wrote an import check on the solution modules — which catches
`def answer(:` and completely misses a model that renames the function the
tests import. The file imports perfectly. Nothing is there.

**What changed.** The question is asked about the baseline instead of about the
solution, because that is the one thing the task contract already pins down:
when a graded run produces a collection error, the hidden tests are re-run
against the untouched starter. If they collect and fail properly there, the
task is intact and the solution broke it — a new status, `solution_error`, which
is evidence and is a failure. If the starter cannot collect either, the task is
broken, and that is the old `collection_error`, still an absence of evidence and
still fatal to the run. The extra pytest run only happens in the rare broken
case.

**What it cost.** Forty minutes, most of it on the wrong fix, and the tell was
that my first version tested the thing I suspected rather than the thing the
contract guaranteed.

## L21 — I calibrated difficulty against my own sense of hard

**What I expected.** Nine tasks, difficulties one through five, written over
several sessions with the hardest ones deliberately last. `online_softmax_attention`
is difficulty five: tiled attention with a running maximum and a rescale, fifty-seven
hidden tests, and the one I was least sure any model would get.

**What happened.** The first full sweep: Claude Opus 5 passed all eight laptop
tasks on the first attempt, including that one, in three and a half minutes for
fifty-four cents. Nothing about the set separates it from anything above it.

**Why that is a finding and not just a good day.** A benchmark that everything
at the top passes has stopped measuring at the top. I had assigned every
difficulty number by how hard the task felt to write, which is a measurement of
me, and never once by how hard it turned out to be for a model — because until
this session no model had been asked. The number was in `meta.yaml` from the day
each task was authored, presented as a property of the task.

**What the failures said that the rate did not.** The contrast row is the useful
one. Haiku 4.5 lost `kv_cache_equivalence` on nineteen of twenty-one tests and
`grad_accumulation` on seventeen of twenty-seven: a mechanism absent. It lost
`softmax_stability` and `bpe_merge_order` on exactly one test each: a convention
guessed differently. Those are different kinds of difficulty and only the first
is worth scaling up. Adding conventions to guess makes a task longer and its
score noisier; adding mechanism to build makes it harder.

**What changed.** The leaderboard page says v1 does not discriminate at the top,
in the same breath as the 100%, rather than letting the number stand on its own.
And the next set gets designed against that distinction instead of against my
intuition.

**What it cost.** Nothing yet, and the near-miss is the whole entry: publishing
`claude-opus-5 — 100%` as a headline, with difficulty numbers I had assigned by
feel, would have been a benchmark quietly claiming a discrimination it does not
have.

## L20 — The same bug, in a second place, because the idea lived in two places

**What I expected.** L11 was the lesson about rounding an absence of evidence
down to zero, and I fixed it: a task with no hardware to run on stays out of
every rate. Three tests cover it. I considered the class closed.

**What happened.** The first attempt at a full sweep ran with no API key in the
environment. Every one of the eight tasks came back `adapter_error`, which is
the adapter saying it never got an answer at all, and the results file said:

```json
"pass_rate": 0.0,
"measured": 8,
"not_measured": 0,
"pass_rate_by_category": {"attention": 0.0, "data": 0.0, "numerics": 0.0, ...}
```

That file says Claude Opus 5 scored zero on eight machine learning tasks. It
never saw one of them. This is the single failure this repository cannot
survive, produced by a missing environment variable.

**Why it survived.** The concept was written down twice, in two files, with two
different memberships. `report.py` had `HARNESS_FAILURES = {collection_error,
adapter_error}` with the comment "the number it produced is not a measurement of
anything" — and used it only for the exit code. The rate arithmetic, four
functions above it, tested `status != "needs_accelerator"` instead. Both are
correct sentences about the same idea and neither knows the other exists. L11
fixed the membership in one of them, which is exactly why it did not fix this:
the fix went where the symptom was rather than where the definition should have
been.

`missing_deps` was leaking the same way, and I only found it while writing the
table.

**What changed, and this time it is the mechanism rather than the value.**
There is one classification now, `STATUSES` in `runner/sandbox.py`, next to the
code that produces the statuses. Each entry declares two things: whether the
solution was actually judged, and whether the run is unusable. `NO_EVIDENCE`
and `HARNESS_FAILURES` are *derived* from it, the printed labels are derived
from it, and `Outcome.__post_init__` refuses to construct an outcome whose
status is not in it. A new status cannot reach the reporting layer
unclassified, which is the only version of this fix that a third occurrence
cannot walk around.

The printed line was lying too, and in a way that would have sent me hunting in
the wrong place: a sweep that died on an API key printed `laptop not run (8
task(s): no accelerator on this machine)`. Two different absences, one message,
written back when only one of them existed.

**What it cost.** An hour, and it was free only because the key happened to be
missing. Had it been present and the sweep merely rate-limited halfway, the file
would have carried a plausible-looking rate computed over a denominator that
included the tasks that never ran, and I would have published it.

## L19 — I ran the same measurement twice and got two different answers

**What I expected.** The cost check needed one more live request, so I ran the
same model against the same task a second time. That was supposed to be
bookkeeping: confirm `usd_cost` matches the tokens, move on.

**What happened.** The arithmetic held exactly — 1182 input at $1 per million
plus 1003 output at $5 is $0.006197, to the last digit. And the task failed. The
first run had passed all twenty-nine tests; the second failed three of them.
Same model, same task, same prompt, same adapter, one attempt each.

**Why.** Everything in this repository is pinned to be deterministic — one
thread, fixed hash seed, no inherited `PYTHONPATH` — and none of that reaches
the part that is not deterministic. The model is sampled. I had spent two
sessions making the *harness* reproducible and quietly assumed that made a
*result* reproducible. It does not, and the pinning is what made the gap
invisible: everything downstream of the model is bit-identical, so a difference
can only have come from the model, and I had no reason to look there.

**What this does not mean.** It is not an argument for retries. A second attempt
*with the test results as feedback* measures the scaffolding, which is why
`max_attempts` is 1 and why that number is written into every results file. Two
independent runs are a different thing: same experiment, sampled twice. The
repository now has a word for each, because I did not have one before.

**What changed.** A single sweep is a sample and the leaderboard says so. The
first published row will name its date, its task-set version, its attempt count
and the fact that it is one draw, not a converged number. Repeated runs are the
honest way to put an error bar on it, and until there are several, no two models
one task apart should be read as distinguishable.

**What it cost.** Six tenths of a cent, and it is the most valuable one this
repository has spent. The near-miss is exact: I was one command away from
publishing a single sample as *the* score for a model, in a repository whose
entire argument is that you should not have to take a number on trust.

## L18 — I published a cost and kept no way to check it

**What I expected.** The adapter computes spend from the published per-token
prices and the usage the API returns, and writes `usd_cost` into the results
file. Session 05 listed the arithmetic as covered: there is an offline test for
it, including the cache multipliers.

**What happened.** The first live run came back `usd_cost: 0.008337`, and the
task for this session was to check that figure against real usage. I could not.
`usd_cost` was the only trace of the usage in the file. The tokens existed
inside the adapter, were used once, and were thrown away.

**Why that matters more here than in an invoice.** A stale price table and a run
that used more tokens than expected produce the same number, and afterwards
there is nothing to tell them apart — not for a reader, and not for me. A cost
on a public leaderboard that nobody can reproduce is exactly the kind of figure
this repository exists to argue against. The offline test proves the function
computes what I told it to; it says nothing about whether what I told it is
still true.

**Where I nearly went wrong.** My first instinct was to check the number by
hand — estimate the prompt at a thousand tokens, work backwards, decide it was
plausible, and write "verified" next to it. That would have been a guess in the
shape of a check.

**What changed.** The four counts the price is computed from — input, output,
cache read, cache write — go into the results file beside the cost. With those
and `PRICES` the figure is reproducible to the cent. The reference solver makes
no calls and reports `tokens: null` rather than four zeros, because zeros there
would read as a measurement.

**What it cost.** Twenty minutes, and it is the fourth entry in this file that
is a version of the same sentence: a number is not evidence unless something
else in the file can contradict it.

## L17 — Every commit was correct and the repository was still red

**What I expected.** One file per commit and a red cross on `main` are the kind
of pair that means somebody force-pushed over somebody else, or squashed a
branch badly. That is what I went looking for, and the first instinct in the
room was to delete the last ten commits and rebuild them.

**What had actually happened.** Nothing was wrong with the history. The commits
were pushed *one at a time*, and GitHub runs the workflow once per push against
the tip of what was pushed. One file per commit guarantees that most tips are
incomplete: a task with a reference and no hidden tests yet, a test module whose
import lands two commits later. Every red cross was correct. `tests/test_runner.py`
went in carrying `import adapters.anthropic_api`, and `adapters/anthropic_api.py`
is still in the queue, so `pytest` exits 2 at collection before `validate` is
ever reached.

**Why it is worth an entry.** Two rules I had written down separately —
"one file per commit" and "CI runs on every commit" — are in direct conflict,
and the conflict is invisible until you look at the *cadence* rather than at any
one commit. I had already written "push once at the end" in `COMMITS.md` as a
tidiness rule. It is not a tidiness rule. It is the only thing that makes the
other two compatible.

**What changed.** `COMMITS.md` now says why, at the top, before the blocks. The
diagnosis cost one `gh run view --log-failed`; the near-miss was the ten minutes
I was about to spend rewriting a history that was fine.

**What it cost.** Nothing, and it would have cost the history. The tell was that
I was reaching for the biggest available tool before reading a single log line.

## L16 — An alias is not a model id, and I checked it as though it were

**What I expected.** The adapter refuses to grade an answer from a model other
than the one asked for; that check is the thing standing between this repository
and publishing one model's score under another's name (L13). `message.model !=
self.model` is what that sentence looks like in code.

**What happened.** The first live request in the project's history came back
`asked claude-haiku-4-5 and was answered by claude-haiku-4-5-20251001`. An alias
resolves to a dated snapshot. The API had done exactly what it promises and my
identity check called it a substitution, so a working request produced
`adapter_error` and measured nothing.

**The second half, which I nearly missed.** The price table is keyed the same
way. Even with the identity check fixed, `price_of("claude-haiku-4-5-20251001")`
misses and the run reports `usd_cost: null` — which does not read as "the lookup
missed", it reads as "this model has no published price". A wrong number would
have been obvious; an absent one looks like a policy.

**What changed.** One rule, applied in both places: an answer is the model that
was asked for if it is that string, or that string plus a trailing date. A
prefix is not enough — `claude-opus-4-5` starts with `claude-opus-4` and is a
different model — so the suffix is matched as a date, not as text. Three tests,
one of them for the prefix case that a looser check would have accepted.

**What it cost.** Twenty minutes, and the reminder that the strictest possible
check is not the safest one. A check that rejects the correct answer does not
fail loudly on the side of caution; it fails as an absence of evidence, which is
the outcome this repository is least able to tell apart from a real one.

## L15 — Two tests that could not fail, for two good reasons and one bad one

**What I expected.** `validate` reports the untouched starter failing every
hidden test. The kernel task's first run on real hardware reported `23 failed,
2 passed`, and the harness called that `ok`.

**What the two were.** One asserted `torch.cuda.is_available()` — deliberate, and
the entire point of L12: no skips, so a missing GPU is loud. The other asserted
that `rmsnorm_fwd_kernel` is a `@triton.jit` function. Both pass against a
starter that implements nothing, because the starter is already decorated and
the machine already has a GPU.

**Why that is not a technicality.** `CONTRIBUTING.md` asks for every test to
fail against the untouched starter, and the reason is L6: under binary scoring a
test that cannot fail is a line of noise in a suite whose only job is to
separate pass from fail. I had two such lines and a validator that shrugged at
them.

**What changed, and it is not the same fix for both.** The `@triton.jit`
assertion is *redundant*: the tests launch the kernel with a grid subscript,
which only a JITFunction supports, so a plain Python function already fails all
nineteen direct-launch tests. Deleted, with a comment saying why. The CUDA
assertion is *not a test at all* — it is a statement about the machine, not
about the solution — so it moved to module scope, where missing hardware is a
collection error rather than a green tick, and where it is no longer counted as
a test that a starter passed. Twenty-five tests became twenty-four, and all
twenty-four fail against the untouched starter.

**What it cost.** Half an hour, and it only surfaced because the mutation script
prints the starter line. `validate` printed the same numbers and called them
`ok`; I read them and moved on.

## L14 — The one claim the task is named after had no test behind it

**What I expected.** `fused_rmsnorm_kernel` exists to ask for float32
accumulation under a float16 input. `prompt.md` says so, and one hidden test was
written specifically to enforce it: 16384 values with a standard deviation of
two, so the sum of squares lands near 65504 and a float16 accumulator saturates
to infinity.

**What happened.** The first mutation run on hardware: `SURVIVED — float16
accumulator`. Four of five mutants caught, and the survivor was the one the task
is about.

**Why.** My own reference design defeated the test. The kernel walks the row in
blocks and keeps a `BLOCK_SIZE`-wide vector of partial sums, so with a block of
1024 no single accumulator lane ever holds more than sixteen squares — about 64,
nowhere near the float16 ceiling. The large number I was reasoning about only
exists inside `tl.sum`, and Triton reduces in float32 regardless. The test was
arithmetic about an implementation I was not running.

**What changed.** A case that does not depend on the block size at all: float16
values around 300 — ordinary numbers in the format — whose *squares* are around
9e4 and past its ceiling. Squared in the input dtype that is an infinity on the
first element, and the row comes back as zeros; in float32 it is unremarkable.
The old long-row test stays, renamed and re-commented to say what it actually
covers, which is the loop rather than the accumulator. The sentence in
`prompt.md` changed too, because it had been licensing the assertion with the
same wrong mechanism.

**What it cost.** An hour, and it is the strongest argument for mutation testing
in this file. Twenty-five passing tests, a licensing pass walked in both
directions, and a prompt sentence pointing at the assertion — all three said the
claim was covered. Only the mutant said otherwise, and only on hardware.

## L13 — The default that is right for an application is wrong for a benchmark

**What I expected.** The Anthropic SDK's guidance is to opt into server-side
fallbacks by default: if the model declines a request, the API re-runs it on
another model inside the same call and you get an answer instead of a refusal.
Good advice, and I was about to take it.

**Why it is wrong here.** An application wants an answer. A benchmark wants an
answer *from a named model*. A fallback that quietly rescues a refused request
would publish Opus's answer under Fable's name — or the reverse — in a results
file whose entire purpose is to say which model did what. The feature does not
misbehave; it does exactly what it promises, and what it promises is
incompatible with the thing being built.

**What changed.** Fallbacks stay off. The adapter checks that the model that
answered is the model that was asked and raises if it is not, and a refusal is
reported as `adapter_error` — an absence of evidence, not a failed task.

**What it cost.** Nothing, and it is here because of how easily it could have
gone the other way: the guidance was explicit, the code was one parameter, and
nothing would have looked wrong afterwards. A default is an answer to somebody
else's question, and the only way to notice that is to re-ask it yourself.

## L12 — A skipped test suite exits zero, and zero means pass

**What I expected.** Hidden tests that need a GPU should skip politely on a
machine without one. Every test suite I have ever written does this.

**What that would have done.** pytest exits 0 when every test is skipped. The
runner reads exit 0 with no failures as a pass. So the polite version of this
task reports itself *solved* on any machine that cannot run a line of it — and
solved by whichever model happened to be pointed at it.

**Where I nearly went wrong.** The instinct is so automatic that I had already
typed `pytest.importorskip`. What stopped it was asking what the harness does
with the result, rather than what a developer sees in a terminal. In a normal
suite a skip is information for a human. Here it is an input to a scoring
function that cannot tell "nothing ran" from "everything passed".

**What changed.** No skips in the hidden tests, and a test that asserts CUDA is
present so the failure is loud if the tests are ever reached without it.
Deciding whether the hardware exists is the runner's job, and it already does
it: missing hardware comes back `needs_accelerator`, which is neither a pass
nor a failure.

**What it cost.** Ten minutes, and it is the most dangerous thing I have not
shipped. Every other bug in this file announces itself as a red run. This one
would have announced itself as a green one.

## L11 — The mechanism was right and the arithmetic still lied

**What I expected.** L10 built the tier separation: a task needing hardware the
machine does not have comes back `needs_accelerator`, which is an absence of
evidence rather than a zero, and I wrote that sentence down as though writing
it made it true.

**What happened.** The first accelerated task went in, and the very first run
printed `accelerated 0% (0/1)`, `kernels 0%`, and — from `validate` —
`9 task(s) validated` when eight had been.

**Why it survived.** `needs_accelerator` was being handled everywhere a
*verdict* was produced and nowhere a *rate* was. The status kept the task out
of the "ok" column and then went straight into the denominator of every
percentage, and the summary line counted rows rather than counting checks. None
of it was reachable while the accelerated tier had no tasks, so the tests I
wrote for the mechanism all passed against a set where the bug could not occur.

**What changed.** Outcomes with no evidence are excluded from the headline
rate, the per-category rates and the per-tier rate; a tier with nothing measured
reports `null` rather than `0`; and `validate` counts what it checked instead of
counting rows. Three tests now cover the case, one of them asserting the printed
line reads `not run` rather than a percentage.

**What it cost.** Half an hour, and the reminder that infrastructure written
before its first user is infrastructure whose tests share the user's blind
spot. The mechanism was fine. Every number downstream of it was wrong, and it
took a real task in the tier to say so.

## L10 — Two promises that look like a contradiction usually want two names

**What I expected.** `kernels` was a hole in the coverage, and holes get filled
by writing the missing task.

**What happened.** I could not write it without breaking something. The
founding constraint of this repository is a laptop, no GPU, five minutes,
reproducible by anyone who clones it — and a CUDA kernel task violates every
clause. But dropping the category means a benchmark about machine learning
engineering that excludes the part of it I find most interesting, and the
README already promises kernels in public.

**The two bad answers.** Quietly drop the category and edit the README down to
what I had already built. Or keep it and let GPU tasks into the headline pass
rate, at which point "you can re-run this benchmark" stops being true and the
number becomes something you take my word for.

**What changed.** Neither. The two claims are different measurements, so they
got different names: a `laptop` tier that the leaderboard rate is computed over
and that the loader refuses to let a GPU into, and an `accelerated` tier
reported beside it and never folded in. A task needing hardware this machine
does not have comes back `needs_accelerator`, which is neither a pass nor a
failure — an absence of evidence, labelled as one.

**What it cost.** An hour of infrastructure to write zero tasks, which felt
wrong while I was doing it. It was not: the alternative was one line of README
edit and a benchmark that quietly means less than it says.

## L9 — Files on disk are not a finished task

**What happened.** I ran two forked agents to write the two hardest tasks in
parallel. One of them died twice — first a server error, then a session limit.
The second death landed after it had written its reference, its hidden tests,
its prompt, its starter and its metadata, and before it had run any of them.

**Why that is the dangerous failure.** The directory looked complete. Every
file the task contract asks for was there, well written, in the right voice. A
`ls` says finished. Nothing about the shape of the work says "this has never
been executed", and the whole premise of a benchmark is that an unvalidated
task does not produce a lower score, it produces a wrong one.

**What changed.** I ran its validation myself and the mutation pass its author
never reached: fifty-seven tests against the reference, fifty-seven failures
against the untouched starter, six wrong implementations all caught. The task
was good. That is not the point — I could not have known that from looking.

**What it cost.** Nothing this time, and it is in here because of how close it
came to costing something. Delegation moves who does the work; it does not move
who is accountable for the evidence. The check that catches this is mechanical
and takes eleven seconds: never let a task into the set on the strength of its
directory listing.

## L8 — A rule in the prompt that no test enforces is not a rule

**What I expected.** The licensing pass runs one way: every assertion needs a
sentence in the prompt. Do that and the prompt and the tests agree.

**What happened.** `sharded_dataloader` passed with fifty-one tests, so I went
looking for wrong implementations it might miss. One of them was to give each
rank a contiguous slab of the epoch order instead of every `world_size`-th
sample. The prompt says stride, in a sentence I had written deliberately. The
slab version passed all fifty-one tests.

**Why.** A contiguous split has every property the tests were checking. The
ranks are disjoint. The batch counts are equal. Every sample is covered. The
resume works. Stride versus slab is only observable if you compare against the
epoch order position by position, and no test did.

**The direction I had missed.** The licensing pass proves no test asserts more
than the prompt promises. It says nothing about the reverse — a promise with no
test behind it. Both gaps break a task, in opposite ways: an unlicensed
assertion fails a model for guessing a different convention, and an unenforced
promise passes two implementations that disagree, so the score stops meaning one
thing.

**What changed.** Two tests, and the count went to fifty-three. The mutant is
now caught by exactly those two, which is how I know the gap was real.

**What it cost.** Twenty minutes, and a second direction on the checklist:
after licensing every assertion, walk every promise in the prompt and name the
test that enforces it. If a promise cannot be tested, it does not belong in the
prompt.

## L7 — A README that promises a command which raises

**What I expected.** The README was written before the code and locked. My job
was to implement it, not to renegotiate it.

**What happened.** It advertised `scratchbench run --model claude-opus-5`. The
model adapters are deliberately skeletons — they raise, with a message
explaining why. So the second command in the quick-start does not work, on a
public repository, to a reader who has just cloned it.

**Where I nearly went wrong.** "Design is locked" and "this sentence is false"
are not the same category, and I spent a moment treating them as one. Locking a
design means not re-litigating the shape of the thing. It does not mean
shipping a claim I have already disproven.

**What changed.** The example is the reference run now, followed by a sentence
saying the adapters land next and why that ordering is deliberate rather than
an omission. The design is untouched.

**What it cost.** Nothing, because it was caught before the first push. It
would have cost the only thing a new repository has, which is that the first
reader believes it.

## L6 — I wrote a test that tested nothing, twice, in two different ways

**What happened.** Two versions of the same mistake, an hour apart.

The first was in `softmax_stability`: a test asserting that `np.exp(1e4)` is
infinity. True, and a decent explanation of why the task exists — but it never
calls the solution, so it passes against an empty starter. Under binary
scoring, a test that cannot fail is not a weak test, it is a line of noise in a
suite whose only job is to separate pass from fail.

The second was worse. Writing the harness tests, I imported `textwrap`, did not
use it, noticed, and instead of deleting the import wrote
`test_textwrap_import_is_used_for_nothing`, asserting the module is not `None`.
That is a test that exists to make an unused import look intentional.

**What changed.** The first became a comment. The second was deleted along with
the import.

**What it cost.** Ten minutes, and the discovery that "run the suite against
the untouched starter and confirm every single test fails" is not a formality —
it is the only check that finds this class of test. It is now in
`CONTRIBUTING.md` as a rule.

## L5 — Fixing the adapter would have been fixing the wrong thing

**What I expected.** Pointing the CLI at `--model claude-opus-5`, whose adapter
is a skeleton, would print a clean message.

**What happened.** A Python traceback, and the sweep died on the first task.

**The fix I almost made.** Make the skeleton adapters return quietly instead of
raising. It is one line and it makes the symptom disappear.

**Why that is wrong.** The skeleton is not the interesting case. The
interesting case is a paid sweep, twenty tasks in, when the API rate-limits or
the connection drops — a real adapter raising for a real reason. A quiet
skeleton does nothing for that. The mechanism at fault is that `grade` called
arbitrary third-party code with no isolation, in the one function whose whole
job is to isolate one task from the next.

**What changed.** `grade` catches anything the solver raises and fails that
task with status `adapter_error`, carrying the exception message. The sweep
continues. A run containing any `adapter_error` exits non-zero, because a
number produced that way is not a measurement of anything.

**What it cost.** Almost the wrong fix. The tell was that my first idea touched
the file where the symptom appeared rather than the file where the rule lives.

## L4 — The flake I would have blamed on the model

**What I expected.** Tasks asserting agreement to 1e-12 in float64 are either
right or wrong, deterministically.

**What I found before it bit me.** numpy and torch change the *order* of their
floating-point reductions with the number of threads available. Same inputs,
same code, different last-bit answers on a machine with a different core count.
My laptop has ten cores; a CI runner has two or four.

**Why that matters more here than in normal code.** A wrong-by-1e-15 result in
an application is invisible. Here it is a task flipping from pass to fail
depending on which machine graded it — and the natural reading of a red run is
"the model got it wrong", not "my harness is not deterministic". The benchmark
would have been quietly lying, in the direction that is hardest to notice,
because a failure looks like a result.

**What changed.** The graded environment pins `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `PYTHONHASHSEED=0`, drops `PYTHONPATH`, and disables
bytecode writing. Determinism is a property of the harness, not a hope about
the tasks.

**What it cost.** Nothing yet, and that is the entry. This is the one I did not
pay for, and I only avoided it because reproducibility is the argument this
repository is making — if you cannot re-run a benchmark, you are trusting a
number.

## L3 — An equivalence test cannot catch a symmetric mistake

**What I expected.** `kv_cache_equivalence` is built around one property:
decoding token by token must equal a single pass over the whole sequence. That
property is the task. Test it hard and the task is covered.

**What happened.** I mutated the reference into the wrong versions a real
implementation would produce, to check the tests would catch them. Three were
caught immediately. The fourth — *delete the rotary embedding entirely* — was
caught by only four tests out of twenty-one, and none of them were the
equivalence tests.

**Why.** Removing RoPE removes it from both paths. The cached path and the full
path are then wrong in exactly the same way, so they agree perfectly. An
equivalence property is invariant under any error that is symmetric across the
two things being compared, and that is a large class of errors, not a corner
case.

**What changed.** The suite now contains `slow_forward`: the same layer
recomputed position by position, with explicit loops and explicit
trigonometry, written to a different shape than the reference. It is the only
test that pins the layer to an absolute answer rather than to itself.

**What it cost.** An hour, and it was the most valuable hour of the session.
Without the mutation pass I would have shipped a difficulty-3 task that a model
could pass by skipping the mechanism the task is named after.

## L2 — When the reference fails my test, my test is the suspect

**What happened.** First run of the `softmax_stability` suite against its own
reference: twenty-nine passed, one failed. The failing test built three rows of
logits at wildly different scales and asserted they produce the same
distribution.

**My first instinct.** Look at the reference. That is where a bug would be
interesting.

**The actual bug.** In the test. I wrote the middle row as
`[-1e4, -1e4 - 2.0]` while the other two rows put the larger logit second. The
rows were not the same distribution; one of them was reversed. The reference
was right and had been right the whole time.

**Why it is worth an entry.** In this repository the tests and the solution are
written by the same person within ten minutes of each other, and the tests are
the newer, less-examined code. On a disagreement the prior should favour the
older artifact, and my instinct did the opposite. Every minute spent reading a
correct reference is a minute not spent on the line that is actually wrong.

**What it cost.** Five minutes, and a rule I now apply: read the assertion
before reading the implementation it accuses.

## L1 — Writing the prompt last is what makes the prompt correct

**What I expected.** The reference-then-tests-then-prompt order in
`TASK_FORMAT.md` was mostly about not leaking the tests into the specification.

**What it actually does.** It surfaces the conventions I was about to leave
unstated. I cannot write a reference without deciding whether the output is
float32 or float64, whether a fully masked attention row is zeros or NaN,
whether the queries sit at the start or the end of the key range, whether the
merge list may be modified. Each of those is invisible while I am writing code
and mandatory once someone else has to reproduce it.

Walking every assertion afterwards and pointing at the sentence in `prompt.md`
that licenses it changed four prompts. The fully masked row returning zeros.
`float64` output regardless of input dtype. The merge list being left alone.
The returned loss being measured before the step rather than after. All four
were decisions I had made in code and nearly failed to write down.

**Why it matters.** A hidden test that can fail for a reason the prompt never
states does not make a task harder. It makes the task measure whether the model
guessed the same convention I did, and reports the answer as if it were
competence.

**What it cost.** Nothing, because the order caught it. That is the point of
having an order.
