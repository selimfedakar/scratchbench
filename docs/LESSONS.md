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
