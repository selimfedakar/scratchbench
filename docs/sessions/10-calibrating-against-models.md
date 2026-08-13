# 10 — Calibrating against models instead of against me (2026-08-09)

## What this step was

`docs/V2_DESIGN.md` had named the two tasks to write first and the rule that
should decide whether they get in. This step wrote both of them end to end, built
the rule alongside them rather than before them, asked two models, and then wrote
a third task because of what the answer was.

The rule refused all three. That is the session, and the refusal is the result
rather than the setback.

## The first two tasks, and what each one was built to separate

**`speculative_decoding_verify`** (attention, laptop, numpy, 24 hidden tests).
A draft model proposes tokens, a target model scores them, and the accept-and-
correct step has to emit tokens distributed *exactly* as the target would have
produced on its own. The prompt never states the acceptance rule or the
correction distribution. It states the guarantee and one more sentence — accept
as often as the guarantee permits — and those two together determine the scheme
uniquely: acceptance can never exceed `p(x)/q(x)` without overshooting, so the
largest permitted rule is `min(1, p/q)`, and the correction has to make up the
difference exactly, which is the renormalised positive residual and nothing else.

Grading is distributional, because that is the only thing that catches the wrong
versions. Two hundred thousand rounds per scenario, proposals drawn from the
draft distributions on the test's own generator so the solution's randomness
cannot move its inputs, and the empirical frequency of the emitted token
compared against the target's at six standard errors. Every plausible wrong
version — resampling from the target on rejection, renormalising `|p - q|`,
dividing the residual by the wrong thing — lands 0.05 to 0.2 away from a band of
0.008. The reference lands at 0.0022.

The mutant that justifies the whole design is `never accepts`: reject every
proposal and sample from the target. It emits *exactly* the right distribution,
it is a correct speculative decoder with the speculation removed, and only the
acceptance-rate test notices. Twelve tests caught it.

**`flash_attention_backward`** (attention, laptop, torch, 49 hidden tests). The
forward is given as `(o, lse)`; the backward is derived. The task splits into
`key_block_gradients`, which is handed one block of keys and never the rest, and
a driver that walks the blocks.

That split is the task. `dS = P * (dP - D)`, and `D = rowsum(dO * O)` is a
property of a whole query row: the columns of `dP` belonging to a key block
depend only on that block, and `D` does not. A blocked implementation that
computes `D` from the columns in hand returns gradients of the right shape, the
right magnitude, and the right value whenever there happens to be one block. The
interface makes that mistake natural and the tests make it fatal.

One test uses no reference at all. If every row of `v` is the same vector, every
query's output is that vector, `dP` and `D` cancel exactly, and `dq` and `dk` are
zero on paper for any `q`, `k` and `do`. The reference lands at 3.5e-16 against a
tolerance of 1e-12, and an implementation that drops the correction fails it with
no autograd involved.

## A third task, written after the first two came back

Both of the above were solved by `claude-opus-5` in every draw, so
`V2_DESIGN.md` §5 says to write something harder from the candidate list. The
useful question was what "harder" means given that the two tasks satisfied every
property §2 asks for, and the answer I arrived at was that the property doing the
work in `fused_rmsnorm_kernel` is not on that list at all: its difficulty is a
fact about a tool, not a step in an argument.

**`activation_checkpointing_rng`** (training, laptop, torch, 34 hidden tests) was
written against that reading. A stack of residual MLP blocks with dropout in the
middle, checkpointed: the forward keeps only the boundaries, the backward
recomputes each block, and the recomputation has to draw the mask the forward
drew. Restoring the generator before each recomputation is the half everyone
writes. The other half is putting it back afterwards, and an implementation that
skips it produces perfect gradients this step and a training run that will not
reproduce on the next one.

That half is graded exactly, with no tolerance in it, because generator state is
a tensor of bytes:

```
mutant: restores before each recompute and never puts it back afterwards
  -> 25 passed, 9 failed
  -> every failure is a generator test; every output and every gradient is right
```

It did not work either. Opus went 10 for 10, and L30 is the entry about why.

## Three gates, three times

`reference/` → `hidden_tests/` → `prompt.md` → `meta.yaml`, then the reference
passes, the untouched starter fails with real assertions, and every mutant is
caught. `tools/mutate_v2_tasks.py` is in the repository because a check nobody
can re-run is a claim.

```
task                          reference  untouched starter  mutants
----------------------------  ---------  -----------------  --------------
speculative_decoding_verify   24 passed  24 failed          8 of 8 caught
flash_attention_backward      49 passed  49 failed          10 of 10 caught
activation_checkpointing_rng  34 passed  34 failed          10 of 10 caught
```

Then the licensing pass, in both directions. It changed one thing, and it was in
`flash_attention_backward`: the prompt said not to reach for `torch.autograd`,
and nothing enforced that. The sentence now says what is true and checkable —
`key_block_gradients` cannot rebuild the forward out of one block of keys, and
nothing returned may carry a gradient history — instead of a rule with no test
behind it (L8).

## The rule, built with its first users rather than before them

`meta.yaml` takes an optional `calibration:` block, `runner/tasks.py` validates
it, and a task in a numbered frozen set from `v2` onward is **refused by the
loader** if the best entry passed every draw. `TaskError`, not a warning. Tasks
that fail the bar go to `frozen_set: warmup`, which keeps the block and stays out
of the headline.

It was deliberately not built in session 07, and the reason is L11: infrastructure
written before its first user has tests that share the user's blind spot, which
has published wrong numbers here twice. Written against two real tasks, the first
thing it did was reject both of them.

## What the models said

Each draw is an independent sweep, no draw sees another's results, and
`max_attempts` is 1 throughout:

```
                              claude-opus-5   claude-sonnet-5   claude-haiku-4-5
speculative_decoding_verify        15/15            10/10             10/20
flash_attention_backward           15/15             8/10              3/19
activation_checkpointing_rng       10/10            10/10              1/10
```

Sonnet was also asked the v1 laptop tier and cleared it, 39 of 39 measured over
five draws at a third of Opus's price. Two of the three models tried now solve
v1 completely, which retires the last open question in `V2_DESIGN.md` §5: the
three tasks that were "solved by both models tried, so they need a third" have
had their third.

Forty draws of Opus 5 across three tasks, and not one failure. Haiku's
nineteenth `flash_attention_backward` draw is missing rather than failed: its
adapter never answered, the hidden tests were never copied in, and an absence of
evidence does not enter a rate.

The rule refuses all three. `frozen_set: warmup` for all three.

Two Haiku sweeps of the same five draws on `speculative_decoding_verify` came
back 2 of 5 and then 5 of 5, same prompt, same settings, hours apart. That is
L19 at scale and it is the reason `CALIBRATION_MIN_DRAWS` is 5 and the blocks
above carry 10 to 20: five draws of a sampled process is a number, not a
measurement, and two of them can disagree completely.

The failure shapes are the part that repays reading, and they were measured by
name rather than inferred from counts (L27). On `speculative_decoding_verify`,
Haiku's wrong versions run, accept at plausible rates, emit plausible tokens, and
fail on the distribution those tokens came from:

```
draw1   4 of 24   both distribution tests only
draw2   6 of 24   distribution, and the rejected token coming back
draw7   6 of 24   distribution, and the acceptance rate
draw4  11 of 24   most of the guarantee
```

Nothing in draw 1 would have been caught by comparing an array against a
reference on one input. On `flash_attention_backward` the split runs the other
way: two draws fail the identical-values probe, which is the row correction being
wrong, and one gets that right and drops the `1/sqrt(d)` on `dq` and `dk`. Same
verdict, different information.

One thing the failure tables say about my own tests, and it is not flattering.
The generator-bracket tests in `activation_checkpointing_rng` — the ones the task
was built around, the ones a mutant proves are load-bearing — **never fired in
twenty real draws.** Opus was right and Haiku was wrong earlier, on shapes and on
the dropout mask itself. The test is not noise: it catches the mutant, and only
it catches the mutant. But it was aimed at a failure mode that sits between the
two models I have, and neither of them is there.

## The design claim that could not survive contact

`V2_DESIGN.md` §3 was written before any of these tasks existed and said
`flash_attention_backward` would enforce non-materialisation structurally,
because the tests require identical results across several block sizes. Sitting
down to write those tests, the sentence is false: block-size invariance is a
property of every correct implementation, including one that builds the whole
N×N matrix and slices it. The claim was plausible, it was in the same register as
the true sentences around it, and there was nothing to check it against because
the task did not exist yet.

The fix is the interface, not the tests. L28.

## A third model, and the sharpest failure shape of the session

Two models cannot tell a property of a task from a property of a model, so
`claude-sonnet-5` was asked the same three tasks, ten draws each:
`speculative_decoding_verify` 10/10, `activation_checkpointing_rng` 10/10,
`flash_attention_backward` **8/10**.

And the split is the interesting part, because it is the same pair of tests both
times:

```
draw5  2 failed, 47 passed  -> test_fewer_queries_than_keys_puts_the_queries_at_the_end
                               test_a_short_query_window_over_a_long_history[True]
draw9  2 failed, 47 passed  -> the same two
```

`2 failed, 47 passed` is the signature of a mutant in `tools/mutate_v2_tasks.py`,
written weeks before Sonnet was asked anything: *queries not put at the end of
the key range*. A model reproduced one of the wrong implementations I had
invented for it, exactly, twice.

Three models, three depths of failure on one task:

| | fails | what the failure is |
|---|---:|---|
| `claude-opus-5` | 0 of 15 | nothing |
| `claude-sonnet-5` | 2 of 10 | the decode-time query offset, the same two tests both times |
| `claude-haiku-4-5` | 16 of 19 | the row correction, or the `1/sqrt(d)` scale |

That is the whole argument for reporting failure shape beside the verdict, in one
table. It also sharpens the admission rule rather than softening it: Sonnet's
failure is a **convention** missed at the edge, not a mechanism absent, and L21
is explicit that convention noise is not the axis a headline set should be
scaling. The task stays in `warmup`.

The rule is still stricter than its stated intent, and that is worth writing
down rather than quietly patching. `V2_DESIGN.md` §4 refuses a task because *the
best model tried* swept it, so a task that separates Sonnet from Opus is refused
on Opus's account. That is deliberate — the headline set exists to rank the top,
and everything else is what `warmup` is for — but it means the set will keep
throwing away tasks that discriminate one rung down. Revisit it when there is a
model between Sonnet and Opus to lose.

## Two defects that a billing failure found

The first calibration attempt ran out of API credit partway through the second of
five draws. Both of the following were sitting in the harness and neither was
reachable without a sweep that dies halfway.

**A partial draw was a draw of the set rate.** `report --variance` printed
`5 draw(s)` and `set pass rate: 100% in every draw` over one complete draw, one
draw that measured half the tasks and solved them, and three that measured
nothing. Every number in the pipeline was correct; the sentence they assembled
into was not. The set rate is now computed over draws that measured everything
the tier was asked for, and the header says how many draws were incomplete.
L20 predicted this failure in as many words two sessions ago. L29.

**`--keep` kept one draw of ten.** Kept workdirs are named after the task, so
five draws of the same task landed on the same directory and the last one
overwrote the rest. What a repeated sweep produces that a single one cannot is
the shape of each failure, and that was exactly what the overwrite threw away —
session 09 had to run its draws one at a time to work around it. Each draw gets
its own directory now, which is how the failure table above exists.

## A third harness change, and I caused the need for it

`warmup` did not exist this morning. The moment it did, the laptop tier stopped
being one published set, and `run --tasks all --tier laptop` — the command in the
README, in `CONTRIBUTING.md` and in `CLAUDE.md` — started covering `v1` and
`warmup` together and reporting one `pass_rate` averaged across them.

That is L23 exactly, one axis over: there it was two tiers blended into a
headline number, here it is two frozen sets. It was found before it published
anything, by asking what the third model's leaderboard row would actually mean,
and the fix is a `--set` filter on `run`, `validate` and `list`. There is no
default filtering — a run that asks for everything gets everything, and its
results file records `task_set` as the list it covered, so a blend is visible
rather than implied. A leaderboard row asks for one set.

The general form is the uncomfortable one: adding a category to a taxonomy
silently changes the meaning of every aggregate computed over it, and nothing in
the code says so. The tier split got a constant, a payload field and three tests
after L23. The set split needed the same and did not have it, because the
category was one line of YAML.

## Cost

$7.43 of API spend, and $0.34 of that bought
nothing: the interrupted sweep, kept because the files it wrote are the evidence
for L29. The credit ran out twice in one session, both times mid-sweep, and both
times the harness reported the hole instead of averaging over it — which is the
fix from L29 doing its job on the same day it was written.

## Still open

- ~~The calibration blocks are claims.~~ Closed in the same session: the 55
  draws are checked in under `calibration/`, `tools/check_calibration.py`
  re-derives all six entries from them, and CI runs it beside `check_cost.py`.
  A number that decides whether a task is allowed into a frozen set is not
  allowed to be the one number nobody can reproduce.
- **The accelerated tier is still the only one measuring at the top**, and it has
  one task.
- **Two models.** A third would say whether Haiku at 10 of 20 on
  `speculative_decoding_verify` is a property of Haiku or of the task.
