# 08 — Five draws (2026-08-09)

## What this step was

Session 07 built `--repeat` and left it unused: the mechanism for an error bar
existed and no published row had one. This step spends the money. Five
independent sweeps of the laptop tier with `claude-opus-5`, one attempt per
task, at the model's own defaults.

It also ran the queue that session 07 wrote. `main` is ninety commits, the
working tree is clean, and every workflow run on it is green. The fifteen red
crosses are gone from the history because the commits they belonged to are.

## The sweep

```
$ python -m runner.cli run --model claude-opus-5 --tasks all --repeat 5
...
claude-opus-5  ·  task set v1  ·  5 draw(s)  ·  laptop tier  ·  8 task(s) asked

task                       passed  spread
-------------------------  ------  ------
attention_causal_mask      5/5     always
bpe_merge_order            5/5     always
grad_accumulation          5/5     always
kv_cache_equivalence       5/5     always
online_softmax_attention   5/5     always
quantization_error_bounds  5/5     always
sharded_dataloader         5/5     always
softmax_stability          5/5     always

set pass rate: 100% in every draw
```

Forty tasks asked, forty passed. Sixteen and a half minutes, $2.64.

## What the five files say that the table does not

The interesting number is not the rate, it is the evidence that the rate is not
a fluke of sampling:

```
draw  rate    cost    wall      in     out
   1  1.00  0.5560   215.4   15858   19070
   2  1.00  0.5328   197.8   15858   18141
   3  1.00  0.5398   201.5   15858   18422
   4  1.00  0.4912   186.5   15858   16476
   5  1.00  0.5225   195.3   15858   17729
```

**Input is identical to the token, five times.** That is the harness holding
still: the same prompt, the same starter files, no cache, nothing drifting.

**Output ranges over 16476 to 19070, a 16% spread.** That is the model not
holding still. Five materially different sets of answers were written, and every
one of them passed all 308 hidden tests.

L19 recorded a task flipping between draws and concluded a single sweep is a
sample. That is still true, and this adds the other half: **the score can be a
converged value while the process producing it is visibly random.** Opus 5 does
not solve v1 by luck. It solves it.

Which is the finding, and it is not good news for v1. A ceiling reached once is
a good day. A ceiling reached five times out of five, with 16% variation in what
the model wrote underneath, is a ceiling. `docs/V2_DESIGN.md` was written
against a suspicion; it is now written against a measurement.

## The column that does have an error bar

$0.49 to $0.56 per draw, a 14% range, entirely from output length. Input is
fixed and output is not, so cost inherits every bit of the model's variance
while the score inherits none of it.

The leaderboard had been about to put an error bar on the score, which turns out
not to need one, and print the cost as a single figure, which does. Both columns
now report what they are: the rate as "100%, in all five", the cost as a range.
`docs/LESSONS.md` L26.

Every one of the five reproduces:

```
$ python tools/check_cost.py
ok           claude-opus-5-20260809-draw1.json  15858 in + 19070 out = $0.556040
ok           claude-opus-5-20260809-draw2.json  15858 in + 18141 out = $0.532815
ok           claude-opus-5-20260809-draw3.json  15858 in + 18422 out = $0.539840
ok           claude-opus-5-20260809-draw4.json  15858 in + 16476 out = $0.491190
ok           claude-opus-5-20260809-draw5.json  15858 in + 17729 out = $0.522515

7 file(s) checked: every cost reproduces from its own tokens.
```

## A bug caught before it cost anything

`report --variance` was run against the existing `results/` directory before the
sweep, out of habit rather than suspicion, and it said:

```
claude-haiku-4-5  ·  task set v1  ·  7 draw(s)  ·  laptop tier
set pass rate: 0% to 100%  ·  mean 20%
```

Seven draws that are not seven of anything: six one-task probe runs from an
afternoon of debugging the adapter, folded in with the eight-task sweep that is
on the leaderboard. The spread measured what had been typed rather than what the
model did. `group_draws` keyed on model and task set and stopped there; the set
of tasks a run actually covered is part of the question too. Fixed, tested, and
written up as L25. The five-draw sweep went straight into the corrected version.

## Still open

- **The Haiku row is one draw.** Five would cost $0.36 and take ten minutes. It
  is the more interesting one: the flip in L19 was Haiku's, so the model with
  something to say about variance is the one that has not been asked.
- **No model has been graded on the accelerated tier.** Unchanged, and unchanged
  for the same reason: this machine has no CUDA device.
- **v2 has no tasks written.** The design is in `docs/V2_DESIGN.md` and the two
  to write first are named there.
