# docs

Four things live here: the design argument, the plan, the mistakes, and the
build log. If you are reading one of them for the first time, read them in that
order.

| File | What it is | Read it when |
|---|---|---|
| [`V2_DESIGN.md`](V2_DESIGN.md) | what makes a task discriminate at the top of the field, and the criterion candidates are written against | you want to know why a task was accepted or refused |
| [`ROADMAP.md`](ROADMAP.md) | every session from here to launch, one section each, with preconditions and costs | you want to know what happens next, or what the open debt is (§9) |
| [`LESSONS.md`](LESSONS.md) | what I got wrong, first person, newest first | you want the parts a changelog would have hidden |
| [`sessions/`](sessions/) | the build log, one entry per working session, with pasted output | you want the evidence behind a number |

## The two documents that carry the most

**`LESSONS.md`** is the other half of this repository. An entry that makes the
author look good is usually the wrong entry, so the entries are the near-misses:
a benchmark that is wrong looks exactly like a benchmark that is right, and the
only defence is writing down every time the difference was nearly missed.

**`ROADMAP.md` §9** is the debt ledger. Anything known to be unfinished is
there rather than in somebody's head.

## Session journals

Each one is what was done, why, and what was verified with pasted output,
written as the step finished rather than reconstructed afterwards.

| # | Session |
|---:|---|
| 00 | [Setup and the quality bar](sessions/00-setup-and-the-quality-bar.md) |
| 01 | [The first five tasks](sessions/01-the-first-five-tasks.md) |
| 02 | [The runner](sessions/02-the-runner.md) |
| 03 | [CI and the broken-task alarm](sessions/03-ci-and-the-broken-task-alarm.md) |
| 04 | [Coverage, tiers, and three tasks at once](sessions/04-coverage-tiers-and-three-tasks-at-once.md) |
| 05 | [The first model adapter and the first kernel](sessions/05-the-first-model-adapter-and-the-first-kernel.md) |
| 06 | [The hardware run, the live API, and a red repository](sessions/06-the-hardware-run-the-live-api-and-a-red-repository.md) |
| 07 | [An audit before the GPU run](sessions/07-an-audit-before-the-gpu-run.md) |
| 08 | [Five draws](sessions/08-five-draws.md) |
| 09 | [The accelerated tier answers](sessions/09-the-accelerated-tier-answers.md) |
| 10 | [Calibrating against models](sessions/10-calibrating-against-models.md) |
| 11 | [A Metal kernel on the machine I already have](sessions/11-a-metal-kernel-on-the-machine-i-already-have.md) |
| 12 | [The rule that read one model](sessions/12-the-rule-that-read-one-model.md) |
| 13 | [A laptop task that is about the framework](sessions/13-a-laptop-task-that-is-about-the-framework.md) |
| 14 | [The second Metal task, and a gate that refused everything](sessions/14-the-second-metal-task-and-a-gate-that-refused-everything.md) |
| 15 | [The stall was never the tasks](sessions/15-the-stall-was-never-the-tasks.md) |
| 16 | [The state the chunk was missing was an argument](sessions/16-the-state-the-chunk-was-missing-was-an-argument.md) |
| 17 | [One loss in thirty was the task](sessions/17-one-loss-in-thirty-was-the-task.md) |

## Elsewhere in the repository

- [`../TASK_FORMAT.md`](../TASK_FORMAT.md) — the task contract, including the
  calibration block and the admission rule
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — how to add a task, and how the
  accelerated tier is verified without a public GPU runner
- [`../leaderboard/`](../leaderboard/) — published results, and the draws every
  row was computed from
- [`../calibration/`](../calibration/) — the draws every `calibration:` block
  was computed from, re-derived in CI
