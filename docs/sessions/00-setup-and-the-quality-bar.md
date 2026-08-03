# 00 — Setup and the quality bar (2026-07-26)

Short journal, one per step of the build. What I did, why, and which piece of
technology is carrying the weight.

## What this step was

Before a single task exists, I fixed how work gets done in this repository.

`CLAUDE.md` is the file every session loads first. It carries four things a
model cannot infer from the code: the intent, the landmines, my workflow rules,
and a dated snapshot of where the repository actually is. The template comes
from my own distillation set at `~/claude_efficiency` — a set of documents I
wrote with the strongest model I had access to, describing how it thinks, so
that whatever model I use next behaves the same way.

Three of those documents do the real work here:

- **THINKING_PROTOCOLS** — read before you write, three hypotheses before any
  fix, root-cause gate, least mechanism, assumption ledger, edge-case grid.
  Its loading paragraph sits verbatim at the top of `CLAUDE.md`, because a
  protocol that never enters the context window is decoration.
- **VERIFICATION_PROTOCOL** — the L0–L3 ladder. L0 is "it compiles", L1 is
  "tests pass", L2 is "I ran it and here is the output", L3 adds an adversarial
  pass.
- **FABLE_BAR_RUBRIC** — an eight-dimension self-review, run before anything is
  presented to me.

## The one line that matters most

In `CLAUDE.md`:

> In this repository L2 means a task was actually run and its output pasted. A
> passing test on its own is L1.

And for a task specifically, L2 has two halves: the reference solution passes
the hidden tests, **and** the untouched starter fails them with real assertion
errors rather than an `ImportError`.

The reason is the whole point of a benchmark. If a task is broken, it does not
produce a smaller number — it produces a *wrong* number, published under my
name, about somebody's model. An import error in the starter and a genuinely
wrong answer both score zero and mean completely different things. So both
halves get run, and both outputs get pasted, before I call anything ready.

## Also in this step

- `.gitignore` — `results/*.json` is generated output; the curated leaderboard
  is not.
- `LICENSE` — MIT, matching what the README already promised.
- `COMMITS.md` — the commit queue. I run every git command in my repositories
  myself; the queue is where the exact one-file-per-commit blocks wait for me.

## Environment this is built on

Anaconda Python 3.10 on an M1 Pro with 16 GB, no GPU. numpy 1.23.5, pytest
7.1.2, torch 2.8.0, PyYAML 6.0. That machine is not a limitation I am working
around — it is the specification. If a task needs more than this, it belongs in
somebody else's benchmark.

## Next

The first five tasks, each written in the order the task contract demands:
reference solution, then hidden tests, then the prompt.
