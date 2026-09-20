# 18 — The badge was red for three sessions

The plan for this session was §5 and §10.3: rent a CUDA box, finish
`fused_rmsnorm_kernel` at ten draws, and take the first real `v2` sweep across
both vendors. The first command I ran was not part of that plan.

```
$ gh run list --limit 3
completed  failure  Record the session 17 second-half commit queue  ci  main  push  34727231092  1m51s  2026-09-13
completed  failure  Record the session 17 commit queue              ci  main  push  34293099048  1m47s  2026-09-09
completed  failure  Record the session 16 second-half commit queue  ci  main  push  34201792249  1m53s  2026-09-08
```

Three consecutive pushes, three red crosses, on a public repository whose
argument is that you should not have to trust its numbers. The badge is in the
first line of the README.

## One test, and it was measuring the runner

The same test failed in every run and on both Python versions:

```
>       assert outcome.status == "mps_stalled"
E       AssertionError: assert 'needs_accelerator' == 'mps_stalled'
tests/test_runner.py:798: AssertionError
FAILED tests/test_runner.py::test_every_attempt_stalling_is_reported_as_no_evidence
1 failed, 130 passed in 64.80s
```

The test guards what L42 is about: a stalled MPS process recorded as a `timeout`
is evidence and a failure, so it would publish a claim about a model that
actually describes this laptop. It fakes a `run_pytest` that always stalls and
asserts the outcome is `mps_stalled`.

It never reached the mechanism on a GPU-less runner, because `grade` checks the
hardware first:

```python
    if task.requires_gpu and task.accelerator not in available_accelerators():
```

The task under test is `metal_cross_entropy_kernel`. On this Mac the gate opens
and the assertion measures the stall guard; on a CI runner the gate closes and
the assertion measures the gate. One line, two meanings, chosen by whoever ran
it — which is the same confusion `needs_accelerator` exists to prevent in the
published results, shipped in the test suite instead.

The fix pins the hardware rather than inheriting it. It is legitimate because
`add_stall_guard` keys on `task.accelerator` and never on the host, so the
mechanism genuinely is machine-independent:

```python
    monkeypatch.setattr(sandbox, "available_accelerators", lambda: {"metal"})
```

## Verified against the machine that found it

Passing locally proves nothing here — locally was always passing. So the CI
environment was reproduced on this laptop with a plugin that forces both
accelerators absent:

```python
def pytest_configure(config):
    import torch
    torch.cuda.is_available = lambda: False
    if getattr(torch.backends, "mps", None) is not None:
        torch.backends.mps.is_available = lambda: False
```

```
$ python -m pytest -q -p nogpu_plugin
131 passed in 51.77s
```

That is the claim worth making: not "the test is fixed" but "the suite does not
depend on the hardware under it", measured on the only machine available.

```
$ gh run list --limit 1
completed  success  Assert the stall guard against the task, not the runner's hardware  ci  main  push  35171958541  3m5s
```

`docs/LESSONS.md` **L47** is the entry. The short version is that a green suite
on one machine is one draw, which is a sentence this repository already
publishes about models and had not applied to itself.

## The claims audit, §6.3

Every factual claim in `README.md` and `leaderboard/README.md` was re-derived
rather than re-read. The public pages had drifted in eight places, all in the
same direction — numbers that were true when written:

| Claim, as published | Command | What it actually says |
|---|---|---|
| "Sixteen tasks" | `ls -d tasks/*/ \| wc -l` | **17** |
| "Thirteen of them run on any laptop; the other three need a GPU" | `python -m runner.cli list` | **14** laptop, 3 accelerated |
| task table lists 14 tasks | `python -m runner.cli list` | 3 missing: `activation_checkpointing_rng`, `chunked_batchnorm_reduction`, `speculative_decoding_verify` |
| `validate --tier all` transcript shows 13 tasks | the transcript below | 17 tasks, 16 validated |
| "re-derives all thirty-two entries" | `python tools/check_calibration.py` | **35 entries from 251 draws** |
| "Forty-four entries" in `LESSONS.md` | `grep -cE '^## L[0-9]+' docs/LESSONS.md` | **46**, now 47 |
| `check_cost.py` transcript shows 2 files | `python tools/check_cost.py` | **251 files** |
| `leaderboard/README.md`: "22 file(s) checked" | `python tools/check_cost.py` | **251 files** |

Two claims held on the first try. The provenance promise — no curriculum code
anywhere — survives a grep for course markers across every tracked file except
the sentence that states the prohibition, and `notes/`, the author's Turkish
working notes, is ignored with zero tracked files under it:

```
$ git grep -niE "cs336|cs224|assignment [0-9]|problem set|stanford|coursera" -- ':!docs/' ':!README.md'
CLAUDE.md:120:- **Do NOT copy anything from any curriculum** — not CS336, not any course:

$ git check-ignore -v notes && git ls-files notes/ | wc -l
.gitignore:18:notes/	notes
       0
```

And no key material has ever been committed, over the whole history rather than
the current tree:

```
$ git log --all -p | grep -nE "sk-ant-[A-Za-z0-9_-]{10,}|sk-proj-[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
(no output)
```

The pattern in the drift is worth naming, because it is not carelessness. Every
one of those eight numbers was correct on the day it was typed, and each became
wrong when a task was added or a draw was taken — which is to say the README was
holding derived values by hand while a tool existed to derive them. §4.3 already
solved this for the leaderboard tables by generating them and failing CI when
the page and its evidence disagree. The README's counts have no such gate, and
that is the real finding rather than the eight corrections.

## The sweep, and what it did to the set it was measuring

Six models, ten draws each, both vendors, `--set v2 --tier all` on this M1 Pro.
Sixty draws, **$8.4807**, against a §10.4 budget of about $30. The CUDA member
returns `needs_accelerator` here and is drawn on a rented box.

| | `flash_attention_backward` | `metal_cross_entropy` | `metal_softmax_backward` |
|---|:---:|:---:|:---:|
| `claude-opus-5` | 10/10 | 9/10 | 8/10 |
| `claude-sonnet-5` | 10/10 | 7/10 | 6/10 |
| `gpt-5.6-sol` | 10/10 | 7/10 | 7/10 |
| `gpt-5.6-terra` | 10/10 | 9/10 | 9/10 |
| `gpt-5.6-luna` | 9/10 | 3/7 | 4/7 |
| `claude-haiku-4-5` | 2/10 | 1/10 | 0/8 |

The sweep's first job was to produce a `v2` row. It produced a decision instead.

`flash_attention_backward` was in `v2` because it was the one laptop task
separating two frontier Claude models — Opus 15/15 against Sonnet 8/10, on
2026-08-09. Sonnet's two losses did not repeat, and two OpenAI models cleared it
outright. Cumulatively the block is now Opus 25/25, `gpt-5.6-terra` 10/10,
`gpt-5.6-sol` 10/10, Sonnet 18/20, `gpt-5.6-luna` 9/10, Haiku 5/29, and
`_check_admission` reads the top two measured entries and refuses a task both
clear. So the rule that admitted it in August removed it in September, on
nothing but fresh draws.

**`v2` therefore ships with no laptop member at all.** The decision and the three
rejected alternatives are `ROADMAP.md` §11; the one worth repeating is that
leaving the OpenAI models out of the calibration blocks would have kept the task
in the set and would have been the exact move L21, L35 and L43 are each about.
A rule that reads only the vendor whose numbers suit the set is not a rule.

### What the accelerated tier did, which is the better half of the news

Both Metal tasks were admitted comfortably, and for the first time a task here
carries six models at six distinct rates:

```
metal_cross_entropy_kernel   terra 90% · opus 77% · sol 70% · sonnet 63% · luna 43% · haiku 14%
metal_softmax_backward       terra 90% · opus 80% · sol 70% · sonnet 60% · luna 57% · haiku 11%
```

And the failure shapes say what the rates cannot. Every failing test instance on
both tasks, grouped by the threadgroup size the launch geometry chose:

```
   size     1:    4
   size     7:   44  <- not a power of two
   size    32:    8
   size    33:   45  <- not a power of two
   size    96:   56  <- not a power of two
   size  1024:    8
```

`claude-sonnet-5` fails **only** at 7, 33 and 96 — never once at 1, 32 or 1024.
That is precisely the hazard both tasks were written around, and it is the first
time a task in this repository has been shown to measure the thing its `probes:`
field claims out of the failure shapes rather than out of my description of it.

### A fix confirmed in the field, six times, by accident

`gpt-5.6-luna` produced six `mps_stalled` draws, three on each Metal task. That
status only appears in this output because of the `sandbox.py` change made
earlier the same session (L48): before it, a baseline that stalled out returned
the same `False` as a baseline that genuinely could not collect, and the run was
published as `collection_error` — *the task is broken*.

The ordering is the evidence. Haiku's two `collection_error` draws were taken at
02:56 and 03:08, before the fix. Every draw after 03:14 ran on the fixed code,
and the same situation now reports `mps_stalled`. No `collection_error` has
appeared since. That is strong support for the L48 diagnosis and it is still not
proof about those two specific draws, which stay exactly as they were recorded.

## §6.1 — the clean-clone test, run for the first time

Everything before today was verified in the working directory, with an editable
install and a shell that already had the dependencies. A reader gets none of
that. The transcript is `~/scratchbench-clean-clone-transcript.txt`; the
summary is that **nothing needed a step the README does not mention**.

```
$ git clone https://github.com/selimfedakar/scratchbench && cd scratchbench
$ python3 -m venv .venv && . .venv/bin/activate && pip install -e .
$ which scratchbench
.../scratchbench/.venv/bin/scratchbench

$ scratchbench validate
...
14 task(s) validated: reference passes, starter fails cleanly.

$ scratchbench run --model reference --tasks all
laptop       100% (14/14)  ·  headline
12.8s wall clock  ·  1 attempt(s) per task

$ python tools/check_cost.py         → 311 file(s) checked
$ python tools/check_calibration.py  → 44 entries from 311 draw(s)
$ python tools/render_leaderboard.py --check → 6 regions match
$ python tools/render_readme.py --check      → 2 regions match
```

Twelve point eight seconds for the whole laptop tier, from a clone, with no GPU.
That is the claim the project is named after and it had never actually been
tested from outside.

**One README bug, and the audit had already flagged it without my fixing it.**
The pasted `validate --tier all` transcript was from a thirteen-task repository.
It is replaced with the fourteen-task default-scope run above — captured in the
clean clone today rather than reconstructed — and the `UNCHECKED` line that
makes the tier point is quoted separately instead of being implied by a table
nobody re-ran.

**What could not be captured, and why it is not in the README.** A real
`validate --tier all` on this machine never finished. It was left running for
nine minutes and killed: the two Metal tasks each stall, retry four times, and
resolve to `BROKEN` — a verdict about the GPU's state after sixty draws, not
about the tasks. Writing the transcript I *expected* would have been the easiest
paragraph of the session and it would have been fiction, so the page carries the
run that happened instead. The stall is L42 and it is the standing cost of
developing this on an M1 Pro.

⚠ This also means `python -m pytest` hangs on this machine after a sweep, because
`test_validate_does_not_count_what_it_could_not_run` dispatches Metal. CI is
unaffected — no MPS there, so those tasks are `UNCHECKED` — and the local
equivalent is the forced no-GPU plugin, which runs the whole suite in 45 seconds:
**143 passed**.
