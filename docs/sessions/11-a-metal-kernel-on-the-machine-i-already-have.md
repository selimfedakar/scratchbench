# 11 — A Metal kernel on the machine I already have

The accelerated tier had one task, needed a rented Linux box, and was the only
tier still separating frontier models. The README promised Metal as well as
Triton and had promised it since the first commit. This session closed that,
and the task it produced is the first one written from the corrected reading of
what makes a task hard (L30): the difficulty is a fact about a tool, and an
obscure one.

## The vehicle: `torch.mps.compile_shader`

The blocker I assumed existed does not. PyTorch 2.8 ships
`torch.mps.compile_shader(source)`, which compiles a Metal Shading Language
string at runtime and hands back a library whose kernels are called straight on
MPS tensors:

```
>>> lib = torch.mps.compile_shader(SOURCE)
>>> lib.cross_entropy_rows(out, logits, targets, n_cols, threads=rows * 128, group_size=128)
```

No Objective-C++, no `cpp_extension` build step, no Xcode project — the toolchain
is `torch` and a Mac. So a `metal` task is verifiable on the same laptop the rest
of the benchmark runs on, and the accelerated tier stops being the tier nobody
can reproduce. It is still reported beside the headline and never inside it.

Four facts about that API, measured rather than assumed, and every one of them
ended up load-bearing in the task:

| Probe | Result |
|---|---|
| `threads=10, group_size=4` | three threadgroups, the last one reporting `threads_per_threadgroup == 2`. Non-uniform threadgroups: the API dispatches threads, not groups. |
| `threadgroup float* [[threadgroup(0)]]` | compiles, runs, and reads back **zeros**. The library binds no threadgroup memory, so the canonical Metal idiom silently produces a wrong answer instead of an error. |
| threadgroup memory, unwritten | garbage, not zeros: `-1.5e+31`, `4.8e+19`. A lane that never publishes an identity is a lane that poisons the reduction. |
| `simd_sum` with `group_size=40` | `32` for the first thirty-two lanes and `8` for the rest. It reduces a simdgroup, never a threadgroup. |
| `group_size=2048` | `ValueError: Threadgroup size exceeds 1024 limit` |

## The task: `metal_cross_entropy_kernel`

Softmax cross-entropy per row, fused: the row maximum, the sum of shifted
exponentials, and the target's logit, in one kernel launch, one threadgroup per
row. The model writes `SOURCE` and nothing else — the harness compiles it and
dispatches it, which is deliberate. It is the `flash_attention_backward` move
one tier over (L28): the launch geometry is the part that makes the task hard,
so it is not something the solution gets to choose.

The caller picks the threadgroup size, anywhere from 1 to 1024, with no
relationship to the row width. That single sentence is what the kernel has to
survive:

- a group narrower than the row, so each lane walks a strided slice;
- a group wider than the row, so lanes with no column still have to publish the
  identity of the reduction and still have to reach every barrier;
- a group that is not a power of two, so the fold has to carry a live range
  rounded up rather than halved;
- a group wider than a simdgroup, so `simd_sum` alone is not a reduction.

The provable properties on top of that: the loss does not move when a whole row
is shifted, a flat row costs exactly `log(n_cols)`, a single-column row costs
exactly zero, and every value is checked against the same loss computed in
float64 on the CPU rather than against the kernel's own reference.

**Verified, both halves, on this machine (M1 Pro, torch 2.8.0, Metal 32023.883):**

```
task                        reference  untouched starter  time   verdict
metal_cross_entropy_kernel  68 passed  68 failed          2.78s  ok
```

Thirteen mutants, and each one carries the verdict it was written to get, so the
script fails in both directions:

```
CAUGHT    threadgroup memory as a bound buffer          66 failed, 2 passed
CAUGHT    simdgroup reduction only                      29 failed, 39 passed
CAUGHT    power-of-two fold                              1 failed, 67 passed
CAUGHT    sum fold bounded by the group                 19 failed, 49 passed
CAUGHT    idle lanes never publish an identity           8 failed, 60 passed
CAUGHT    out-of-range lanes return early               16 failed, 52 passed
CAUGHT    barrier inside the branch                     28 failed, 40 passed
CAUGHT    no barrier after publishing the maxima         7 failed, 61 passed
CAUGHT    no shift before the exponential               68 failed
CAUGHT    shift never added back                        68 failed
SURVIVED  max fold bounded by the group                 68 passed
SURVIVED  thread 0 does the whole row                   68 passed
SURVIVED  no barrier before reusing the scratch         68 passed
```

Three survivors, three different reasons, and none of them is "the tests are
thin". They are L31, L32 and L33.

## What the models did with it

Ten independent draws each, one attempt per draw, every model at its own
defaults, every results file checked in.

| Model | Passed | Cost per draw |
|---|---:|---|
| `claude-opus-5` | **8 / 10** | $0.078 to $0.108 |
| `claude-sonnet-5` | **4 / 10** | $0.033 to $0.091 |
| `claude-haiku-4-5` | **1 / 9** | $0.006 to $0.117 |

**This is the first task in the repository that a frontier model fails on the
laptop it is developed on.** Opus 5 has gone 40 for 40 on the v1 laptop tier and
40 for 40 on the three v2 candidates; here it drops two draws out of ten, and
Sonnet sits clearly below it rather than beside it.

### The failure has a name, and it is not a mechanism

Every one of Opus's two failures and five of Sonnet's six are the same compile
error:

```
E   SyntaxError: program_source:39:14: error: cannot combine with previous
    'type-name' declaration specifier
E           uint half = (n + 1u) / 2u;
E                ^
```

`half` is a type in Metal Shading Language — the 16-bit float — so it cannot be
a variable name. Both models reach for it for the same reason: the reduction
folds a range in half, and `half` is what that variable is called in every CUDA
tutorial ever written. The reduction logic underneath is right in every one of
those draws.

I made the identical mistake in my first draft of the reference, an hour before
either model was asked. That is L34, and it is the sharpest evidence I have for
L30's claim that the difficulty which survives is a fact about a tool rather
than a step in an argument.

The one Sonnet draw that ran and still failed is the more familiar shape:
`25 failed, 43 passed`, and the fold is `for (uint offset = tpg >> 1; offset > 0;
offset >>= 1)` — the power-of-two halving that drops entries whenever the
threadgroup size is not a power of two. That is `mutate_metal_task.py`'s third
mutant, reproduced by a real model, and its failures land where the mutant's do:
eleven in the shape-and-group-size sweep, five in the non-power-of-two test, four
in the wide row of large logits.

Haiku fails a third way, the same way it failed the CUDA task: six of its eight
failures call Metal functions that do not exist — `simdgroup_max`,
`atomic_fetch_max` on a float, `atomic_compare_exchange_weak_explicit` with
argument types nothing matches. One draw wrote

```metal
for (uint step = (tpg + 1) / 2; step > 0; step = (step + 1) / 2)
```

which never reaches zero. The GPU hung, and the task's own 300 second limit
killed it: the first `timeout` this repository has recorded, and a reminder that
the time limit is a safety property on this tier rather than a formality. Its
tenth draw came back `adapter_error` — the API returned `overloaded` — so it
measured nothing, and the rate is over nine draws rather than ten.

### What that means for the task's shelf life

The honest read is uncomfortable in the way L30 predicted. The signal separating
these three models right now is dominated by one reserved word. A model that
learns it — and any model that sees this repository will — moves several draws up
without becoming better at threadgroup reductions. The rest of the difficulty is
still there and still real: the group size the kernel does not choose, the idle
lanes, the fold over a range that is not a power of two, the shift. But the
number this task publishes today is not the number it will publish next year, and
the reason is contamination wearing a lexical coat.

## The three v1 tasks that had been waiting

`attention_causal_mask`, `sharded_dataloader` and `quantization_error_bounds`
were solved by all three models on every draw and `V2_DESIGN.md` §5 had already
concluded they belong in `warmup`. They are there now, calibrated out of draws
that were already published rather than out of new ones: six for Opus and Haiku
(the five-draw sweep of 2026-08-09 plus the single sweep of 2026-08-03), five
for Sonnet.

That needed one change to the machinery. `tools/check_calibration.py` read
`calibration/` only, so the alternative was copying seventeen results files that
are already checked in under `leaderboard/` — the same evidence in the
repository twice, with two chances to drift. It now walks both directories, and
`tests/test_runner.py` re-derives every block from both.

`v1` is five laptop tasks today and was eight when every published row was
measured. The leaderboard page says so, in the section the rows sit in, because
a set that changes shape after it was published is exactly the kind of thing a
reader has to be told rather than left to notice.
