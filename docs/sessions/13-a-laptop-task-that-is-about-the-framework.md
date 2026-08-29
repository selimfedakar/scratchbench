# 13 — A laptop task that is about the framework, not the derivation

Two things happened in this session. The headline question that sections 09
through 12 kept walking past got decided, and the first laptop task written
against that decision got built.

## The gap that had to be named first

`v2` had three members and one of them was on the laptop tier:

```
$ python -m runner.cli list --set v2 --tier laptop
flash_attention_backward    attention  laptop  5  180s  torch  v2
```

The leaderboard's headline rate is the laptop tier by construction —
`HEADLINE_TIER` in `runner/report.py` — so publishing `v2` in that shape would
have published a headline computed over a single task. That is not a small
reporting wrinkle. It is the accumulated consequence of the accelerated tier
turning out to be the only one that discriminates, and it had never been written
down in one place.

## The decision, and the option that was tempting

`V2_DESIGN.md` §4b now carries it. `v2` does not ship until it has at least
three laptop tasks; the tier code does not change; the accelerated members are
reported beside the headline with the hardware each needs named.

The tempting alternative was to re-cut the tiers into `portable | apple | cuda`
and let the headline span the first two, on the grounds that a Mac is a laptop.
It costs no new tasks and it would have produced a three-member headline the
same afternoon. It also replaces "reproducible by anyone who clones this" with
"reproducible by anyone who owns the right laptop" while the README goes on
saying the first thing. Redefining the set to fit the inventory is L21 with a
different noun.

What would reopen it is written down with the decision: two independent attempts
at a discriminating laptop task, both cleared by the top of the field. That
makes the laptop tier's saturation a measurement instead of a suspicion.

## The task: `custom_autograd_double_backward`

§2.0 says the difficulty has to be a fact about a tool, and an obscure one.
Three laptop tasks built against the reasoning properties in §2.1 through §2.4
are in `warmup` proving the point. So this one is not about deriving anything —
the derivative of `w * x * sigmoid(alpha * x)` is undergraduate calculus. It is
about the `autograd.Function` contract:

- The backward is written by hand, and **the backward will itself be
  differentiated**. A backward built out of operations the framework can still
  see gives correct second derivatives; one that reaches around it with
  `.detach()`, `.data`, `no_grad` or `@once_differentiable` gives first
  derivatives that are exactly right and second derivatives that are quietly
  missing terms.
- The sigmoid the forward computed is a constant. A `Function`'s forward runs
  with differentiation switched off, so a value carried across from it arrives
  with no graph, and reusing it is the same bug wearing a different shape.
- Tensors the backward needs go through `save_for_backward`, because that is
  what gives the framework a version to check. An implementation holding its own
  reference computes a gradient from a tensor the caller modified in place, and
  nothing anywhere raises.
- Gradients come back one per argument, including a `None` for the argument that
  was a Python number, and nothing for an input that did not ask.

Everything on that list is exactly checkable against autograd on the plain
expression, in float64, on a CPU, in under two seconds.

## Both halves of L2

```
$ python -m runner.cli validate --tasks custom_autograd_double_backward
custom_autograd_double_backward  116 passed  116 failed  2.35s  ok

1 task(s) validated: reference passes, starter fails cleanly.
```

The first version of the test file had one test that passed against the
untouched starter — a structural `issubclass` check the starter satisfies
because the starter is handed the skeleton. That is L15, and the fix was not to
delete the check but to fold it into the test that already exercises the class,
where it can still fail against a solution that replaces the Function with the
plain expression.

## The mutants, and the one that changed the task

Twelve, in `tools/mutate_v2_tasks.py`. Nine caught, three expected to survive.

Two of the three survivors are the same fact and it is not one I knew when I
started: **torch sums a broadcast gradient down to its input's shape itself**,
silently, bit-identically to doing it by hand. I had built the reduction into
the task as one of its mechanisms, written `_reduce_to`, covered seven shape
pairs broadcasting both directions, and written two mutants that drop the
reduction. Both passed all 116 tests. Twenty lines outside the repository
answered the question the survivors raised:

```
torch 2.8.0
x.grad shape (1, 5) expected (1, 5)
warnings: []
matches autograd: True True
scalar-for-vector rejected: Function BadBackward returned an invalid gradient
at index 0 - got [] but expected shape compatible with [3]
```

So the task measures two mechanisms, not three, and `prompt.md` — which had not
been written yet, because the writing order puts it last — promises nothing
about reducing anything. `docs/LESSONS.md` L36.

The third survivor ignores `ctx.needs_input_grad` and computes a gradient nobody
asked for. Autograd discards it, so the only cost is work, and no black-box test
sees wasted work. Same shape as L33 one tier over.

Holding an expected survivor meant teaching `tools/mutate_v2_tasks.py` the
two-way expectation `tools/mutate_metal_task.py` already had: a mutant may
declare `SURVIVES`, and the script fails if such a mutant is *caught*, not only
when one expected to be caught escapes.

## The calibration, and the refusal

Ten draws per model, thirty draws, **$1.37**.

| model | passed | draws |
|---|---:|---:|
| `claude-opus-5` | 10 | 10 |
| `claude-sonnet-5` | 10 | 10 |
| `claude-haiku-4-5` | 0 | 10 |

The top two entries are both at the ceiling, so the admission rule refuses the
task and it is `frozen_set: warmup`. That is the rule working. It is also the
first of the two attempts `V2_DESIGN.md` §4b names as the condition for
reopening the headline decision, which makes the next laptop task carry more
weight than this one did.

## The number that is worth more than the refusal

Haiku failed all ten draws and failed **the same seventeen tests of 116 every
time**:

```
FAILED test_scaled_swish.py::test_gradgradcheck
FAILED test_scaled_swish.py::test_second_derivative_in_x_matches_autograd
```

Two test functions, seventeen parametrisations, identical across draws one
through ten. Every other test passed in every draw: the forward, the first-order
gradients against autograd, the broadcasting, `needs_input_grad`, the return
arity, the version-counter check. A completely correct first-order backward, and
the second derivative missing.

The cause is the same in all ten, and it is mutant three:

```python
x, w, sigmoid_alpha_x = ctx.saved_tensors
...
grad_x = grad_output * w * sigmoid_alpha_x * (1 + alpha * x * one_minus_sigmoid)
```

The sigmoid is saved from the forward, where it was computed with
differentiation switched off, so it enters the backward as a constant and takes
every term that should have flowed through it. One draw wrote the comment
*"For second-order derivatives to work, we must use tensor operations throughout
and not extract scalar values"* directly above the line that does it.

`tools/mutate_v2_tasks.py`'s third mutant for this task is that exact
substitution, written before any model was asked. This is the second time a
model has reproduced one of this repository's invented wrong implementations
verbatim — `flash_attention_backward` was the first — and it is the strongest
argument I have for writing mutants at all.

## What it means, and what §2.0 now says

The task measures what it was built to measure and does not measure at the top,
because the fact it is built on is *documented*: PyTorch has a page about double
backward in custom Functions. The Metal task's `half` has no page, because it is
not a topic. §2.0 now carries the question that separates the two —
*does the framework have a page about exactly this mistake?* — and
`docs/LESSONS.md` L37 is the uncomfortable version, which is that L30 already
told me this two sessions ago in my own words.

## One thing found on the way past

`tools/check_cost.py` defaulted to `leaderboard/` alone, while
`tools/check_calibration.py` walks `leaderboard/` **and** `calibration/`. So 95
checked-in results files carried costs that nothing re-derived. It now walks
both, which is the same argument as L11 and L20: one check, one place, no second
list.

```
before: 57 file(s) checked
after: 152 file(s) checked: every cost reproduces from its own tokens
```

Every one of the 95 reproduced, so this found no error. It closed a gap that
would have hidden one.

## Verified

| Claim | Level | Evidence |
|---|---|---|
| Reference passes, starter fails | L2 | `validate --tasks custom_autograd_double_backward` → `116 passed  116 failed  ok` |
| Every task still validates | L2 | `validate --tier all` → `13 task(s) validated`, 1 not checked here |
| Twelve mutants behave as written | L2 | `mutate_v2_tasks.py --task custom_autograd_double_backward` → `all 12 mutants behaved as expected` |
| Harness suite | L1 | `pytest -q` → `105 passed` |
| Calibration re-derives | L2 | `check_calibration.py` → `26 calibration entries re-derived from 152 draw(s)` |
| Every checked-in cost re-derives | L2 | `check_cost.py` → `152 file(s) checked` |
| Spend | L2 | `$1.3704`, summed from the thirty results files |

## Not verified

- **The task has never been run by anyone but me and three models.** No third
  party has cloned the repository and re-run it.
- **`v2` is unchanged by this session**: still three members, one of them on the
  laptop tier, still unpublished.
- **The sharpened §2.0 question has not been applied to anything yet.** It was
  derived from two data points, `metal_cross_entropy_kernel` and this task, and
  the next laptop candidate is its first real test.
