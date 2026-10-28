# 20 — The contract, tested on the easy inputs

Two of the review's findings land on the same two tasks — `chunked_batchnorm_reduction`
and `chunked_batchnorm_backward` — and on the same uncomfortable fact: both have
a test for the promise that was broken, and both tests exercised the promise on
an input that could not break it.

## F04 — the no-grad test used detached inputs

Each reference promises that nothing it returns carries a gradient history, and
each has `test_the_gradients_carry_no_autograd_history` to enforce it. The test
builds its case with detached tensors. So it checks whether the reference *adds*
history and never whether it *passes history through*. Hand in an input that
requires a gradient:

```
{"probe": "no_grad_contract", "task": "chunked_batchnorm_reduction",
 "requires_grad": [true, true, true], "grad_fn": ["MulBackward0", ...]}
```

Every returned gradient carried the input's `grad_fn`. The fix is the whole of
the contract rather than a line of it: every function in both references runs
under `@torch.no_grad()`. A new test hands in grad-requiring tensors and checks
the result and the inter-pass buffer come back clean, and that the inputs are
untouched. It has teeth — strip the decorators and it fails:

```
$ grep -v '@torch.no_grad()' reference.py > mutant.py && pytest -k no_grad_contract
E   assert True is False   # grad_fn=<MulBackward0>
1 failed
```

## F05 — the variance tolerance was swept where the variance is slack

The reduction recovers the variance from raw moments, `sum(x*x)/M - mean*mean`,
and the tolerance is swept over inputs whose variance is order one — where the
cancellation costs three digits and the tolerance sits four orders above it.
Cluster the inputs at `1 ± 1e-9` and the raw variance comes out **negative**,
`-2.2e-16` against a true `1.5e-18`, and with an `eps` small enough not to lift
it, `sqrt` returns NaN. The prompt had only ever said "`eps` is positive".

This one is a contract gap, not a reference bug, and the honest fix is to state
the contract. The prompt now says what the single-reduction interface can keep:
inputs of order one and an `eps` of conventional size, over which `var + eps`
stays positive — and explicitly not a variance driven below the arithmetic's
floor by a smaller `eps`. A new test confirms the finite result on near-constant
inputs and claims only finiteness, because `1/sqrt(var + eps)` amplifies the
cancellation and accuracy there is ill-conditioning the "order one" clause
already scopes out. The clamp I did not add is the point: `var.clamp_min(0)`
would turn the NaN into a quiet wrong number, which is the trade this repository
exists not to make.

## Verified

```
$ python -m runner.cli validate --tasks chunked_batchnorm_reduction,chunked_batchnorm_backward
chunked_batchnorm_reduction  59 passed  59 failed  ok
chunked_batchnorm_backward   45 passed  45 failed  ok
2 task(s) validated: reference passes, starter fails cleanly.

$ python tools/mutate_v2_tasks.py --task chunked_batchnorm_reduction
chunked_batchnorm_reduction: all 16 mutants behaved as expected
$ python tools/mutate_v2_tasks.py --task chunked_batchnorm_backward
chunked_batchnorm_backward: all 16 mutants behaved as expected
```

Both tasks are **version 2**. Their version-1 calibration is kept, labelled as
belonging to version 1, and a re-sweep is pending — it would have graded against
the old tests, and presenting it as a version-2 measurement is the thing §5.1 and
F07 are about. Total hidden tests 805 to 808. The lesson is `docs/LESSONS.md` L51:
every promise had a test; not every test had an input with teeth.
