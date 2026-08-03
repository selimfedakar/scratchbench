# 01 — The first five tasks (2026-07-26)

## What this step was

Five tasks, spanning four of the six categories, written in the order the task
contract insists on: `reference/` first, then `hidden_tests/`, then `prompt.md`,
then `meta.yaml`.

That order is not bureaucracy. Writing the reference first is what makes the
unstated conventions visible — I cannot write the solution without deciding
whether the output is float32 or float64, whether a fully masked row is zeros
or NaN, whether the queries sit at the start or the end of the key range. Every
one of those decisions has to end up in the prompt, and I only know the full
list once the code exists.

| Task | Category | Difficulty | What it isolates |
|---|---|---|---|
| `softmax_stability` | numerics | 1 | shifting by the max; keeping log-probabilities below the underflow floor; `-inf` masks as data |
| `bpe_merge_order` | tokenization | 2 | encoding follows the learned merge order, not frequency and not position |
| `attention_causal_mask` | attention | 2 | mask direction, the `-inf` convention, padding composed with causality |
| `kv_cache_equivalence` | attention | 3 | incremental decode identical to a full pass: rotation offset and mask alignment |
| `grad_accumulation` | training | 3 | four micro-batches producing exactly the update of one large batch |

## The technology, and why each piece

**numpy for four of the five.** No framework does the thinking, the failure
modes stay visible, and a laptop runs the whole thing in under a second.
`softmax_stability`, `attention_causal_mask` and `kv_cache_equivalence` are
pure numpy; `bpe_merge_order` is the standard library alone, because a
tokenizer that reaches for a tokenizer library is answering a different
question.

**PyTorch for `grad_accumulation`, on CPU, in float64.** Gradient equivalence
cannot be measured without autograd, and asking for a hand-written backward
pass would measure a different skill. float64 is what lets the test assert
"identical to the full-batch step" at 1e-12 instead of hedging at 1e-5. The
whole task, twenty-seven tests including Adam and momentum, runs in 3.4
seconds — the five-minute rule is not under any pressure here.

**Rotary position embeddings, spelled out in the prompt.** `kv_cache_equivalence`
needs RoPE because the offset bug lives in it, but inventing RoPE is a separate
task, so the prompt pins the convention exactly: halves rather than adjacent
pairs, `base ** (-i / (head_dim / 2))`, position taken from the full sequence.
What is left to get right is the thing being measured — that the position comes
from the length of the cache and not from the index inside the chunk.

## The rule I kept checking against

> If a hidden test can fail for a reason the prompt never states, the task is
> broken.

After writing each test file I walked every assertion and pointed at the
sentence in `prompt.md` that licenses it. That pass changed the prompts several
times: the fully masked row returning zeros, `float64` output regardless of
input dtype, the merge list being left unmodified, the returned loss being
measured before the step rather than after. All of those were conventions I had
decided in the reference and nearly failed to write down.

It also deleted a test. `softmax_stability` had one that asserted
`np.exp(1e4)` is infinity — true, and a decent comment, but it never calls the
solution, so it passed against an empty starter. A test that cannot fail is
noise in a binary score. It became a comment.

## Verified

Every task was run both ways: the reference against the hidden tests, and the
untouched starter against the same tests. Both outputs pasted below. This is
what `CLAUDE.md` calls L2 for a task.

| Task | Reference | Untouched starter |
|---|---|---|
| `softmax_stability` | 29 passed in 0.22s | 29 failed in 0.32s |
| `bpe_merge_order` | 30 passed in 0.02s | 30 failed in 0.09s |
| `attention_causal_mask` | 26 passed in 0.15s | 26 failed in 0.35s |
| `kv_cache_equivalence` | 21 passed in 0.18s | 21 failed in 0.35s |
| `grad_accumulation` | 27 passed in 3.36s | 27 failed in 1.61s |

The starter failures are `NotImplementedError` raised from the stub bodies —
not import errors, not collection errors. That distinction is the difference
between "the model got it wrong" and "the task was broken", and they score the
same, so it gets checked rather than assumed.

## Then I tried to break the tasks

Passing tests prove the reference is right. They do not prove the tests would
catch a plausible wrong answer, and a task that cannot fail is worse than no
task at all — it inflates every score published against it.

So I mutated the two hardest references into the wrong versions a real
implementation would produce, and ran the hidden tests against each.

`kv_cache_equivalence`:

| Mutant | Result |
|---|---|
| rotate at the chunk index instead of the absolute position | 8 failed, 13 passed |
| skip the rotation entirely | 4 failed, 17 passed |
| anchor the mask at the start of the key range | 8 failed, 13 passed |
| transpose the mask so a token reads the future | 14 failed, 7 passed |

`grad_accumulation`:

| Mutant | Result |
|---|---|
| weight each micro-batch by one over the chunk count | 11 failed, 16 passed |
| step the optimiser once per micro-batch | 20 failed, 7 passed |
| never clear the leftover gradients | 1 failed, 26 passed |
| sum the micro-batch losses unscaled | 17 failed, 10 passed |

Scoring is binary, so every one of those is a failed task. The "skip the
rotation entirely" mutant is the interesting one: it is perfectly
self-consistent between its own cached and uncached paths, so every equivalence
test in the file passes it. It is caught only by the test that recomputes the
layer position by position with explicit trigonometry — which is why that test
is in there.

## Next

The runner. Until it exists, all of the above is a pile of directories that I
verified by hand with `cp` and `cd`, which is not a benchmark.
