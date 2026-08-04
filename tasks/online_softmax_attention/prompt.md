# Attention, one block of keys at a time

Fill in `tiled_attention.py`. numpy only:

```python
def combine_states(left, right) -> tuple[np.ndarray, np.ndarray, np.ndarray]
def attention_tiled(q, k, v, block_size: int, causal: bool = False) -> tuple[np.ndarray, np.ndarray]
```

Softmax looks like the one thing in attention that cannot be streamed. Every
weight is divided by a sum over every key, so the normaliser has to exist before
any single output does — which is why ordinary attention builds the whole
`n_queries` by `n_keys` score matrix first, and why long contexts were expensive
long before anyone wrote a fused kernel.

They can be streamed, and the trick is to hold a partial answer that later
blocks are allowed to correct. Walk the keys in blocks of `block_size`, carrying
three running quantities per query: the largest score seen so far, the sum of
the exponentials measured against that largest score, and the same weighted sum
applied to the values. When a block turns up holding a bigger score, everything
already accumulated was measured against a maximum that is no longer the
maximum, and it has to be brought onto the new one before the block can be added
to it.

That correction is the task. Everything else here is bookkeeping around it.

## The state

For a group of keys and a given query, write `s` for the scaled scores of that
group — a masked key has a score of negative infinity. The **state** of the
group is the triple:

- `m`, the largest `s` in the group;
- `l`, the sum of `exp(s - m)` over the group;
- `o`, the sum of `exp(s - m) * v` over the group, one vector per query.

An **empty group** — no keys, or a group in which every key is masked away from
this query — has `m` of negative infinity, `l` of zero, and `o` of zeros. A
group in which every key is masked behaves in every respect like a group with no
keys in it.

Shapes: `m` and `l` are `(batch, heads, n_queries)`, and `o` is
`(batch, heads, n_queries, head_dim_v)`. All three are `float64`.

## combine_states

`combine_states(left, right)` takes two states, each a tuple `(m, l, o)`, and
returns the state that the union of the two groups would have. Both sides
describe the same queries, so the shapes match and the result keeps them.

It is the arithmetic the whole task rests on. Note what it has to do with the
side whose maximum loses: that side's `l` and `o` are both sums of terms
measured against a maximum that is about to change, and both of them are
rescaled, by the same factor, or the result stops being a state at all. When one
of the sides is empty the answer is the other side, untouched — including when
both are empty, which is the case where the obvious expression for the rescaling
factor is negative infinity minus negative infinity, and that is not a number.

The merge is associative: combining three states left-to-right gives the same
answer as combining the last two first. That property is what makes the
algorithm useful, since it lets separate machines each reduce a slice of the
keys and merge their partial states afterwards. It returns `float64` whatever it
was given, and it leaves the arrays it was handed alone.

## attention_tiled

`q` has shape `(batch, heads, n_queries, head_dim)`; `k` and `v` have shape
`(batch, heads, n_keys, head_dim)` and `(batch, heads, n_keys, head_dim_v)`.
`q` and `k` share their last dimension, `v` need not.

Scores are dot products divided by the square root of the query/key head
dimension. Each batch entry and each head is scored independently of the others.
The function returns the pair `(output, logsumexp)`:

- `output`, shape `(batch, heads, n_queries, head_dim_v)`: the softmax-weighted
  average of the values, with weights that are non-negative and sum to one over
  the keys each query is allowed to read.
- `logsumexp`, shape `(batch, heads, n_queries)`: the logarithm of the sum of
  the exponentials of that query's scaled scores. It is the normaliser the
  output was divided by, in log space, and it is returned because a backward
  pass needs it and because recomputing it later would mean building the matrix
  this function exists to avoid. A query's attention weight on a key is
  therefore `exp(score - logsumexp)`.

`block_size` is at least one, it need not divide `n_keys` — the last block is
short whenever it does not — and it may be larger than `n_keys`, in which case
there is one block. It is a performance knob and nothing else: the numbers
coming out must not depend on it.

`causal` defaults to false. When it is true, a query may read only the keys at
or before its own position, and the queries are the **last** `n_queries`
positions of the key range, the same convention a KV-cache forces and the one
the rest of this benchmark uses. Two queries against five keys are therefore at
positions three and four. Combined with blocking, this produces blocks that lie
entirely in the future of the queries they were just scored against, and a block
like that must contribute nothing rather than poison the accumulator.

## Conventions

- `n_queries` never exceeds `n_keys`, and every dimension is at least one, so
  every query has at least one key it may read. There is no padding mask.
- Inputs may be `float32` or `float64`. Both returned arrays are `float64`, and
  `q`, `k` and `v` are not modified.
- Scores can be far from zero in either direction — tens of thousands is
  ordinary for a model late in training — and neither the output nor the
  logsumexp may overflow, underflow or otherwise lose the answer there.
- No `torch`, no `scipy`. numpy is enough.
