# Causal attention over a padded batch

Fill in `causal_attention.py`. Two functions, numpy only:

```python
def causal_mask(n_queries: int, n_keys: int) -> np.ndarray
def attention(q, k, v, key_padding_mask=None) -> np.ndarray
```

`attention` is scaled dot-product attention: score every query against every
key, divide by the square root of the head dimension, turn the scores into a
distribution with a softmax over the keys, and take that weighted average of the
values. It is always causal — there is no flag to turn it off.

`causal_mask` is the additive mask that makes it causal, and it is returned
separately because getting it right is most of the work. It has shape
`(n_queries, n_keys)`, holds `0.0` at every position a query is allowed to
read, and negative infinity everywhere else. Adding it to the scores before the
softmax drives the forbidden positions to zero probability.

A query may read the keys at or before its own position. When the two lengths
differ, the queries are the **last** `n_queries` positions of the key range —
that is the situation a KV-cache creates, where one fresh query arrives against
a long history, and the mask has to place it at the end of that history rather
than at the start. Attention over a trailing slice of the queries therefore
gives exactly the same answer as attention over all of them, read back on the
same rows.

`key_padding_mask` handles batches of unequal sequence lengths. It is a boolean
array of shape `(batch, n_keys)` where `True` marks a real token and `False`
marks filler. Filler keys receive no attention from any query of that sequence
and from any head — the same padding applies across all heads. This composes
with causality rather than replacing it: a key must be both visible and real.

Between the two of them, causality and padding can leave a query with nothing
at all to read: filler at the front of a sequence does exactly that to the
first query. The row of scores is then entirely negative infinity, and the
softmax you would normally write returns NaN for it, which spreads through the
rest of the network on the next matrix multiply. Define that query's output to
be all zeros.

## Conventions

- `q` has shape `(batch, heads, n_queries, head_dim)`; `k` and `v` have shape
  `(batch, heads, n_keys, head_dim_k)` and `(batch, heads, n_keys, head_dim_v)`.
  `q` and `k` share their last dimension; `v` need not.
- The output has shape `(batch, heads, n_queries, head_dim_v)`.
- `n_queries` never exceeds `n_keys`. Every dimension is at least one.
- The scale is one over the square root of the query/key head dimension.
- Inputs may be `float32` or `float64`; both functions return `float64`, and
  neither modifies anything it was given.
- `key_padding_mask` defaults to `None`, which means every key is real.
- The scores contain no NaN and no `+inf` before masking, so the only
  non-finite value you have to reason about is the `-inf` you introduce.
- No `torch`, no `scipy`. numpy is enough.
