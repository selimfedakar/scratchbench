# A KV-cache that changes the speed and nothing else

Fill in `kv_cache.py`. numpy only:

```python
class CachedAttention:
    def __init__(self, wq, wk, wv, wo, n_heads: int, rope_base: float = 10_000.0)
    def forward(self, x, cache=None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]
```

One causal multi-head attention layer with rotary position embeddings, able to
run over a whole sequence at once or over one token at a time while carrying
the keys and values forward.

The contract is equivalence. Feeding a sequence to `forward` in chunks of any
size, passing each returned cache into the next call, must produce exactly what
one call over the whole sequence produces — to floating-point noise, not to
"close enough". The cache is an optimisation, and an optimisation that changes
the answer is a bug that no shape check and no loss curve will find for you.

## The layer

`forward` returns `(output, cache)`. The output has shape
`(batch, seq_len, d_model)`. The cache is the pair `(keys, values)`, each of
shape `(batch, n_heads, total_keys, head_dim)`, where `total_keys` counts
everything seen so far — the history that came in plus the chunk just
processed. Passing `cache=None` starts a fresh sequence at position zero.

The steps, in order:

1. Project with `x @ wq`, `x @ wk`, `x @ wv`. All four weight matrices are
   `(d_model, d_model)`, there are no biases, and `head_dim` is
   `d_model // n_heads`.
2. Split into heads by cutting the projection into `n_heads` contiguous blocks
   of `head_dim` columns: head `h` owns columns `h * head_dim` through
   `(h + 1) * head_dim`.
3. Apply the rotation below to the queries and the keys — not to the values.
4. Append the new keys and values to the ones from the cache.
5. Attend: scaled dot product with the scale one over the square root of
   `head_dim`, causal masking, softmax over the keys, weighted sum of the
   values.
6. Put the heads back together in the order they were split, and finish with
   `@ wo`.

## The rotation

RoPE reads each head vector as `head_dim // 2` pairs and turns every pair by an
angle proportional to the token's position. `head_dim` is even. For pair index
`i` counting from zero and absolute position `p`:

```
theta = p * rope_base ** (-i / (head_dim / 2))
```

The two coordinates of pair `i` are the entries at `i` and at
`i + head_dim // 2` — the vector is split into halves, not into neighbouring
couples. Writing `a` for the first and `b` for the second, the rotation sends
them to `a * cos(theta) - b * sin(theta)` and `b * cos(theta) + a * sin(theta)`.

`p` is the position in the **full sequence**, not in the chunk. This is the
whole reason `forward` needs the cache before it can rotate anything.

## Conventions

- Causal means a query may read the keys at or before its own position. Within
  a chunk that arrives after a history, the chunk's first query sits directly
  after the last cached key.
- The cache handed to `forward` must come back unmodified. A caller may keep an
  earlier cache and continue from it more than once; that is how beam search
  and speculative decoding work.
- Inputs may be `float32` or `float64`. The output and the cache are `float64`,
  and `x` is not modified.
- Every dimension is at least one. There is no padding, no dropout, no bias,
  and no attention over anything but this one layer.
- No `torch`, no `scipy`. numpy is enough.
