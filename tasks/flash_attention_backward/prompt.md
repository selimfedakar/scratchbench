# The backward pass, one block of keys at a time

Fill in `flash_backward.py`. torch only:

```python
def key_block_gradients(q, k_block, v_block, o, lse, do,
                        key_offset: int, query_offset: int,
                        causal: bool = False) -> tuple[Tensor, Tensor, Tensor]

def flash_attention_backward(q, k, v, o, lse, do, block_size: int,
                             causal: bool = False) -> tuple[Tensor, Tensor, Tensor]
```

Attention became affordable on long sequences when someone noticed that the
softmax normaliser can be carried as a running quantity, so the forward pass
never has to hold a matrix quadratic in the sequence length. The backward pass
has to keep that promise or the memory comes straight back, which means it
cannot differentiate the thing it is differentiating: it has to be derived, and
then evaluated a block of keys at a time, recomputing what it needs from
quantities the forward pass already returned.

That is the shape of this task. `key_block_gradients` is handed exactly one
block of keys and never the rest of them, which is what makes the constraint
real rather than a request, and `flash_attention_backward` walks the blocks and
assembles the answer.

## The forward pass you are differentiating

`q` has shape `(batch, heads, n_queries, head_dim)`; `k` and `v` have shape
`(batch, heads, n_keys, head_dim)` and `(batch, heads, n_keys, head_dim_v)`.
Scores are dot products of queries with keys divided by the square root of the
query/key head dimension, and each batch entry and head is scored on its own.
`o`, shape `(batch, heads, n_queries, head_dim_v)`, is the softmax-weighted
average of the values over the keys each query is allowed to read, and `lse`,
shape `(batch, heads, n_queries)`, is the logarithm of the sum of the
exponentials of that query's scaled scores — so a query's weight on a key is
`exp(score - lse)`, and the pair `(o, lse)` is exactly what a blocked forward
pass hands over. Both were computed from the same `q`, `k` and `v` you are given.

`do` has the same shape as `o` and is the gradient of the scalar loss with
respect to `o`. You are asked for the gradient of that same loss with respect to
`q`, `k` and `v`, returned in that order.

Everything is `float64` and nothing requires grad. Nothing you return may carry
a gradient history either, and none of the tensors you are handed is yours to
modify. `torch.autograd` is no help here and there is no version of it that
would be: `key_block_gradients` holds one block of keys and cannot rebuild the
forward pass that produced `o` out of it. Deriving the thing is the task.

## Masking and where things sit

`causal` defaults to false. When it is true, a query may read only the keys at
or before its own position, and the queries are the **last** `n_queries`
positions of the key range, which is the convention a KV-cache forces and the
one the rest of this benchmark uses. Two queries against five keys are therefore
at positions three and four.

`key_block_gradients` cannot work that out for itself, because it is looking at
a slice: `key_offset` is the position of `k_block[..., 0, :]` in the full key
range, and `query_offset` is the position of `q[..., 0, :]` in that same range.
A block can lie beyond the reach of some of the queries it was just scored
against, in which case it contributes nothing to them, and "nothing" here means
zeros rather than something that is not a number.

## The two functions

`key_block_gradients` returns a triple. The first element has the shape of `q`
and is that block's **contribution** to the query gradient — the blocks add up
to it, they do not each compute it. The second and third have the shapes of
`k_block` and `v_block` and are the finished gradients for those keys and
values, since a key appears in exactly one block.

`flash_attention_backward` takes `block_size` keys at a time, starting at the
first key, and returns `(dq, dk, dv)` with the shapes of `q`, `k` and `v`.
`block_size` is at least one, need not divide `n_keys` — the last block is short
whenever it does not — and may be larger than `n_keys`, in which case there is a
single block. It is a memory knob and nothing else: the numbers coming out must
not depend on it.

## Conventions

- torch only, on the CPU.
- `n_queries` never exceeds `n_keys`, and every dimension is at least one, so
  every query has at least one key it may read. There is no padding mask.
- `head_dim_v` need not equal `head_dim`.
- The gradients are exact, not approximate: whatever this returns should agree
  with the analytic answer to the precision `float64` allows, which is many
  digits rather than a few.
