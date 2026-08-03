# Softmax that does not fall over

Fill in `stable_softmax.py`. Three functions, numpy only:

```python
def logsumexp(x, axis: int = -1, keepdims: bool = False) -> np.ndarray
def softmax(x, axis: int = -1) -> np.ndarray
def log_softmax(x, axis: int = -1) -> np.ndarray
```

`logsumexp` returns the logarithm of the sum of the exponentials along `axis`.
`softmax` returns the probabilities that sum to one along `axis`. `log_softmax`
returns their logarithms.

These are the three functions every attention implementation and every
cross-entropy loss ends up calling, and they are called on real logits, which
means they are called on numbers that a direct `exp()` cannot represent. A
transformer late in training will hand you a logit of ten thousand. Softmax is
mathematically invariant to a constant shift along its axis and completely
undone by overflow, and the gap between those two facts is the whole task.

The other half is masking. Attention masks arrive as additive negative
infinity, so `-inf` is a normal input value here, not an error: a masked
position must come out with probability exactly zero and must not disturb the
probabilities of the positions around it. When a whole slice along `axis` is
`-inf` — every key masked, which happens on padded batches — the answer is
defined as all zeros from `softmax`, `-inf` from `log_softmax`, and `-inf` from
`logsumexp`. That slice must not produce NaN.

`log_softmax` must not be a wrapper around `log(softmax(x))`. The point of
having it is that it stays accurate for entries far below the maximum: a
position eight hundred nats down has a probability that is exactly zero in
float64, but its log-probability is about −800, and that number has to survive.

## Conventions

- `x` is array-like: a numpy array of any dtype, or a nested list. It has at
  least one dimension and at least one element along `axis`.
- `axis` is a single integer and may be negative.
- All three functions return `float64` arrays whatever the input dtype was, and
  none of them modify the input.
- `keepdims` on `logsumexp` behaves the way it does everywhere else in numpy:
  false drops the reduced axis, true leaves it in place with length one.
- You may assume the input contains no NaN and no `+inf`. `-inf` is expected.
- Accuracy: the finite results are compared against exact values to within a
  small multiple of float64 precision, so shifting by the maximum is not merely
  advisable, it is the only thing that gets you there.
