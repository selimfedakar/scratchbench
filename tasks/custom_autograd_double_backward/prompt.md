# A Function the framework can differentiate twice

Fill in `scaled_swish.py`. torch only, on the CPU:

```python
class ScaledSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, alpha): ...
    @staticmethod
    def backward(ctx, grad_output): ...

def scaled_swish(x, w, alpha) -> Tensor
```

The operation is `w * x * sigmoid(alpha * x)`: a swish gate with an
adjustable slope, scaled by a weight. Written as an ordinary expression it would
be one line and the framework would differentiate it for you. Written as an
`autograd.Function` it is yours to differentiate, and `scaled_swish` is the
helper the rest of the world calls, so it has to go through the class rather
than around it.

That is the whole constraint, and it is the point of the exercise: a
hand-written backward is what a fused kernel, a memory-saving recomputation and
a custom operator all eventually need, and the framework only checks the parts
of the contract it can see.

## What the two arguments are

`x` and `w` are tensors that broadcast against each other, in either direction —
a weight per column against a batch of rows, a single number against everything,
a batch against a weight that has a one where the batch has a length. The result
has the broadcast shape and the dtype it was given, which is `float32` or
`float64`. `alpha` is a Python number, not a tensor.

Either tensor may or may not ask for a gradient, and the pair changes from call
to call. Return gradients for the ones that asked and nothing for the ones that
did not; `alpha` never asks, because a number has nothing to accumulate into.
The backward returns one thing for each argument the forward took, in that
order, and the gradient of an input has the shape of that input. What arrives at
the backward is a tensor of the output's shape; nothing is promised about how it
is laid out in memory.

Nothing you are handed is yours to modify, and the values may be large enough
that the gate saturates. Saturation is not an error and nothing may come back
non-finite because of it.

## The part that is easy to get half right

The gradient this returns will itself be differentiated. Training a model whose
loss involves a gradient — a penalty on the input gradient, an unrolled inner
step, anything second-order — differentiates the backward pass, and a backward
that produces the right numbers is not automatically a backward that can be
differentiated in turn.

So the second derivatives have to come out right, not just the first. What that
asks of the code is a property rather than an extra formula: the backward has to
be written the way the forward would have been, out of operations the framework
can still see, on the tensors it handed you. Anything that reaches around the
framework to get at a tensor's values will give you first derivatives that are
exactly right and second derivatives that are quietly missing terms.

The same applies to the sigmoid. Whatever the forward computed, it computed with
differentiation switched off — that is what a `Function`'s forward is for — so a
value carried across from it arrives as a constant.

## What the backward reads

Hand the tensors the backward needs to the framework rather than keeping your
own references to them. Between a forward and the backward that follows it a
caller can modify a tensor in place, and the gradient computed from the modified
one is wrong. The framework detects exactly that and raises, but only for
tensors it was given to hold.

## Conventions

- torch only, CPU, no other dependencies.
- The gradients are exact rather than approximate: at `float64` they should
  agree with the analytic answer to many digits rather than a few.
- `scaled_swish(x, w, alpha)` returns the tensor.
