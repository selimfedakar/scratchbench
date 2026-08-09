"""Reference solution — the verification half of speculative decoding.

A small draft model proposes several tokens in the time the large target model
needs for one, and the large model scores all of them in a single forward pass.
The whole scheme is only worth anything if the tokens that come out are
distributed exactly as the target model would have produced them on its own. Not
approximately, not usually: exactly, or the speedup was bought by quietly
sampling from a different model than the one the user asked for.

Two quantities carry that guarantee, and both are easy to get subtly wrong.

The first is the acceptance rule. A proposal `x` was drawn from `q`, and we want
the survivor to look as though it came from `p`. Accepting `x` with probability
`a(x)` leaves `q(x) * a(x)` mass on that token, so `a(x)` can never exceed
`p(x) / q(x)` without overshooting, and it can never exceed one because it is a
probability. `min(1, p(x) / q(x))` is therefore the largest acceptance rule the
guarantee permits, and any smaller one throws away speedup for nothing. That
maximality is not decoration — "always reject and sample from the target" also
emits exactly the right distribution, and is worth nothing at all.

The second is the correction. After acceptance, the token has mass
`min(p(x), q(x))` — short of `p(x)` by exactly the positive part `p(x) - q(x)`
where the target wanted more than the draft offered. So the rejection branch has
to sample from the *positive residual* `max(0, p - q)`, renormalised, and that
distribution puts zero mass on the token that was just rejected: rejection only
happens where `p < q`, and the residual is zero there. Its total mass is
`1 - sum(min(p, q))`, which is exactly the rejection probability, and that is
what makes the two branches add up to `p` on the nose.

Three wrong versions of that second quantity all look reasonable and none of
them is right. Sampling from the target directly on rejection leaves
`min(p, q) + (1 - alpha) * p`, which is not `p`. Renormalising `p - q` without
taking the positive part is renormalising something with negative entries.
Dividing the residual by anything other than its own sum leaves a distribution
that does not sum to one, or does so only when the two models happen to agree.
Every one of them produces plausible tokens at a plausible acceptance rate, and
the only thing that catches them is asking what distribution the output actually
has.

Everything after that is bookkeeping. The first rejection ends the round,
because every token after it was drafted conditioned on a prefix that no longer
exists. If nothing is rejected the target's own distribution for the position
after the proposal is already available — it was computed in the same forward
pass and would otherwise be thrown away — so one extra token comes free, which
is why a round of `n` proposals can emit `n + 1` tokens.
"""

from __future__ import annotations

import numpy as np


def _sample(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    """One draw from a distribution over the vocabulary."""
    return int(np.searchsorted(np.cumsum(probabilities), rng.random()))


def verify_draft(draft_tokens, draft_probs, target_probs, rng) -> np.ndarray:
    """Accept a prefix of the proposal and emit one corrected token after it."""
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64)
    draft_probs = np.asarray(draft_probs, dtype=np.float64)
    target_probs = np.asarray(target_probs, dtype=np.float64)

    emitted: list[int] = []
    for position, token in enumerate(draft_tokens):
        proposed = draft_probs[position, token]
        wanted = target_probs[position, token]
        # min(1, wanted / proposed) is the largest acceptance probability that
        # still leaves the survivor distributed as the target.
        if rng.random() * proposed < wanted:
            emitted.append(int(token))
            continue

        # The target wanted more of some token than the draft was offering.
        # That deficit, and nothing else, is what the replacement comes from —
        # note it is zero at the token just rejected, which is why the rejected
        # token can never come back.
        residual = np.maximum(target_probs[position] - draft_probs[position], 0.0)
        emitted.append(_sample(residual / residual.sum(), rng))
        return np.array(emitted, dtype=np.int64)

    # Nothing was rejected, so the target's distribution for the position after
    # the proposal is still valid and already paid for.
    emitted.append(_sample(target_probs[len(draft_tokens)], rng))
    return np.array(emitted, dtype=np.int64)
