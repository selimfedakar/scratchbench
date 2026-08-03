"""Reference solution — incremental decoding that matches a full forward pass.

A KV-cache is an optimisation with a correctness contract: feeding a sequence
one token at a time, carrying the keys and values forward, must produce exactly
what a single pass over the whole sequence produces. When it does not, nothing
crashes. The model still generates fluent text, slightly worse text, and the
training metrics never mention it — which is why this equivalence is worth a
test rather than an inspection.

Two places lose the equivalence, and both are about position.

The rotation offset. RoPE encodes a token's position in the angle of its query
and key vectors. During a full pass, token four is rotated by four. During
incremental decoding it arrives alone, in a chunk whose local index is zero, and
rotating it by zero writes the wrong position into the only place position is
recorded. The chunk has to know where it starts, and the length of the cache is
what tells it.

The mask alignment. Inside a chunk of new tokens the mask is still causal, but
the chunk sits at the *end* of a key range that includes the whole history, so
query row i covers keys up to `past + i`. Aligning the mask to the start of the
key range instead lets a token read the history but not its own neighbours,
which is subtly wrong in exactly the way that survives a shape check.

The cache itself is returned, never mutated: the caller may legitimately hold on
to an earlier cache and branch from it, which is how beam search and speculative
decoding work.
"""

from __future__ import annotations

import numpy as np


def rope(x: np.ndarray, offset: int, base: float) -> np.ndarray:
    """Rotate pairs of coordinates by an angle proportional to position.

    `x` is (batch, heads, seq, head_dim) and the chunk starts at `offset` in
    the full sequence.
    """
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    half = head_dim // 2

    positions = np.arange(offset, offset + seq_len, dtype=np.float64)
    inv_freq = np.power(base, -np.arange(half, dtype=np.float64) / half)

    angles = positions[:, None] * inv_freq[None, :]  # (seq, half)
    cos, sin = np.cos(angles), np.sin(angles)

    first, second = x[..., :half], x[..., half:]
    return np.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )


class CachedAttention:
    """One causal attention layer that can decode with or without a cache."""

    def __init__(self, wq, wk, wv, wo, n_heads: int, rope_base: float = 10_000.0) -> None:
        self.wq = np.asarray(wq, dtype=np.float64)
        self.wk = np.asarray(wk, dtype=np.float64)
        self.wv = np.asarray(wv, dtype=np.float64)
        self.wo = np.asarray(wo, dtype=np.float64)
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.head_dim = self.wq.shape[0] // n_heads

    def forward(self, x, cache=None):
        """Run the layer over `x`, continuing from `cache` if one is given."""
        x = np.asarray(x, dtype=np.float64)
        batch, seq_len, d_model = x.shape

        def split_heads(projected: np.ndarray) -> np.ndarray:
            reshaped = projected.reshape(batch, seq_len, self.n_heads, self.head_dim)
            return np.transpose(reshaped, (0, 2, 1, 3))

        queries = split_heads(x @ self.wq)
        keys = split_heads(x @ self.wk)
        values = split_heads(x @ self.wv)

        # Where this chunk starts in the full sequence. Everything below that
        # depends on position reads it from here.
        past = 0 if cache is None else cache[0].shape[2]

        queries = rope(queries, past, self.rope_base)
        keys = rope(keys, past, self.rope_base)

        if cache is not None:
            keys = np.concatenate([cache[0], keys], axis=2)
            values = np.concatenate([cache[1], values], axis=2)
        new_cache = (keys, values)

        n_keys = keys.shape[2]
        # Query row i is at absolute position past + i, so it may read keys 0
        # through past + i.
        query_pos = np.arange(n_keys - seq_len, n_keys)[:, None]
        key_pos = np.arange(n_keys)[None, :]
        mask = np.where(key_pos <= query_pos, 0.0, -np.inf)

        scores = (queries @ np.swapaxes(keys, -1, -2)) / np.sqrt(self.head_dim)
        scores = scores + mask

        peak = np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores - peak)
        weights /= np.sum(weights, axis=-1, keepdims=True)

        attended = weights @ values
        merged = np.transpose(attended, (0, 2, 1, 3)).reshape(batch, seq_len, d_model)
        return merged @ self.wo, new_cache
