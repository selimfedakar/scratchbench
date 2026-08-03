"""Reference solution — byte-pair encoding that respects the merge order.

The list of merges is not a set of rewrite rules that may be applied in any
convenient order. It is a *transcript*: merge zero was learned first, on the
raw byte sequence, and every later merge was learned on text where all the
earlier merges had already been applied. Replaying it in a different order
replays a vocabulary the tokenizer never had.

Two orderings look reasonable and are both wrong. Taking the most frequent
adjacent pair in the current sequence is the rule used during *training*, not
during encoding, and the training corpus is not this string. Scanning left to
right and merging the first pair that happens to be in the table is faster and
produces different tokens for the same text, which means a model trained on
one and served by the other sees ids that mean nothing.

So each round finds the single lowest-ranked pair present anywhere in the
sequence, applies it everywhere at once, and starts over.
"""

from __future__ import annotations


def encode(text: str, merges: list[tuple[bytes, bytes]]) -> list[bytes]:
    """Encode `text` into tokens by replaying `merges` in learned order."""
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    tokens = [bytes([byte]) for byte in text.encode("utf-8")]

    while len(tokens) > 1:
        # The earliest-learned pair present anywhere in the sequence wins,
        # regardless of how many times it or its rivals occur.
        best_pair = None
        best_rank = len(ranks)
        for pair in zip(tokens, tokens[1:]):
            rank = ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_pair, best_rank = pair, rank

        if best_pair is None:
            break

        left, right = best_pair
        merged = left + right

        # One left-to-right sweep, non-overlapping: in "aaa" the pair (a, a)
        # consumes the first two bytes, and the third is left for a later
        # round rather than being merged backwards into the token just built.
        rebuilt: list[bytes] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                rebuilt.append(merged)
                i += 2
            else:
                rebuilt.append(tokens[i])
                i += 1
        tokens = rebuilt

    return tokens


def decode(tokens: list[bytes]) -> str:
    """Join tokens back into text. Exactly inverts `encode`."""
    return b"".join(tokens).decode("utf-8")
