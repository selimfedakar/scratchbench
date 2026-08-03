# Encoding with a BPE merge table

Fill in `bpe_encode.py`. Two functions, standard library only:

```python
def encode(text: str, merges: list[tuple[bytes, bytes]]) -> list[bytes]
def decode(tokens: list[bytes]) -> str
```

The training half of byte-pair encoding is already done. You are given its
output: an ordered list of merges, each a pair of byte strings that the trainer
decided to fuse. Your job is the encoder that replays them on new text.

Encoding starts at the bytes. Encode the text as UTF-8 and begin with one token
per byte — not per character, so anything outside ASCII starts as several
tokens.

Then apply the merge table. The list is ordered: index 0 was learned first and
has the highest priority, and priority is the only thing that decides which
merge happens next. In each round, look at every adjacent pair currently in the
sequence, take the one that appears earliest in the merge list, and replace
every occurrence of that pair with the fused token. Then look again, because
fusing tokens creates pairs that did not exist a moment ago, and a merge learned
later may be waiting for exactly them. Stop when no adjacent pair appears in the
table at all.

Replacing "every occurrence" means one sweep from left to right, and a token
consumed by a merge is not available to the next one: in `aaa`, the pair `(a, a)`
takes the first two bytes and leaves the third alone. That third byte is free to
merge in a later round.

There is no pre-tokenization here. The text is not split on whitespace or
punctuation first, so a merge is free to fuse a space with the letter after it,
or to fuse two bytes that live inside a single non-ASCII character. That is what
the merge table describes and the encoder does not second-guess it.

`decode` is the exact inverse: concatenate the tokens and read the result back
as UTF-8. `decode(encode(text, merges))` returns `text` unchanged, for any text.

## Conventions

- Tokens are `bytes` objects, and the returned list contains `bytes`, never
  `int` and never `str`.
- Empty text encodes to an empty list; an empty token list decodes to `""`.
- The merge list may contain pairs that never occur in the text. It contains no
  duplicate pairs. Do not modify it.
- The text is arbitrary valid Unicode: whitespace, punctuation, accents, emoji.
- `decode` is only ever called on a token list whose bytes concatenate to valid
  UTF-8, so it does not need an error strategy.
- No `regex`, no `tiktoken`, no `transformers` — the standard library is enough
  and reaching for a tokenizer library answers a different question.
