"""Byte-pair encoding: replaying a learned merge table on new text."""

from __future__ import annotations


def encode(text: str, merges: list[tuple[bytes, bytes]]) -> list[bytes]:
    """Encode `text` into tokens by replaying `merges` in learned order."""
    raise NotImplementedError


def decode(tokens: list[bytes]) -> str:
    """Join tokens back into text. Exactly inverts `encode`."""
    raise NotImplementedError
