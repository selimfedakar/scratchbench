"""The verification step of speculative decoding."""

from __future__ import annotations

import numpy as np


def verify_draft(draft_tokens, draft_probs, target_probs, rng) -> np.ndarray:
    """Accept a prefix of the proposal and emit one corrected token after it."""
    raise NotImplementedError
