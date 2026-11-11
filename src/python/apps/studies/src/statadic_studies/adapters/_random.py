"""Scientific adapter functions grouped by computational ownership."""

from __future__ import annotations

import hashlib

import numpy as np


def rng(study: str, scenario: str, replicate: int, *, seed: int, stream: str = "dgp"):
    """Use stable named streams, independent of worker count and campaign length."""
    if min(seed, replicate) < 0:
        raise ValueError("Seed and replicate must be nonnegative")
    digest = hashlib.sha256(f"{study}\0{scenario}\0{stream}".encode()).digest()
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, replicate, *words]))
