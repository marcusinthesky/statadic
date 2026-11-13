# jaxtyping shape strings are runtime contracts.
"""Point clouds and their distance matrices, shared by the geometry suites.

A plain module rather than fixtures (t49 C3): these are pure functions of a
seed, with no state to set up or tear down. ``tests/geometry/test_embedding.py``,
``tests/geometry/test_median.py`` and ``tests/association/test_dispersion.py``
each need the same clustered cloud, and a fixture would only add indirection —
plus a fixture is scoped to the directory its ``conftest.py`` sits in, and these
three consumers span two.
"""

from __future__ import annotations

import numpy as np


def euclidean(points: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances of a point cloud.

    Args:
        points: Array of shape ``(n, d)``.

    Returns:
        The ``(n, n)`` distance matrix.

    """
    diffs = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=-1))


def clustered(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs of sizes 4, 5 and 6.

    Separation is 100x the within-group scale, so every group statistic these
    suites compute is far from its degenerate case and a failure means the code
    changed rather than the draw being unlucky.

    Args:
        seed: Seed for the NumPy generator.

    Returns:
        ``(distances, codes)`` — the ``(15, 15)`` Euclidean distance matrix and
        the group code of each point.

    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    sizes = [4, 5, 6]
    chunks = [
        c + rng.normal(scale=0.1, size=(s, 3))
        for c, s in zip(centres, sizes, strict=True)
    ]
    codes = np.concatenate([np.full(s, i, dtype=np.int32) for i, s in enumerate(sizes)])
    return euclidean(np.vstack(chunks)), codes
