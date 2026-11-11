"""Independent validation DGP and shipped qualification accounting imports."""

import numpy as np
from statadic.testing.qualification import (
    DEFAULT_DRAWS,
    DEFAULT_TRIALS,
    Qualification,
    QualificationReport,
    TrialOutcome,
    summarize_trials,
)


def trial_rng(scenario: int, trial: int, *, stream: int = 0) -> np.random.Generator:
    """Independent NumPy validation DGP, separate from shipped JAX trial keys."""
    if min(scenario, trial, stream) < 0:
        message = "Validation RNG identifiers must be nonnegative"
        raise ValueError(message)
    return np.random.default_rng(
        np.random.SeedSequence([20261028, scenario, trial, stream])
    )


__all__ = [
    "DEFAULT_DRAWS",
    "DEFAULT_TRIALS",
    "Qualification",
    "QualificationReport",
    "TrialOutcome",
    "summarize_trials",
    "trial_rng",
]
