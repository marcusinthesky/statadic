"""Thresholds and the AR(1) panel shared by the moment-condition suites.

Split out of the former ``test_moment_conditions.py`` (2026-08-01), which held
one suite per inference module in a single file and so named no module — the
layout gate's ``test_every_test_module_is_named_for_what_it_covers`` failed on
it. The tests moved to ``test_{gmm,clustering,wolak,kappa,gel,gms}.py``, one per
module they cover, and the constants that more than one of them reads live here.

A plain module rather than a ``conftest.py``, following ``tests/_cases.py``: this
is data, not lifecycle, and a leading underscore keeps it out of the layout
gate's ``_public`` walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

#: Nominal test level used by every size and power assertion below.
SIGNIFICANCE_LEVEL = 0.05
#: Degrees of freedom for the three- and two-moment fixtures.
THREE_MOMENT_DOF = 3
TWO_MOMENT_DOF = 2
#: Empirical size ceilings — the hard bound, and the softer one the
#: better-calibrated statistics are held to.
MAX_EMPIRICAL_SIZE = 0.15
SOFT_MAX_EMPIRICAL_SIZE = 0.12
#: Tolerance on the chi-bar-squared tail calibration.
TAIL_CALIBRATION_TOLERANCE = 0.015
#: Margin either side of the Wolak decision boundary.
WOLAK_DECISION_MARGIN = 0.1
#: First-stage F below which identification is called weak.
WEAK_IDENTIFICATION_THRESHOLD = 10.0
#: Trust-region cap for the GEL inner solver.
GEL_TRUST_REGION_CAP = 1e4
#: Expected degrees of freedom in the collinear and non-converged GEL fixtures.
EXPECTED_COLLINEAR_DOF = 4
EXPECTED_NONCONVERGED_DOF = 3
#: Power floors: the generic one, and the looser one for indicator GMS, whose
#: conservatism is the point of the statistic.
MINIMUM_POWER = 0.3
MINIMUM_GMS_POWER = 0.4
#: A statistic below this magnitude counts as numerically zero.
NEAR_ZERO_STATISTIC = 1e-8
#: Label of the second non-empty cluster in the ``cluster_pairs`` fixtures.
SECOND_NONEMPTY_CLUSTER_LABEL = 2


def ar1_panel(
    n_observations: int, m: int, phi: float, mean: float, seed: int
) -> np.ndarray:
    """Draw an AR(1) moment panel with a known mean.

    Args:
        n_observations: Number of rows.
        m: Number of moment columns.
        phi: Autoregressive coefficient.
        mean: Constant added to every entry.
        seed: PRNG seed.

    Returns:
        An ``(n_observations, m)`` panel.

    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_observations, m))
    x = np.empty((n_observations, m))
    x[0] = e[0]
    for t in range(1, n_observations):
        x[t] = phi * x[t - 1] + e[t]
    return x + mean


_BASIS_AMPLITUDE = 2.0
_NOISE_BOUND = 1.0


def full_rank_panels(n_observations: int, m: int) -> SearchStrategy[np.ndarray]:
    """Draw ``(n_observations, m)`` moment panels with a full-rank covariance.

    Generated noise within ``+-1`` rides on a deterministic basis of ``m`` sines of
    distinct frequency and amplitude 2, so no column can be constant or a copy of
    another, by construction rather than by filtering. HAC-based tests document a
    full-rank moment covariance as their domain.

    Args:
        n_observations: Number of rows, fixed per test so JAX traces once.
        m: Number of moment columns.

    Returns:
        A strategy for float64 NumPy panels.

    """
    time = np.arange(n_observations, dtype=np.float64)[:, None]
    frequency = 0.37 * (np.arange(m, dtype=np.float64)[None, :] + 1.0)
    basis = _BASIS_AMPLITUDE * np.sin(frequency * time + 0.5 * np.arange(m))
    return hnp.arrays(
        np.float64,
        (n_observations, m),
        elements=st.floats(-_NOISE_BOUND, _NOISE_BOUND),
    ).map(lambda noise: basis + noise)


# Rank-crossing profile fixture (t65.4). A rank-4 factor structure at m=15 whose
# fifth Omega direction crosses `rcond * lambda_max` as theta moves, which is the
# configuration that made the profiled objective discontinuous.
PROFILE_FROZEN_RANK = 4
PROFILE_CROSSED_RANK = 5
PROFILE_CROSSING_THETA = 5.0e-3
# Across the crossing the retired objective jumped from ~0.68 to ~1999; a frozen
# subspace keeps `2T*P` order one and its grid-to-grid variation small.
PROFILE_MAX_OBJECTIVE = 1.0
PROFILE_MAX_OBJECTIVE_STEP = 0.05


def rank_deficient_profile_panel() -> tuple[np.ndarray, np.ndarray]:
    """Supply the same numerical rank-crossing data to distinct GEL contracts.

    A rank-four factor structure at m=15 gains a fifth retained direction
    near theta=1.26e-3. Objective continuity and inferential eligibility remain
    separate assertions in their respective owning test modules.
    """
    rng = np.random.default_rng(0)
    factors = rng.standard_normal((2000, 4))
    loadings = rng.standard_normal((4, 15))
    panel = factors @ loadings + 1e-4 * rng.standard_normal((2000, 15))
    return panel, np.ones((15, 1))
