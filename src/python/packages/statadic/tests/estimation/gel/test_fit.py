"""Retained GEL fitting state before reference-distribution selection."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem

from ..._budgets import examples


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
@examples(8)
@example(seed=778)
@given(seed=st.integers(0, 2**32 - 1))
def test_fixed_fit_retains_actual_moments_and_multiplier(
    seed: int,
    criterion: Literal["cue", "el", "et"],
) -> None:
    """Retained coordinates describe the actual fitted dual objective."""
    moments = np.random.default_rng(seed).standard_normal((120, 3)) + 0.1
    fit = FixedGEL(criterion=criterion).estimate(FixedMomentProblem(moments))
    assert fit.converged
    assert fit.criterion == criterion
    np.testing.assert_array_equal(fit.moments, moments)
    assert int(np.asarray(fit.keep).sum()) == fit.effective_rank
    reduced_multiplier = np.asarray(fit.basis).T @ fit.lam
    np.testing.assert_allclose(
        reduced_multiplier * (1.0 - np.asarray(fit.keep)), 0.0, atol=1e-12
    )
    assert np.isfinite(fit.objective)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
@examples(8)
@example(seed=111)
@given(seed=st.integers(0, 2**32 - 1))
def test_affine_fit_retains_solver_and_rank_evidence(
    seed: int,
    criterion: Literal["cue", "el", "et"],
) -> None:
    """Affine fitting retains its parameter and frozen profile ingredients."""
    rng = np.random.default_rng(seed)
    noise = 0.2 * rng.standard_normal((600, 4))
    noise -= noise.mean(axis=0, keepdims=True)
    derivative = np.array([[1.0], [0.6], [-0.4], [0.8]])
    theta_true = 0.7
    g0 = noise - theta_true * derivative[:, 0]
    fit = AffineGEL(criterion=criterion).estimate(AffineMomentProblem(g0, derivative))
    assert fit.converged
    assert fit.inner_converged
    assert fit.outer_converged
    assert fit.rank_stable == (fit.effective_rank == fit.rank_at_parameters)
    np.testing.assert_allclose(fit.theta, [theta_true], atol=1e-5)
    np.testing.assert_allclose(
        fit.moments, fit.moments0 + np.einsum("tmk,k->tm", fit.jacobian, fit.theta)
    )
    np.testing.assert_allclose(fit.objective, 0.0, atol=1e-10)
