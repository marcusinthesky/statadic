"""Independent panel variance-component and retained-inference witnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.api.panel import _predict_panel
from statadic.estimation.panel import (
    FamaMacBeth,
    PanelProblem,
    RandomEffects,
)
from statadic.inference.panel_covariance import fama_macbeth_covariance, panel_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.operators.panel import prepare_panel_index
from scipy import stats

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from jax import Array
    from statadic.estimation.panel import (
        FamaMacBethFit,
        PanelFit,
    )

_MISSING_PERIOD = 2


def _case(
    *,
    balanced: bool,
    weighting: Literal["unit", "entity", "observation"],
    invariant: bool = False,
    seed: int = 738,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep explicit inputs with heteroskedastic idiosyncratic errors."""
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(12), 6)
    time = np.tile(np.arange(6), 12)
    keep = (
        np.ones(len(entity), dtype=bool)
        if balanced
        else ~((entity % 3 == 0) & (time == _MISSING_PERIOD))
    )
    entity, time = entity[keep], time[keep]
    x = np.column_stack([np.ones(len(entity)), rng.normal(size=(len(entity), 2))])
    if invariant:
        x[:, -1] = rng.normal(size=12)[entity]
    w = np.ones(len(entity))
    if weighting == "entity":
        w = np.linspace(0.4, 2.5, 12)[entity]
    elif weighting == "observation":
        w = rng.uniform(0.3, 2.7, len(entity))
    y = (
        x @ np.array([0.5, -0.4, 0.7])
        + rng.normal(size=12)[entity]
        + rng.normal(size=len(entity)) / np.sqrt(w)
    )
    return x, y, w, entity, time


def _residual(
    x: np.ndarray, y: np.ndarray, scale: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """NumPy SVD reference with original-scale absorption cutoff."""
    norms = np.ones(x.shape[1]) if scale is None else scale
    z = x / np.where(norms > 0, norms, 1)
    cutoff = max(x.shape) * np.finfo(x.dtype).eps
    # Fix the absolute cutoff before solving; noise in an absorbed column is
    # measured relative to its original scale, not its residual norm.
    singular = np.linalg.svd(z, compute_uv=False)
    rcond = cutoff * max(singular[0], 1) / singular[0] if singular[0] > 0 else cutoff
    if scale is None:
        rcond = cutoff
    beta, _, rank, _ = np.linalg.lstsq(z, y, rcond=rcond)
    return y - z @ beta, int(rank)


def _re_reference(  # noqa: PLR0914  # named independent component arithmetic
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    entity: np.ndarray,
    *,
    small_sample: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent dense-dummy projection, components and quasi-demeaned fit."""
    w = weights / weights.mean()
    root = np.sqrt(w)
    dummy = np.eye(len(np.unique(entity)))[entity]
    counts = dummy.sum(axis=0)
    data = np.column_stack([y, x])
    means = np.linalg.lstsq(root[:, None] * dummy, root[:, None] * data, rcond=None)[0]
    within = root[:, None] * (data - dummy @ means)
    eps, within_rank = _residual(
        within[:, 1:], within[:, 0], np.linalg.norm(root[:, None] * x, axis=0)
    )
    between, between_rank = _residual(means[:, 1:], means[:, 0])
    sigma_e = eps @ eps / (len(y) - len(counts) - within_rank)
    df_between = len(counts) - between_rank
    if small_sample and np.ptp(counts) != 0:
        weighted, _ = _residual(
            np.sqrt(counts)[:, None] * means[:, 1:], np.sqrt(counts) * means[:, 0]
        )
        sums = dummy.T @ (root[:, None] * x)
        repeated = dummy @ (sums / counts[:, None])
        trace = np.trace(
            np.linalg.pinv(repeated.T @ repeated, hermitian=True) @ (sums.T @ sums)
        )
        sigma_u = max(
            0, (weighted @ weighted - df_between * sigma_e) / (len(y) - trace)
        )
    else:
        harmonic = len(counts) / np.sum(1 / counts)
        sigma_u = max(0, between @ between / df_between - sigma_e / harmonic)
    theta = 1 - np.sqrt(sigma_e / (counts * sigma_u + sigma_e))
    transformed = root[:, None] * (data - theta[entity, None] * means[entity])
    beta = np.linalg.lstsq(transformed[:, 1:], transformed[:, 0], rcond=None)[0]
    return np.array([sigma_e, sigma_u]), theta, transformed, beta


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.regression
@pytest.mark.parametrize("balanced", [True, False])
@pytest.mark.parametrize("weighting", ["unit", "entity", "observation"])
@pytest.mark.parametrize("small_sample", [False, True])
@examples(8)
@example(seed=738)
@given(seed=st.integers(0, 2**32 - 1))
def test_random_effects_components_and_transform_have_independent_witnesses(  # noqa: PLR0914
    seed: int,
    *,
    balanced: bool,
    weighting: Literal["unit", "entity", "observation"],
    small_sample: bool,
) -> None:
    """Catch omitted root weights, unbalanced trace errors and wrong component df."""
    x, y, w, entity, time = _case(balanced=balanced, weighting=weighting, seed=seed)
    components, theta, transformed, beta = _re_reference(
        x, y, w, entity, small_sample=small_sample
    )
    fit = RandomEffects(small_sample=small_sample).estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            w,
            effects="random",
            intercept_column=0,
        )
    )
    assert fit.valid
    np.testing.assert_allclose(
        fit.variance_components, components, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(fit.theta, theta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        fit.linear.design, transformed[:, 1:], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        fit.linear.outcome, transformed[:, 0], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(fit.parameters, beta, rtol=1e-10, atol=1e-12)
    group_residual = np.array(
        [
            np.average((y - x @ beta)[entity == group], weights=w[entity == group])
            for group in np.unique(entity)
        ]
    )
    counts = np.bincount(entity)
    effects = (
        counts
        * components[1]
        / (counts * components[1] + components[0])
        * group_residual
    )[entity]
    np.testing.assert_allclose(fit.effects, effects, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        fit.residuals, y - x @ beta - effects, rtol=1e-10, atol=1e-12
    )
    residual = transformed[:, 0] - transformed[:, 1:] @ beta
    assert fit.df_resid == len(y) - x.shape[1]
    covariance = (
        residual
        @ residual
        / int(fit.df_resid)
        * np.linalg.inv(transformed[:, 1:].T @ transformed[:, 1:])
    )
    uncertainty = panel_covariance(fit)
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, covariance, rtol=1e-10, atol=1e-12
    )
    intervals = coefficient_intervals(
        uncertainty, reference="t", degrees_of_freedom=fit.df_resid
    )
    critical = stats.t.ppf(0.975, int(fit.df_resid))
    expected_ci = np.stack(
        [
            beta - critical * np.sqrt(np.diag(covariance)),
            beta + critical * np.sqrt(np.diag(covariance)),
        ]
    )
    np.testing.assert_allclose(
        intervals.estimate.ci, expected_ci, rtol=1e-9, atol=1e-10
    )
    restriction = np.array([[0.0, 1.0, -0.5]])
    test = linear_restriction_test(uncertainty, restriction, 0.0, reference="chi2")
    quadratic = float(
        (restriction @ beta)[0] ** 2 / (restriction @ covariance @ restriction.T)[0, 0]
    )
    assert test.valid
    np.testing.assert_allclose(test.test.statistic, quadratic, rtol=1e-10)
    np.testing.assert_allclose(
        test.test.pvalue, stats.chi2.sf(quadratic, 1), rtol=1e-9, atol=1e-12
    )


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("small_sample", [False, True])
@examples(8)
@example(seed=738)
@given(seed=st.integers(0, 2**32 - 1))
def test_re_auxiliary_rank_deficiency_does_not_invalidate_identified_slopes(
    seed: int, *, small_sample: bool
) -> None:
    """Time-invariant regressors use no within-regression degree of freedom."""
    x, y, w, entity, time = _case(
        balanced=False, weighting="observation", invariant=True, seed=seed
    )
    components, theta, _, beta = _re_reference(
        x, y, w, entity, small_sample=small_sample
    )
    fit = RandomEffects(small_sample=small_sample).estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            w,
            effects="random",
            intercept_column=0,
        )
    )
    assert fit.valid
    np.testing.assert_allclose(
        fit.variance_components, components, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(fit.theta, theta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.parameters, beta, rtol=1e-10, atol=1e-12)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("small_sample", [False, True])
@pytest.mark.parametrize("invariant", [False, True])
@pytest.mark.parametrize("intercept", [False, True])
@examples(8)
@example(seed=738)
@given(seed=st.integers(0, 2**32 - 1))
def test_corrected_random_effects_matches_current_linearmodels(
    optional_reference: Callable[[str], ModuleType],
    seed: int,
    *,
    small_sample: bool,
    invariant: bool,
    intercept: bool,
) -> None:
    """Compare common targets, keeping BLUP effects separate from quasi residuals."""
    optional_reference("linearmodels")
    pytest.importorskip("linearmodels", minversion="7.1")
    lm, pd = optional_reference("linearmodels.panel"), optional_reference("pandas")
    x, y, w, entity, time = _case(
        balanced=False, weighting="observation", invariant=invariant, seed=seed
    )
    if not intercept:
        x = x[:, 1:]
    fit = RandomEffects(small_sample=small_sample).estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            w,
            effects="random",
            intercept_column=0 if intercept else None,
        )
    )
    index = pd.MultiIndex.from_arrays([entity, time])
    reference = lm.RandomEffects(
        pd.Series(y, index=index),
        pd.DataFrame(x, index=index),
        weights=pd.Series(w, index=index),
    ).fit(small_sample=small_sample)
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, reference.params, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        fit.variance_components,
        reference.variance_decomposition.iloc[:2].to_numpy()[::-1],
        rtol=1e-7,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit.theta, reference.theta.to_numpy().ravel(), rtol=1e-7, atol=1e-8
    )
    np.testing.assert_allclose(
        fit.linear.residuals, reference.wresids, rtol=1e-7, atol=1e-8
    )


@pytest.mark.integration
@pytest.mark.contract
def test_fama_macbeth_native_df_counts_identified_periods_under_transforms() -> None:
    """A failed cross section is retained but cannot contribute inferential df."""
    entity = np.tile(np.arange(5), 5)
    time = np.repeat(np.arange(5), 5)
    x = np.column_stack([np.ones(len(entity)), np.tile(np.linspace(-1, 1, 5), 5)])
    x[time == _MISSING_PERIOD, 1] = 0
    y = np.sin(np.arange(len(entity))) + x[:, 1] * (time + 1)
    index = prepare_panel_index(entity, time)
    fit = FamaMacBeth().estimate(PanelProblem(x, y, index, effects="pooled"))
    assert fit.valid
    np.testing.assert_array_equal(fit.period_valid, [True, True, False, True, True])
    assert fit.df_resid == np.count_nonzero(fit.period_valid) - 1

    def evaluate_stack_leaves(*leaves: object) -> Array:
        return jnp.stack([jnp.asarray(leaf) for leaf in leaves])

    stacked = jax.tree.map(
        evaluate_stack_leaves,
        PanelProblem(x, y, index, effects="pooled"),
        PanelProblem(x, y * 2, index, effects="pooled"),
    )

    def evaluatefit(p: PanelProblem) -> FamaMacBethFit:
        return FamaMacBeth().estimate(p)

    batch = jax.jit(jax.vmap(evaluatefit))(stacked)
    np.testing.assert_array_equal(batch.df_resid, [3, 3])
    uncertainty = fama_macbeth_covariance(fit)
    intervals = coefficient_intervals(
        uncertainty, reference="t", degrees_of_freedom=fit.df_resid
    )
    assert intervals.valid
    assert float(intervals.critical_value) == pytest.approx(
        stats.t.ppf(0.975, 3), rel=1e-10
    )


@pytest.mark.component
@pytest.mark.regression
def test_re_zero_entity_component_reduces_to_weighted_linear_fit() -> None:
    """Negative component estimates are clipped without invalidating point fitting."""
    x, _, weights, entity, time = _case(balanced=False, weighting="observation")
    rng = np.random.default_rng(798)
    error = rng.normal(size=len(entity))
    dummy = np.eye(len(np.unique(entity)))[entity]
    means = np.linalg.lstsq(
        np.sqrt(weights)[:, None] * dummy, np.sqrt(weights) * error, rcond=None
    )[0]
    error -= dummy @ means
    y = x @ np.array([0.5, -0.4, 0.7]) + error
    fit = RandomEffects().estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            weights,
            effects="random",
            intercept_column=0,
        )
    )
    assert fit.valid
    assert fit.variance_components[1] == 0
    np.testing.assert_array_equal(fit.theta, np.zeros(len(np.unique(entity))))
    np.testing.assert_array_equal(fit.effects, np.zeros(len(entity)))
    expected = np.linalg.lstsq(
        np.sqrt(weights)[:, None] * x, np.sqrt(weights) * y, rcond=None
    )[0]
    np.testing.assert_allclose(fit.parameters, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.component
@pytest.mark.numerical
def test_re_rank_deficient_between_projection_uses_its_actual_rank() -> None:
    """Mean-zero period regressors need not identify auxiliary between slopes."""
    entity = np.repeat(np.arange(10), 6)
    time = np.tile(np.arange(6), 10)
    rng = np.random.default_rng(798)
    x = np.column_stack([np.ones(len(entity)), time - time.mean()])
    y = (
        x @ np.array([0.5, -0.4])
        + rng.normal(size=10)[entity]
        + rng.normal(size=len(entity))
    )
    components, theta, _, beta = _re_reference(
        x, y, np.ones(len(y)), entity, small_sample=True
    )
    fit = RandomEffects(small_sample=True).estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            effects="random",
            intercept_column=0,
        )
    )
    assert fit.valid
    np.testing.assert_allclose(
        fit.variance_components, components, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(fit.theta, theta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.parameters, beta, rtol=1e-10, atol=1e-12)


@pytest.mark.component
@pytest.mark.property
def test_re_relative_weights_and_joint_row_labels_are_invariant() -> None:
    """Preserve alignment under reordered rows, renamed entities and scaled weights."""
    x, y, weights, entity, time = _case(balanced=False, weighting="observation")
    original = RandomEffects(small_sample=True).estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            weights,
            effects="random",
            intercept_column=0,
        )
    )
    order = np.random.default_rng(738).permutation(len(y))
    reordered = RandomEffects(small_sample=True).estimate(
        PanelProblem(
            x[order],
            y[order],
            prepare_panel_index(100 - entity[order], time[order]),
            weights[order] * 7.3,
            effects="random",
            intercept_column=0,
        )
    )
    assert original.valid
    assert reordered.valid
    np.testing.assert_allclose(
        reordered.parameters, original.parameters, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        reordered.variance_components,
        original.variance_components,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reordered.fitted_values, original.fitted_values[order], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        reordered.effects, original.effects[order], rtol=1e-10, atol=1e-12
    )


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_re_regression_distinguishes_omitted_root_multiplier(
    dtype: type,
) -> None:
    """The witness remains effective at caller-owned single and double precision."""
    x, y, weights, entity, time = _case(balanced=False, weighting="observation")
    components, theta, _, expected = _re_reference(
        x, y, weights, entity, small_sample=False
    )
    data = np.column_stack([y, x])
    means = np.stack(
        [
            np.average(data[entity == group], axis=0, weights=weights[entity == group])
            for group in np.unique(entity)
        ]
    )
    legacy = (
        np.sqrt(weights / weights.mean())[:, None] * data
        - theta[entity, None] * means[entity]
    )
    legacy_beta = np.linalg.lstsq(legacy[:, 1:], legacy[:, 0], rcond=None)[0]
    assert not np.allclose(legacy_beta, expected, rtol=1e-7, atol=1e-8)
    fit = RandomEffects().estimate(
        PanelProblem(
            x.astype(dtype),
            y.astype(dtype),
            prepare_panel_index(entity, time),
            weights.astype(dtype),
            effects="random",
            intercept_column=0,
        )
    )
    assert fit.valid
    assert fit.parameters.dtype == dtype
    tolerance = 1e-4 if dtype is np.float32 else 1e-10
    np.testing.assert_allclose(
        fit.parameters, expected, rtol=tolerance, atol=tolerance * 0.01
    )
    np.testing.assert_allclose(
        fit.variance_components, components, rtol=tolerance, atol=tolerance * 0.01
    )


@pytest.mark.component
@pytest.mark.contract
def test_re_unavailable_variance_component_df_fails_closed() -> None:
    """A full-rank point design cannot manufacture between residual degrees."""
    entity = np.repeat(np.arange(2), 4)
    time = np.tile(np.arange(4), 2)
    x = np.column_stack([np.ones(len(entity)), np.arange(len(entity))])
    y = np.sin(np.arange(len(entity)))
    fit = RandomEffects().estimate(
        PanelProblem(
            x,
            y,
            prepare_panel_index(entity, time),
            effects="random",
            intercept_column=0,
        )
    )
    assert not fit.valid
    assert np.all(np.isnan(fit.parameters))


@pytest.mark.component
@pytest.mark.contract
def test_re_batched_public_parameters_and_prediction_preserve_valid_lanes() -> None:
    """Batch validity broadcasts over parameters and rows, never their columns."""
    x, y, weights, entity, time = _case(balanced=False, weighting="observation")
    index = prepare_panel_index(entity, time)

    def evaluate_stack_leaves(*leaves: object) -> Array:
        return jnp.stack([jnp.asarray(leaf) for leaf in leaves])

    stacked = jax.tree.map(
        evaluate_stack_leaves,
        PanelProblem(x, y, index, weights, effects="random", intercept_column=0),
        PanelProblem(x, y * 2, index, -weights, effects="random", intercept_column=0),
    )

    def evaluate_fit(p: PanelProblem) -> PanelFit:
        return RandomEffects().estimate(p)

    evaluate_fit = jax.jit(jax.vmap(evaluate_fit))(stacked)
    np.testing.assert_array_equal(evaluate_fit.valid, [True, False])
    assert np.all(np.isfinite(evaluate_fit.parameters[0]))
    assert np.all(np.isnan(evaluate_fit.parameters[1]))
    predictions = _predict_panel(evaluate_fit, x)
    np.testing.assert_allclose(
        predictions[0], x @ evaluate_fit.parameters[0], rtol=1e-10, atol=1e-12
    )
    assert np.all(np.isnan(predictions[1]))
