"""Observed spatial likelihood against independently assembled Gaussian blocks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, reject
from hypothesis import strategies as st
from statadic.estimation.spatial import (
    SAR,
    ObservedSpatialPanelProblem,
    observed_panel_profile,
)
from statadic.operators.spatial import ObservedSpatialPeriod, prepare_spatial_weights
from scipy.linalg import block_diag

from ..._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy import ndarray


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("lag", [True, False])
@pytest.mark.parametrize("effects", ["pooled", "unit", "time", "two-way", "random"])
@examples(4)
@example(seed=603)
@given(seed=st.integers(0, 2**32 - 1))
def test_selected_likelihood_matches_joint_gaussian(  # noqa: PLR0914, PLR0915
    *,
    lag: bool,
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
    seed: int,
) -> None:
    """Compare selected blocks against independent full covariance."""
    rng = np.random.default_rng(seed)
    roster = 4
    active = [(0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 3), (0, 1, 2, 3)]
    selected = [(0, 1, 2), (0, 1, 2, 3), (0, 1, 2), (0, 2, 3)]
    periods = []
    designs = []
    outcomes = []
    full_means = []
    blocks = []
    effect_maps = []
    spatial = 0.23
    ratio = 0.4
    with jax.enable_x64():
        for t, (entities, obs) in enumerate(zip(active, selected, strict=True)):
            n = len(entities)
            w = rng.uniform(size=(n, n))
            np.fill_diagonal(w, 0)
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1)
            x = rng.normal(size=(n, 1))
            yy = rng.normal(size=len(obs))
            periods.append(
                ObservedSpatialPeriod(prepare_spatial_weights(w), entities, obs)
            )
            designs.append(x)
            outcomes.append(yy)
            a = np.eye(n) - (spatial if lag else 0) * w
            b = np.eye(n) - (0 if lag else spatial) * w
            propagation = np.linalg.solve(a, np.eye(n))
            noise = propagation @ np.linalg.solve(b, np.eye(n))
            blocks.append((noise @ noise.T)[np.ix_(obs, obs)])
            d = np.eye(roster)[np.asarray(entities)]
            effect_maps.append((propagation @ d)[list(obs)])
            pieces = [x]
            if effects in ("unit", "two-way"):
                pieces.append(d)
            if effects in ("time", "two-way"):
                width = len(active) if effects == "time" else len(active) - 1
                col = np.zeros((n, width))
                index = t if effects == "time" else t - 1
                if index >= 0:
                    col[:, index] = 1
                pieces.append(col)
            full_means.append((propagation @ np.column_stack(pieces))[list(obs)])
        problem = ObservedSpatialPanelProblem(
            tuple(periods),
            tuple(designs),
            tuple(outcomes),
            roster,
            process=SAR(),
            effects="pooled",
        )
        actual = observed_panel_profile(
            problem,
            jnp.asarray([spatial]),
            lag=lag,
            effects=effects,
            variance_ratio=jnp.asarray(ratio),
        )
        cov = block_diag(*blocks)
        if effects == "random":
            h = np.concatenate(effect_maps)
            cov += ratio * h @ h.T
        root = np.linalg.cholesky(cov)
        xx = np.linalg.solve(root, np.concatenate(full_means))
        yy = np.linalg.solve(root, np.concatenate(outcomes))
        coefficients = np.linalg.lstsq(xx, yy, rcond=None)[0]
        residual = yy - xx @ coefficients
        sigma2 = residual @ residual / len(yy)
        log_likelihood = -0.5 * (
            2 * np.log(np.diag(root)).sum() + len(yy) * (np.log(2 * np.pi * sigma2) + 1)
        )
        np.testing.assert_allclose(actual[0], coefficients[:1], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual[1], coefficients[1:], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual[2], sigma2, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual[3], log_likelihood, rtol=1e-9, atol=1e-9)


@pytest.mark.component
@pytest.mark.property
def test_random_fit_does_not_select_inferior_zero_variance() -> None:
    """A strong shared effect must improve on the zero-variance boundary."""
    from statadic.estimation.spatial import ObservedPanelML  # noqa: PLC0415

    rng = np.random.default_rng(221)
    n, t = 20, 6
    w = np.roll(np.eye(n, dtype=np.float32), 1, axis=1)
    periods = tuple(
        ObservedSpatialPeriod(
            prepare_spatial_weights(w), tuple(range(n)), tuple(range(n))
        )
        for _ in range(t)
    )
    shared = rng.normal(size=n).astype(np.float32)
    design = tuple(rng.normal(size=(n, 1)).astype(np.float32) for _ in range(t))
    outcome = tuple(
        shared + rng.normal(scale=0.4, size=n).astype(np.float32) for _ in range(t)
    )
    problem = ObservedSpatialPanelProblem(
        periods, design, outcome, n, process=SAR(), effects="pooled"
    )
    fit = ObservedPanelML(max_steps=100, tolerance=1e-05).estimate(
        replace(problem, process=SAR(), slx_columns=(), effects="random")
    )
    better = observed_panel_profile(
        problem,
        jnp.asarray([0.0]),
        lag=True,
        effects="random",
        variance_ratio=jnp.asarray(1.0),
    )[3]
    assert bool(fit.valid)
    assert not bool(fit.variance_boundary)
    assert float(fit.log_likelihood) >= float(better) - 0.01


@pytest.mark.component
@pytest.mark.reference
def test_random_effects_preserve_small_residual_in_float32() -> None:  # noqa: PLR0914
    """Woodbury whitening must not cancel a nearly fitted response."""
    rng = np.random.default_rng(51)
    n, t, ratio = 5, 4, 1000.0
    w = np.roll(np.eye(n), 1, axis=1)
    inverse = np.linalg.inv(np.eye(n) - 0.2 * w)
    periods = tuple(
        ObservedSpatialPeriod(
            prepare_spatial_weights(w.astype(np.float32)),
            tuple(range(n)),
            tuple(range(n - 1)),
        )
        for _ in range(t)
    )
    x = [rng.normal(size=(n, 1)) for _ in range(t)]
    design = np.vstack([(inverse @ item)[:-1] for item in x])
    y = 2 * design[:, 0] + 0.001 * rng.normal(size=t * (n - 1))
    h = np.vstack([inverse[:-1]] * t)
    covariance = block_diag(*[(inverse @ inverse.T)[:-1, :-1]] * t) + ratio * h @ h.T
    root = np.linalg.cholesky(covariance)
    xx, yy = np.linalg.solve(root, design), np.linalg.solve(root, y)
    beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
    residual = yy - xx @ beta
    sigma2 = residual @ residual / len(y)
    expected = -0.5 * (
        2 * np.log(np.diag(root)).sum() + len(y) * (np.log(2 * np.pi * sigma2) + 1)
    )
    problem = ObservedSpatialPanelProblem(
        periods,
        tuple(item.astype(np.float32) for item in x),
        tuple(item.astype(np.float32) for item in np.split(y, t)),
        n,
        process=SAR(),
        effects="pooled",
    )
    actual = observed_panel_profile(
        problem,
        jnp.asarray([0.2], dtype=jnp.float32),
        lag=True,
        effects="random",
        variance_ratio=jnp.asarray(ratio, dtype=jnp.float32),
    )
    assert np.isfinite(actual[3])
    np.testing.assert_allclose(actual[3], expected, rtol=0, atol=0.03)


def _assert_boundary_profile_maximum(
    w: np.ndarray, x: tuple[np.ndarray, ...], y: tuple[np.ndarray, ...]
) -> None:
    """Independently show the concentrated likelihood peaks at |rho| -> 1."""
    n, t = w.shape[0], len(y)
    design = np.concatenate(x)

    def profile(rho: float) -> float:
        a = np.eye(n) - rho * w
        lagged = np.concatenate([a @ yt for yt in y])
        coefficient = np.linalg.lstsq(design, lagged, rcond=None)[0]
        residual = lagged - design @ coefficient
        log_det = np.linalg.slogdet(a)[1]
        return t * log_det - 0.5 * n * t * np.log(residual @ residual / (n * t))

    grid = np.linspace(-0.999, 0.999, 401)
    best = int(np.argmax([profile(rho) for rho in grid]))
    assert best in {0, grid.size - 1}


@pytest.mark.component
@pytest.mark.reference
@examples(4)
@example(seed=33)
@example(seed=1015)  # Profile maximum on the stability boundary.
@given(seed=st.integers(0, 2**32 - 1))
def test_observed_information_matches_independent_finite_difference(seed: int) -> None:
    """Compare the observed information against numerical differences.

    A sample whose likelihood has no interior maximum must fail closed instead.
    """
    from statadic.estimation.spatial import ObservedPanelML  # noqa: PLC0415
    from scipy.optimize._numdiff import approx_derivative  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n, t = 5, 3
    w = np.roll(np.eye(n), 1, axis=1)
    periods = tuple(
        ObservedSpatialPeriod(
            prepare_spatial_weights(w), tuple(range(n)), tuple(range(n))
        )
        for _ in range(t)
    )
    x = tuple(rng.normal(size=(n, 1)) for _ in periods)
    y = tuple(rng.normal(size=n) for _ in periods)

    def objective(point: np.ndarray) -> float:
        beta, rho, log_sigma2 = point
        a = np.eye(n) - rho * w
        inverse = np.linalg.solve(a, np.eye(n))
        covariance = inverse @ inverse.T
        root = np.linalg.cholesky(covariance)
        residual = np.concatenate(
            [
                np.linalg.solve(root, yt - inverse @ xt[:, 0] * beta)
                for xt, yt in zip(x, y, strict=True)
            ]
        )
        return 0.5 * (
            t * 2 * np.log(np.diag(root)).sum()
            + t * n * (np.log(2 * np.pi) + log_sigma2)
            + residual @ residual * np.exp(-log_sigma2)
        )

    def gradient(point: np.ndarray) -> np.ndarray:
        def evaluate_objective_vector(q: np.ndarray) -> ndarray:
            return np.array([objective(q)])

        return approx_derivative(
            evaluate_objective_vector, point, method="3-point"
        ).reshape(-1)

    with jax.enable_x64():
        fit = ObservedPanelML(max_steps=40).estimate(
            ObservedSpatialPanelProblem(
                periods, x, y, n, process=SAR(), slx_columns=(), effects="pooled"
            )
        )
        if not fit.valid:
            _assert_boundary_profile_maximum(w, x, y)
            reject()
        assert fit.information_valid
        state = np.array(
            [
                float(fit.parameters[0]),
                float(fit.parameters[1]),
                np.log(float(fit.sigma2)),
            ]
        )

        hessian = approx_derivative(gradient, state, method="3-point")
        expected = (
            hessian[:2, :2] - np.outer(hessian[:2, 2], hessian[2, :2]) / hessian[2, 2]
        )
        np.testing.assert_allclose(fit.information, expected, rtol=3e-4, atol=2e-4)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("lag", [True, False])
@pytest.mark.parametrize("effects", ["pooled", "unit", "random"])
@examples(4)
@example(seed=64)
@given(seed=st.integers(0, 2**32 - 1))
def test_complete_mask_reduces_to_balanced_profile(
    *,
    lag: bool,
    effects: Literal["pooled", "unit", "random"],
    seed: int,
) -> None:
    """A complete selected panel reproduces the retained balanced likelihood."""
    from statadic.estimation.spatial.panel import spatial_panel_profile  # noqa: PLC0415
    from statadic.operators.spatial import prepare_spatial_panel_weights  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n, t = 5, 3
    w = np.roll(np.eye(n), 1, axis=1)
    x = rng.normal(size=(t, n, 1))
    y = rng.normal(size=(t, n))
    with jax.enable_x64():
        graph = prepare_spatial_weights(w)
        periods = tuple(
            ObservedSpatialPeriod(graph, tuple(range(n)), tuple(range(n)))
            for _ in range(t)
        )
        problem = ObservedSpatialPanelProblem(
            periods, tuple(x), tuple(y), n, process=SAR(), effects="pooled"
        )
        observed = observed_panel_profile(
            problem,
            jnp.array([0.2]),
            lag=lag,
            effects=effects,
            variance_ratio=jnp.array(0.3),
        )
        balanced = spatial_panel_profile(
            jnp.array([0.2]),
            jnp.asarray(x),
            jnp.asarray(y),
            prepare_spatial_panel_weights(w, period_ids=np.arange(t)),
            lag=lag,
            error=not lag,
            effects=effects,
            variance_ratio=jnp.array(0.3),
        )
        np.testing.assert_allclose(observed[0], balanced[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(observed[2:4], balanced[2:4], rtol=1e-9, atol=1e-9)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.parametrize("lag", [True, False])
@examples(4)
@example(seed=534)
@given(seed=st.integers(0, 2**32 - 1))
def test_complete_unit_fe_durbin_matches_spreg(  # noqa: PLR0914
    *,
    lag: bool,
    optional_reference: Callable[[str], ModuleType],
    seed: int,
) -> None:
    """Compare Durbin ML with spreg given identical explicit WX regressors."""
    from statadic.estimation.spatial import SAR, SEM, ObservedPanelML  # noqa: PLC0415

    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(seed)
    t, n = 5, 8
    w = 0.5 * np.roll(np.eye(n), 1, axis=1) + 0.5 * np.roll(np.eye(n), -1, axis=1)
    x = rng.normal(size=(t, n, 1))
    alpha = rng.normal(size=n)
    noise = rng.normal(size=(t, n)) * 0.4
    wx = np.einsum("ij,tjp->tip", w, x)
    mean = x[:, :, 0] * 0.5 + wx[:, :, 0] * 0.2 + alpha
    y = (
        np.linalg.solve(np.eye(n) - 0.2 * w, (mean + noise).T).T
        if lag
        else mean + np.linalg.solve(np.eye(n) + 0.2 * w, noise.T).T
    )
    with jax.enable_x64():
        graph = prepare_spatial_weights(w)
        periods = tuple(
            ObservedSpatialPeriod(graph, tuple(range(n)), tuple(range(n)))
            for _ in range(t)
        )
        problem = ObservedSpatialPanelProblem(
            periods, tuple(x), tuple(y), n, process=SAR(), effects="pooled"
        )
        fit = ObservedPanelML().estimate(
            replace(
                problem,
                process=SAR() if lag else SEM(),
                slx_columns=(0,),
                effects="unit",
            )
        )
        oracle_class = spreg.Panel_FE_Lag if lag else spreg.Panel_FE_Error
        oracle = oracle_class(
            y.reshape(-1, 1),
            np.concatenate([x, wx], axis=2).reshape(t * n, 2),
            libpysal.weights.full2W(w),
            epsilon=1e-9,
        )
        assert fit.valid
        np.testing.assert_allclose(
            fit.parameters, np.asarray(oracle.betas).ravel(), atol=3e-5
        )
        np.testing.assert_allclose(
            fit.log_likelihood, oracle.logll + (t * n) / 2 * np.log(t * n), atol=2e-5
        )


@pytest.mark.component
@pytest.mark.contract
def test_selected_panel_rejects_unidentified_unit_effects() -> None:
    """A missing never-observed SEM unit makes its mean effect unidentified."""
    with jax.enable_x64():
        w = np.roll(np.eye(4), 1, axis=1)
        graph = prepare_spatial_weights(w)
        periods = tuple(
            ObservedSpatialPeriod(graph, (0, 1, 2, 3), (0, 1, 2)) for _ in range(3)
        )
        x = tuple(np.arange(4, dtype=float)[:, None] + t for t in range(3))
        y = tuple(np.array([0.5, 0.1, -0.3]) + t * 0.2 for t in range(3))
        problem = ObservedSpatialPanelProblem(
            periods, x, y, 4, process=SAR(), effects="pooled"
        )
        result = observed_panel_profile(
            problem, jnp.array([0.2]), lag=False, effects="unit"
        )
        assert result[-1] < 5  # noqa: PLR2004
        assert jnp.isneginf(result[3])


@pytest.mark.component
@pytest.mark.property
@examples(4)
@example(seed=37)
@given(seed=st.integers(0, 2**32 - 1))
def test_selected_panel_is_invariant_to_graph_node_permutation(seed: int) -> None:
    """Reordering nodes and observations together preserves the likelihood."""
    rng = np.random.default_rng(seed)
    n = 5
    w = np.roll(np.eye(n), 1, axis=1)
    x = rng.normal(size=(n, 2))
    selected = (0, 2, 4)
    y = rng.normal(size=3)
    order = np.array([3, 1, 4, 0, 2])
    remapped = tuple(int(np.flatnonzero(order == i)[0]) for i in selected)
    with jax.enable_x64():
        original = ObservedSpatialPanelProblem(
            (
                ObservedSpatialPeriod(
                    prepare_spatial_weights(w), tuple(range(n)), selected
                ),
            ),
            (x,),
            (y,),
            n,
            process=SAR(),
            effects="pooled",
        )
        reordered = ObservedSpatialPanelProblem(
            (
                ObservedSpatialPeriod(
                    prepare_spatial_weights(w[np.ix_(order, order)]),
                    tuple(order.tolist()),
                    remapped,
                ),
            ),
            (x[order],),
            (y,),
            n,
            process=SAR(),
            effects="pooled",
        )
        one = observed_panel_profile(original, jnp.array([0.2]), lag=True)
        two = observed_panel_profile(reordered, jnp.array([0.2]), lag=True)
        np.testing.assert_allclose(one[0], two[0], atol=1e-10)
        np.testing.assert_allclose(one[2:4], two[2:4], atol=1e-10)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("effects", ["pooled", "random"])
def test_observed_profile_respects_input_precision(
    effects: Literal["pooled", "random"],
) -> None:
    """A small selected panel preserves float32 and agrees on its natural scale."""
    n, t = 5, 3
    with jax.enable_x64():
        outputs = []
        for dtype in (np.float32, np.float64):
            w = np.roll(np.eye(n, dtype=dtype), 1, axis=1)
            graph = prepare_spatial_weights(w)
            periods = tuple(
                ObservedSpatialPeriod(graph, tuple(range(n)), (0, 1, 2, 3))
                for _ in range(t)
            )
            local_rng = np.random.default_rng(77)
            x = tuple(local_rng.normal(size=(n, 1)).astype(dtype) for _ in periods)
            y = tuple(local_rng.normal(size=4).astype(dtype) for _ in periods)
            problem = ObservedSpatialPanelProblem(
                periods, x, y, n, process=SAR(), effects="pooled"
            )
            result = observed_panel_profile(
                problem,
                jnp.asarray([0.2], dtype=dtype),
                lag=True,
                effects=effects,
                variance_ratio=jnp.asarray(0.3, dtype=dtype),
            )
            assert result[0].dtype == jnp.dtype(dtype)
            outputs.append(result)
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            outputs[0][2:4], outputs[1][2:4], rtol=2e-5, atol=2e-6
        )


@pytest.mark.component
@pytest.mark.reference
@examples(4)
@example(seed=70)
@given(seed=st.integers(0, 2**32 - 1))
def test_random_effect_information_matches_independent_finite_difference(
    seed: int,
) -> None:
    """Profile the shared variance and scale in the observed information."""
    from statadic.estimation.spatial import ObservedPanelML  # noqa: PLC0415
    from scipy.optimize._numdiff import approx_derivative  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n, t = 5, 5
    w = np.roll(np.eye(n), 1, axis=1)

    def objective(point: np.ndarray) -> float:
        beta, rho, log_sigma2, log_ratio = point
        inverse = np.linalg.solve(np.eye(n) - rho * w, np.eye(n))
        noise = block_diag(*([inverse @ inverse.T] * t))
        effect = np.concatenate([inverse] * t)
        covariance = noise + np.exp(log_ratio) * effect @ effect.T
        root = np.linalg.cholesky(covariance)
        residual = np.concatenate(
            [yt - inverse @ xt[:, 0] * beta for xt, yt in zip(x, y, strict=True)]
        )
        residual = np.linalg.solve(root, residual)
        return 0.5 * (
            2 * np.log(np.diag(root)).sum()
            + t * n * (np.log(2 * np.pi) + log_sigma2)
            + residual @ residual * np.exp(-log_sigma2)
        )

    def gradient(point: np.ndarray) -> np.ndarray:
        def evaluate_objective_vector(q: np.ndarray) -> ndarray:
            return np.array([objective(q)])

        return approx_derivative(
            evaluate_objective_vector, point, method="3-point"
        ).reshape(-1)

    with jax.enable_x64():
        graph = prepare_spatial_weights(w)
        periods = tuple(
            ObservedSpatialPeriod(graph, tuple(range(n)), tuple(range(n)))
            for _ in range(t)
        )
        alpha = rng.normal(size=n)
        x = tuple(rng.normal(size=(n, 1)) for _ in periods)
        y = tuple(z[:, 0] * 0.8 + alpha + rng.normal(size=n) * 0.5 for z in x)
        fit = ObservedPanelML(max_steps=60, tolerance=1e-06).estimate(
            ObservedSpatialPanelProblem(
                periods, x, y, n, process=SAR(), slx_columns=(), effects="random"
            )
        )
        assert fit.valid
        assert fit.information_valid
        state = np.array(
            [
                float(fit.parameters[0]),
                float(fit.parameters[1]),
                np.log(float(fit.sigma2)),
                np.log(float(fit.variance_ratio)),
            ]
        )

        hessian = approx_derivative(gradient, state, method="3-point")
        expected = hessian[:2, :2] - hessian[:2, 2:] @ np.linalg.solve(
            hessian[2:, 2:], hessian[2:, :2]
        )
        np.testing.assert_allclose(fit.information, expected, rtol=1e-3, atol=3e-4)


def _impact_problem(seed: int) -> tuple[np.ndarray, ObservedSpatialPanelProblem]:
    """Construct the small observed panel used by fit-availability witnesses."""
    rng = np.random.default_rng(seed)
    n, t = 5, 3
    w = np.roll(np.eye(n), 1, axis=1)
    graph = prepare_spatial_weights(w)
    periods = tuple(
        ObservedSpatialPeriod(graph, tuple(range(n)), (0, 1, 2, 3)) for _ in range(t)
    )
    x = tuple(rng.normal(size=(n, 1)) for _ in periods)
    y = tuple(rng.normal(size=4) for _ in periods)
    return w, ObservedSpatialPanelProblem(
        periods, x, y, n, process=SAR(), effects="pooled"
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("lag", [True, False])
def test_observed_covariance_and_period_impacts(*, lag: bool) -> None:
    """An interior-fit witness composes observed information and mean derivatives."""
    from statadic.estimation.spatial import SAR, SEM, ObservedPanelML  # noqa: PLC0415
    from statadic.inference.spatial import (  # noqa: PLC0415
        spatial_covariance,
        spatial_panel_impacts,
    )

    with jax.enable_x64():
        w, problem = _impact_problem(19)
        n, t = problem.n_units, len(problem.periods)
        fit = ObservedPanelML(max_steps=40).estimate(
            replace(
                problem,
                process=SAR() if lag else SEM(),
                slx_columns=(0,),
                effects="pooled",
            )
        )
        assert fit.valid
        assert fit.information_valid
        uncertainty = spatial_covariance(fit, method="observed")
        assert uncertainty.valid
        assert uncertainty.sample_capacity == 12  # noqa: PLR2004
        np.testing.assert_allclose(
            uncertainty.covariance.values,
            np.linalg.inv(fit.information),
            rtol=1e-8,
            atol=1e-8,
        )
        impacts = spatial_panel_impacts(uncertainty)
        assert impacts.valid
        assert impacts.uncertainty_valid
        beta, theta, spatial = np.asarray(fit.parameters)
        inverse = np.linalg.solve(np.eye(n) - (spatial if lag else 0) * w, np.eye(n))
        derivative = inverse @ (beta * np.eye(n) + theta * w)
        direct = np.trace(derivative) / n
        total = derivative.sum() / n
        np.testing.assert_allclose(
            impacts.values,
            np.tile([direct, total - direct, total], (t, 1, 1)),
            atol=1e-9,
        )
        with pytest.raises(ValueError, match="no debiasing"):
            spatial_covariance(fit, method="observed", debiased=True)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("max_steps", [40, 256])
def test_observed_boundary_fit_is_unavailable(max_steps: int) -> None:
    """Seed 744 converges at the boundary; extra iterations cannot qualify it."""
    from statadic.estimation.spatial import SAR, ObservedPanelML  # noqa: PLC0415
    from statadic.inference.spatial import spatial_covariance  # noqa: PLC0415

    with jax.enable_x64():
        _, problem = _impact_problem(744)
        fit = ObservedPanelML(max_steps=max_steps).estimate(
            replace(problem, process=SAR(), slx_columns=(0,), effects="pooled")
        )
        assert fit.solver.converged
        assert fit.identified
        assert fit.boundary
        assert not fit.valid
        assert not fit.information_valid
        assert np.isnan(fit.parameters).all()
        assert not spatial_covariance(fit, method="observed").valid


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("lag", [True, False])
@examples(4)
@example(seed=744)
@given(seed=st.integers(0, 2**32 - 1))
def test_observed_covariance_preserves_fit_availability(
    *, lag: bool, seed: int
) -> None:
    """Generated samples may be unavailable; inference must retain that outcome."""
    from statadic.estimation.spatial import SAR, SEM, ObservedPanelML  # noqa: PLC0415
    from statadic.inference.spatial import spatial_covariance  # noqa: PLC0415

    with jax.enable_x64():
        _, problem = _impact_problem(seed)
        fit = ObservedPanelML(max_steps=40).estimate(
            replace(
                problem,
                process=SAR() if lag else SEM(),
                slx_columns=(0,),
                effects="pooled",
            )
        )
        if fit.valid:
            assert np.isfinite(fit.parameters).all()
        else:
            assert np.isnan(fit.parameters).all()
            assert not spatial_covariance(fit, method="observed").valid
