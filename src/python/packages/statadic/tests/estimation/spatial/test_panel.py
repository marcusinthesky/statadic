# standalone package-maintenance entry point
"""Spatial panels against independently assembled joint likelihoods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.results import SolverResult, SolverStatus
from statadic.estimation.spatial import SAC, SAR, SEM, PanelML, SpatialPanelProblem, panel
from statadic.estimation.spatial.panel import spatial_panel_profile
from statadic.operators.spatial import prepare_spatial_panel_weights
from scipy.linalg import block_diag

from ..._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.estimation.spatial.panel import SpatialPanelFit
    from statadic.operators.spatial import SpatialPanelWeights


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(126)
    t, n = 5, 8
    graphs = []
    for _ in range(t):
        w = rng.uniform(size=(n, n))
        np.fill_diagonal(w, 0)
        w /= w.sum(axis=1, keepdims=True)
        graphs.append(w)
    x = rng.normal(size=(t, n, 2))
    y = rng.normal(size=(t, n))
    return x, y, np.stack(graphs)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("model", ["sar", "sem", "sac"])
@pytest.mark.parametrize("effects", ["pooled", "unit", "random"])
def test_time_varying_profile_matches_full_joint_gls(  # noqa: PLR0914
    model: Literal["sar", "sem", "sac"], effects: Literal["pooled", "unit", "random"]
) -> None:
    """Use actual noncommuting W_t and an independent NT-square oracle only in tests."""
    x, y, w = _sample()
    t, n, p = x.shape
    weights = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    rho = 0.3 if model != "sem" else 0.0
    lam = -0.2 if model != "sar" else 0.0
    spatial = jnp.asarray(
        [rho, lam] if model == "sac" else [rho if model == "sar" else lam]
    )
    native = spatial_panel_profile(
        spatial,
        jnp.asarray(x),
        jnp.asarray(y),
        weights,
        lag=model != "sem",
        error=model != "sar",
        effects=effects,
        variance_ratio=jnp.asarray(0.7),
    )
    a = block_diag(*[np.eye(n) - rho * g for g in w])
    b = block_diag(*[np.eye(n) - lam * g for g in w])
    d = np.tile(np.eye(n), (t, 1))
    transformed_y = a @ y.ravel()
    design = x.reshape(t * n, p)
    if effects == "random":
        inverse_b = np.linalg.inv(b)
        covariance = inverse_b @ inverse_b.T + 0.7 * d @ d.T
        root = np.linalg.cholesky(covariance)
        xx = np.linalg.solve(root, design)
        yy = np.linalg.solve(root, transformed_y)
        logdet = np.linalg.slogdet(a)[1] - np.linalg.slogdet(root)[1]
    else:
        xx = b @ (np.column_stack([design, d]) if effects == "unit" else design)
        yy = b @ transformed_y
        logdet = np.linalg.slogdet(a)[1] + np.linalg.slogdet(b)[1]
    coefficient = np.linalg.lstsq(xx, yy, rcond=None)[0]
    residual = yy - xx @ coefficient
    sigma2 = residual @ residual / (t * n)
    likelihood = logdet - t * n / 2 * (np.log(2 * np.pi * sigma2) + 1)
    np.testing.assert_allclose(native[0], coefficient[:p], atol=1e-10)
    np.testing.assert_allclose(native[2], sigma2, atol=1e-10)
    np.testing.assert_allclose(native[3], likelihood, atol=1e-10)
    if effects == "unit":
        np.testing.assert_allclose(native[1], coefficient[p:], atol=1e-10)
    order = np.array([2, 0, 4, 1, 7, 3, 6, 5])
    assert not np.allclose(w[0] @ w[1], w[1] @ w[0])
    reordered = spatial_panel_profile(
        spatial,
        jnp.asarray(x[:, order]),
        jnp.asarray(y[:, order]),
        prepare_spatial_panel_weights(
            w[:, order][:, :, order], period_ids=np.arange(t), unit_ids=order
        ),
        lag=model != "sem",
        error=model != "sar",
        effects=effects,
        variance_ratio=jnp.asarray(0.7),
    )
    np.testing.assert_allclose(reordered[0], native[0], atol=1e-10)
    np.testing.assert_allclose(reordered[1], native[1][order], atol=1e-10)
    np.testing.assert_allclose(reordered[2:4], native[2:4], atol=1e-10)
    np.testing.assert_allclose(reordered[4], native[4][:, order], atol=1e-10)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.parametrize("model", ["sar", "sem"])
@examples(5)
@example(seed=53)
@given(seed=st.integers(0, 2**32 - 1))
def test_fixed_weights_fixed_effects_match_spreg(
    optional_reference: Callable[[str], ModuleType],
    model: Literal["sar", "sem", "sac"],
    seed: int,
) -> None:
    """Constant W reduces to the matched fixed-effects spatial panel model."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(seed)
    t, n, p = 8, 12, 2
    w = 0.5 * np.roll(np.eye(n), 1, axis=1) + 0.5 * np.roll(np.eye(n), -1, axis=1)
    x = rng.normal(size=(t, n, p))
    alpha = rng.normal(size=n)
    innovation = rng.normal(size=(t, n)) * 0.4
    mean = x @ np.array([0.5, -0.4]) + alpha
    y = (
        np.linalg.solve(np.eye(n) - 0.3 * w, (mean + innovation).T).T
        if model == "sar"
        else mean + np.linalg.solve(np.eye(n) + 0.2 * w, innovation.T).T
    )
    weights = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    fit = PanelML().estimate(
        SpatialPanelProblem(
            x,
            y,
            weights,
            process=SAR() if model == "sar" else SEM(),
            slx_columns=(),
            effects="unit",
        )
    )
    oracle = (spreg.Panel_FE_Lag if model == "sar" else spreg.Panel_FE_Error)(
        y.reshape(-1, 1), x.reshape(t * n, p), libpysal.weights.full2W(w), epsilon=1e-9
    )
    assert fit.valid
    np.testing.assert_allclose(
        fit.parameters, np.asarray(oracle.betas).ravel(), atol=2e-6
    )
    # spreg's panel profile uses log(SSE), not log(SSE/NT), retaining
    # the parameter-independent NT/2*log(NT) offset in its reported logll.
    np.testing.assert_allclose(
        fit.log_likelihood, oracle.logll + (t * n) / 2 * np.log(t * n), atol=1e-7
    )
    np.testing.assert_allclose(
        np.linalg.inv(fit.information)[:-1, :-1], oracle.vm, rtol=2e-5, atol=2e-7
    )


@pytest.mark.component
@pytest.mark.contract
def test_varying_weights_fit_is_transformable_and_rejects_reordering() -> None:
    """JIT/vmap preserve period-specific graphs and unit keys must agree."""
    x, y, w = _sample()
    weights = prepare_spatial_panel_weights(w, period_ids=np.arange(x.shape[0]))

    def evaluate_fit(z: jax.Array) -> SpatialPanelFit:
        return PanelML().estimate(
            SpatialPanelProblem(
                jnp.asarray(x),
                z,
                weights,
                process=SAR(),
                slx_columns=(),
                effects="unit",
            )
        )

    evaluate_fit = jax.jit(evaluate_fit)(jnp.asarray(y))
    assert evaluate_fit.valid

    def evaluate_fit_3(z: jax.Array) -> SpatialPanelFit:
        return PanelML().estimate(
            SpatialPanelProblem(
                jnp.asarray(x),
                z,
                weights,
                process=SAR(),
                slx_columns=(),
                effects="unit",
            )
        )

    batched = jax.jit(jax.vmap(evaluate_fit_3))(jnp.asarray(np.stack([y, y * 2])))
    assert np.all(batched.valid)
    keys = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    keys[-1] = keys[-1, ::-1]
    with pytest.raises(ValueError, match="ordering"):
        prepare_spatial_panel_weights(
            w, period_ids=np.arange(x.shape[0]), unit_ids=keys
        )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("model", ["sar", "sem"])
def test_random_effects_varying_weights_fit_matches_independent_marginal_ml(  # noqa: PLR0914
    model: Literal["sar", "sem", "sac"],
) -> None:
    """Optimize an independently assembled full covariance, including changing W_t."""
    from scipy.optimize import minimize  # noqa: PLC0415

    rng = np.random.default_rng(38)
    t, n, p = 6, 8, 2
    graphs = []
    for _ in range(t):
        graph = rng.uniform(size=(n, n))
        np.fill_diagonal(graph, 0)
        graph /= graph.sum(axis=1, keepdims=True)
        graphs.append(graph)
    w = np.stack(graphs)
    x = np.concatenate([np.ones((t, n, 1)), rng.normal(size=(t, n, 1))], axis=2)
    alpha = rng.normal(size=n) * 0.8
    errors = rng.normal(size=(t, n)) * 0.4
    mean = x @ np.array([0.4, -0.6]) + alpha
    y = np.stack(
        [
            np.linalg.solve(np.eye(n) - 0.3 * w[i], mean[i] + errors[i])
            if model == "sar"
            else mean[i] + np.linalg.solve(np.eye(n) + 0.2 * w[i], errors[i])
            for i in range(t)
        ]
    )
    weights = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    native = PanelML(tolerance=1e-08).estimate(
        SpatialPanelProblem(
            x,
            y,
            weights,
            process=SAR() if model == "sar" else SEM(),
            slx_columns=(),
            effects="random",
        )
    )
    assert native.valid
    d = np.tile(np.eye(n), (t, 1))
    design = x.reshape(t * n, p)

    def objective(state: np.ndarray) -> float:
        spatial = 0.999 * np.tanh(state[0])
        ratio = np.exp(state[1])
        a = block_diag(
            *[np.eye(n) - spatial * g if model == "sar" else np.eye(n) for g in w]
        )
        b = block_diag(
            *[np.eye(n) - spatial * g if model == "sem" else np.eye(n) for g in w]
        )
        bi = np.linalg.inv(b)
        covariance = bi @ bi.T + ratio * d @ d.T
        root = np.linalg.cholesky(covariance)
        xx = np.linalg.solve(root, design)
        yy = np.linalg.solve(root, a @ y.ravel())
        beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
        residual = yy - xx @ beta
        return float(
            np.linalg.slogdet(root)[1]
            - np.linalg.slogdet(a)[1]
            + t * n / 2 * (np.log(2 * np.pi * np.mean(residual**2)) + 1)
        )

    oracle = minimize(
        objective, np.array([0.0, 0.0]), method="BFGS", options={"gtol": 1e-7}
    )
    np.testing.assert_allclose(
        native.parameters[-1], 0.999 * np.tanh(oracle.x[0]), atol=3e-5
    )
    np.testing.assert_allclose(
        native.variance_ratio, np.exp(oracle.x[1]), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(native.log_likelihood, -oracle.fun, atol=1e-6)


@pytest.mark.component
@pytest.mark.reference
def test_random_error_profile_gradient_at_zero_has_no_eigenspace_singularity() -> None:
    """The identity filter has a repeated spectrum but a smooth likelihood."""
    x, y, w = _sample()
    weights = prepare_spatial_panel_weights(w, period_ids=np.arange(x.shape[0]))

    def objective(lam: jax.Array) -> jax.Array:
        return spatial_panel_profile(
            lam[None],
            jnp.asarray(x),
            jnp.asarray(y),
            weights,
            lag=False,
            error=True,
            effects="random",
            variance_ratio=jnp.asarray(0.7),
        )[3]

    derivative = jax.jit(jax.grad(objective))(jnp.asarray(0.0))
    h = 1e-5
    finite_difference = (objective(jnp.asarray(h)) - objective(jnp.asarray(-h))) / (
        2 * h
    )
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, finite_difference, atol=1e-7)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("model", ["sar", "sem", "sac"])
@examples(5)
@example(seed=0)
@example(seed=7)
@example(seed=42)
@example(seed=2779)
@given(seed=st.integers(0, 2**32 - 1))
def test_pooled_fixed_graph_path_equals_per_period_path(
    model: Literal["sar", "sem", "sac"],
    seed: int,
) -> None:
    """The fixed-W fast path reproduces the broadcast per-period likelihood.

    A (T,N,N) stack of identical graphs takes the general path. W is a
    nonsymmetric cyclic-plus-skew operator with complex eigenvalues, and one
    evaluation sits next to the stability bound.
    """
    rng = np.random.default_rng(seed)
    t, n = 30, 9
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    assert np.max(np.abs(np.linalg.eigvals(w).imag)) > 0.1  # noqa: PLR2004
    x = np.stack([np.ones((t, n)), rng.normal(size=(t, n))], axis=2)
    y = rng.normal(size=(t, n)) @ np.linalg.inv(np.eye(n) - 0.4 * w).T
    fixed = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    stacked = prepare_spatial_panel_weights(
        np.broadcast_to(w, (t, n, n)), period_ids=np.arange(t)
    )
    lag, error = model != "sem", model != "sar"
    for spatial in (
        ([0.4], [-0.3], [0.995], [-0.99])
        if model != "sac"
        else (
            [0.4, -0.3],
            [0.99, 0.5],
        )
    ):
        args = (jnp.asarray(spatial), jnp.asarray(x), jnp.asarray(y))
        fast = spatial_panel_profile(
            *args, fixed, lag=lag, error=error, effects="pooled"
        )
        general = spatial_panel_profile(
            *args, stacked, lag=lag, error=error, effects="pooled"
        )
        for a, b in zip(fast, general, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("model", ["sar", "sem", "sac"])
def test_pooled_fixed_graph_optimizer_parity(
    model: Literal["sar", "sem", "sac"],
) -> None:
    """The original seed compares the two optimizer paths."""
    rng = np.random.default_rng(7)
    t, n = 30, 9
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    x = np.stack([np.ones((t, n)), rng.normal(size=(t, n))], axis=2)
    y = rng.normal(size=(t, n)) @ np.linalg.inv(np.eye(n) - 0.4 * w).T
    fixed = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    stacked = prepare_spatial_panel_weights(
        np.broadcast_to(w, (t, n, n)), period_ids=np.arange(t)
    )
    estimator = PanelML()
    fit_fast = estimator.estimate(
        SpatialPanelProblem(
            x,
            y,
            fixed,
            process={"sar": SAR(), "sem": SEM(), "sac": SAC()}[model],
            slx_columns=(),
            effects="pooled",
        )
    )
    fit_general = estimator.estimate(
        SpatialPanelProblem(
            x,
            y,
            stacked,
            process={"sar": SAR(), "sem": SEM(), "sac": SAC()}[model],
            slx_columns=(),
            effects="pooled",
        )
    )
    # SAC lag and error are nearly collinear on this circulant graph, so its
    # multistart optimiser stops at different points of a flat ridge; the
    # profile values (above, rtol 1e-10) are the exact comparison and the fitted
    # log-likelihoods agree to the optimiser tolerance, rtol 1e-6.
    if model != "sac":
        np.testing.assert_allclose(
            fit_fast.parameters, fit_general.parameters, rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(fit_fast.sigma2, fit_general.sigma2, rtol=1e-8)
    if model == "sac":
        # Compare the numerical paths at retained iterates, including when a
        # fit is unavailable. This is not a comparison of usable fitted states.
        for fit in (fit_fast, fit_general):
            if bool(fit.solver.converged):
                assert float(fit.solver.residual_norm) <= estimator.tolerance
            else:
                assert not bool(fit.valid)
                assert np.isnan(fit.log_likelihood)
        likelihoods = [
            spatial_panel_profile(
                fit.solver.value,
                jnp.asarray(x),
                jnp.asarray(y),
                weights,
                lag=True,
                error=True,
                effects="pooled",
            )[3]
            for fit, weights in ((fit_fast, fixed), (fit_general, stacked))
        ]
    else:
        likelihoods = [fit_fast.log_likelihood, fit_general.log_likelihood]
    np.testing.assert_allclose(*likelihoods, rtol=1e-8 if model != "sac" else 1e-6)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("model", [SAR, SEM])
def test_random_effects_acceptance_requires_average_score(
    model: type[SAR | SEM],
) -> None:
    """A converged random-effects fit has a small score in lambda and log ratio."""
    x, y, ws = _sample()
    t, n, _ = x.shape
    weights = prepare_spatial_panel_weights(ws[0], period_ids=np.arange(t))
    estimator = PanelML()
    fit = estimator.estimate(
        SpatialPanelProblem(
            x, y, weights, process=model(), slx_columns=(), effects="random"
        )
    )
    limit = weights.limit * 0.999

    def average_negative_profile(state: jax.Array) -> jax.Array:
        return -spatial_panel_profile(
            limit * state[:1],
            jnp.asarray(x),
            jnp.asarray(y),
            weights,
            lag=model is SAR,
            error=model is SEM,
            effects="random",
            variance_ratio=jnp.exp(state[1]),
        )[3] / (t * n)

    state = jnp.stack([fit.parameters[-1] / limit, jnp.log(fit.variance_ratio)])
    score = jnp.max(jnp.abs(jax.grad(average_negative_profile)(state)))
    boundary = bool(fit.variance_ratio <= 0.0)
    assert (
        boundary
        or not bool(fit.solver.converged)
        or float(score) <= 10 * (estimator.tolerance)
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("stacked", [False, True])
def test_pooled_sac_seed_zero_acceptance_requires_average_score(
    *, stacked: bool
) -> None:
    """Check accepted fits against an independent physical-coordinate score."""
    rng = np.random.default_rng(0)
    t, n = 30, 9
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    x = np.stack([np.ones((t, n)), rng.normal(size=(t, n))], axis=2)
    y = rng.normal(size=(t, n)) @ np.linalg.inv(np.eye(n) - 0.4 * w).T
    graph = np.broadcast_to(w, (t, n, n)) if stacked else w
    weights = prepare_spatial_panel_weights(graph, period_ids=np.arange(t))
    estimator = PanelML()
    fit = estimator.estimate(
        SpatialPanelProblem(
            x, y, weights, process=SAC(), slx_columns=(), effects="pooled"
        )
    )
    limit = weights.limit * 0.999

    def average_negative_profile(u: jax.Array) -> jax.Array:
        return -spatial_panel_profile(
            limit * u,
            jnp.asarray(x),
            jnp.asarray(y),
            weights,
            lag=True,
            error=True,
            effects="pooled",
        )[3] / (t * n)

    score = jnp.max(
        jnp.abs(jax.grad(average_negative_profile)(fit.solver.value / limit))
    )
    np.testing.assert_allclose(fit.solver.residual_norm, score, rtol=1e-5, atol=1e-10)
    assert not bool(fit.solver.converged) or float(score) <= estimator.tolerance
    assert bool(fit.valid) <= bool(fit.solver.converged)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(("slope", "roundoff"), [(4e-7, True), (1e-4, False)])
def test_panel_score_acceptance_respects_precision(
    monkeypatch: pytest.MonkeyPatch, *, x64: bool, slope: float, roundoff: bool
) -> None:
    """Float32 rounding is tolerated, but a substantive score still rejects success."""

    def backend(
        objective: object, initial: jax.Array, **kwargs: object
    ) -> SolverResult[jax.Array]:
        del objective, kwargs
        return SolverResult(
            jnp.zeros_like(initial),
            jnp.asarray(True),  # noqa: FBT003
            jnp.asarray(10),
            jnp.asarray(SolverStatus.SUCCESS),
            jnp.asarray(0.0),
            jnp.asarray(1),
            jnp.asarray(10),
        )

    def profile(
        spatial: jax.Array, ratio: jax.Array | None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        del ratio
        value = slope * spatial.sum()
        return spatial, spatial, value, -value, spatial

    monkeypatch.setattr(panel, "minimise_bfgs_result", backend)
    with jax.enable_x64(new_val=x64):
        selected, _ = panel._multistart_panel_search(  # noqa: SLF001
            profile,
            2,
            jnp.asarray(1.0),
            random_effects=False,
            max_steps=10,
            tolerance=1e-7,
            score_sample_size=1,
        )
    assert bool(selected.converged) == (not x64 and roundoff)
    assert float(selected.residual_norm) == pytest.approx(slope)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("converged", [False, True])
@pytest.mark.parametrize("boundary_gain", [-1.0, 0.0, 1.0])
def test_random_boundary_selection_preserves_likelihood(
    monkeypatch: pytest.MonkeyPatch, *, converged: bool, boundary_gain: float
) -> None:
    """A converged boundary cannot replace a better positive-variance candidate."""

    def result(*, success: bool) -> SolverResult[jax.Array]:
        return SolverResult(
            jnp.asarray([0.2]),
            jnp.asarray(success),
            jnp.asarray(1),
            jnp.asarray(SolverStatus.SUCCESS if success else SolverStatus.STALLED),
            jnp.asarray(0.0),
            jnp.asarray(1),
            jnp.asarray(10),
        )

    def positive(
        *args: object, **kwargs: object
    ) -> tuple[SolverResult[jax.Array], jax.Array]:
        del args, kwargs
        return result(success=converged), jnp.asarray(2.0)

    def boundary(*args: object, **kwargs: object) -> SolverResult[jax.Array]:
        del args, kwargs
        return result(success=True)

    def profile(
        spatial: jax.Array, ratio: jax.Array | None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        likelihood = 10 + jnp.where(ratio == 0, boundary_gain, 0)
        return spatial, spatial, spatial, likelihood, spatial

    monkeypatch.setattr(panel, "_multistart_panel_search", positive)
    monkeypatch.setattr(panel, "bounded_scalar_minimise", boundary)
    selected, ratio = panel._panel_search(  # noqa: SLF001
        profile, 1, jnp.asarray(1.0), effects="random", max_steps=10, tolerance=1e-7
    )
    assert float(ratio) == pytest.approx(0 if boundary_gain >= 0 else 2)
    assert bool(selected.converged) == (converged or boundary_gain >= 0)


@pytest.mark.component
@pytest.mark.contract
def test_pooled_sac_rejects_false_success_independent_of_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject backend success with a large score and screen before selection."""
    tolerance = 0.01

    def false_success(
        objective: object, initial: jax.Array, **kwargs: object
    ) -> SolverResult[jax.Array]:
        del objective, kwargs
        return SolverResult(
            initial,
            jnp.asarray(True),  # noqa: FBT003
            jnp.asarray(1),
            jnp.asarray(SolverStatus.SUCCESS),
            jnp.asarray(0.0),
            jnp.asarray(2),
            jnp.asarray(10),
        )

    monkeypatch.setattr(panel, "minimise_bfgs_result", false_success)

    def search(offset: float) -> SolverResult[jax.Array]:
        def profile(
            s: jax.Array, ratio: jax.Array | None
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            del ratio
            value = jnp.sum((s - 0.2) ** 2) + offset
            return s, s, value, -value, s

        return panel._multistart_panel_search(  # noqa: SLF001
            profile,
            2,
            jnp.asarray(1.0),
            random_effects=False,
            max_steps=10,
            tolerance=tolerance,
            score_sample_size=1,
        )[0]

    original, shifted = search(0.0), search(100.0)
    for result in (original, shifted):
        assert not bool(result.converged)
        assert int(result.status) == SolverStatus.STALLED
        assert float(result.residual_norm) > tolerance
    np.testing.assert_allclose(original.value, shifted.value)
    np.testing.assert_allclose(original.residual_norm, shifted.residual_norm)

    def mixed_profile(
        s: jax.Array, ratio: jax.Array | None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        del ratio
        value = jnp.sum(s**2 - 3 * s**3)
        return s, s, value, -value, s

    selected, _ = panel._multistart_panel_search(  # noqa: SLF001
        mixed_profile,
        2,
        jnp.asarray(1.0),
        random_effects=False,
        max_steps=10,
        tolerance=tolerance,
        score_sample_size=1,
    )
    assert bool(selected.converged)
    np.testing.assert_allclose(selected.value, [0.0, 0.0])


@pytest.mark.component
@pytest.mark.contract
def test_stalled_recovery_preserves_budget_and_failure_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlled backend isolates retry eligibility under JIT and batching."""
    budget = 6
    tolerance = 1e-7

    def restarted(
        objective: object,
        initial: jax.Array,
        *,
        max_steps: int,
        rtol: float,
        atol: float,
    ) -> SolverResult[jax.Array]:
        del objective, rtol, atol
        target = jnp.full_like(initial, jnp.arctanh(0.2))
        return SolverResult(
            target,
            jnp.asarray(True),  # noqa: FBT003
            jnp.asarray(2),
            jnp.asarray(SolverStatus.SUCCESS),
            jnp.asarray(0.0, dtype=initial.dtype),
            jnp.asarray(-1),
            jnp.asarray(max_steps),
        )

    monkeypatch.setattr(panel, "minimise_bfgs_result", restarted)

    def recover(
        initial: jax.Array, status: jax.Array, iterations: jax.Array
    ) -> SolverResult[jax.Array]:
        def average_objective(u: jax.Array) -> jax.Array:
            return jnp.sum((u - 0.2) ** 2)

        def objective(z: jax.Array) -> jax.Array:
            return average_objective(jnp.tanh(z))

        previous = SolverResult(
            initial,
            jnp.asarray(False),  # noqa: FBT003
            iterations,
            status,
            jnp.asarray(0.4, dtype=initial.dtype),
            jnp.asarray(-1),
            jnp.asarray(budget),
        )
        return panel._recover_stalled_panel_search(  # noqa: SLF001
            objective,
            average_objective,
            previous,
            jnp.asarray(1.0, dtype=initial.dtype),
            dimension=2,
            max_steps=budget,
            tolerance=tolerance,
        )

    initials = jnp.asarray([[0.0, 0.0]] * 3 + [[jnp.nan, 0.0], [20.0, 20.0]])
    statuses = jnp.asarray(
        [SolverStatus.STALLED] * 2
        + [SolverStatus.MAX_STEPS]
        + [SolverStatus.STALLED] * 2
    )
    steps = jnp.asarray([2, 4, 2, 2, 2])
    result = jax.jit(jax.vmap(recover))(initials, statuses, steps)
    np.testing.assert_array_equal(result.converged, [True, False, False, False, False])
    np.testing.assert_array_equal(result.iterations, [4, 4, 2, 2, 2])
    assert bool(jnp.all(result.iterations <= result.max_steps))
    np.testing.assert_allclose(jnp.tanh(result.value[0]), [0.2, 0.2])
    np.testing.assert_allclose(result.value[1:], initials[1:], equal_nan=True)
    assert float(result.residual_norm[0]) <= tolerance


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("model", ["sar", "sem", "sac"])
def test_fixed_graph_entity_information_matches_nuisance_elimination(
    model: str,
) -> None:
    """Time-centering equals the full conditional Fisher Schur complement."""
    x, y, ws = _sample()
    t, n, _ = x.shape
    w = ws[0]
    fixed = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    stacked = prepare_spatial_panel_weights(
        np.broadcast_to(w, (t, n, n)), period_ids=np.arange(t)
    )
    lag, error = model != "sem", model != "sar"
    spatial = jnp.asarray([0.3, -0.2] if lag and error else [0.3])
    args = (jnp.asarray([0.25, -0.3]), jnp.asarray(y[0]), spatial, jnp.asarray(0.7))

    def information(weights: SpatialPanelWeights) -> jax.Array:
        return panel._fixed_information(  # noqa: SLF001 -- compare two mathematical reductions
            jnp.asarray(x), weights, *args, lag=lag, error=error, fixed=True
        )

    np.testing.assert_allclose(
        information(fixed), information(stacked), rtol=1e-10, atol=1e-11
    )
