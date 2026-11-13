"""Independent matrix references for retained IV weights and covariance."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import OneWayCluster, PlainLongRun
from statadic.estimation.iv import (
    IVProblem,
    TwoStageLeastSquares,
    TwoStepIVGMM,
    iv_first_stage_diagnostics,
)
from statadic.inference.iv import hansen_j, iv_covariance
from statadic.operators.longrun import newey_west_kernel


@pytest.mark.component
@pytest.mark.reference
def test_iv_actual_weight_sandwich_and_two_step_j() -> None:  # noqa: PLR0914
    """Check the retained IV contract against the selected witness."""
    rng = np.random.default_rng(741)
    n = 128
    z = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
    x = np.column_stack((z[:, 0], z[:, 1] + 0.5 * rng.normal(size=n)))
    y = x @ np.array([1.0, 2.0]) + rng.normal(size=n)
    problem = IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    initial = TwoStageLeastSquares().estimate(problem)
    g0 = z.T @ y / n
    d = z.T @ x / n
    w0 = np.linalg.inv(z.T @ z / n)
    expected = np.linalg.solve(d.T @ w0 @ d, d.T @ w0 @ g0)
    np.testing.assert_allclose(initial.parameters, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(initial.affine.jacobian.values, -d, rtol=2e-5, atol=2e-5)
    assert not any(
        leaf.shape == (n, z.shape[1], x.shape[1])
        for leaf in jax.tree_util.tree_leaves(initial)
    )
    residuals = y - x @ expected
    scores = z * residuals[:, None]
    omega = scores.T @ scores / n
    w1 = np.linalg.inv(omega)
    two_expected = np.linalg.solve(d.T @ w1 @ d, d.T @ w1 @ g0)
    two = TwoStepIVGMM(PlainLongRun).estimate(problem)
    np.testing.assert_allclose(two.parameters, two_expected, rtol=3e-5, atol=3e-5)
    assert bool(two.valid)
    # Two-step completes its single prescribed update; it has no stopping rule.
    assert int(two.iterations) == 1
    assert bool(two.converged)
    u = iv_covariance(two)
    scores2 = z * (y - x @ two_expected)[:, None]
    omega2 = scores2.T @ scores2 / n
    a = np.linalg.solve(d.T @ w1 @ d, d.T @ w1)
    np.testing.assert_allclose(
        u.covariance.values, a @ omega2 @ a.T / n, rtol=1e-4, atol=1e-6
    )
    result = hansen_j(two, assume_correct_specification=True)
    assert bool(result.valid)
    gbar = scores2.mean(axis=0)
    np.testing.assert_allclose(result.statistic, n * gbar @ w1 @ gbar, rtol=1e-4)
    assert not bool(hansen_j(initial, assume_correct_specification=True).valid)
    assert not bool(hansen_j(two, assume_correct_specification=False).valid)
    partial_r2, instrument_rank = iv_first_stage_diagnostics(initial, 1)
    restricted = x[:, 1] - x[:, 1].mean()
    complete = x[:, 1] - z @ np.linalg.lstsq(z, x[:, 1], rcond=None)[0]
    np.testing.assert_allclose(
        partial_r2, [1 - complete @ complete / (restricted @ restricted)], rtol=2e-5
    )
    assert int(instrument_rank) == np.linalg.matrix_rank(z)


@pytest.mark.component
@pytest.mark.reference
def test_instrument_units_preserve_2sls_scores_weights_and_covariance() -> None:
    """An invertible instrument rescaling preserves an independently solved span."""
    t = np.arange(-2.0, 3.0)
    x = np.column_stack((np.ones(5), t))
    y = 1 + 2 * t + np.array([0.1, -0.1, 0.2, -0.1, 0.1])
    z = x.copy()
    scaling = np.array([2.0**15, 2.0**-24])
    q = np.linalg.qr(z)[0]
    projected = q @ q.T
    influence = np.linalg.solve(x.T @ projected @ x, x.T @ projected)
    expected = influence @ y
    residuals = y - x @ expected
    covariance = (influence * residuals**2) @ influence.T
    base = TwoStageLeastSquares().estimate(
        IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    )
    changed = TwoStageLeastSquares().estimate(
        IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z * scaling))
    )
    assert bool(base.valid)
    assert bool(changed.valid)
    np.testing.assert_allclose(changed.parameters, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        changed.affine.weight,
        np.asarray(base.affine.weight) / scaling[:, None] / scaling[None, :],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        changed.affine.scores.values,
        (z * scaling) * (y - x @ np.asarray(changed.parameters))[:, None],
        rtol=2e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        iv_covariance(changed).covariance.values, covariance, rtol=2e-4, atol=1e-6
    )


@pytest.mark.component
@pytest.mark.contract
def test_rank_deficient_instruments_are_unavailable() -> None:
    """Check the retained IV contract against the selected witness."""
    z = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    fit = TwoStageLeastSquares().estimate(IVProblem(z, jnp.arange(3.0), z))
    assert not bool(fit.valid)
    two = TwoStepIVGMM(PlainLongRun).estimate(IVProblem(z, jnp.arange(3.0), z))
    assert not bool(two.valid)
    assert not bool(two.converged)


@pytest.mark.component
@pytest.mark.reference
def test_cluster_and_hac_use_actual_moment_map() -> None:
    """Check the retained IV contract against the selected witness."""
    rng = np.random.default_rng(83)
    n = 60
    z = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
    x = z[:, :2] + np.column_stack((np.zeros(n), rng.normal(size=n) * 0.2))
    y = x @ [1.0, 0.5] + rng.normal(size=n)
    groups = np.repeat(np.arange(6), 10)
    fit = TwoStepIVGMM(OneWayCluster).estimate(
        IVProblem(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), jnp.asarray(groups), 6
        )
    )
    assert bool(fit.valid)
    scores = np.asarray(fit.affine.scores.values)
    a = np.asarray(fit.affine.moment_map)
    totals = np.stack([scores[groups == group].sum(axis=0) for group in range(6)])
    cluster = iv_covariance(
        fit, kind="cluster", groups=jnp.asarray(groups), n_clusters=6
    )
    np.testing.assert_allclose(
        cluster.covariance.values,
        a @ (totals.T @ totals / n) @ a.T / n,
        rtol=2e-4,
        atol=1e-6,
    )
    hac = iv_covariance(fit, kind="HAC", bandwidth=2)
    omega = np.asarray(newey_west_kernel(jnp.asarray(scores), 2))
    np.testing.assert_allclose(
        hac.covariance.values, a @ omega @ a.T / n, rtol=2e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        fit.preliminary_variability,
        np.linalg.inv(np.asarray(fit.affine.weight)),
        rtol=3e-4,
        atol=1e-5,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_exactly_identified_j_is_unavailable() -> None:
    """Check the retained IV contract against the selected witness."""
    rng = np.random.default_rng(3)
    z = np.column_stack((np.ones(40), rng.normal(size=40)))
    y = z @ [1.0, 2.0] + rng.normal(size=40)
    fit = TwoStepIVGMM(PlainLongRun).estimate(
        IVProblem(jnp.asarray(z), jnp.asarray(y), jnp.asarray(z))
    )
    assert bool(fit.valid)
    assert not bool(hansen_j(fit, assume_correct_specification=True).valid)


@pytest.mark.integration
def test_iterated_uses_two_stage_initialization_and_records_iteration() -> None:
    """A bounded iteration begins at the same two-stage fit and records the stop."""
    from statadic.estimation.iv import IteratedIVGMM  # noqa: PLC0415

    rng = np.random.default_rng(18)
    z = np.column_stack((np.ones(70), rng.normal(size=(70, 2))))
    x = np.column_stack((z[:, 0], z[:, 1] + 0.3 * rng.normal(size=70)))
    y = x @ [1.0, 0.4] + rng.normal(size=70)
    problem = IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    start = TwoStageLeastSquares().estimate(problem)
    first = IteratedIVGMM(PlainLongRun, max_iter=1).estimate(problem)
    np.testing.assert_allclose(first.preliminary_parameters, start.parameters)
    assert int(first.iterations) == 1
    assert bool(first.valid)
    if not bool(first.converged):
        assert not bool(hansen_j(first, assume_correct_specification=True).valid)


@pytest.mark.integration
@pytest.mark.contract
def test_invalid_cluster_codes_do_not_create_valid_inference() -> None:
    """Invalid or missing cluster labels cannot silently enter the sandwich."""
    rng = np.random.default_rng(62)
    z = np.column_stack((np.ones(32), rng.normal(size=(32, 2))))
    x = z[:, :2]
    y = x @ [1.0, 0.2] + rng.normal(size=32)
    fit = TwoStageLeastSquares().estimate(
        IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    )
    codes = jnp.asarray([0] * 16 + [2] * 16)
    result = iv_covariance(fit, kind="cluster", groups=codes, n_clusters=2)
    assert not bool(result.valid)


@pytest.mark.component
@pytest.mark.contract
def test_jit_vmap_preserves_rank_failure_as_invalid_lane() -> None:
    """Fixed-shape batched problems preserve unavailable rank-deficient lanes."""
    import jax  # noqa: PLC0415

    x = jnp.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
    y = jnp.array([1.0, 2.0, 4.0, 4.0, 6.0])
    bad = jnp.ones_like(x)
    problems = IVProblem(jnp.stack((x, x)), jnp.stack((y, y)), jnp.stack((x, bad)))
    fits = jax.jit(jax.vmap(TwoStageLeastSquares().estimate))(problems)
    assert bool(fits.valid[0])
    assert not bool(fits.valid[1])
    assert bool(jnp.all(jnp.isnan(fits.parameters[1])))


@pytest.mark.component
@pytest.mark.contract
def test_jitted_uncertainty_and_hansen_return_pytrees() -> None:
    """Native inference remains composable under JAX transformations."""
    import jax  # noqa: PLC0415

    rng = np.random.default_rng(43)
    z = np.column_stack((np.ones(64), rng.normal(size=(64, 2))))
    x = z[:, :2] + np.column_stack((np.zeros(64), rng.normal(size=64) * 0.2))
    y = x @ [1.0, 0.4] + rng.normal(size=64)
    fit = TwoStepIVGMM(PlainLongRun).estimate(
        IVProblem(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    )
    uncertainty = jax.jit(iv_covariance)(fit)
    test = jax.jit(partial(hansen_j, assume_correct_specification=True))(fit)
    assert bool(uncertainty.valid)
    assert bool(test.valid)
    assert len(jax.tree.leaves(uncertainty)) > 1
    assert len(jax.tree.leaves(test)) > 1


@pytest.mark.component
@pytest.mark.contract
def test_cluster_rank_failure_is_lane_safe_under_jit_vmap() -> None:
    """Fewer independent cluster sums than moments cannot yield an inverse."""
    import jax  # noqa: PLC0415

    rng = np.random.default_rng(45)
    n = 30
    z = np.stack(
        [np.column_stack((np.ones(n), rng.normal(size=(n, 2)))) for _ in range(20)]
    )
    x = z[:, :, :2]
    y = x @ np.array([0.5, 1.0]) + rng.normal(size=(20, n))
    groups = np.tile(np.repeat([0, 1], 15), (20, 1))
    problems = IVProblem(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), jnp.asarray(groups), 2
    )
    fits = jax.jit(jax.vmap(TwoStepIVGMM(OneWayCluster).estimate))(problems)
    assert not bool(jnp.any(fits.valid))
    assert bool(jnp.all(jnp.isnan(fits.parameters)))
    control_groups = np.tile(np.repeat(np.arange(5), 6), (20, 1))
    control = IVProblem(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), jnp.asarray(control_groups), 5
    )
    assert bool(
        jnp.all(jax.jit(jax.vmap(TwoStepIVGMM(OneWayCluster).estimate))(control).valid)
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("max_iter", "tolerance"),
    [
        (True, None),
        (1.5, None),
        (0, None),
        (2, True),
        (2, float("nan")),
        (2, float("inf")),
        (2, 0.0),
    ],
)
def test_invalid_iteration_controls_rejected(
    max_iter: float, tolerance: float | None
) -> None:
    """Reject malformed or nonfinite static iteration controls."""
    from statadic.estimation.iv import IteratedIVGMM  # noqa: PLC0415

    x = jnp.array([[1.0], [2.0], [3.0]])
    with pytest.raises((TypeError, ValueError)):
        IteratedIVGMM(
            PlainLongRun, max_iter=cast("int", max_iter), tolerance=tolerance
        ).estimate(IVProblem(x, x[:, 0], x))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("bandwidth", "n_clusters"), [(True, 0), (1.5, 0), (0, True), (0, -1)]
)
def test_invalid_static_bandwidth_and_cluster_count(
    bandwidth: float, n_clusters: int
) -> None:
    """Static bandwidth and cluster count are nonboolean integers in range."""
    x = jnp.array([[1.0], [2.0], [3.0]])
    with pytest.raises((TypeError, ValueError)):
        TwoStepIVGMM(PlainLongRun, bandwidth=cast("int", bandwidth)).estimate(
            IVProblem(x, x[:, 0], x, n_clusters=n_clusters)
        )
