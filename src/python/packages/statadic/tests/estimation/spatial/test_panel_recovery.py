"""Random-panel recovery against the seed-42 rounded float32 witness.

The frozen outcome's first element was moved one ULP down; on the recording
host the retained interior iterate stalled after 72 steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import minimize
from statadic.core.results import SolverResult, SolverStatus
from statadic.estimation.spatial import SAR, PanelML, SpatialPanelProblem, panel
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import prepare_spatial_panel_weights

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray
    from statadic.estimation.spatial.panel import SpatialPanelFit
    from statadic.operators.spatial import SpatialPanelWeights

pytestmark = [pytest.mark.numerical]
WITNESS = Path(__file__).resolve().parents[2] / "_validation/panel_float32_witness.npz"
STALLED_STEPS = 72
EXPECTED_STEPS = 3
RATIO_MIN, RATIO_MAX = 5.3, 5.4
TWO_ULPS = 2


def _witness() -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    with np.load(WITNESS) as data:
        return tuple(data[key].copy() for key in ("x", "y", "graphs"))


def _problem() -> tuple[
    jax.Array,
    jax.Array,
    SpatialPanelWeights,
    jax.Array,
    Callable[[jax.Array], jax.Array],
    Callable[[jax.Array], jax.Array],
    SolverResult[jax.Array],
]:
    x, y, graphs = _witness()
    x, y = jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
    weights = prepare_spatial_panel_weights(
        jnp.asarray(graphs, dtype=jnp.float32), period_ids=np.arange(len(x))
    )
    limit = weights.limit * jnp.float32(0.999)

    def profile_objective(u: jax.Array) -> jax.Array:
        return -panel.spatial_panel_profile(
            limit * u[:1],
            x,
            y,
            weights,
            lag=True,
            error=False,
            effects="random",
            variance_ratio=jnp.exp(u[1]),
        )[3]

    def average(u: jax.Array) -> jax.Array:
        return profile_objective(u) / y.size

    def objective(z: jax.Array) -> jax.Array:
        u = jnp.concatenate((jnp.tanh(z[:1]), z[1:]))
        return profile_objective(u)

    u = jnp.asarray([0.24193380773067474, 1.6752272844314575], dtype=jnp.float32)
    z = jnp.stack((jnp.arctanh(u[0]), u[1]))
    score = jnp.max(jnp.abs(jax.grad(average)(u)))
    selected = SolverResult(
        z,
        jnp.asarray(0, dtype=jnp.bool_),
        jnp.asarray(72, dtype=jnp.int32),
        jnp.asarray(SolverStatus.STALLED, dtype=jnp.int32),
        score,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(256, dtype=jnp.int32),
    )
    return x, y, weights, limit, average, objective, selected


def _oracle(
    x: ArrayLike, y: ArrayLike, graphs: ArrayLike, spatial: float, ratio: float
) -> float:
    """Full covariance Cholesky GLS, independent of Woodbury profiling."""
    x, y, graphs = (np.asarray(a, dtype=np.float64) for a in (x, y, graphs))
    t, n, p = x.shape
    a = block_diag(*(np.eye(n) - spatial * graph for graph in graphs))
    d = np.tile(np.eye(n), (t, 1))
    root = np.linalg.cholesky(np.eye(t * n) + ratio * d @ d.T)
    xx = solve_triangular(root, x.reshape(t * n, p), lower=True)
    yy = solve_triangular(root, a @ y.ravel(), lower=True)
    beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
    residual = yy - xx @ beta
    return float(
        np.linalg.slogdet(root)[1]
        - np.linalg.slogdet(a)[1]
        + t * n / 2 * (np.log(2 * np.pi * np.mean(residual**2)) + 1)
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.regression
def test_float32_stalled_candidate_recovers_against_full_covariance() -> None:  # noqa: PLR0914
    """Compare recovery with an independently assembled covariance."""
    x, y, weights, limit, average, objective, selected = _problem()

    def recover(value: SolverResult[jax.Array]) -> SolverResult[jax.Array]:
        return panel._recover_panel_score_root(  # noqa: SLF001
            value, objective, average, limit, dimension=1, max_steps=256, tolerance=1e-7
        )

    result = jax.jit(recover)(selected)
    assert STALLED_STEPS < int(result.iterations) <= STALLED_STEPS + 8
    threshold = max(1e-7, 16 * float(np.finfo(np.float32).eps))
    if bool(result.converged):
        assert int(result.status) == SolverStatus.SUCCESS
        returned = jnp.concatenate((jnp.tanh(result.value[:1]), result.value[1:]))
        returned_score = jnp.max(jnp.abs(jax.grad(average)(returned)))
        assert float(returned_score) <= threshold
        assert float(result.residual_norm) <= threshold
        assert np.linalg.eigvalsh(np.asarray(jax.hessian(average)(returned))).min() > 0
        # Re-evaluating a reduction outside the solve can lower differently.
        # Constant-objective controls below test the exact one-ULP policy.
        assert _oracle(
            x,
            y,
            weights.values,
            float(limit * returned[0]),
            float(jnp.exp(returned[1])),
        ) < _oracle(
            x,
            y,
            weights.values,
            float(limit * jnp.tanh(selected.value[0])),
            float(jnp.exp(selected.value[1])),
        )
    else:
        assert int(result.status) == SolverStatus.STALLED
        np.testing.assert_array_equal(result.value, selected.value)
        np.testing.assert_array_equal(result.residual_norm, selected.residual_norm)

    def solve(value: jax.Array) -> SolverResult[jax.Array]:
        return panel._newton_root_result(  # noqa: SLF001
            jax.grad(average), value, max_steps=8, rtol=1e-7, atol=threshold
        )

    initial = jnp.concatenate((jnp.tanh(selected.value[:1]), selected.value[1:]))
    root = jax.jit(solve)(initial)
    assert bool(root.converged)
    encoded = jnp.concatenate((jnp.arctanh(root.value[:1]), root.value[1:]))
    u = jnp.concatenate((jnp.tanh(encoded[:1]), encoded[1:]))
    score = jnp.max(jnp.abs(jax.grad(average)(u)))
    assert float(score) <= threshold
    assert np.linalg.eigvalsh(np.asarray(jax.hessian(average)(u))).min() > 0
    old = np.asarray(selected.value)
    new = np.asarray(encoded)
    old_nll = _oracle(
        x, y, weights.values, float(limit * jnp.tanh(old[0])), np.exp(old[1])
    )
    new_nll = _oracle(
        x, y, weights.values, float(limit * jnp.tanh(new[0])), np.exp(new[1])
    )
    assert new_nll < old_nll

    def oracle_objective(v: NDArray[np.float64]) -> float:
        return _oracle(x, y, weights.values, float(limit) * v[0], np.exp(v[1]))

    oracle = minimize(
        oracle_objective,
        np.asarray(u, dtype=np.float64),
        method="BFGS",
        options={"gtol": 1e-8},
    )
    np.testing.assert_allclose(u, oracle.x, atol=3e-5, rtol=0)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "case", ["success", "large_score", "outside", "nonfinite", "failed"]
)
def test_root_candidate_is_certified_before_acceptance(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Only valid backend candidates passing the score can succeed."""
    target = jnp.asarray([0.2, 0.3], dtype=jnp.float32)

    def average(u: jax.Array) -> jax.Array:
        return jnp.sum((u - target) ** 2)

    def objective(z: jax.Array) -> jax.Array:
        return average(jnp.stack((jnp.tanh(z[0]), z[1])))

    values = {
        "success": target,
        "large_score": target + 0.01,
        "outside": jnp.asarray([1.1, 0.3], dtype=jnp.float32),
        "nonfinite": jnp.asarray([jnp.nan, 0.3], dtype=jnp.float32),
        "failed": target,
    }

    def root(
        score: Callable[[jax.Array], jax.Array],
        initial: jax.Array,
        *,
        max_steps: int,
        rtol: float,
        atol: float,
    ) -> SolverResult[jax.Array]:
        del score, initial, rtol, atol
        success = case != "failed"
        return SolverResult(
            values[case],
            jnp.asarray(success),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(
                SolverStatus.SUCCESS if success else SolverStatus.STALLED,
                dtype=jnp.int32,
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(max_steps, dtype=jnp.int32),
        )

    monkeypatch.setattr(panel, "_newton_root_result", root)
    selected = SolverResult(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.bool_),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(SolverStatus.STALLED, dtype=jnp.int32),
        jnp.asarray(0.6, dtype=jnp.float32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
    )
    result = panel._recover_panel_score_root(  # noqa: SLF001
        selected,
        objective,
        average,
        jnp.float32(1.0),
        dimension=1,
        max_steps=20,
        tolerance=1e-7,
    )
    assert bool(result.converged) == (case == "success")
    assert int(result.iterations) == EXPECTED_STEPS
    if case != "success":
        np.testing.assert_array_equal(result.value, selected.value)
        assert int(result.status) == SolverStatus.STALLED
        assert float(result.residual_norm) == pytest.approx(0.6)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("status", "steps"), [(SolverStatus.MAX_STEPS, 1), (SolverStatus.STALLED, 13)]
)
def test_root_recovery_skips_ineligible_lanes(status: SolverStatus, steps: int) -> None:
    """Preserve exhausted or otherwise ineligible solver results."""

    def average(u: jax.Array) -> jax.Array:
        return jnp.sum(u**2)

    def objective(z: jax.Array) -> jax.Array:
        return average(jnp.stack((jnp.tanh(z[0]), z[1])))

    selected = SolverResult(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.bool_),
        jnp.asarray(steps, dtype=jnp.int32),
        jnp.asarray(status, dtype=jnp.int32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
    )
    result = panel._recover_panel_score_root(  # noqa: SLF001
        selected,
        objective,
        average,
        jnp.float32(1.0),
        dimension=1,
        max_steps=20,
        tolerance=1e-7,
    )
    assert int(result.iterations) == steps
    assert int(result.status) == status
    np.testing.assert_array_equal(result.value, selected.value)


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.parametrize(
    ("dtype", "direction"),
    [("float32", 0), ("float32", -1), ("float32", 1), ("float64", 0)],
)
def test_witness_fit_is_valid_in_eager_and_jit(dtype: str, direction: int) -> None:
    """Fit the witness and its float32 neighbors in both execution modes."""
    x, y, graphs = _witness()
    if direction:
        y.flat[0] = np.nextafter(y.flat[0], np.float32(np.inf * direction))
    with jax.enable_x64(new_val=dtype == "float64"):
        weights = prepare_spatial_panel_weights(
            jnp.asarray(graphs, dtype=dtype), period_ids=np.arange(len(x))
        )
        design, outcome = jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)
        assert str(outcome.dtype) == dtype

        def fit(values: jax.Array) -> SpatialPanelFit:
            return PanelML().estimate(
                SpatialPanelProblem(
                    design, values, weights, process=SAR(), effects="random"
                )
            )

        eager, compiled = fit(outcome), jax.jit(fit)(outcome)
        for result in (eager, compiled):
            assert bool(result.valid & result.solver.converged)
            assert bool(jnp.isfinite(result.log_likelihood))
            assert bool(spatial_covariance(result).valid)
            assert RATIO_MIN < float(result.variance_ratio) < RATIO_MAX
        np.testing.assert_allclose(
            eager.parameters, compiled.parameters, atol=3e-4, rtol=3e-4
        )


@pytest.mark.component
@pytest.mark.contract
def test_recovery_keeps_ineligible_lane_under_vmap() -> None:
    """Independent batched lanes retain their own outcomes."""
    _, _, _, limit, average, objective, selected = _problem()
    other = SolverResult(
        selected.value,
        selected.converged,
        selected.iterations,
        jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32),
        selected.residual_norm,
        selected.rank,
        selected.max_steps,
    )

    def stack(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.stack((a, b))

    def recover(value: SolverResult[jax.Array]) -> SolverResult[jax.Array]:
        return panel._recover_panel_score_root(  # noqa: SLF001
            value, objective, average, limit, dimension=1, max_steps=256, tolerance=1e-7
        )

    batch = jax.tree.map(stack, selected, other)
    result = jax.jit(jax.vmap(recover))(batch)
    assert STALLED_STEPS < int(result.iterations[0]) <= STALLED_STEPS + 8
    if bool(result.converged[0]):
        u = jnp.stack((jnp.tanh(result.value[0, 0]), result.value[0, 1]))
        assert float(jnp.max(jnp.abs(jax.grad(average)(u)))) <= max(
            1e-7, 16 * np.finfo(np.float32).eps
        )
        assert int(result.status[0]) == SolverStatus.SUCCESS
    else:
        assert int(result.status[0]) == SolverStatus.STALLED
        np.testing.assert_array_equal(result.value[0], selected.value)
        np.testing.assert_array_equal(result.residual_norm[0], selected.residual_norm)
    assert not bool(result.converged[1])
    assert int(result.status[1]) == SolverStatus.MAX_STEPS
    np.testing.assert_array_equal(result.value[1], other.value)


@pytest.mark.component
@pytest.mark.contract
def test_quadratic_vmap_recovers_only_stalled_lane() -> None:
    """A recoverable lane succeeds independently of an exhausted lane."""
    target = jnp.asarray([0.2, 0.3], dtype=jnp.float32)

    def average(u: jax.Array) -> jax.Array:
        return jnp.sum((u - target) ** 2)

    def objective(z: jax.Array) -> jax.Array:
        return average(jnp.stack((jnp.tanh(z[0]), z[1])))

    selected = SolverResult(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.bool_),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(SolverStatus.STALLED, dtype=jnp.int32),
        jnp.float32(0.6),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
    )
    other = SolverResult(
        selected.value,
        selected.converged,
        selected.iterations,
        jnp.asarray(SolverStatus.MAX_STEPS, dtype=jnp.int32),
        selected.residual_norm,
        selected.rank,
        selected.max_steps,
    )

    def stack(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.stack((a, b))

    def recover(value: SolverResult[jax.Array]) -> SolverResult[jax.Array]:
        return panel._recover_panel_score_root(  # noqa: SLF001
            value,
            objective,
            average,
            jnp.float32(1),
            dimension=1,
            max_steps=20,
            tolerance=1e-7,
        )

    result = jax.jit(jax.vmap(recover))(jax.tree.map(stack, selected, other))
    assert bool(result.converged[0])
    assert not bool(result.converged[1])
    assert int(result.status[1]) == SolverStatus.MAX_STEPS
    np.testing.assert_array_equal(result.value[1], other.value)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("ulps", [1, 2])
def test_objective_guard_allows_at_most_one_ulp(
    monkeypatch: pytest.MonkeyPatch, ulps: int
) -> None:
    """Allow only one representable step in the objective."""
    target = jnp.asarray([0.2, 0.3], dtype=jnp.float32)

    def average(u: jax.Array) -> jax.Array:
        return jnp.sum((u - target) ** 2)

    old = jnp.float32(1.0)
    raised = jnp.nextafter(old, jnp.float32(jnp.inf))
    if ulps == TWO_ULPS:
        raised = jnp.nextafter(raised, jnp.float32(jnp.inf))

    def objective(z: jax.Array) -> jax.Array:
        return jnp.where(z[0] == 0, old, raised)

    def root(
        score: Callable[[jax.Array], jax.Array],
        initial: jax.Array,
        *,
        max_steps: int,
        rtol: float,
        atol: float,
    ) -> SolverResult[jax.Array]:
        del score, initial, rtol, atol
        return SolverResult(
            target,
            jnp.asarray(1, dtype=jnp.bool_),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(SolverStatus.SUCCESS, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(max_steps, dtype=jnp.int32),
        )

    monkeypatch.setattr(panel, "_newton_root_result", root)
    selected = SolverResult(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.bool_),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(SolverStatus.STALLED, dtype=jnp.int32),
        jnp.float32(0.6),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
    )
    result = panel._recover_panel_score_root(  # noqa: SLF001
        selected,
        objective,
        average,
        jnp.float32(1),
        dimension=1,
        max_steps=20,
        tolerance=1e-7,
    )
    assert bool(result.converged) == (ulps == 1)
    if ulps == TWO_ULPS:
        np.testing.assert_array_equal(result.value, selected.value)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("singular_at_end", [False, True])
def test_nonpositive_curvature_never_certifies_root(
    monkeypatch: pytest.MonkeyPatch, *, singular_at_end: bool
) -> None:
    """Reject a root without positive full-rank curvature."""
    target = jnp.asarray([0.2, 0.3], dtype=jnp.float32)

    def average(u: jax.Array) -> jax.Array:
        if singular_at_end:
            return (u[0] - target[0]) ** 4 + (u[1] - target[1]) ** 2
        return -((u[0] - target[0]) ** 2) + (u[1] - target[1]) ** 2

    def root(
        score: Callable[[jax.Array], jax.Array],
        initial: jax.Array,
        *,
        max_steps: int,
        rtol: float,
        atol: float,
    ) -> SolverResult[jax.Array]:
        del score, initial, rtol, atol
        return SolverResult(
            target,
            jnp.asarray(1, dtype=jnp.bool_),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(SolverStatus.SUCCESS, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(max_steps, dtype=jnp.int32),
        )

    monkeypatch.setattr(panel, "_newton_root_result", root)
    selected = SolverResult(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.bool_),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(SolverStatus.STALLED, dtype=jnp.int32),
        jnp.float32(0.6),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
    )

    def objective(z: jax.Array) -> jax.Array:
        return average(jnp.stack((jnp.tanh(z[0]), z[1])))

    result = panel._recover_panel_score_root(  # noqa: SLF001
        selected,
        objective,
        average,
        jnp.float32(1),
        dimension=1,
        max_steps=20,
        tolerance=1e-7,
    )
    assert not bool(result.converged)
    np.testing.assert_array_equal(result.value, selected.value)
