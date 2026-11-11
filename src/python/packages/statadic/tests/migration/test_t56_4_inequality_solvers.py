# jaxtyping shape strings are runtime contracts.
"""T56.4 parity and failure boundaries for batched inequality projections."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import statadic.inference._projection as projection_module
import statadic.inference.gms as gms_module
import statadic.inference.wolak as wolak_module
import numpy as np
import pytest
from statadic.core.random import cell_key
from statadic.inference.gms import (
    indicator_gms,
)
from statadic.inference.wolak import (
    chi_bar_squared_weights_mc,
    wolak_statistic,
    wolak_test,
)
from statadic.optimize.simplex import batched_nonnegative_quadratic_program

if TYPE_CHECKING:
    from statadic.optimize.simplex import NonnegativeQPSolution

_GMS_KEY = (2871141828, 1470508057)
_WOLAK_KEY = (2304444120, 1138204623)


def _baseline_panel() -> np.ndarray:
    """Return the fixed ``n=400, k=8`` pre-migration benchmark panel."""
    rng = np.random.default_rng(932)
    n_observations = 400
    dimension = 8
    correlation = 0.45
    covariance = (1.0 - correlation) * np.eye(dimension) + correlation * np.ones(
        (dimension, dimension)
    )
    draws = (
        rng.standard_normal((n_observations, dimension))
        @ np.linalg.cholesky(covariance).T
    )
    means = np.zeros(dimension)
    means[0] = -0.12
    means[dimension // 2 :] = 0.30
    return draws + means


def _brute_force_binding_counts(
    covariance: np.ndarray, *, draw_count: int
) -> np.ndarray:
    """Return the binding-coordinate histogram from an independently solved QP."""
    inverse: np.ndarray = np.linalg.pinv(covariance)
    quadratic: np.ndarray = 0.5 * (inverse + inverse.T)
    key = cell_key(0, "inference", "chi_bar_squared_weights_mc")
    with jax.enable_x64(new_val=True):
        normals: np.ndarray = np.asarray(
            jax.random.normal(
                key, shape=(draw_count, covariance.shape[0]), dtype=np.float64
            )
        )
        factor: np.ndarray = np.linalg.cholesky(
            covariance + 1e-12 * np.eye(covariance.shape[0])
        )
        standardized: np.ndarray = (factor @ normals.T).T
        brute = batched_nonnegative_quadratic_program(
            jnp.asarray(quadratic),
            jnp.asarray(-(standardized @ quadratic)),
            active_tolerance=1e-10,
            max_sweeps=16384,
        )
    assert bool(np.all(np.asarray(brute.converged)))
    binding = np.sum(np.asarray(brute.active_mask), axis=1)
    return np.bincount(binding, minlength=covariance.shape[0] + 1).astype(float)


@pytest.mark.component
@pytest.mark.regression
def test_gms_wolak_match_scipy_baseline_and_preserve_draw_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve STATADIC's key schedule and statistics across JAX numerical changes.

    The historical node name remains stable for retained test evidence. Exact
    floating-point random bytes are outside JAX's cross-release contract.
    """
    recorded: list[tuple[tuple[int, ...], tuple[int, ...], np.dtype]] = []
    original_normal = jax.random.normal

    def recording_normal(
        key: jax.Array,
        *,
        shape: tuple[int, ...],
        dtype: type[np.float64] | None = None,
    ) -> jax.Array:
        # Pin our stream identity and caller-selected precision, not a JAX
        # implementation's floating-point representation of normal draws.
        draw = (
            original_normal(key, shape=shape)
            if dtype is None
            else original_normal(key, shape=shape, dtype=dtype)
        )
        recorded.append(
            (tuple(map(int, jax.random.key_data(key))), tuple(draw.shape), draw.dtype)
        )
        return draw

    monkeypatch.setattr(jax.random, "normal", recording_normal)
    panel = _baseline_panel()
    with jax.enable_x64(new_val=True):
        gms = indicator_gms(panel, n_mc=256, seed=17)
        wolak = wolak_test(panel, n_mc=256, seed=17)

    assert recorded == [
        (_GMS_KEY, (8, 256), np.dtype("float64")),
        (_WOLAK_KEY, (256, 8), np.dtype("float64")),
    ]
    assert gms.stat == pytest.approx(4.055997273074623, abs=2e-13)
    assert gms.crit == pytest.approx(4.868485772670151, abs=2e-13)
    assert gms.pvalue == pytest.approx(0.07392996108949416, abs=0.0)
    assert not gms.reject
    assert wolak.stat == pytest.approx(4.055997273074624, abs=2e-13)
    assert wolak.pvalue == pytest.approx(0.15028534291536164, abs=2e-13)
    expected_binding = 2
    assert wolak.n_binding == expected_binding
    np.testing.assert_array_equal(
        wolak.weights,
        [
            0.078125,
            0.28515625,
            0.33203125,
            0.20703125,
            0.08203125,
            0.015625,
            0.0,
            0.0,
            0.0,
        ],
    )


@pytest.mark.component
@pytest.mark.contract
def test_gms_eager_boundary_rejects_unconverged_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GMS never calibrates from an unconverged compiled projection lane."""
    original_solver = gms_module.batched_nonnegative_quadratic_program

    def forced_unconverged(
        quadratic: jax.Array,
        linear: jax.Array,
        *,
        active_tolerance: float,
        tolerance: float = 1e-10,
        max_sweeps: int = 256,
    ) -> NonnegativeQPSolution:
        # ``max_sweeps`` and ``tolerance`` are accepted and forwarded rather
        # than dropped: both GMS and Wolak pass explicit values because the
        # shared defaults are too small for their projections, and a double
        # that did not accept them would stop matching the real signature.
        solution = original_solver(
            quadratic,
            linear,
            active_tolerance=active_tolerance,
            tolerance=tolerance,
            max_sweeps=max_sweeps,
        )
        return solution._replace(converged=jnp.zeros_like(solution.converged))

    monkeypatch.setattr(
        gms_module,
        "batched_nonnegative_quadratic_program",
        forced_unconverged,
    )
    with pytest.raises(RuntimeError, match=r"unconverged=[1-9]"):
        indicator_gms(_baseline_panel(), n_mc=8, seed=17)


@pytest.mark.component
@pytest.mark.contract
def test_wolak_eager_boundary_rejects_unconverged_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wolak never reports a statistic from an unconverged projection lane."""
    original_solver = wolak_module.batched_nonnegative_quadratic_program

    def forced_unconverged(
        quadratic: jax.Array,
        linear: jax.Array,
        *,
        active_tolerance: float,
        tolerance: float = 1e-10,
        max_sweeps: int = 256,
    ) -> NonnegativeQPSolution:
        # ``max_sweeps`` and ``tolerance`` are accepted and forwarded rather
        # than dropped: a caller may pass an explicit budget and an explicit
        # projected-KKT tolerance when the shared defaults do not suit its
        # projections (``wolak._PROJECTION_MAX_SWEEPS`` /
        # ``wolak._PROJECTION_TOLERANCE``), and a double that did not accept
        # those arguments would stop matching the real signature.
        solution = original_solver(
            quadratic,
            linear,
            active_tolerance=active_tolerance,
            tolerance=tolerance,
            max_sweeps=max_sweeps,
        )
        return solution._replace(converged=jnp.zeros_like(solution.converged))

    monkeypatch.setattr(
        wolak_module,
        "batched_nonnegative_quadratic_program",
        forced_unconverged,
    )
    # The eager boundary still fails closed on an injected non-convergence even
    # though ``_refine_projection_solution`` recomputes the verdict: the
    # refinement only overturns the compiled solver for a lane it measurably
    # improved, and this lane's iterate is already optimal.
    with pytest.raises(RuntimeError, match=r"Wolak statistic.*unconverged=1"):
        wolak_statistic(np.array([-1.0, 0.2]), np.eye(2), 20)


#: ``K = 3`` projection with ``cond(Q) = 149`` on which the shared solver's
#: default budget of 256 cyclic sweeps is demonstrably insufficient. Pinned as
#: literals rather than regenerated so the case cannot drift with a PRNG change.
_COND149_QUADRATIC = np.array(
    [
        [9608.690666650124, -9474.60004101172, -8004.228046831918],
        [-9474.60004101172, 13313.51931430514, -89.5185155355942],
        [-8004.228046831918, -89.5185155355942, 25608.746179338723],
    ]
)
_COND149_LINEAR = np.array(
    [[-5943.416861055922, -7368.128679724461, -3658.988751335546]]
)

#: ``optimize/simplex.py``'s own cyclic-sweep default. The raised solver budgets
#: are asserted to exceed it, so it is named rather than repeated as a literal.
_DEFAULT_PROJECTION_MAX_SWEEPS = 256
#: Projected-gradient norm the ``cond(Q) = 149`` case is still above at the
#: default budget, and below at the raised one. Two thresholds, not one, because
#: the test asserts a separation rather than a single tolerance.
_UNCONVERGED_GRADIENT_NORM = 1e-3
_CONVERGED_GRADIENT_NORM = 1e-9
#: KKT residual at the true constrained optimum after active-set refinement.
_OPTIMUM_KKT_RESIDUAL = 1e-12


@pytest.mark.unit
@pytest.mark.regression
def test_cond149_projection_needs_more_than_the_default_sweep_budget() -> None:
    """Pin the sweep budget that ``wolak._PROJECTION_MAX_SWEEPS`` exists for.

    t56.4's acceptance asks for the budget to be pinned by a test rather than a
    comment. ``optimize/simplex.py`` runs its sweeps under ``lax.fori_loop``
    with no early exit, so the budget is paid in full on every call; this test
    is the evidence that paying it is necessary.
    """
    expected_condition = 149.0
    assert np.linalg.cond(_COND149_QUADRATIC) == pytest.approx(
        expected_condition, rel=1e-9
    )

    with jax.enable_x64(new_val=True):
        default_budget = batched_nonnegative_quadratic_program(
            jnp.asarray(_COND149_QUADRATIC),
            jnp.asarray(_COND149_LINEAR),
            max_sweeps=_DEFAULT_PROJECTION_MAX_SWEEPS,
        )
        raised_budget = batched_nonnegative_quadratic_program(
            jnp.asarray(_COND149_QUADRATIC),
            jnp.asarray(_COND149_LINEAR),
            max_sweeps=wolak_module._PROJECTION_MAX_SWEEPS,  # the constant under test  # noqa: SLF001
        )

    assert not bool(np.asarray(default_budget.converged)[0])
    assert (
        float(np.asarray(default_budget.projected_gradient_norm)[0])
        > _UNCONVERGED_GRADIENT_NORM
    )
    assert bool(np.asarray(raised_budget.converged)[0])
    assert (
        float(np.asarray(raised_budget.projected_gradient_norm)[0])
        < _CONVERGED_GRADIENT_NORM
    )


@pytest.mark.component
@pytest.mark.regression
def test_wolak_chi_bar_calibration_survives_the_illconditioned_covariance() -> None:
    """Regression for the t56.4 conditioning defect found on real Paper 3 data.

    The 15x15 HAC moment covariance shipped alongside this test is the exact one
    that made ``dvc repro p3_empirical`` fail with

        Wolak chi-bar calibration nonnegative-QP solver rejected 940/3000 lanes

    at ``_PROJECTION_MAX_SWEEPS = 4096``. Cyclic coordinate descent converges
    linearly at rate ``~(1 - 1/cond(Q))`` per sweep, which at ``cond(Q) = 1840``
    is too slow for a fixed budget; ``_polish_active_set`` restores the finite
    active-set termination the retired SciPy NNLS path had. The projection is
    the same mathematical program either way, so the weights must equal an
    independently converged answer exactly, not merely approximate it.
    """
    fixture = np.load(
        pathlib.Path(__file__).parent
        / "fixtures"
        / "t56_4_wolak_illconditioned_covariance.npz"
    )
    covariance: np.ndarray = fixture["covariance"]
    gbar: np.ndarray = fixture["gbar"]
    expected_condition = 1840.4
    assert np.linalg.cond(covariance) == pytest.approx(expected_condition, rel=1e-4)

    # Fails closed before the fix; must now complete for all 3000 lanes.
    weights = np.asarray(chi_bar_squared_weights_mc(covariance, n_mc=3000, seed=0))
    assert weights.shape == (covariance.shape[0] + 1,)
    assert float(weights.sum()) == pytest.approx(1.0)

    # Independent reference: brute-force the same projections with a sweep
    # budget large enough to converge on their own (16384 reaches 0/3000 with
    # max projected-KKT 8.5e-11), and require an exact match of the binding
    # tally the weights are a histogram of.
    counts = _brute_force_binding_counts(covariance, draw_count=3000)
    # The *tally* is the claim, and it is still asserted exactly: recovering the
    # integer histogram from the returned weights must reproduce this reference
    # bin for bin. The normalization is compared to a float64 tolerance instead
    # of bit-for-bit, because t65.1 moved the division inside `jnp` and the two
    # libraries round `counts / counts.sum()` differently in the last place —
    # measured at 4.34e-19 absolute on one of 16 bins (~1 ulp, 1.3e-16
    # relative), which the precision policy rates a non-event. Asserting the
    # histogram exactly and the quotient to tolerance tests the invariant that
    # matters rather than a division's rounding mode.
    np.testing.assert_array_equal(
        np.rint(np.asarray(weights) * counts.sum()),
        counts,
    )
    np.testing.assert_allclose(weights, counts / counts.sum(), rtol=1e-15, atol=0.0)

    stat, n_binding = wolak_statistic(gbar, covariance, 1207)
    assert np.isfinite(stat)
    assert stat > 0.0
    assert 0 <= n_binding <= covariance.shape[0]


@pytest.mark.unit
@pytest.mark.property
def test_active_set_polish_never_worsens_the_solver_iterate() -> None:
    """The refinement is monotone in the projected-KKT residual by construction."""
    rng = np.random.default_rng(4711)
    dimension = 6
    root = rng.standard_normal((dimension, dimension))
    quadratic = root @ root.T + 0.05 * np.eye(dimension)
    quadratic = 0.5 * (quadratic + quadratic.T)
    linear = rng.standard_normal((64, dimension))

    with jax.enable_x64(new_val=True):
        solution = batched_nonnegative_quadratic_program(
            jnp.asarray(quadratic),
            jnp.asarray(linear),
            max_sweeps=256,
        )
    before = projection_module.projected_kkt_residual(
        quadratic, linear, np.asarray(solution.primal, dtype=np.float64)
    )
    primal, after, refined = projection_module.polish_active_set(
        quadratic,
        linear,
        np.asarray(solution.primal, dtype=np.float64),
        np.asarray(solution.flat_mask, dtype=bool),
    )
    # Monotone by construction: a candidate is kept only if it strictly reduces
    # the projected-KKT residual, so the refinement can never return a worse
    # iterate than the compiled solver's, and it only claims to have refined a
    # lane it actually improved.
    assert np.all(after <= before)
    assert np.all(primal >= 0.0)
    assert np.all(refined == (after < before))

    # On a well-conditioned problem the compiled solver already converges, and
    # the refined primal must agree with it to a tight tolerance.
    np.testing.assert_allclose(
        primal, np.asarray(solution.primal, dtype=np.float64), atol=1e-12, rtol=0.0
    )


@pytest.mark.unit
@pytest.mark.regression
def test_gms_qlr_projection_budget_is_pinned_above_the_shared_default() -> None:
    """GMS's QLR projection needs the same raised budget Wolak's does.

    ``gms.py`` was left on the shared 256-sweep default by t56.4 and carries the
    identical defect: fixed-budget cyclic coordinate descent converges linearly
    at rate ``~(1 - 1/cond(Q))``. Measured on a ``k = 15`` QLR-shaped batch of
    3001 lanes with ``cond(omega_inv) = 2052``, rejected lanes were 2898 (256
    sweeps), 1908 (1024), 959 (4096), 0 (16384) for the solver alone, and 104 /
    6 / 0 / 0 once the active-set refinement runs.

    **The two constants decoupled on 2026-08-08 and the assertion is now an
    inequality.** It previously pinned them *equal*, which encoded an accident:
    both happened to be 4096, not because the two callers solve the same
    problem. They do not. GMS recenters slack coordinates by ``1e6 * max(sig)``,
    putting ``‖c‖∞`` at 4.2e+10 where Wolak's is O(1), and on the
    qwen3-embedding-8b ablation batch 4096 sweeps left one lane at a
    projected-KKT residual of 31.1464 that no amount of refinement could rescue
    (32, 128 and 384 polish rounds return the identical value). GMS is now
    16384; Wolak stays 4096, where its own docstring measures the iteration
    count as saturated -- 4096 and 65536 sweeps reproduce the same 2.274e-12
    residual, so raising it would buy nothing and cost 4x.

    What must not regress is that each caller sits *above* the shared default,
    which is the gap t56.4's acceptance flagged. That is asserted directly.
    """
    assert (
        gms_module._PROJECTION_MAX_SWEEPS >= wolak_module._PROJECTION_MAX_SWEEPS  # noqa: SLF001
    )  # the constants under test
    assert (
        gms_module._PROJECTION_MAX_SWEEPS > _DEFAULT_PROJECTION_MAX_SWEEPS  # noqa: SLF001
    )  # the constant under test
    assert (
        wolak_module._PROJECTION_MAX_SWEEPS > _DEFAULT_PROJECTION_MAX_SWEEPS  # noqa: SLF001
    )  # the constant under test

    # The same literal-pinned cond(Q) = 149 projection the Wolak budget test
    # uses: the GMS budget must clear it where the shared default does not.
    with jax.enable_x64(new_val=True):
        at_gms_budget = batched_nonnegative_quadratic_program(
            jnp.asarray(_COND149_QUADRATIC),
            jnp.asarray(_COND149_LINEAR),
            max_sweeps=gms_module._PROJECTION_MAX_SWEEPS,  # the constant under test  # noqa: SLF001
        )
    assert bool(np.asarray(at_gms_budget.converged)[0])


@pytest.mark.component
@pytest.mark.regression
def test_gms_qlr_survives_the_real_paper3_moment_panel() -> None:
    """Regression for the GMS half of the t56.4 conditioning defect.

    This is the exact clustered moment panel reaching ``indicator_gms`` from
    ``robustness_gel_gms`` in ``p3_empirical``, the call that failed with

        indicator GMS QLR nonnegative-QP solver rejected 774/3001 lanes
        (... unconverged=774, max_projected_kkt=59.441)

    Its ``cond(Omega_inv) = 1840.4`` is *identical* to the Wolak covariance's,
    so conditioning is not what distinguishes it. What does is GMS's slack
    recentering (``standardized[:, slack] += 1e6 * max(sig)``), which inflates
    the linear term to ``||c||_inf ~ 5.1e10`` and makes almost every free-face
    solve infeasible. Clipping to the orthant then stalls -- all 774 stalled
    lanes had a negative free-face component -- which is why the refinement
    takes a Lawson--Hanson ratio-test step instead.
    """
    panel: np.ndarray = np.load(
        pathlib.Path(__file__).parent / "fixtures" / "t56_4_gms_qlr_moment_panel.npz"
    )["panel"]
    assert panel.shape == (1207, 15)

    # Fails closed before the fix; must now complete for all 3001 lanes.
    with jax.enable_x64(new_val=True):
        result = indicator_gms(panel, alpha=0.05, n_mc=3000, seed=0)

    assert np.isfinite(result.stat)
    assert np.isfinite(result.crit)
    assert 0.0 <= result.pvalue <= 1.0
    assert isinstance(result.reject, bool)


@pytest.mark.unit
@pytest.mark.numerical
def test_active_set_polish_takes_a_ratio_test_step_when_the_face_is_infeasible() -> (
    None
):
    """Clipping stalls where the ratio test makes progress.

    Constructed so the unconstrained free-face minimum is infeasible: clipping
    it yields no residual improvement, so a clip-and-reject refinement would
    stall, while the ratio test drives the blocking coordinate to zero.
    """
    quadratic = np.array([[4.0, 3.9, 0.0], [3.9, 4.0, 0.0], [0.0, 0.0, 1.0]])
    # Free-face minimum has a negative coordinate for this linear term.
    linear = np.array([[-1.0, 2.0, -1.0]])
    start = np.array([[0.5, 0.5, 1.0]])

    primal, residual, refined = projection_module.polish_active_set(
        quadratic,
        linear,
        start,
        np.zeros(start.shape, dtype=bool),
    )
    before = projection_module.projected_kkt_residual(quadratic, linear, start)

    assert np.all(primal >= 0.0)
    assert np.all(residual <= before)
    assert bool(refined[0])
    # Converged to the true constrained optimum.
    assert float(residual[0]) < _OPTIMUM_KKT_RESIDUAL
