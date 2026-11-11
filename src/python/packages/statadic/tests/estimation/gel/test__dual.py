# jaxtyping shape strings are runtime contracts.
"""Numerical GEL dual, frozen-profile and transformation contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.estimation.gel import FixedGEL
from statadic.estimation.gel._dual import (
    _GEL_RCOND,
    _gel_identified_subspace,
    _gel_inner_sup,
    _gel_inner_sup_kernel,
    _gel_profile_objective_kernel,
    _gel_rho_kernel,
)
from statadic.estimation.moments import FixedMomentProblem

from ..._budgets import examples
from ...inference._moments import (
    PROFILE_CROSSED_RANK,
    PROFILE_CROSSING_THETA,
    PROFILE_FROZEN_RANK,
    PROFILE_MAX_OBJECTIVE,
    PROFILE_MAX_OBJECTIVE_STEP,
    rank_deficient_profile_panel,
)


def _carrier(values: np.ndarray, criterion: str) -> tuple[np.ndarray, ...]:
    """Evaluate the traced carrier in float64 and hand back host arrays.

    Stands in for the retired ``_gel_rho`` eager door: the solver stopped
    calling it once the Newton loop became traceable, so the float64 decision it
    used to own now belongs to whoever evaluates the kernel — here, the test.
    """
    with jax.enable_x64(new_val=True):
        parts = _gel_rho_kernel(jnp.asarray(values, jnp.float64), criterion)
    return tuple(np.asarray(part, dtype=np.float64) for part in parts)


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
@examples(100)
@example(values=np.array([-0.4, 0.0, 0.4]))
@given(values=hnp.arrays(np.float64, 3, elements=st.floats(-0.4, 0.4)))
def test_gel_carrier_autodiff_matches_retired_interior_algebra(
    values: np.ndarray,
    criterion: Literal["cue", "el", "et"],
) -> None:
    """Scalar carrier autodiff reproduces analytic interior derivatives."""
    rho, rho1, rho2 = _carrier(values, criterion)
    if criterion == "cue":
        expected = -values - 0.5 * values**2
        expected1 = -1.0 - values
        expected2 = -np.ones_like(values)
    elif criterion == "et":
        exponential = np.exp(values)
        expected = -exponential + 1.0
        expected1 = -exponential
        expected2 = -exponential
    else:
        argument = 1.0 - values
        expected = np.log(argument)
        expected1 = -1.0 / argument
        expected2 = -1.0 / argument**2
    np.testing.assert_allclose(rho, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rho1, expected1, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rho2, expected2, atol=1e-12, rtol=1e-12)

    step = 1e-5
    plus = _carrier(values + step, criterion)[0]
    minus = _carrier(values - step, criterion)[0]
    finite_first = (plus - minus) / (2.0 * step)
    finite_second = (plus - 2.0 * rho + minus) / step**2
    np.testing.assert_allclose(rho1, finite_first, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(rho2, finite_second, atol=5e-6, rtol=5e-6)


@pytest.mark.unit
@pytest.mark.numerical
def test_gel_carrier_native_derivatives_retain_unclipped_tails() -> None:
    """Native AD differentiates the stated carriers without straight-through clamps."""
    et_values = np.array([-60.0, 60.0])
    _, et_first, et_second = _carrier(et_values, "et")
    et_expected = -np.exp(et_values)
    np.testing.assert_allclose(et_first, et_expected, rtol=1e-12)
    np.testing.assert_allclose(et_second, et_expected, rtol=1e-12)

    values = np.array([1.0 - 1e-6])
    _, el_first, el_second = _carrier(values, "el")
    np.testing.assert_allclose(el_first, -1 / (1 - values), rtol=1e-12)
    np.testing.assert_allclose(el_second, -1 / (1 - values) ** 2, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.contract
def test_gel_rank_decision_requires_a_caller_owned_x64() -> None:
    """The float64 rank obligation is the caller's to meet, and is transparent.

    :func:`_gel_identified_subspace` is where the obligation bites: the retained
    rank propagates to ``dof`` and so to a chi-square reference. statadic opens no
    scope of its own, so an x64-disabled caller gets float32 carriers rather
    than a silent upcast — the transparency that makes the loss detectable
    instead of merely wrong.
    """
    rng = np.random.default_rng(4)
    base = rng.standard_normal((256, 2))
    moments = np.column_stack([base, base[:, 0] + 1e-9 * rng.standard_normal(256)])

    with jax.enable_x64(new_val=True):
        basis, keep, rank = _gel_identified_subspace(moments)
    with jax.enable_x64(new_val=False):
        narrowed, _, _ = _gel_identified_subspace(moments)
        assert not jax.config.read("jax_enable_x64")

    assert basis.dtype == jnp.float64
    assert keep.dtype == jnp.float64
    assert rank >= 1
    assert narrowed.dtype == jnp.float32


@pytest.mark.integration
@pytest.mark.contract
def test_gel_carrier_kernel_transforms_but_solver_rejects_tracing() -> None:
    """Only scalar carrier calculus is traced; dynamic-rank GEL is a named door."""
    values = jnp.asarray([-0.2, 0.0, 0.2])

    def evaluate_gel_rho_kernel(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _gel_rho_kernel(x, "el")

    compiled = jax.jit(evaluate_gel_rho_kernel)(values)

    def evaluate_gel_rho_kernel_3(
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _gel_rho_kernel(x, "cue")

    batched = jax.vmap(evaluate_gel_rho_kernel_3)(jnp.stack((values, values + 0.1)))

    def evaluate_gel_rho_kernel_4(
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _gel_rho_kernel(x, "et")

    jax.make_jaxpr(evaluate_gel_rho_kernel_4)(values)
    assert compiled[0].shape == (3,)
    assert batched[0].shape == (2, 3)

    def evaluate_get_objective(panel: jax.Array) -> float:
        return FixedGEL().estimate(FixedMomentProblem(panel)).objective

    with pytest.raises(RuntimeError, match="host-only GEL solver boundary"):
        jax.jit(evaluate_get_objective)(jnp.ones((10, 2)))


@pytest.mark.component
@pytest.mark.property
def test_linear_gel_profile_objective_is_continuous_in_theta() -> None:
    """Freezing the identified subspace removes the rank-crossing jump.

    Regression: deciding ``keep`` from ``Omega(theta)`` *inside* the objective
    made the profile discontinuous. Across the crossing the objective jumped by
    ~3000x (``2T*P`` from 0.68 to 1999), and the outer BFGS halted exactly on
    the jump and reported ``converged=True`` — publishing a garbage statistic.
    """
    panel, derivative = rank_deficient_profile_panel()
    n_observations = panel.shape[0]
    basis, keep, rank = _gel_identified_subspace(panel, _GEL_RCOND)
    moments = jnp.asarray(panel)
    jacobian = jnp.asarray(
        np.broadcast_to(derivative[None], (n_observations, *derivative.shape))
    )
    subspace = jnp.asarray(basis, dtype=moments.dtype)
    mask = jnp.asarray(keep, dtype=moments.dtype)

    grid = np.array([0.0, 1.0e-3, 1.2578e-3, 2.0e-3, 5.0e-3, 1.0e-2])
    values = np.array(
        [
            2.0
            * n_observations
            * float(
                _gel_profile_objective_kernel(
                    moments, jacobian, jnp.asarray([t]), subspace, mask, "cue", 100
                )
            )
            for t in grid
        ]
    )

    # The rank genuinely changes across this grid -- that is the fixture.
    assert rank == PROFILE_FROZEN_RANK
    crossed = panel + PROFILE_CROSSING_THETA * derivative[:, 0]
    assert _gel_identified_subspace(crossed, _GEL_RCOND)[2] == PROFILE_CROSSED_RANK
    # ...but the frozen-subspace profile stays smooth and O(1) throughout.
    assert np.all(values < PROFILE_MAX_OBJECTIVE), (
        f"objective jumped across the crossing: {values}"
    )
    assert np.max(np.abs(np.diff(values))) < PROFILE_MAX_OBJECTIVE_STEP


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_gel_profile_gradient_matches_finite_difference(criterion: str) -> None:
    """The envelope-theorem gradient is exact, not an approximation.

    ``beta*`` is stationary in every coordinate -- in the retained ones by
    convergence, in the dropped ones because their columns of ``reduced`` are
    exactly zero -- so ``dP/dtheta = partial P/partial theta`` at the argmax and
    the solver need not be taped.
    """
    rng = np.random.default_rng(21)
    n_observations = 600
    noise = 0.2 * rng.standard_normal((n_observations, 4))
    noise -= noise.mean(axis=0, keepdims=True)
    derivative = np.array([[1.0], [0.6], [-0.4], [0.8]])
    panel = noise - 0.7 * derivative[:, 0]
    jacobian_panel = np.broadcast_to(
        derivative[None], (n_observations, *derivative.shape)
    )

    def objective(theta: jax.Array) -> jax.Array:
        return _gel_profile_objective_kernel(
            moments, jacobian, theta, basis, keep, criterion, 100
        )

    with jax.enable_x64(new_val=True):
        moments = jnp.asarray(panel, jnp.float64)
        jacobian = jnp.asarray(jacobian_panel, jnp.float64)
        basis, keep, _ = _gel_identified_subspace(panel, _GEL_RCOND)

        point = jnp.asarray([0.4], jnp.float64)
        analytic = float(jax.grad(objective)(point)[0])
        step = 1e-6
        finite = float(
            (objective(point + step) - objective(point - step)) / (2.0 * step)
        )
    assert np.isfinite(analytic)
    np.testing.assert_allclose(analytic, finite, rtol=1e-6)


@pytest.mark.component
@pytest.mark.property
def test_gel_profile_gradient_survives_degenerate_omega() -> None:
    """Repeated eigenvalues of ``Omega`` must not NaN the outer gradient.

    ``eigh``'s JVP carries ``1/(lambda_i - lambda_j)`` terms, so differentiating
    through the rank decision returned NaN whenever ``Omega`` had exact ties --
    reachable for whitened or orthonormalized moment panels. Freezing the basis
    keeps ``eigh`` off the tape entirely.
    """
    rng = np.random.default_rng(7)
    basis_q, _ = np.linalg.qr(rng.standard_normal((400, 4)))
    panel = basis_q * np.sqrt(400.0)  # Omega == I exactly: four-fold tie
    jacobian_panel = np.broadcast_to(np.ones((4, 1))[None], (400, 4, 1))

    with jax.enable_x64(new_val=True):
        moments = jnp.asarray(panel, jnp.float64)
        jacobian = jnp.asarray(jacobian_panel, jnp.float64)
        subspace, mask, _ = _gel_identified_subspace(panel, _GEL_RCOND)
        for criterion in ("cue", "el", "et"):

            def evaluate_gel_profile_objective_kernel(
                theta: jax.Array, c: str = criterion
            ) -> jax.Array:
                return _gel_profile_objective_kernel(
                    moments, jacobian, theta, subspace, mask, c, 100
                )

            gradient = jax.grad(evaluate_gel_profile_objective_kernel)(
                jnp.zeros(1, jnp.float64)
            )
            assert np.isfinite(float(gradient[0])), f"{criterion}: NaN gradient"


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
@examples(8)
@example(seed=51)
@given(seed=st.integers(0, 2**32 - 1))
def test_gel_inner_sup_converges_under_float32(
    seed: int, criterion: Literal["cue", "el", "et"]
) -> None:
    """The gradient test is precision-portable.

    A fixed absolute tolerance (historically 1e-9) is unreachable in float32 --
    the attainable gradient norm floors near 1e-8 -- so every fit would report
    ``converged=False`` and every statistic would be NaN. The stopping rule is
    ``sqrt(eps)`` relative to the moment scale.
    """
    rng = np.random.default_rng(seed)
    panel = rng.standard_normal((500, 4))
    basis, keep, _ = _gel_identified_subspace(panel, _GEL_RCOND)

    single = jnp.asarray(panel, jnp.float32)
    reduced32 = (single @ jnp.asarray(basis, jnp.float32)) * jnp.asarray(
        keep, jnp.float32
    )
    _, objective32, converged32 = _gel_inner_sup_kernel(
        reduced32, jnp.asarray(keep, jnp.float32), criterion, 200
    )
    assert bool(converged32), f"{criterion}: float32 fit never certified"

    _, objective64, converged64, _ = _gel_inner_sup(panel, criterion)
    assert converged64
    # The statistic is second-order insensitive to the multiplier, so the
    # precision split must not move it at the resolution of a chi-square test.
    statistic32 = 2.0 * panel.shape[0] * float(objective32)
    statistic64 = 2.0 * panel.shape[0] * objective64
    np.testing.assert_allclose(statistic32, statistic64, rtol=0.0, atol=1e-4)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_probability_carriers_reject_infinite_hull_optima_in_compiled_batches(
    dtype: type, criterion: str
) -> None:
    """Small unnormalized gradients cannot certify nonexistent EL/ET optima."""
    panels = jnp.asarray(
        [
            [[-1.0], [-0.2], [0.3], [1.2]],
            [[0.2], [0.5], [0.8], [1.2]],
            [[0.0], [0.0], [0.8], [1.2]],
        ],
        dtype=dtype,
    )
    keep = jnp.ones(1, dtype=dtype)

    def solve(panel: jax.Array) -> jax.Array:
        return _gel_inner_sup_kernel(panel, keep, criterion, 200)[2]

    actual = jax.jit(jax.vmap(solve))(panels)
    expected = [True, True, True] if criterion == "cue" else [True, False, False]
    np.testing.assert_array_equal(actual, expected)
    for panel, converged in zip(panels, expected, strict=True):
        assert bool(solve(panel)) == converged


@pytest.mark.integration
@pytest.mark.contract
@examples(8)
@example(seed=61)
@given(seed=st.integers(0, 2**32 - 1))
def test_gel_inner_sup_kernel_survives_jit_and_vmap(seed: int) -> None:
    """The new numerical core carries the transform obligations of its door.

    ``numerical-host-boundaries.md`` requires a core to be covered under direct
    execution, nested ``jit`` and leading-axis ``vmap``, with the eager door
    still rejecting tracers. The leading axis here is the replication axis a
    Monte-Carlo size/power study runs on, which is the reason the inner
    supremum was made traceable at all.
    """
    rng = np.random.default_rng(seed)
    replications = 4
    panels = rng.standard_normal((replications, 300, 4))
    basis, keep, _ = _gel_identified_subspace(panels[0], _GEL_RCOND)
    stacked = jnp.asarray(panels)
    subspace = jnp.asarray(basis, dtype=stacked.dtype)
    mask = jnp.asarray(keep, dtype=stacked.dtype)

    for criterion in ("cue", "el", "et"):

        def solve(panel: jax.Array, c: str = criterion) -> jax.Array:
            return _gel_inner_sup_kernel((panel @ subspace) * mask, mask, c, 200)[1]

        direct = solve(stacked[0])

        def compiled_solve(
            p: jax.Array, s: Callable[[jax.Array], jax.Array] = solve
        ) -> jax.Array:
            return jax.jit(s)(p)

        nested = jax.jit(compiled_solve)(stacked[0])
        batched = jax.jit(jax.vmap(solve))(stacked)

        assert batched.shape == (replications,)
        assert bool(jnp.all(jnp.isfinite(batched))), f"{criterion}: non-finite batch"
        np.testing.assert_allclose(float(nested), float(direct), rtol=1e-6)
        np.testing.assert_allclose(float(batched[0]), float(direct), rtol=1e-6)

    # The eager door keeps rejecting tracers even though its core no longer must.
    def fit_objective(panel: jax.Array) -> float:
        return _gel_inner_sup(panel, "cue")[1]

    with pytest.raises(RuntimeError, match="host-only GEL solver boundary"):
        jax.jit(fit_objective)(jnp.ones((10, 2)))
