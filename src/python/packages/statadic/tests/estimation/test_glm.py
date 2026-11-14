# standalone package-maintenance entry point
"""Independent likelihood witnesses and availability checks for canonical GLMs."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    PoissonMLE,
    PoissonProblem,
)
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.optimize.glm import ExistenceStatus

from .._budgets import examples
from .._validation.glm import interior_outcomes

if TYPE_CHECKING:
    from statadic.model.glm import BinomialLogitEquation, PoissonLogEquation


def _fit_binomial(
    x: jax.Array, y: jax.Array, *, trials: jax.Array, offset: jax.Array
) -> GLMFit[BinomialLogitEquation]:
    return BinomialLogitMLE().estimate(
        BinomialProblem(x, y, trials=trials, offset=offset)
    )


_GENERATED_BINOMIAL_FIT = jax.jit(_fit_binomial)


def _fit_poisson(
    x: jax.Array, y: jax.Array, *, offset: jax.Array, exposure: jax.Array
) -> GLMFit[PoissonLogEquation]:
    return PoissonMLE().estimate(PoissonProblem(x, y, offset=offset, exposure=exposure))


_GENERATED_POISSON_FIT = jax.jit(_fit_poisson)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType

    from jax import Array
    from jax.typing import (
        DTypeLike,
    )
    from statadic.estimation.glm import GLMFit
    from statadic.model.glm import GLMEquation


def _data(dtype: DTypeLike = jnp.float64) -> tuple[jax.Array, ...]:
    x = jnp.column_stack(
        (jnp.ones(12, dtype=dtype), jnp.linspace(-1.5, 1.5, 12, dtype=dtype))
    )
    successes = jnp.asarray([2, 4, 3, 5, 4, 7, 6, 7, 7, 8, 7, 9], dtype=dtype)
    trials = jnp.asarray([10, 11, 10, 12, 10, 11, 10, 12, 10, 11, 10, 12], dtype=dtype)
    counts = jnp.asarray([0, 1, 2, 1, 3, 2, 4, 3, 6, 4, 5, 7], dtype=dtype)
    offset = jnp.linspace(-0.15, 0.2, 12, dtype=dtype)
    exposure = jnp.linspace(0.75, 1.75, 12, dtype=dtype)
    return x, successes, trials, counts, offset, exposure


_RETAINED_SUCCESSES = [2, 4, 3, 5, 4, 7, 6, 7, 7, 8, 7, 9]
_RETAINED_TRIALS = [10, 11, 10, 12, 10, 11, 10, 12, 10, 11, 10, 12]
_RETAINED_COUNTS = [0, 1, 2, 1, 3, 2, 4, 3, 6, 4, 5, 7]


def _interior_outcomes() -> st.SearchStrategy[tuple[list[int], list[int]]]:
    """Draw shared interior outcomes using the retained trial counts."""
    return interior_outcomes(tuple(_RETAINED_TRIALS))


_RETAINED_OUTCOMES = (_RETAINED_SUCCESSES, _RETAINED_COUNTS)


@pytest.mark.component
@pytest.mark.performance
@pytest.mark.regression
def test_repeated_glm_fits_reuse_the_solver_compilation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Changing counts with a fixed shape must not rebuild the solver program."""
    design = jnp.column_stack((jnp.ones(9), jnp.linspace(-1.0, 1.0, 9)))
    counts = jnp.asarray([1, 2, 1, 3, 2, 4, 3, 5, 4])
    estimator = PoissonMLE(max_steps=79)
    with caplog.at_level(logging.WARNING, logger="jax"), jax.log_compiles():
        first = estimator.estimate(PoissonProblem(design, counts))
        jax.block_until_ready(first.parameters)
        initial = sum("Compiling jit(_fit_glm)" in r.message for r in caplog.records)
        second = estimator.estimate(PoissonProblem(design, counts + 1))
        jax.block_until_ready(second.parameters)
        repeated = sum("Compiling jit(_fit_glm)" in r.message for r in caplog.records)
    assert bool(first.valid)
    assert bool(second.valid)
    assert initial == 1
    assert repeated == initial


@pytest.mark.component
@pytest.mark.contract
def test_mixed_precision_roles_set_solver_working_precision() -> None:
    """An x64 offset must not leave the optimizer at float32 precision."""
    if not jax.config.x64_enabled:
        pytest.skip("JAX x64 is disabled")
    design = jnp.stack(
        [jnp.ones(6, dtype=jnp.float32), jnp.arange(6, dtype=jnp.float32)], axis=1
    )
    outcome = jnp.asarray([1, 2, 1, 3, 4, 5], dtype=jnp.float32)
    offset = jnp.full(6, 0.1, dtype=jnp.float64)
    mixed = PoissonMLE().estimate(PoissonProblem(design, outcome, offset=offset))
    reference = PoissonMLE().estimate(
        PoissonProblem(
            design.astype(jnp.float64), outcome.astype(jnp.float64), offset=offset
        )
    )
    assert bool(mixed.valid)
    assert bool(reference.valid)
    assert mixed.parameters.dtype == jnp.float64
    np.testing.assert_allclose(
        mixed.parameters, reference.parameters, rtol=1e-6, atol=1e-7
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_full_fit_and_covariance_match_statsmodels(  # noqa: PLR0914
    optional_reference: Callable[[str], ModuleType], family: str
) -> None:
    """Match count encoding, offset, dispersion and full likelihood constants."""
    sm = optional_reference("statsmodels.api")
    x, successes, trials, counts, offset, exposure = _data()
    if family == "binomial":
        fit = BinomialLogitMLE().estimate(
            BinomialProblem(x, successes, trials=trials, offset=offset)
        )
        model = sm.GLM(
            np.column_stack((successes, trials - successes)),
            np.asarray(x),
            family=sm.families.Binomial(),
            offset=np.asarray(offset),
        )
        response = fit.probabilities
    else:
        fit = PoissonMLE().estimate(
            PoissonProblem(x, counts, offset=offset, exposure=exposure)
        )
        model = sm.GLM(
            np.asarray(counts),
            np.asarray(x),
            family=sm.families.Poisson(),
            offset=np.asarray(offset),
            exposure=np.asarray(exposure),
        )
        response = fit.fitted_means
    reference = model.fit(maxiter=200, tol=1e-12)
    assert bool(fit.valid)
    assert int(fit.rank) == x.shape[1]
    assert int(fit.df_resid) == reference.df_resid
    np.testing.assert_allclose(fit.parameters, reference.params, rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(response, reference.fittedvalues, rtol=1e-6, atol=2e-7)
    np.testing.assert_allclose(
        fit.scores.values, model.score_obs(fit.parameters), atol=1e-10
    )
    np.testing.assert_allclose(
        -fit.jacobian.values * x.shape[0], -model.hessian(fit.parameters), rtol=1e-12
    )
    assert float(fit.log_likelihood) == pytest.approx(reference.llf, rel=1e-10)
    assert float(fit.deviance) == pytest.approx(reference.deviance, rel=1e-10)
    uncertainty = glm_covariance(fit)
    assert bool(uncertainty.valid)
    np.testing.assert_allclose(
        uncertainty.covariance.values, reference.cov_params(), rtol=2e-6
    )
    robust = model.fit(maxiter=200, tol=1e-12, cov_type="HC0")
    hc0, hc1 = glm_covariance(fit, method="HC0"), glm_covariance(fit, method="HC1")
    np.testing.assert_allclose(hc0.covariance.values, robust.cov_params(), rtol=2e-6)
    np.testing.assert_allclose(
        hc1.covariance.values, hc0.covariance.values * 12 / 10, rtol=1e-12
    )
    assert hc0.fit is fit
    intervals = coefficient_intervals(uncertainty, reference="normal")
    np.testing.assert_allclose(
        np.asarray(intervals.estimate.ci).T, reference.conf_int(), rtol=1e-5, atol=3e-7
    )
    restriction = np.asarray([[0.3, 1.0]])
    tested = linear_restriction_test(uncertainty, restriction, 0.2, reference="chi2")
    oracle_test = reference.wald_test((restriction, np.asarray([0.2])), scalar=True)
    assert tested.valid
    np.testing.assert_allclose(tested.test.statistic, oracle_test.statistic, rtol=1e-5)
    np.testing.assert_allclose(tested.test.pvalue, oracle_test.pvalue, rtol=1e-5)
    held_out = np.asarray([[1.0, -1.2], [1.0, 0.3], [1.0, 1.8]])
    held_offset = np.asarray([-0.1, 0.2, 0.3])
    held_trials = np.asarray([7.0, 9.0, 11.0])
    held_exposure = np.asarray([0.7, 1.1, 1.5])
    if family == "binomial":
        binomial_fit = cast("GLMFit[BinomialLogitEquation]", fit)
        expected = reference.predict(held_out, offset=held_offset)
        np.testing.assert_allclose(
            binomial_fit.predict(held_out, offset=held_offset), expected, rtol=1e-6
        )
        np.testing.assert_allclose(
            binomial_fit.predict(
                held_out,
                trials=held_trials,
                offset=held_offset,
                scale="mean",
            ),
            held_trials * expected,
            rtol=1e-6,
        )
    else:
        poisson_fit = cast("GLMFit[PoissonLogEquation]", fit)
        expected = reference.predict(
            held_out, offset=held_offset, exposure=held_exposure
        )
        np.testing.assert_allclose(
            poisson_fit.predict(held_out, offset=held_offset, exposure=held_exposure),
            expected,
            rtol=1e-6,
        )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_scores_and_information_have_an_independent_finite_difference_witness(  # noqa: PLR0914
    family: str,
) -> None:
    """Differentiate a plain NumPy/Python count likelihood rather than STATADIC code."""
    x, successes, trials, counts, offset, exposure = _data()
    if family == "binomial":
        fit = BinomialLogitMLE().estimate(
            BinomialProblem(x, successes, trials=trials, offset=offset)
        )
    else:
        fit = PoissonMLE().estimate(
            PoissonProblem(x, counts, offset=offset, exposure=exposure)
        )
    arrays = [
        np.asarray(value) for value in (x, successes, trials, counts, offset, exposure)
    ]
    xx, yy, mm, cc, oo, ee = arrays

    def likelihood(beta: np.ndarray) -> np.ndarray:
        eta = xx @ beta + oo
        if family == "binomial":
            constants = np.asarray(
                [
                    math.lgamma(m + 1) - math.lgamma(y + 1) - math.lgamma(m - y + 1)
                    for y, m in zip(yy, mm, strict=True)
                ]
            )
            return (
                constants
                - yy * np.logaddexp(0, -eta)
                - (mm - yy) * np.logaddexp(0, eta)
            )
        eta = eta + np.log(ee)
        return cc * eta - np.exp(eta) - np.asarray([math.lgamma(y + 1) for y in cc])

    beta = np.asarray(fit.parameters)
    step = 2e-5
    directions = np.eye(beta.size) * step
    numerical_scores = np.column_stack(
        [
            (likelihood(beta + direction) - likelihood(beta - direction)) / (2 * step)
            for direction in directions
        ]
    )
    np.testing.assert_allclose(
        fit.scores.values, numerical_scores, rtol=1e-7, atol=1e-9
    )
    numerical_information = np.empty((2, 2))
    for i, di in enumerate(directions):
        for k, dk in enumerate(directions):
            numerical_information[i, k] = -np.sum(
                likelihood(beta + di + dk)
                - likelihood(beta + di - dk)
                - likelihood(beta - di + dk)
                + likelihood(beta - di - dk)
            ) / (4 * step**2)
    np.testing.assert_allclose(
        -fit.jacobian.values * 12, numerical_information, rtol=2e-5, atol=1e-5
    )


@pytest.mark.integration
@pytest.mark.contract
@examples(8)
@example(outcomes=_RETAINED_OUTCOMES, dtype=jnp.float32)
@example(outcomes=_RETAINED_OUTCOMES, dtype=jnp.float64)
@given(outcomes=_interior_outcomes(), dtype=st.sampled_from([jnp.float32, jnp.float64]))
def test_fixed_shape_fit_covariance_prediction_jit_and_vmap(
    outcomes: tuple[list[int], list[int]], dtype: DTypeLike
) -> None:
    """Qualify array transforms separately from any fitted-solution derivative."""
    x, _, trials, _, offset, exposure = _data(dtype)
    successes = jnp.asarray(outcomes[0], dtype=dtype)
    counts = jnp.asarray(outcomes[1], dtype=dtype)

    def fit_binomial_outcome(y: jax.Array) -> GLMFit[BinomialLogitEquation]:
        return BinomialLogitMLE().estimate(
            BinomialProblem(x, y, trials=trials, offset=offset)
        )

    binomial = jax.jit(fit_binomial_outcome)(successes)

    def fit_poisson(y: jax.Array) -> GLMFit[PoissonLogEquation]:
        return PoissonMLE().estimate(
            PoissonProblem(x, y, offset=offset, exposure=exposure)
        )

    poisson = jax.jit(fit_poisson)(counts)
    assert bool(binomial.valid)
    assert bool(poisson.valid)
    assert binomial.parameters.dtype == x.dtype == poisson.parameters.dtype

    batch = jax.jit(jax.vmap(fit_poisson))(jnp.stack((counts, counts + 1)))
    assert np.asarray(batch.valid).all()
    covariance = jax.jit(jax.vmap(glm_covariance))(batch)
    assert np.asarray(covariance.valid).all()

    def evaluate_predict(fit: GLMFit[GLMEquation]) -> Array:
        return fit.predict(x, offset=offset, exposure=exposure)

    prediction = jax.jit(evaluate_predict)(poisson)
    np.testing.assert_allclose(prediction, poisson.fitted_means, rtol=1e-5)
    np.testing.assert_allclose(
        binomial.predict(x, offset=offset), binomial.probabilities, rtol=1e-6
    )
    np.testing.assert_allclose(
        binomial.predict(x, trials=trials, offset=offset, scale="mean"),
        binomial.fitted_means,
        rtol=1e-6,
    )


@pytest.mark.component
@pytest.mark.property
@examples(8)
@example(counts=_RETAINED_COUNTS)
@given(counts=st.lists(st.integers(1, 8), min_size=12, max_size=12))
def test_exposure_is_an_additive_log_offset_and_names_remain_ordered(
    counts: list[int],
) -> None:
    """Exposure changes expected counts, while the shared ordered schema is explicit."""
    x, _, _, _, offset, exposure = _data()
    count_array = jnp.asarray(counts, dtype=x.dtype)
    names = ("intercept", "slope")
    fit = PoissonMLE().estimate(
        PoissonProblem(
            x, count_array, offset=offset, exposure=exposure, parameter_names=names
        )
    )
    equivalent = PoissonMLE().estimate(
        PoissonProblem(x, count_array, offset=offset + jnp.log(exposure))
    )
    np.testing.assert_allclose(fit.parameters, equivalent.parameters, atol=1e-10)
    np.testing.assert_array_equal(fit.exposure, exposure)
    np.testing.assert_array_equal(fit.offset, offset)
    with pytest.raises(ValueError, match="column order"):
        fit.predict(x, parameter_names=tuple(reversed(names)))
    np.testing.assert_allclose(
        fit.predict(
            x,
            parameter_names=names,
            scale="link",
            offset=offset,
            exposure=exposure,
        ),
        fit.linear_predictor,
    )


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_complete_quasi_and_poisson_separation_with_offset_invariance(
    dtype: DTypeLike,
) -> None:
    """A strict recession direction cannot become an ordinary finite point estimate."""
    complete_x = jnp.asarray([[1, -2], [1, -1], [1, 1], [1, 2]], dtype=dtype)
    quasi_x = jnp.asarray([[1, -1], [1, 0], [1, 0], [1, 1]], dtype=dtype)
    y = jnp.asarray([0, 0, 1, 1], dtype=dtype)
    for x in (complete_x, quasi_x):
        for offset in (
            jnp.zeros(4, dtype=dtype),
            jnp.asarray([2, -1, 1, 4], dtype=dtype),
        ):
            fit = BinomialLogitMLE().estimate(BinomialProblem(x, y, offset=offset))
            assert fit.existence.status == ExistenceStatus.SEPARATED
            assert bool(fit.identified)
            assert not bool(fit.valid)
            assert np.isnan(np.asarray(fit.parameters)).all()
            assert not bool(glm_covariance(fit).valid)
    separated_poisson = PoissonMLE().estimate(
        PoissonProblem(quasi_x, jnp.asarray([0, 2, 1, 0], dtype=dtype))
    )
    # This fixture has zero rows at both slope extremes, and positive interior
    # rows: its only possible boundary direction is zero, so it is finite.
    assert bool(separated_poisson.valid)
    zero = PoissonMLE().estimate(PoissonProblem(complete_x, jnp.zeros(4, dtype=dtype)))
    assert zero.existence.status == ExistenceStatus.SEPARATED
    assert not bool(zero.valid)
    poisson_quasi = PoissonMLE().estimate(
        PoissonProblem(
            jnp.asarray([[1, 0], [1, 1], [1, 2]], dtype=dtype),
            jnp.asarray([2, 0, 0], dtype=dtype),
        )
    )
    assert poisson_quasi.existence.status == ExistenceStatus.SEPARATED
    assert not bool(poisson_quasi.valid)


@pytest.mark.component
@pytest.mark.numerical
def test_binary_overlap_and_all_zero_poisson_without_intercept_can_be_finite() -> None:
    """Use the cone criterion, including zero counts without an intercept."""
    x = jnp.asarray([[1.0, -1], [1, -1], [1, 1], [1, 1]])
    overlap = BinomialLogitMLE().estimate(BinomialProblem(x, jnp.asarray([0, 1, 0, 1])))
    assert bool(overlap.valid)
    np.testing.assert_allclose(overlap.parameters, 0.0, atol=1e-12)
    zero = PoissonMLE().estimate(
        PoissonProblem(jnp.asarray([[-1.0], [1.0]]), jnp.zeros(2))
    )
    assert bool(zero.valid)
    np.testing.assert_allclose(zero.parameters, 0.0, atol=1e-12)


@pytest.mark.component
@pytest.mark.property
def test_interior_binomial_existence_does_not_depend_on_observation_count() -> None:
    """A relative dual-correction bound retains clearly finite moderate samples."""
    fit = BinomialLogitMLE().estimate(
        BinomialProblem(
            jnp.ones((400, 1), dtype=jnp.float32),
            jnp.ones(400, dtype=jnp.float32),
            trials=2,
        )
    )
    assert bool(fit.existence.finite)
    assert bool(fit.valid)
    assert bool(glm_covariance(fit).valid)
    np.testing.assert_allclose(fit.probabilities, 0.5, atol=1e-6)


@pytest.mark.component
@pytest.mark.property
@examples(8)
@example(outcomes=_RETAINED_OUTCOMES, factor=1e-5)
@example(outcomes=_RETAINED_OUTCOMES, factor=1.0)
@example(outcomes=_RETAINED_OUTCOMES, factor=1e5)
@given(
    outcomes=_interior_outcomes(),
    factor=st.sampled_from([1e-5, 1e-3, 1.0, 1e3, 1e5]),
)
def test_column_rescaling_preserves_fitting_and_separation(
    outcomes: tuple[list[int], list[int]], factor: float
) -> None:
    """Units of a nonzero column cannot change existence or the fitted means."""
    x, _, trials, _, offset, exposure = _data(jnp.float32)
    successes = jnp.asarray(outcomes[0], dtype=x.dtype)
    counts = jnp.asarray(outcomes[1], dtype=x.dtype)
    scaled = x * jnp.asarray([factor, 1 / factor], dtype=x.dtype)

    def fit_binomial_design(
        design: jax.Array, y: jax.Array
    ) -> GLMFit[BinomialLogitEquation]:
        return BinomialLogitMLE().estimate(
            BinomialProblem(design, y, trials=trials, offset=offset)
        )

    def fit_poisson_design(
        design: jax.Array, y: jax.Array
    ) -> GLMFit[PoissonLogEquation]:
        return PoissonMLE().estimate(
            PoissonProblem(design, y, exposure=exposure, offset=offset)
        )

    for fitting, outcome in (
        (
            fit_binomial_design,
            successes,
        ),
        (
            fit_poisson_design,
            counts,
        ),
    ):
        original = fitting(x, outcome)
        result = fitting(scaled, outcome)
        assert bool(original.valid)
        assert bool(result.valid)
        np.testing.assert_allclose(
            result.fitted_means, original.fitted_means, rtol=1e-4
        )
    separated = BinomialLogitMLE().estimate(
        BinomialProblem(
            jnp.full((4, 1), factor, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32)
        )
    )
    assert separated.existence.status == ExistenceStatus.SEPARATED
    assert not bool(separated.valid)
    assert not bool(glm_covariance(separated).valid)


@pytest.mark.integration
@pytest.mark.contract
def test_tiny_complete_separation_is_rejected_without_the_annotation_hook() -> None:
    """Ordinary default-f32 imports cannot mistake small units for finite overlap."""
    code = """\
import jax  # noqa: PLC0415, TC002
import jax.numpy as jnp
from statadic.estimation.glm import (
    BinomialLogitMLE, BinomialProblem, PoissonMLE, PoissonProblem,
)
from statadic.inference.glm_covariance import glm_covariance
from statadic.optimize.glm import ExistenceStatus
assert not jax.config.x64_enabled
for scale in (1e-20, 1e-5, 1e5):
    fit = BinomialLogitMLE().estimate(
        BinomialProblem(jnp.full((4, 1), scale), jnp.ones(4))
    )
    assert fit.existence.status == ExistenceStatus.SEPARATED
    assert not fit.valid
    assert not glm_covariance(fit).valid
    zero = PoissonMLE().estimate(PoissonProblem(jnp.full((4, 1), scale), jnp.zeros(4)))
    assert zero.existence.status == ExistenceStatus.SEPARATED
    assert not zero.valid
    assert not glm_covariance(zero).valid
"""
    result = subprocess.run(  # fixed interpreter and local regression  # noqa: S603
        [sys.executable, "-c", code],
        env=dict(os.environ, JAX_ENABLE_X64="false"),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    ["fractional", "negative", "trials", "exposure", "nonfinite", "rank", "budget"],
)
def test_invalid_and_nonconverged_fits_do_not_authorize_inference(case: str) -> None:
    """Keep input, rank, solver and inferential failure evidence distinct."""
    x, successes, trials, counts, offset, exposure = _data()
    if case == "fractional":
        fit = PoissonMLE().estimate(PoissonProblem(x, counts + 0.1))
    elif case == "negative":
        fit = PoissonMLE().estimate(PoissonProblem(x, counts.at[0].set(-1)))
    elif case == "trials":
        fit = BinomialLogitMLE().estimate(BinomialProblem(x, successes, trials=0))
    elif case == "exposure":
        fit = PoissonMLE().estimate(
            PoissonProblem(x, counts, exposure=exposure.at[0].set(0))
        )
    elif case == "nonfinite":
        fit = PoissonMLE().estimate(
            PoissonProblem(x, counts, offset=offset.at[0].set(jnp.inf))
        )
    elif case == "rank":
        fit = BinomialLogitMLE().estimate(
            BinomialProblem(jnp.ones_like(x), successes, trials=trials)
        )
        assert not bool(fit.identified)
    else:
        fit = PoissonMLE(max_steps=1).estimate(PoissonProblem(x, counts))
        assert bool(fit.input_valid)
        assert not bool(fit.solver.converged)
    assert not bool(fit.valid)
    assert not bool(glm_covariance(fit).valid)
    assert np.isnan(np.asarray(fit.parameters)).all()


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_separation_cone_agrees_with_independent_linear_program(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """A bounded SciPy LP independently witnesses complete and quasi recession."""
    scipy = optional_reference("scipy.optimize")
    for x, y in (
        (np.asarray([[1.0, -2], [1, -1], [1, 1], [1, 2]]), np.asarray([0, 0, 1, 1])),
        (np.asarray([[1.0, -1], [1, 0], [1, 0], [1, 1]]), np.asarray([0, 0, 1, 1])),
        (np.asarray([[1.0, -1], [1, -1], [1, 1], [1, 1]]), np.asarray([0, 1, 0, 1])),
    ):
        oriented = x * (2 * y - 1)[:, None]
        oracle = scipy.linprog(
            -oriented.sum(axis=0),
            A_ub=-oriented,
            b_ub=np.zeros(len(y)),
            bounds=[(-1, 1)] * 2,
            method="highs",
        )
        assert oracle.success
        fit = BinomialLogitMLE().estimate(
            BinomialProblem(jnp.asarray(x), jnp.asarray(y))
        )
        oracle_tolerance = 1e-9
        assert bool(fit.existence.finite) == (oracle.fun >= -oracle_tolerance)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize(("dtype", "delta"), [(jnp.float32, 1e-4), (jnp.float64, 1e-8)])
def test_small_separating_margin_cannot_be_classified_as_finite(
    optional_reference: Callable[[str], ModuleType], dtype: DTypeLike, delta: float
) -> None:
    """An absolute dual residual cannot certify overlap in nearly collinear data."""
    scipy = optional_reference("scipy.optimize")
    x = jnp.asarray(
        [[1, 1 - delta], [1, 1 - delta], [1, 1 + delta], [1, 1 + delta]], dtype=dtype
    )
    y = jnp.asarray([0, 0, 1, 1], dtype=dtype)
    oriented = np.asarray(x) * np.asarray(2 * y - 1)[:, None]
    oracle = scipy.linprog(
        -oriented.sum(axis=0),
        A_ub=-oriented,
        b_ub=np.zeros(4),
        bounds=[(-1, 1)] * 2,
        method="highs",
    )
    assert oracle.success
    assert oracle.fun < 0
    fit = BinomialLogitMLE().estimate(BinomialProblem(x, y))
    assert bool(fit.identified)
    assert not bool(fit.existence.finite)
    assert not bool(fit.valid)
    assert not bool(glm_covariance(fit).valid)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("glum")
def test_poisson_matches_glum_when_available(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match an unpenalized log-link count model with an explicit design intercept."""
    glum = optional_reference("glum")
    x, _, _, counts, offset, exposure = _data()
    reference = glum.GeneralizedLinearRegressor(
        family="poisson",
        link="log",
        alpha=0,
        fit_intercept=False,
        gradient_tol=1e-10,
        max_iter=200,
    ).fit(
        np.asarray(x), np.asarray(counts), offset=np.asarray(offset + jnp.log(exposure))
    )
    fit = PoissonMLE().estimate(
        PoissonProblem(x, counts, offset=offset, exposure=exposure)
    )
    assert bool(fit.valid)
    np.testing.assert_allclose(fit.parameters, reference.coef_, rtol=1e-5, atol=2e-7)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("family", ["binomial", "poisson"])
@examples(8)
@given(
    outcomes=st.lists(st.integers(2, 6), min_size=8, max_size=8),
    offsets=st.lists(st.integers(-4, 4), min_size=4, max_size=4),
    exposures=st.lists(st.integers(2, 8), min_size=4, max_size=4),
    scale=st.sampled_from([0.75, 1.0, 1.5]),
)
def test_generated_interior_glm_matches_independent_fit_scores_and_information(  # noqa: PLR0914
    optional_reference: Callable[[str], ModuleType],
    family: str,
    outcomes: list[int],
    offsets: list[int],
    exposures: list[int],
    scale: float,
) -> None:
    """Interior unequal paired counts guarantee finite fits and resolved score noise."""
    sm = optional_reference("statsmodels.api")
    x = np.column_stack((np.ones(8), np.repeat(scale * np.linspace(-1.5, 1.5, 4), 2)))
    y = np.asarray(outcomes, dtype=float)
    y[::2], y[1::2] = 2 + y[::2] % 2, 5 + y[1::2] % 2
    offset, exposure = np.repeat(offsets, 2) / 8, np.repeat(exposures, 2) / 4
    if family == "binomial":
        trials = np.full(8, 8.0)
        model = sm.GLM(
            np.column_stack((y, trials - y)),
            x,
            family=sm.families.Binomial(),
            offset=offset,
        )
        reference = model.fit(maxiter=200, tol=1e-12)
        beta = np.asarray(reference.params)
        probability = 1 / (1 + np.exp(-(x @ beta + offset)))
        means = trials * probability
        variance = trials * probability * (1 - probability)
        actual = _GENERATED_BINOMIAL_FIT(x, y, trials=trials, offset=offset)
    else:
        model = sm.GLM(
            y, x, family=sm.families.Poisson(), offset=offset, exposure=exposure
        )
        reference = model.fit(maxiter=200, tol=1e-12)
        beta = np.asarray(reference.params)
        means = np.exp(x @ beta + offset + np.log(exposure))
        variance = means
        actual = _GENERATED_POISSON_FIT(x, y, offset=offset, exposure=exposure)
    scores = x * (y - means)[:, None]
    information = x.T @ (variance[:, None] * x)
    model_covariance = np.linalg.inv(information)
    hc0_covariance = model_covariance @ scores.T @ scores @ model_covariance

    assert actual.valid
    assert actual.identified
    assert actual.existence.finite
    assert int(actual.rank) == x.shape[1]
    assert int(actual.df_resid) == x.shape[0] - x.shape[1]
    assert int(actual.df_resid) == reference.df_resid
    np.testing.assert_allclose(
        actual.parameters, reference.params, rtol=2e-5, atol=3e-7
    )
    np.testing.assert_allclose(actual.fitted_means, means, rtol=2e-6, atol=5e-7)
    np.testing.assert_allclose(actual.scores.values, scores, rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(
        -8 * actual.jacobian.values, information, rtol=2e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        glm_covariance(actual).covariance.values, model_covariance, rtol=3e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        glm_covariance(actual, method="HC0").covariance.values,
        hc0_covariance,
        rtol=2e-5,
        atol=1e-7,
    )
