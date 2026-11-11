"""Covariance doors return exactly the native owner's arrays and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.api.estimand import Covariance, Estimand
from statadic.estimation.covariance import (
    LedoitWolf,
    SampleCovariance,
    ledoit_wolf_sample_kernel,
    sample_covariance,
)

from .._budgets import examples
from .._typing import ill_typed

if TYPE_CHECKING:
    from typing import SupportsIndex

    from hypothesis.strategies import SearchStrategy


def _panel(n_obs: int = 60, n_vars: int = 5, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(n_vars, n_vars))
    return rng.normal(size=(n_obs, n_vars)) @ mixing


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("ddof", [0, 1])
def test_sample_covariance_equals_native_and_numpy(ddof: int) -> None:
    """Sample covariance matches the owner exactly and NumPy to float32 accuracy."""
    returns = _panel()
    estimate = Estimand(Covariance(), estimator=SampleCovariance(ddof)).estimate(
        returns
    )
    np.testing.assert_array_equal(estimate.sigma, sample_covariance(returns, ddof))
    np.testing.assert_allclose(
        estimate.sigma, np.cov(returns, rowvar=False, ddof=ddof), rtol=1e-4
    )
    assert (estimate.n_observations, estimate.n_variables) == (60, 5)
    assert estimate.shrinkage is None
    assert estimate.target is None


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize(("declared", "ddof"), [(np.int32(0), 0), (np.int64(1), 1)])
def test_sample_covariance_accepts_integer_scalar_divisors(
    declared: SupportsIndex, ddof: int
) -> None:
    """Index-valued caller controls retain the same divisor as the eager owner."""
    estimator = SampleCovariance(declared)
    assert type(estimator.ddof) is int
    values = _panel()
    result = Estimand(Covariance(), estimator=estimator).estimate(values)
    np.testing.assert_allclose(
        result.sigma, np.cov(values, rowvar=False, ddof=ddof), rtol=1e-4
    )
    np.testing.assert_array_equal(result.sigma, sample_covariance(values, ddof))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("ddof", [True, -1, 0.5])
def test_sample_covariance_rejects_invalid_divisor_controls(ddof: object) -> None:
    """Accepting integer scalar classes does not accept booleans or fractions."""
    with pytest.raises((TypeError, ValueError)):
        ill_typed(SampleCovariance)(ddof)


@pytest.mark.integration
def test_ledoit_wolf_equals_native_and_kernel() -> None:
    """Ledoit-Wolf matches the eager owner and its pure kernel."""
    returns = _panel()
    estimate = Estimand(Covariance(), estimator=LedoitWolf()).estimate(returns)
    native = ledoit_wolf_sample_kernel(jnp.asarray(returns))
    np.testing.assert_array_equal(estimate.sigma, native.sigma)
    np.testing.assert_array_equal(estimate.target, native.target)
    assert estimate.shrinkage == native.shrinkage
    kernel = ledoit_wolf_sample_kernel(jnp.asarray(returns))
    np.testing.assert_array_equal(estimate.sigma, kernel.sigma)
    assert estimate.shrinkage is not None
    assert 0.0 < estimate.shrinkage < 1.0


_MIN_VARIANCE = 1e-3


def _panels() -> SearchStrategy[np.ndarray]:
    """Panels inside the documented domain: every column has positive variance."""
    return hnp.arrays(
        np.float64,
        (60, 5),
        elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    ).filter(lambda panel: bool(np.all(np.var(panel, axis=0) > _MIN_VARIANCE)))


@pytest.mark.component
@pytest.mark.contract
@example(panel=_panel(), model=SampleCovariance())
@example(panel=_panel(), model=LedoitWolf())
@given(panel=_panels(), model=st.sampled_from([SampleCovariance(), LedoitWolf()]))
@examples(20)
def test_prepared_panel_is_retained_and_model_bound(
    panel: np.ndarray,
    model: SampleCovariance | LedoitWolf,
) -> None:
    """Preparation snapshots the panel and rejects another model's token."""
    panel = panel.copy()
    specification = Estimand(Covariance(), estimator=model)
    expected = specification.estimate(panel).sigma
    prepared = specification.prepare(panel)
    panel[:] = 0
    np.testing.assert_array_equal(
        specification.estimate_prepared(prepared).sigma, expected
    )
    with pytest.raises(ValueError, match="another model"):
        Estimand(Covariance(), estimator=type(model)()).estimate_prepared(prepared)


@pytest.mark.integration
@pytest.mark.contract
@example(returns=_panel())
@given(returns=_panels())
@examples(20)
def test_ledoit_wolf_float64_equals_native(returns: np.ndarray) -> None:
    """The double-precision door equals the owner."""
    with jax.enable_x64(new_val=True):
        estimate = Estimand(Covariance(), estimator=LedoitWolf()).estimate(returns)
        native = ledoit_wolf_sample_kernel(jnp.asarray(returns))
        assert estimate.sigma.dtype == jnp.float64
        np.testing.assert_array_equal(estimate.sigma, native.sigma)


@pytest.mark.unit
@pytest.mark.contract
def test_invalid_panels_fail_in_the_owner() -> None:
    """The class validates panels eagerly before calling the kernel."""
    with pytest.raises(ValueError, match="observation"):
        Estimand(Covariance(), estimator=LedoitWolf()).estimate(np.ones((1, 3)))
