"""Quantity workflows retain native producers, aligned identity and model ownership."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations
from statadic.api.estimand import Covariance, Estimand, MomentConditions
from statadic.api.moments import GMMSandwich, MomentData
from statadic.core.equations import PlainLongRun
from statadic.estimation.covariance import LedoitWolf, SampleCovariance
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

    import jax
    from statadic.estimation.covariance import CovarianceProblem

pytestmark = pytest.mark.component


@pytest.mark.reference
def test_named_covariance_preserves_column_order_and_row_identity() -> None:
    """An independent covariance checks explicit feature selection and ddof."""
    rows = Observations(
        {"a": [1.0, 4.0, 2.0, 8.0], "b": [0.0, 3.0, 7.0, 2.0], "unused": [9.0] * 4},
        row_ids=("u", "v", "w", "x"),
    )
    fitted = Estimand(Covariance(("b", "a")), estimator=SampleCovariance()).estimate(
        rows
    )
    expected = np.cov(
        np.column_stack((np.asarray(rows.numeric("b")), np.asarray(rows.numeric("a")))),
        rowvar=False,
    )
    np.testing.assert_allclose(fitted.sigma, expected, rtol=2e-6)
    assert fitted.context.columns == ("b", "a")
    assert fitted.context.row_ids == rows.row_ids
    assert fitted.valid
    with pytest.raises(ValueError, match="explicit columns"):
        Estimand(Covariance(), estimator=SampleCovariance()).estimate(rows)


@pytest.mark.contract
def test_covariance_preparation_is_owned_and_model_bound() -> None:
    """The model snapshots mutable input and a copied specification has a new token."""
    values = np.array([[1.0, 2.0], [2.0, 0.0], [4.0, 3.0], [5.0, 8.0]])
    model = Estimand(Covariance(), estimator=SampleCovariance(ddof=0))
    expected = np.cov(values, rowvar=False, ddof=0)
    prepared = model.prepare(values)
    values[:] = -100
    np.testing.assert_allclose(model.estimate_prepared(prepared).sigma, expected)
    with pytest.raises(ValueError, match="another model"):
        replace(model).estimate_prepared(prepared)


@pytest.mark.contract
def test_supplied_moments_use_one_data_owner_and_shared_model_lifecycle() -> None:
    """Re-estimation consumes the prepared moment snapshot, not caller mutation."""
    moments = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0], [4.0, 2.0]])
    jacobian = np.array([[1.0], [0.0]])
    ids = (10, 20, 30, 40)
    model = Estimand(MomentConditions(), estimator=FixedWeightGMM(PlainLongRun))
    prepared = model.prepare(
        MomentData.from_arrays(
            AffineMomentProblem(moments, jacobian, weight=np.eye(2)), row_ids=ids
        )
    )
    moments[:] = 100
    jacobian[:] = 0
    fit = model.estimate_prepared(prepared)
    assert fit.context.row_ids == ids
    np.testing.assert_allclose(fit.parameters, [-2.5])
    assert fit.infer(covariance=GMMSandwich()).uncertainty.fit is fit.core
    with pytest.raises(ValueError, match="another model"):
        replace(model).estimate_prepared(prepared)


@pytest.mark.contract
def test_structural_covariance_estimator_returns_its_actual_native_array() -> None:
    """Orchestration does not recompute or replace a producer's fitted values."""
    retained: list[jax.Array] = []

    class Estimator:
        def estimate(self, problem: CovarianceProblem) -> jax.Array:
            result = SampleCovariance().estimate(problem)
            retained.append(result)
            return result

    fit = Estimand(Covariance(), estimator=Estimator()).estimate(
        jnp.arange(12.0).reshape(4, 3)
    )
    assert fit.core is retained[0]
    assert fit.valid  # Singular covariance is allowed; no inverse is promised.


@pytest.mark.contract
def test_covariance_domains_remain_estimator_specific() -> None:
    """Shrinkage does not silently manufacture constant-column correlations."""
    values = jnp.array([[1.0, 0.0], [1.0, 2.0], [1.0, 3.0]])
    assert Estimand(Covariance(), estimator=SampleCovariance()).estimate(values).valid
    with pytest.raises(ValueError, match="positive sample variances"):
        Estimand(Covariance(), estimator=LedoitWolf()).estimate(values)
    with pytest.raises(ValueError, match="Too few observations"):
        Estimand(Covariance(), estimator=SampleCovariance(ddof=3)).estimate(values)


@pytest.mark.contract
def test_numerically_unavailable_covariance_remains_reportable() -> None:
    """A failed numerical producer is not replaced by a fabricated covariance."""

    class Unavailable:
        def estimate(self, problem: CovarianceProblem) -> jax.Array:
            width = jnp.asarray(problem.values).shape[1]
            return jnp.full((width, width), jnp.nan)

    result = Estimand(Covariance(), estimator=Unavailable()).estimate(
        jnp.arange(8.0).reshape(4, 2)
    )
    assert not result.valid
    assert result.report().warnings
    assert result.report().metadata["fit_valid"] is False


@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
@pytest.mark.contract
def test_quantity_constructor_preserves_concrete_result_types(tmp_path: Path) -> None:
    """The bounded constructor grammar retains all four native result families."""
    check_types(
        """
from typing import assert_type
from statadic.api.covariance import CovarianceEstimate
from statadic.api.data import Observations
from statadic.api.estimand import Covariance, Estimand, MomentConditions
from statadic.api.moments import AffineGELResult, FixedGELResult, GMMResult, MomentData
from statadic.core.equations import PlainLongRun
from statadic.estimation.covariance import SampleCovariance
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem

def check(
    rows: Observations,
    affine: MomentData[AffineMomentProblem],
    fixed: MomentData[FixedMomentProblem],
) -> None:
    covariance = Estimand(Covariance(("a",)), estimator=SampleCovariance())
    assert_type(covariance.estimate(rows), CovarianceEstimate)
    gmm = Estimand(MomentConditions(), estimator=FixedWeightGMM(PlainLongRun))
    assert_type(gmm.estimate(affine), GMMResult[PlainLongRun])
    affine_gel = Estimand(MomentConditions(), estimator=AffineGEL())
    assert_type(affine_gel.estimate(affine), AffineGELResult)
    fixed_gel = Estimand(MomentConditions(), estimator=FixedGEL())
    assert_type(fixed_gel.estimate(fixed), FixedGELResult)
""",
        "quantity_results.py",
        tmp_path,
    )
