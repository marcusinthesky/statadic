"""Public structural workflows against independently prepared reference models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.inference import HAC, HC, Clustered, ModelBased
from statadic.api.ordination import Ordination
from statadic.api.panel import PanelCovariance, PanelResult
from statadic.api.regression import IVResult, LinearResult, Regression
from statadic.api.representation import ObservedDissimilarity
from statadic.api.spatial.graphs import PanelData
from statadic.estimation.iv import TwoStageLeastSquares
from statadic.estimation.panel import PanelFit, Within

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.api.inference import CovarianceSpec

_MISSING_PERIOD = 2


def _tolerance(dtype: type) -> float:
    return 4e-4 if dtype is np.float32 else 2e-8


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_fixed_effects_matches_linearmodels_without_intercept(  # noqa: PLR0914 -- matched rows, independent peer, covariance and predictions
    optional_reference: Callable[[str], ModuleType], dtype: type
) -> None:
    """Host effects select the same unbalanced within model and covariance."""
    lm = optional_reference("linearmodels.panel")
    pd = optional_reference("pandas")
    rng = np.random.default_rng(927)
    entity = np.repeat(np.arange(14), 6)
    time = np.tile(np.arange(6), 14)
    keep = ~((entity % 3 == 0) & (time == _MISSING_PERIOD))
    entity, time = entity[keep], time[keep]
    features = rng.normal(size=(len(entity), 2)).astype(dtype)
    outcome = (
        features @ np.array([0.7, -0.4], dtype=dtype)
        + rng.normal(size=14)[entity]
        + 0.2 * rng.normal(size=len(entity))
    ).astype(dtype)
    tolerance = _tolerance(dtype)
    with jax.enable_x64():
        rows = Observations(
            {
                "unit": entity,
                "time": time,
                "x": jnp.asarray(features[:, 0]),
                "z": jnp.asarray(features[:, 1]),
                "y": jnp.asarray(outcome),
            }
        )
        layout = PanelLayout("unit", "time")
        fitted = Regression(
            "y",
            Linear("x") + Linear("z"),
            structure=layout,
            effects="unit",
            # The stopping criterion is chosen at the supplied working precision.
            estimator=Within(tolerance=float(max(1e-9, 16 * np.finfo(dtype).eps))),
        ).estimate(PanelData(rows, layout))
        assert isinstance(fitted, PanelResult)
        assert isinstance(fitted.core, PanelFit)
        index = pd.MultiIndex.from_arrays([entity, time])
        peer = lm.PanelOLS(
            pd.Series(outcome, index=index),
            pd.DataFrame(features, index=index),
            entity_effects=True,
        ).fit(cov_type="unadjusted", debiased=True, auto_df=False, count_effects=True)
        np.testing.assert_allclose(
            fitted.parameters, peer.params, rtol=tolerance, atol=tolerance
        )
        covariance = fitted.infer(
            covariance=PanelCovariance(
                method="classical", debiased=True, count_effects=True
            )
        )
        assert covariance.uncertainty.fit is fitted.core
        np.testing.assert_allclose(
            covariance.uncertainty.covariance.values,
            peer.cov,
            rtol=tolerance,
            atol=tolerance,
        )
        dummy = np.eye(14)[entity]
        expected_df = len(outcome) - np.linalg.matrix_rank(
            np.column_stack((features, dummy))
        )
        assert int(fitted.core.df_resid) == expected_df == peer.df_resid
        expected = peer.fitted_values.iloc[:, 0] + peer.estimated_effects.iloc[:, 0]
        np.testing.assert_allclose(
            fitted.predict(rows, kind="conditional"),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_two_stage_least_squares_matches_linearmodels(  # noqa: PLR0914 -- independent IV inputs, estimates, covariance and prediction targets
    optional_reference: Callable[[str], ModuleType], dtype: type
) -> None:
    """The public IV route matches structural coefficients, covariance and queries."""
    lm = optional_reference("linearmodels.iv")
    rng = np.random.default_rng(928)
    count = 96
    x = rng.normal(size=count).astype(dtype)
    instruments = rng.normal(size=(count, 2)).astype(dtype)
    disturbance = rng.normal(size=count).astype(dtype)
    endogenous = (
        0.8 * instruments[:, 0] - 0.4 * instruments[:, 1] + 0.3 * x + 0.5 * disturbance
    ).astype(dtype)
    outcome = (
        0.4
        - 0.2 * x
        + 1.3 * endogenous
        + 0.6 * disturbance
        + 0.3 * rng.normal(size=count)
    ).astype(dtype)
    exogenous = np.column_stack((np.ones(count, dtype=dtype), x))
    tolerance = _tolerance(dtype)
    with jax.enable_x64():
        rows = Observations(
            {
                "x": jnp.asarray(x),
                "endog": jnp.asarray(endogenous),
                "z1": jnp.asarray(instruments[:, 0]),
                "z2": jnp.asarray(instruments[:, 1]),
                "y": jnp.asarray(outcome),
            }
        )
        fitted = Regression(
            "y",
            Intercept() + Linear("x") + Linear("endog"),
            endogenous=("endog",),
            instruments=("z1", "z2"),
            estimator=TwoStageLeastSquares(),
        ).estimate(rows)
        assert isinstance(fitted, IVResult)
        peer = lm.IV2SLS(outcome, exogenous, endogenous[:, None], instruments).fit(
            cov_type="unadjusted", debiased=False
        )
        np.testing.assert_allclose(
            fitted.parameters, peer.params, rtol=tolerance, atol=tolerance
        )
        inferred = fitted.infer(covariance=ModelBased())
        assert inferred.uncertainty.fit is fitted.core
        np.testing.assert_allclose(
            inferred.uncertainty.covariance.values,
            peer.cov,
            rtol=tolerance,
            atol=tolerance,
        )
        outcome_test = inferred.test(
            Hypothesis(LinearRestriction.zero("endog")), reference="chi2"
        )
        peer_test = peer.wald_test(np.array([[0.0, 0.0, 1.0]]))
        assert outcome_test.available
        assert outcome_test.statistic == pytest.approx(
            peer_test.stat, rel=tolerance, abs=tolerance
        )
        assert outcome_test.pvalue == pytest.approx(
            peer_test.pval, rel=tolerance, abs=tolerance
        )
        query = np.array([[-0.7, 0.4], [0.2, -0.3], [1.1, 0.9]], dtype=dtype)
        expected = peer.predict(
            exog=np.column_stack((np.ones(3, dtype=dtype), query[:, 0])),
            endog=query[:, 1, None],
        )
        np.testing.assert_allclose(
            fitted.predict(
                {"x": jnp.asarray(query[:, 0]), "endog": jnp.asarray(query[:, 1])}
            ),
            np.asarray(expected).ravel(),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["HC0", "HC1", "HAC", "clustered"])
def test_public_two_stage_least_squares_robust_covariance_matches_linearmodels(  # noqa: PLR0914 -- independent IV inputs, four covariance requests and peer options
    optional_reference: Callable[[str], ModuleType], dtype: type, kind: str
) -> None:
    """Robust 2SLS covariance matches linearmodels under matched conventions."""
    lm = optional_reference("linearmodels.iv")
    rng = np.random.default_rng(928)
    count = 96
    x = rng.normal(size=count).astype(dtype)
    instruments = rng.normal(size=(count, 2)).astype(dtype)
    disturbance = rng.normal(size=count).astype(dtype)
    endogenous = (
        0.8 * instruments[:, 0] - 0.4 * instruments[:, 1] + 0.3 * x + 0.5 * disturbance
    ).astype(dtype)
    # Heteroskedastic structural error so the robust meats differ from model-based.
    outcome = (
        0.4
        - 0.2 * x
        + 1.3 * endogenous
        + (0.6 + 0.5 * np.abs(x)) * disturbance
        + 0.3 * rng.normal(size=count)
    ).astype(dtype)
    clusters = np.arange(count) % 12
    exogenous = np.column_stack((np.ones(count, dtype=dtype), x))
    tolerance = _tolerance(dtype)
    bandwidth = 4
    # Conventions, each checked to agree to rounding in float64:
    #  - HC0 is the plain sandwich, linearmodels "robust" with debiased=False.
    #  - HC1 scales by n / (n - k) with k = 3 coefficients, linearmodels
    #    "robust" with debiased=True. linearmodels' debiased flag is the only
    #    small-sample switch, so it selects HC1 here.
    #  - HAC is Bartlett with weights 1 - j / (bandwidth + 1) for lags
    #    j <= bandwidth. statadic's bandwidth is that largest included lag and
    #    linearmodels' "bandwidth" is the same number; both use uncentred scores
    #    and no small-sample factor (debiased=False).
    #  - clustered is the one-way sum of cluster-summed scores with no G/(G-1)
    #    or (n-1)/(n-k) factor, linearmodels "clustered" with debiased=False.
    specification: CovarianceSpec
    peer_options: dict[str, object]
    if kind == "HC0":
        specification = HC("HC0")
        peer_options = {"cov_type": "robust", "debiased": False}
    elif kind == "HC1":
        specification = HC("HC1")
        peer_options = {"cov_type": "robust", "debiased": True}
    elif kind == "HAC":
        specification = HAC(bandwidth, order_by="t")
        peer_options = {
            "cov_type": "kernel",
            "kernel": "bartlett",
            "bandwidth": bandwidth,
            "debiased": False,
        }
    else:
        specification = Clustered("g")
        peer_options = {
            "cov_type": "clustered",
            "clusters": clusters,
            "debiased": False,
        }
    with jax.enable_x64():
        rows = Observations(
            {
                "x": jnp.asarray(x),
                "endog": jnp.asarray(endogenous),
                "z1": jnp.asarray(instruments[:, 0]),
                "z2": jnp.asarray(instruments[:, 1]),
                "y": jnp.asarray(outcome),
                "t": jnp.arange(count, dtype=jnp.float64),
                "g": jnp.asarray(clusters),
            }
        )
        fitted = Regression(
            "y",
            Intercept() + Linear("x") + Linear("endog"),
            endogenous=("endog",),
            instruments=("z1", "z2"),
            estimator=TwoStageLeastSquares(),
        ).estimate(rows)
        assert isinstance(fitted, IVResult)
        peer = lm.IV2SLS(outcome, exogenous, endogenous[:, None], instruments).fit(
            **peer_options
        )
        inferred = fitted.infer(covariance=specification)
        np.testing.assert_allclose(
            fitted.parameters, peer.params, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            inferred.uncertainty.covariance.values,
            peer.cov,
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-learn")
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_principal_component_regression_matches_sklearn_and_ols(  # noqa: PLR0914 -- independent PCA/OLS training and held-out comparisons
    optional_reference: Callable[[str], ModuleType], dtype: type
) -> None:
    """Independent PCA and OLS check held-out predictions without sign assumptions."""
    decomposition = optional_reference("sklearn.decomposition")
    sm = optional_reference("statsmodels.api")
    rng = np.random.default_rng(929)
    features = (
        rng.normal(size=(32, 3))
        @ np.array([[2.0, 0.1, -0.2], [0.0, 0.8, 0.3], [0.0, 0.0, 0.2]])
    ).astype(dtype)
    outcome = (
        0.7
        + features @ np.array([0.6, -0.4, 0.2], dtype=dtype)
        + 0.2 * rng.normal(size=32)
    ).astype(dtype)
    query = rng.normal(size=(5, 3)).astype(dtype)
    tolerance = _tolerance(dtype)
    with jax.enable_x64():
        rows = Observations(
            {
                **{f"x{i}": jnp.asarray(features[:, i]) for i in range(3)},
                "y": jnp.asarray(outcome),
            }
        )
        coordinates = Ordination.pca(("x0", "x1", "x2"), dim=2, name="principal")
        fitted = Regression.ols("y", Intercept() + coordinates).estimate(rows)
        assert isinstance(fitted, LinearResult)
        pca = decomposition.PCA(n_components=2, svd_solver="full")
        scores = pca.fit_transform(features)
        peer = sm.OLS(
            outcome, np.column_stack((np.ones(len(scores), dtype=dtype), scores))
        ).fit()
        actual_scores = fitted.design.ordination("principal").inspection.fit.coordinates
        np.testing.assert_allclose(
            actual_scores @ actual_scores.T,
            scores @ scores.T,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            fitted.core.fitted_values, peer.fittedvalues, rtol=tolerance, atol=tolerance
        )
        peer_query = np.column_stack(
            (np.ones(len(query), dtype=dtype), pca.transform(query))
        )
        actual = fitted.predict({f"x{i}": jnp.asarray(query[:, i]) for i in range(3)})
        np.testing.assert_allclose(
            actual, peer.predict(peer_query), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            fitted.design.ordination("principal").explained_variance_ratio,
            pca.explained_variance_ratio_,
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("negative", [False, True])
def test_signed_spectrum_supports_consumer_table_without_legacy_embedding(
    *, negative: bool
) -> None:
    """The retained five-field spectrum supports strict and descriptive consumers."""
    points = np.arange(5.0)
    distances = np.abs(points[:, None] - points[None])
    if negative:
        distances = distances**2
    with jax.enable_x64():
        fitted = Ordination.pcoa(
            dim=None, negative_inertia="positive-subspace" if negative else "raise"
        ).estimate(
            ObservedDissimilarity.from_array(
                distances, row_ids=("a", "b", "c", "d", "e")
            )
        )
        spectrum = fitted.spectrum
        centering = np.eye(5) - np.ones((5, 5)) / 5
        expected = np.linalg.eigvalsh(centering @ (-0.5 * distances**2) @ centering)[
            ::-1
        ]
        threshold = (
            8 * len(points) * np.finfo(np.float64).eps * np.max(np.abs(expected))
        )
        table = tuple(
            (
                axis + 1,
                float(value),
                bool(value > spectrum.threshold),
                bool(value < -spectrum.threshold),
            )
            for axis, value in enumerate(spectrum.eigenvalues)
        )
        np.testing.assert_allclose(
            [row[1] for row in table], expected, rtol=1e-12, atol=1e-12
        )
        assert spectrum.threshold == pytest.approx(threshold, rel=1e-12)
        assert spectrum.n_positive == sum(row[2] for row in table)
        assert spectrum.n_negative == sum(row[3] for row in table)
        assert spectrum.material_negative_inertia == negative
        assert spectrum.n_negative == int(np.count_nonzero(expected < -threshold))
        assert fitted.context.construction_warrant is None
        assert fitted.valid  # Descriptive negative inertia is not numerical failure.
