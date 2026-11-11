# standalone package-maintenance entry point
"""Panel covariance methods against matched linearmodels conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from statadic.estimation.panel import FamaMacBeth, PanelProblem, Within
from statadic.inference.panel_covariance import fama_macbeth_covariance, panel_covariance
from statadic.operators.panel import prepare_panel_index

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_FINAL_PERIOD = 9


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("balanced", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    "method",
    [
        "classical",
        "heteroskedastic",
        "cluster-entity",
        "cluster-time",
        "cluster-two-way",
        "driscoll-kraay",
        "autocorrelation",
    ],
)
def test_panel_covariance_matches_linearmodels(
    optional_reference: Callable[[str], ModuleType],
    *,
    balanced: bool,
    weighted: bool,
    method: Literal[
        "classical",
        "heteroskedastic",
        "cluster-entity",
        "cluster-time",
        "cluster-two-way",
        "driscoll-kraay",
        "autocorrelation",
    ],
) -> None:
    """Match slopes/effect DoF and explicit covariance/bandwidth choices."""
    optional_reference("linearmodels")
    pytest.importorskip("linearmodels", minversion="7.1")
    lm = optional_reference("linearmodels.panel")
    pd = optional_reference("pandas")
    rng = np.random.default_rng(126)
    e = np.repeat(np.arange(20), 10)
    t = np.tile(np.arange(10), 20)
    # AC peers use observed row lags. Terminal omissions align those with STATADIC's
    # actual-period lags; interior-gap semantics have an independent sum witness.
    keep = (
        np.ones(len(e), dtype=bool)
        if balanced
        else ~((e % 3 == 0) & (t == _FINAL_PERIOD))
    )
    e, t = e[keep], t[keep]
    x = rng.normal(size=(len(e), 2))
    y = x @ np.array([0.5, -0.3]) + rng.normal(size=20)[e] + rng.normal(size=len(e))
    weights = rng.uniform(0.4, 2.5, len(e)) if weighted else np.ones(len(e))
    index = prepare_panel_index(e, t)
    fit = Within().estimate(PanelProblem(x, y, index, weights, effects="unit"))
    actual = panel_covariance(fit, method=method, bandwidth=2)
    keys = pd.MultiIndex.from_arrays([e, t])
    model = lm.PanelOLS(
        pd.Series(y, index=keys),
        pd.DataFrame(x, index=keys),
        weights=pd.Series(weights, index=keys),
        entity_effects=True,
    )
    if method == "classical":
        oracle = model.fit(
            cov_type="unadjusted", debiased=True, auto_df=False, count_effects=True
        )
    elif method == "heteroskedastic":
        oracle = model.fit(
            cov_type="robust", debiased=True, auto_df=False, count_effects=True
        )
    elif method.startswith("cluster"):
        oracle = model.fit(
            cov_type="clustered",
            cluster_entity=method != "cluster-time",
            cluster_time=method != "cluster-entity",
            debiased=True,
            auto_df=False,
            count_effects=True,
        )
    else:
        oracle = model.fit(
            cov_type="kernel" if method == "driscoll-kraay" else "ac",
            kernel="bartlett",
            bandwidth=2,
            debiased=True,
            auto_df=False,
            count_effects=True,
        )
    assert actual.valid
    np.testing.assert_allclose(
        actual.covariance.values, oracle.cov, rtol=1e-7, atol=1e-9
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("bandwidth", [0, 2])
def test_fama_macbeth_hac_matches_linearmodels(
    bandwidth: int, optional_reference: Callable[[str], ModuleType]
) -> None:
    """Coefficient-series covariance includes its declared finite-T correction."""
    optional_reference("linearmodels")
    pytest.importorskip("linearmodels", minversion="7.1")
    lm = optional_reference("linearmodels.panel")
    pd = optional_reference("pandas")
    rng = np.random.default_rng(126)
    e = np.repeat(np.arange(20), 10)
    t = np.tile(np.arange(10), 20)
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    y = rng.normal(size=200)
    fit = FamaMacBeth().estimate(
        PanelProblem(x, y, prepare_panel_index(e, t), effects="pooled")
    )
    actual = fama_macbeth_covariance(fit, bandwidth=bandwidth)
    keys = pd.MultiIndex.from_arrays([e, t])
    oracle = lm.FamaMacBeth(pd.Series(y, index=keys), pd.DataFrame(x, index=keys)).fit(
        cov_type="kernel", bandwidth=bandwidth, debiased=True
    )
    assert actual.valid
    np.testing.assert_allclose(
        actual.covariance.values, oracle.cov, rtol=1e-7, atol=1e-9
    )
