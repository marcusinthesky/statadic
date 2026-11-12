# standalone package-maintenance entry point
"""Native static panels against actual linearmodels public estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.panel import (
    Between,
    FamaMacBeth,
    FamaMacBethFit,
    FirstDifference,
    PanelProblem,
    Pooled,
    RandomEffects,
    Within,
)
from statadic.inference.panel_covariance import (
    fama_macbeth_covariance,
    panel_covariance,
)
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.operators.panel import prepare_panel_index

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from jax import Array
    from statadic.estimation.panel import (
        PanelFit,
    )

_MISSING_PERIOD = 2


def _expected_df(
    method: str, x: np.ndarray, entity: np.ndarray, time: np.ndarray
) -> int:
    """Count retained sample/effect rank without consulting a fitted object."""
    n, width = x.shape
    groups, periods = len(np.unique(entity)), len(np.unique(time))
    if method == "fmb":
        return periods - 1
    if method == "between":
        return groups - width
    if method == "fd":
        keys = set(zip(entity.tolist(), time.tolist(), strict=True))
        consecutive = sum((group, period - 1) in keys for group, period in keys)
        return consecutive - (width - 1)
    if method == "fe":
        return n - width - (groups - 1)
    if method == "two-way":
        dummy = np.column_stack([np.eye(groups)[entity], np.eye(periods)[time]])
        return n - width - (int(np.linalg.matrix_rank(dummy)) - 1)
    return n - width


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize(
    "method", ["pooled", "fe", "two-way", "re", "between", "fd", "fmb"]
)
@pytest.mark.parametrize("weighted", [False, True])
@examples(6)
@example(seed=126)
@example(seed=12000)
@given(seed=st.integers(0, 2**32 - 1))
def test_panel_coefficients_match_linearmodels(  # noqa: PLR0914, PLR0915
    optional_reference: Callable[[str], ModuleType],
    seed: int,
    *,
    method: str,
    weighted: bool,
) -> None:  # independent native/oracle state
    """Match complete aligned unbalanced samples and explicit constant conventions."""
    optional_reference("linearmodels")
    pytest.importorskip("linearmodels", minversion="7.1")
    lm = optional_reference("linearmodels.panel")
    pd = optional_reference("pandas")
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(16), 8)
    time = np.tile(np.arange(8), 16)
    keep = ~((entity % 3 == 0) & (time == _MISSING_PERIOD))
    entity, time = entity[keep], time[keep]
    x = np.column_stack([np.ones(len(entity)), rng.normal(size=(len(entity), 2))])
    y = (
        x @ np.array([0.3, -0.5, 0.8])
        + rng.normal(size=16)[entity]
        + rng.normal(size=len(entity)) * 0.4
    )
    w = rng.uniform(0.5, 2.0, len(entity)) if weighted else np.ones(len(entity))
    index = prepare_panel_index(entity, time)
    mi = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])
    df = pd.DataFrame(x, index=mi).sort_index(level="time")
    series = pd.Series(y, index=mi).sort_index(level="time")
    weight_series = pd.Series(w, index=mi).sort_index(level="time")
    if method == "pooled":
        native = Pooled().estimate(PanelProblem(x, y, index, w, effects="pooled"))
        oracle = lm.PooledOLS(series, df, weights=weight_series).fit()
    elif method in ("fe", "two-way"):
        native = Within().estimate(
            PanelProblem(
                x,
                y,
                index,
                w,
                effects="two-way" if method == "two-way" else "unit",
                intercept_column=0,
            )
        )
        oracle = lm.PanelOLS(
            series,
            df,
            weights=weight_series,
            entity_effects=True,
            time_effects=method == "two-way",
        ).fit()
    elif method == "re":
        native = RandomEffects().estimate(
            PanelProblem(x, y, index, w, effects="random", intercept_column=0)
        )
        oracle = lm.RandomEffects(series, df, weights=weight_series).fit()
    elif method == "between":
        native = Between().estimate(PanelProblem(x, y, index, w, effects="pooled"))
        oracle = lm.BetweenOLS(series, df, weights=weight_series).fit()
    elif method == "fd":
        native = FirstDifference().estimate(
            PanelProblem(x[:, 1:], y, index, w, effects="pooled")
        )
        oracle = lm.FirstDifferenceOLS(
            series, df.iloc[:, 1:], weights=weight_series
        ).fit()
    else:
        native = FamaMacBeth().estimate(PanelProblem(x, y, index, w, effects="pooled"))
        oracle = lm.FamaMacBeth(series, df, weights=weight_series).fit()
    assert native.valid
    np.testing.assert_allclose(native.parameters, oracle.params, rtol=1e-7, atol=1e-8)
    if isinstance(native, FamaMacBethFit):
        uncertainty = fama_macbeth_covariance(native)
    else:
        uncertainty = panel_covariance(native)
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, oracle.cov, rtol=2e-7, atol=1e-9
    )
    expected_df = _expected_df(method, x, entity, time)
    assert native.df_resid == expected_df
    assert native.df_resid == oracle.df_resid
    intervals = coefficient_intervals(
        uncertainty, reference="t", degrees_of_freedom=native.df_resid
    )
    assert intervals.valid
    np.testing.assert_allclose(
        intervals.estimate.ci, np.asarray(oracle.conf_int()).T, rtol=2e-7, atol=1e-8
    )
    restriction = np.zeros((1, len(oracle.params)))
    restriction[0, -1] = 1
    test = linear_restriction_test(uncertainty, restriction, 0.0, reference="chi2")
    oracle_test = oracle.wald_test(restriction)
    np.testing.assert_allclose(
        test.test.statistic, oracle_test.stat, rtol=2e-7, atol=1e-8
    )
    np.testing.assert_allclose(
        test.test.pvalue, oracle_test.pval, rtol=2e-7, atol=1e-10
    )
    assert test.numerator_df == oracle_test.df


@pytest.mark.component
@pytest.mark.contract
@examples(8)
@example(seed=17)
@given(seed=st.integers(0, 2**32 - 1))
def test_panel_fixed_effects_jit_vmap_and_absorbed_failure(seed: int) -> None:
    """Fixed-shape effects remain transformable and fail closed on absorption."""
    entity = np.repeat(np.arange(4), 5)
    time = np.tile(np.arange(5), 4)
    index = prepare_panel_index(entity, time)
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(20, 2)))
    y = (
        x @ jnp.array([0.5, -0.3])
        + jnp.asarray(entity)
        + jnp.asarray(rng.normal(size=20)) * 0.1
    )

    def evaluate_fit(p: PanelProblem) -> PanelFit:
        return Within().estimate(p)

    evaluate_fit = jax.jit(evaluate_fit)(PanelProblem(x, y, index, effects="unit"))
    assert evaluate_fit.valid

    def evaluate_stack_leaves(*leaves: jax.Array) -> Array:
        return jnp.stack(leaves)

    stacked = jax.tree.map(
        evaluate_stack_leaves,
        PanelProblem(x, y, index, effects="unit"),
        PanelProblem(x, y * 2, index, effects="unit"),
    )

    def evaluate_fit_3(p: PanelProblem) -> PanelFit:
        return Within().estimate(p)

    batch = jax.jit(jax.vmap(evaluate_fit_3))(stacked)
    np.testing.assert_allclose(batch.parameters[1], evaluate_fit.parameters * 2)
    absorbed = Within().estimate(
        PanelProblem(
            jnp.column_stack([x, jnp.asarray(entity)]), y, index, effects="unit"
        )
    )
    assert not absorbed.valid
