"""Named order keys bind post-estimation calculations to retained source rows."""

from __future__ import annotations

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import BreuschGodfrey, DurbinWatson
from statadic.api.hypotheses import TestOutcome
from statadic.api.inference import HAC
from statadic.api.regression import LinearResult, Regression
from statadic.inference.linear_covariance import hac_covariance
from statadic.inference.linear_diagnostics import breusch_godfrey, durbin_watson

pytestmark = pytest.mark.integration


def _identity(rows: tuple[int | str, ...] | list[int | str]) -> dict[str, object]:
    """Independently re-hash ordered row ids as reports promise."""
    payload = json.dumps(list(rows), separators=(",", ":"), ensure_ascii=False)
    return {"count": len(rows), "sha256": hashlib.sha256(payload.encode()).hexdigest()}


def _columns() -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([2, 0, 5, 1, 8, 3, 7, 4, 11, 6, 10, 9], dtype=np.float32),
        "y": np.asarray([2, 0, 4, 1, 3, 7, 4, 8, 9, 6, 12, 9], dtype=np.float32),
        "time": np.asarray([7, 2, 11, 0, 5, 4, 1, 10, 3, 9, 6, 8], dtype=np.float32),
    }


@pytest.mark.contract
@pytest.mark.parametrize("arrow", [False, True])
def test_ordering_uses_owned_deferred_columns_and_preserves_fit(*, arrow: bool) -> None:
    """Ordering metadata and arithmetic agree after caller-owned buffers mutate."""
    columns = _columns()
    order = jnp.asarray(np.argsort(columns["time"]))
    source = pytest.importorskip("pyarrow").table(columns) if arrow else columns
    fitted = Regression.ols("y", Intercept() + Linear("x")).estimate(source)
    core, rows = fitted.core, fitted.context.row_ids
    columns["time"][:] = np.nan
    outcomes = fitted.diagnose(
        BreuschGodfrey(2, order_by="time"), DurbinWatson(order_by="time")
    )
    expected = (
        breusch_godfrey(core, 2, row_order=order),
        durbin_watson(core, row_order=order),
    )
    for outcome, native in zip(outcomes, expected, strict=True):
        assert isinstance(outcome, TestOutcome)
        assert outcome.statistic == pytest.approx(float(native.statistic), rel=1e-5)
        assert outcome.construction["rows"] == _identity(rows)
        assert outcome.construction["calculation_order"] == _identity(
            [rows[int(index)] for index in order]
        )
        assert outcome.construction["order_by"] == "time"
    inferred = fitted.infer(covariance=HAC(2, order_by="time"))
    assert inferred.uncertainty.fit is core
    np.testing.assert_allclose(
        inferred.uncertainty.covariance.values,
        hac_covariance(core, 2, row_order=order).covariance.values,
    )
    assert inferred.report().metadata["calculation_order"] == _identity(
        [rows[int(index)] for index in order]
    )
    assert fitted.core is core
    assert fitted.context.row_ids == rows


@pytest.mark.contract
@pytest.mark.parametrize(
    "keys", [[0.0] * 12, [True] * 12, [1j] * 12, [float("nan")] * 12, [None] * 12]
)
def test_ordering_rejects_unsupported_or_ambiguous_deferred_keys(
    keys: list[object],
) -> None:
    """An unused bad key permits fitting but never silently defines lag adjacency."""
    fitted = Regression.ols("y", Intercept() + Linear("x")).estimate(
        {**_columns(), "bad": keys}
    )
    with pytest.raises((TypeError, ValueError)):
        fitted.diagnose(DurbinWatson(order_by="bad"))
    with pytest.raises((TypeError, ValueError)):
        fitted.infer(covariance=HAC(1, order_by="bad"))
    assert fitted.diagnose(DurbinWatson(order_by=None))["durbin_watson"].available


@pytest.mark.contract
def test_ordering_preserves_selected_observation_to_source_mapping() -> None:
    """A subset resolves only its selected numeric source values in its own order."""
    original = Regression.ols("y", Intercept() + Linear("x")).prepare(_columns())
    selected = original.context.observations.take((9, 3, 7, 0, 5, 2, 1, 10))
    fitted = Regression.ols("y", Intercept() + Linear("x")).estimate(selected)
    assert isinstance(fitted, LinearResult)
    keys = fitted.context.observations.resolve_numeric("time")
    expected = tuple(selected.row_ids[int(index)] for index in jnp.argsort(keys))
    outcome = fitted.diagnose(DurbinWatson(order_by="time"))["durbin_watson"]
    assert isinstance(outcome, TestOutcome)
    assert outcome.construction["rows"] == _identity(selected.row_ids)
    assert outcome.construction["calculation_order"] == _identity(expected)
    with pytest.raises(KeyError):
        fitted.diagnose(DurbinWatson(order_by="missing"))
