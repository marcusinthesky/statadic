"""The Mantel workflow equals the native Mantel kernels for the same inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.association import (
    Mantel,
    MantelInput,
    MatrixExchangeability,
)
from statadic.api.comparison import PermutationPlan
from statadic.api.hypotheses import Hypothesis
from statadic.association.mantel import (
    mantel_correlation_kernel,
    partial_mantel_correlation_kernel,
)
from statadic.core.random import permutation_batch
from statadic.inference.mantel import mantel_permutation_kernel, mantel_permutation_key

if TYPE_CHECKING:
    from statadic.api.association import MantelResult, MantelTestEvidence
    from statadic.association.mantel import MantelMethod

_N = 9
_DRAWS = 99
_HALF = 0.5


def _mantel_evidence(fitted: MantelResult, plan: PermutationPlan) -> MantelTestEvidence:
    """Exercise the hypothesis path and retain the exact native schedule for checks."""
    outcome = fitted.infer(resampling=plan).test(Hypothesis(MatrixExchangeability()))
    assert outcome.evidence is not None
    return outcome.evidence


def _matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    points = rng.normal(size=(_N, 3))
    x = np.linalg.norm(points[:, None] - points[None], axis=-1)
    noise = rng.normal(size=(_N, _N)) * 0.3
    y = x + (noise + noise.T)
    np.fill_diagonal(y, 0.0)
    control = np.triu((rng.random((_N, _N)) < _HALF).astype(float), 1)
    return x, y, control + control.T


@pytest.mark.component
@pytest.mark.contract
def test_mantel_binding_defers_draws_and_partial_remains_descriptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound ordinary Mantel null never silently becomes partial Mantel."""
    x, y, control = _matrices()
    result = Mantel().estimate(MantelInput(x, y, control=control))
    bound = result.infer(resampling=PermutationPlan(n=3))
    assert bound.observed is result
    assert bound.report().metadata["test_status"] == "not requested"
    outcome = bound.test(Hypothesis(MatrixExchangeability()))
    assert outcome.construction["partial_correlation"] == "descriptive only"
    assert outcome.evidence is not None
    assert outcome.evidence.observed is result

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        message = "draws executed"
        raise AssertionError(message)

    monkeypatch.setattr("statadic.api.association.mantel_permutation_kernel", forbidden)
    assert result.infer(resampling=PermutationPlan(n=3)).observed is result


@pytest.mark.component
@pytest.mark.contract
def test_prepared_mantel_is_model_bound() -> None:
    """Prepared matrices reproduce the direct fit only on their owner."""
    x, y, control = _matrices()
    model = Mantel()
    prepared = model.prepare(MantelInput(x, y, control=control))
    expected = model.estimate(MantelInput(x=x, y=y, control=control))
    x[:] = 0
    y[:] = 0
    control[:] = 0
    result = model.estimate_prepared(prepared)
    np.testing.assert_array_equal(result.statistic, expected.statistic)
    np.testing.assert_array_equal(result.partial, expected.partial)
    np.testing.assert_array_equal(result.x, expected.x)
    np.testing.assert_array_equal(result.y, expected.y)
    with pytest.raises(ValueError, match="another model"):
        Mantel().estimate_prepared(prepared)


@pytest.mark.integration
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_random_plan_reproduces_native_mantel_kernel(method: MantelMethod) -> None:
    """Statistic, null draws and p-value equal the kernel on the native stream."""
    x, y, _ = _matrices()
    schedule = permutation_batch(mantel_permutation_key(5), jnp.arange(_N), _DRAWS)
    observed, null = mantel_permutation_kernel(
        jnp.asarray(x),
        jnp.asarray(y),
        schedule,
        method,
    )
    pvalue = float((1 + jnp.sum(null >= observed)) / (1 + _DRAWS))
    fit = Mantel(method).estimate(MantelInput(x, y))
    inference = _mantel_evidence(fit, PermutationPlan(n=_DRAWS, seed=5))
    assert float(fit.statistic) == pytest.approx(float(observed), rel=1e-12)
    np.testing.assert_array_equal(np.asarray(inference.null), np.asarray(null))
    assert inference.pvalue == pvalue
    assert inference.valid


@pytest.mark.integration
def test_supplied_schedule_equals_native_kernel_and_tail() -> None:
    """A supplied schedule is used verbatim with the add-one tail."""
    x, y, _ = _matrices()
    rng = np.random.default_rng(0)
    schedule = np.stack([rng.permutation(_N) for _ in range(_DRAWS)])
    ids = tuple(f"f{k}" for k in range(_N))
    inference = _mantel_evidence(
        Mantel().estimate(MantelInput(x, y, row_ids=ids)),
        PermutationPlan(n=_DRAWS, indices=schedule, row_ids=ids),
    )
    observed, null = mantel_permutation_kernel(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(schedule)
    )
    np.testing.assert_array_equal(np.asarray(inference.null), np.asarray(null))
    expected = float((1 + jnp.sum(null >= observed)) / (1 + _DRAWS))
    assert inference.pvalue == pytest.approx(expected, abs=0)
    (row,) = inference.report().tests
    assert row.name == "mantel"
    assert row.pvalue == pytest.approx(expected, abs=0)
    assert (row.law, row.law_parameters) == ("resampled", (_DRAWS,))
    assert row.statistic == pytest.approx(float(observed))
    assert all(
        metric.name not in {"pvalue", "correlation"}
        for metric in inference.report().metrics
    )


@pytest.mark.component
@pytest.mark.reference
def test_partial_equals_residual_correlation_and_is_pearson_only() -> None:
    """The partial kernel matches an explicit intercept-and-control residualization."""
    x, y, control = _matrices()
    a, b = np.triu_indices(_N, 1)
    design = np.column_stack([np.ones(len(a)), control[a, b]])

    def resid(v: np.ndarray) -> np.ndarray:
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    expected = np.corrcoef(resid(x[a, b]), resid(y[a, b]))[0, 1]
    fit = Mantel().estimate(MantelInput(x, y, control=control))
    assert fit.partial is not None
    assert float(fit.partial) == pytest.approx(expected, rel=1e-10)
    kernel = partial_mantel_correlation_kernel(
        jnp.asarray(x[a, b]), jnp.asarray(y[a, b]), jnp.asarray(control[a, b])
    )
    assert float(kernel) == pytest.approx(float(fit.partial), abs=0)
    plain = mantel_correlation_kernel(jnp.asarray(x[a, b]), jnp.asarray(y[a, b]))
    assert float(plain) == pytest.approx(float(fit.statistic), rel=1e-12)
    with pytest.raises(ValueError, match="Pearson-only"):
        Mantel("spearman").estimate(MantelInput(x, y, control=control))


@pytest.mark.component
@pytest.mark.contract
def test_rejects_asymmetric_input() -> None:
    """Invalid geometry fails at fit rather than yielding a statistic."""
    x, y, _ = _matrices()
    bad = x.copy()
    bad[0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        Mantel().estimate(MantelInput(bad, y))


@pytest.mark.component
@pytest.mark.contract
def test_generated_random_schedule_rejects_mismatched_row_ids() -> None:
    """Native random streams obey the retained observation ordering."""
    x, y, _ = _matrices()
    ids = tuple(f"f{k}" for k in range(_N))
    fit = Mantel().estimate(MantelInput(x, y, row_ids=ids))
    with pytest.raises(ValueError, match="row IDs"):
        _mantel_evidence(fit, PermutationPlan(n=5, seed=3, row_ids=ids[::-1]))


@pytest.mark.component
@pytest.mark.contract
def test_undefined_correlation_reports_are_unavailable() -> None:
    """Constant strict triangles retain undefined observed and null evidence."""
    _, y, control = _matrices()
    fit = Mantel().estimate(MantelInput(np.zeros_like(y), y, control=control))
    observed = fit.report()
    for metric in observed.metrics:
        assert metric.value is None
        assert not metric.available
        assert metric.unavailable_reason
    inference = _mantel_evidence(fit, PermutationPlan(n=5, seed=3))
    assert not inference.valid
    assert inference.pvalue is None
    report = inference.report()
    (row,) = report.tests
    assert not row.available
    assert row.statistic is None
    assert row.pvalue is None
    assert row.unavailable_reason
    partial = report.metrics[0]
    assert partial.name == "partial_correlation"
    assert partial.value is None
    assert not partial.available
    assert partial.unavailable_reason
