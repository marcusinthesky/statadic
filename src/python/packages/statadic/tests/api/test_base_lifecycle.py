"""The shared lifecycle supports non-regression input and context types."""

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field, replace
from typing import override

import jax.numpy as jnp
import pytest
import statadic
from statadic.api.base import InvalidFitError, Model, Result
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.reporting import Report


@dataclass(frozen=True)
class PreparedNumber:
    """Prepared scalar bound to its producing model."""

    value: int
    model_token: object = field(repr=False)


@dataclass(frozen=True)
class NumberResult(Result[int, str]):
    """Scalar result with a non-regression context."""

    @override
    def report(self) -> Report:
        return Report(title=self.context)


@dataclass(frozen=True)
class NumberModel(Model[int, PreparedNumber, NumberResult]):
    """A minimal model over scalar input."""

    @override
    def prepare(self, data: int) -> PreparedNumber:
        return PreparedNumber(data, self._token)

    @override
    def _estimate_prepared(self, prepared: PreparedNumber) -> NumberResult:
        return NumberResult(prepared.value * 2, "numbers")


@pytest.mark.component
@pytest.mark.contract
def test_typed_non_regression_lifecycle() -> None:
    """Fit and retained preparation accept family-specific data and context."""
    model = NumberModel()
    prepared = model.prepare(3)
    result = model.estimate_prepared(prepared)
    assert result.core == model.estimate(3).core
    assert result.report().title == "numbers"
    with pytest.raises(ValueError, match="another model"):
        NumberModel().estimate_prepared(prepared)


@pytest.mark.component
@pytest.mark.contract
def test_replaced_model_has_its_own_preparation() -> None:
    """Replacing a model creates a new owner even when its options match."""
    model = NumberModel()
    prepared = model.prepare(3)
    replacement = replace(model)
    with pytest.raises(ValueError, match=r"dataclasses.replace.*prepare"):
        replacement.estimate_prepared(prepared)
    assert model.estimate_prepared(prepared).core == model.estimate(3).core
    assert (
        replacement.estimate_prepared(replacement.prepare(3)).core
        == model.estimate(3).core
    )


@pytest.mark.component
@pytest.mark.contract
def test_require_valid_returns_result_or_explains_invalid_fit() -> None:
    """Reject a real rank-deficient fit using its retained diagnostics."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    data = Observations({"x": x, "y": x + 1})
    valid = Regression("y", Intercept() + Linear("x")).estimate(data)
    assert valid.require_valid() is valid
    invalid_data = Observations({"x": jnp.ones(4), "y": x + 1})
    invalid = Regression("y", Intercept() + Linear("x")).estimate(invalid_data)
    report = invalid.report()
    retained = invalid.core
    with pytest.raises(
        InvalidFitError, match=r"Regression.*solver_status.*identified"
    ) as error:
        invalid.require_valid()
    assert isinstance(error.value, ValueError)
    for warning in report.warnings:
        assert warning in str(error.value)
    for name in ("solver_status", "identified", "finite_optimum", "existence_status"):
        if name in report.metadata:
            assert f"{name}={report.metadata[name]}" in str(error.value)
    assert invalid.core is retained
    assert not invalid.valid


@pytest.mark.unit
@pytest.mark.contract
def test_estimate_is_the_only_estimation_method() -> None:
    """No public class keeps a ``fit`` method beside the canonical ``estimate``.

    A retained ``fit`` field or property names fitted state and remains allowed.
    """
    offenders = []
    for info in pkgutil.walk_packages(statadic.__path__, "statadic."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if not inspect.isclass(value):
                continue
            for verb in ("fit", "fit_prepared"):
                member = inspect.getattr_static(value, verb, None)
                if inspect.isfunction(member):
                    offenders.append(f"{info.name}.{name}.{verb}")
    assert offenders == []
