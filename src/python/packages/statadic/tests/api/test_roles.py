"""Shared input roles preserve declarations across designs and models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from statadic.api.roles import merge_input_roles

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.contract
def test_merge_roles_rejects_conflicting_consumers() -> None:
    """Compatible consumers share a column; incompatible meanings fail."""
    assert merge_input_roles(
        {"x": "numeric"}, {"x": "numeric", "g": "categorical"}
    ) == {
        "x": "numeric",
        "g": "categorical",
    }
    with pytest.raises(ValueError, match="conflicting input roles"):
        merge_input_roles({"x": "numeric"}, {"x": "measure"})


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.typing
def test_model_roles_and_retained_columns_are_typed(tmp_path: Path) -> None:
    """Consumers see declared roles and normalized arrays without casts."""
    check_types(
        """
from jax import Array
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression, Observed, Poisson
from statadic.api.roles import InputRoleProvider, merge_input_roles

def consume(provider: InputRoleProvider) -> None: ...

terms = Intercept() + Linear("x")
consume(terms)
consume(Regression('y', terms))
consume(Regression(Observed('y'), terms, exposure='exposure', law=Poisson()))
consume(Regression(
    'y', terms + Linear('endog'),
    endogenous=('endog',), instruments=('instrument',),
))
observations = Observations({"x": [1.0, 2.0]})
array: Array = observations.columns["x"]
bad: list[float] = observations.columns["x"]  # E: not assignable
merge_input_roles({"x": "unknown"})  # E: not assignable
""",
        "host_roles_fixture.py",
        tmp_path,
    )
