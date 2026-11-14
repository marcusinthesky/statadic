"""Diagnostic study contracts execute the named procedure and retain decisions."""

import pytest
from statadic_studies.adapters import run_trial
from statadic_studies.contracts import STUDIES
from statadic_studies.definitions.diagnostics import DIAGNOSTIC_METHODS


@pytest.mark.parametrize("name", DIAGNOSTIC_METHODS)
@pytest.mark.parametrize("scenario", ["gaussian-null", "declared-alternative"])
def test_native_diagnostic_trial_matches_its_declared_decision(name, scenario):
    study = f"linear.diagnostic-{name}"
    definition = STUDIES[study]
    assert definition.null_scenarios == ("gaussian-null",)
    assert definition.decision_only_methods == ("test",)
    result = run_trial(study, scenario, 0, 20261109)["test"]
    assert result["valid"]
    assert 0 <= result["pvalue"] <= 1
    assert result["rejected"] == (result["pvalue"] < definition.alpha)
