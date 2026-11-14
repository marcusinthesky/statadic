"""The teaching report is the registered campaign, not a second simulation."""

import pytest
from statadic_docs_labs import study_tutorial
from statadic_docs_labs.study_tutorial import run_study


def test_registered_campaign_preserves_counts_and_truth():
    report = run_study(20, 123)
    scenarios = report["summary"]["scenarios"]
    assert set(scenarios) == {
        "null-equal",
        "null-unequal",
        "alternative-equal",
        "alternative-unequal",
    }
    assert set(report["population_truth"]) == set(scenarios)
    for scenario, cell in scenarios.items():
        assert cell["attempted"] == 20 and cell["pending"] == 0
        assert report["population_truth"][scenario] == pytest.approx(
            0 if scenario.startswith("null") else 0.09
        )
        assert all(m["valid"] + m["failed"] == 20 for m in cell["methods"].values())
    assert report["manifest"]["payload"]["installation_scope"] == "imported-source-only"


def test_teaching_wrapper_uses_registered_manifest_summary_and_trial_truth(monkeypatch):
    manifest = {"registered": "manifest"}
    summary = {"registered": "summary"}
    visited = []

    def create(root, study, *, trials, seed, profile):
        assert study == "mmd.finite-bias"
        assert (trials, seed, profile) == (100, 17, "smoke")
        visited.append(("create", root))
        return manifest

    def execute(root):
        assert visited == [("create", root)]
        visited.append(("execute", root))

    def summarize(root):
        assert visited[-1] == ("execute", root)
        return summary

    monkeypatch.setattr(study_tutorial, "create", create)
    monkeypatch.setattr(study_tutorial, "execute", execute)
    monkeypatch.setattr(study_tutorial, "summarize", summarize)
    monkeypatch.setattr(
        study_tutorial,
        "verify",
        lambda root: [
            {
                "scenario": "known",
                "outcomes": {
                    "valid": {"valid": True, "truth": 0.25},
                    "failed": {"valid": False, "truth": 999},
                },
            }
        ],
    )
    report = run_study(100, 17)
    assert report["manifest"] is manifest
    assert report["summary"] is summary
    assert report["population_truth"] == {"known": 0.25}


@pytest.mark.parametrize("trials,seed", [(19, 1), (20, -1), (20, 10001), (True, 1)])
def test_rejects_out_of_scope_inputs(trials, seed):
    with pytest.raises(ValueError):
        run_study(trials, seed)
