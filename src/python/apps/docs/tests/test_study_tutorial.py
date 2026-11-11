"""The teaching report is the registered campaign, not a second simulation."""

import pytest
from statadic_docs_labs.study_tutorial import run_study
from statadic_studies.analysis import summarize
from statadic_studies.campaign import create, execute


def test_reuses_campaign_and_preserves_counts_and_truth(tmp_path):
    root = tmp_path / "campaign"
    manifest = create(root, "mmd.finite-bias", trials=20, seed=123, profile="smoke")
    execute(root)
    expected = summarize(root)
    report = run_study(20, 123)
    assert report["manifest"] == manifest
    assert report["summary"] == expected
    assert set(report["population_truth"]) == set(expected["scenarios"])
    for scenario, cell in expected["scenarios"].items():
        assert cell["attempted"] == 20 and cell["pending"] == 0
        assert report["population_truth"][scenario] == pytest.approx(
            0 if scenario.startswith("null") else 0.09
        )
        assert all(m["valid"] + m["failed"] == 20 for m in cell["methods"].values())
    assert report["manifest"]["payload"]["installation_scope"] == "imported-source-only"


def test_seed_reproducibility_and_variation():
    first = run_study(20, 11)
    assert first == run_study(20, 11)
    other = run_study(20, 12)
    assert first["summary"]["scenarios"] != other["summary"]["scenarios"]


@pytest.mark.parametrize("trials,seed", [(19, 1), (20, -1), (20, 10001), (True, 1)])
def test_rejects_out_of_scope_inputs(trials, seed):
    with pytest.raises(ValueError):
        run_study(trials, seed)
