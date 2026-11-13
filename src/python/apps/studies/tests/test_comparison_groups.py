"""Comparisons require a frozen common-estimand contract."""

import copy

import pytest
from statadic_studies import analysis
from statadic_studies.contracts import STUDIES
from statadic_studies.study_types import validate_definition


def test_only_frozen_groups_produce_pairs_even_when_null_truths_coincide(monkeypatch):
    study = STUDIES["mmd.kernel-estimation"].record()
    study["scenarios"] = ("null-equal",)
    outcomes = {
        method: {"valid": True, "estimate": 0.0, "truth": 0.0}
        for method in study["methods"]
    }
    monkeypatch.setattr(
        analysis,
        "manifest",
        lambda _: {
            "id": "frozen",
            "payload": {"study": study, "profile": "smoke", "trials": 1},
        },
    )
    monkeypatch.setattr(
        analysis, "verify", lambda _: [{"scenario": "null-equal", "outcomes": outcomes}]
    )
    report = analysis.summarize(None)
    paired = report["scenarios"]["null-equal"]["paired"]
    assert report["analysis_version"] == 2
    assert paired["RBF-U-minus-RBF-V"]["comparison_group"] == "RBF-mmd-squared"
    assert "RBF-U-minus-RBF-root-V" not in paired
    assert "RBF-U-minus-linear-U" not in paired
    study.pop("comparison_groups")
    assert analysis.summarize(None)["scenarios"]["null-equal"]["paired"] == {}


@pytest.mark.parametrize(
    "groups",
    [
        (("same", ("U", "missing")),),
        (("same", ("U", "V")), ("same", ("U", "V"))),
        (("first", ("U", "V")), ("second", ("U", "V"))),
        (("empty", ("U",)),),
        (("bad-container", "UV"),),
    ],
)
def test_invalid_comparison_groups_rejected(groups):
    study = copy.deepcopy(STUDIES["mmd.finite-bias"].record())
    study["comparison_groups"] = groups
    with pytest.raises(ValueError, match="comparison groups"):
        validate_definition(study)


def test_older_publication_reports_analysis_version_change(tmp_path, monkeypatch):
    import json

    (tmp_path / "evidence-manifest.json").write_text(json.dumps({"files": {}}))
    (tmp_path / "summary.json").write_text(json.dumps({"campaign": "old"}))
    monkeypatch.setattr(analysis, "verify_inventory", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="older analysis version"):
        analysis.verify_export(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == {"campaign": "old"}
