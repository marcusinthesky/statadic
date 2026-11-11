"""Executed wheel lanes must cover the exact declared numerical witnesses."""

import json

import pytest
from statadic_docs_labs import numerical_contracts as contracts


def test_class_organized_tests_count_as_their_exact_module(tmp_path):
    report = tmp_path / "results.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.energy.test_weights.TestSolver" '
        'name="test_kkt"/></testsuite>'
    )
    assert contracts.read_results(report, ["tests/energy/test_weights.py"])
    with pytest.raises(ValueError, match="was not executed"):
        contracts.read_results(report, ["tests/test_weights.py"])


@pytest.mark.parametrize("outcome", ["<skipped/>", "<failure/>", "<error/>"])
def test_unexecuted_numerical_witnesses_cannot_qualify(tmp_path, outcome):
    report = tmp_path / "results.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.test_example" name="test_it">'
        f"{outcome}</testcase></testsuite>"
    )
    with pytest.raises(ValueError, match="Incomplete numerical"):
        contracts.read_results(report, ["tests/test_example.py"])


def test_test_source_changes_invalidate_retained_passing_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(contracts, "CONTRACT_FILES", {"example": ("test_example.py",)})
    package = tmp_path / "package"
    (package / "tests").mkdir(parents=True)
    source = package / "tests/test_example.py"
    source.write_text("independent control\n")
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"candidate bytes")
    (tmp_path / "numerical-contracts.xml").write_text(
        '<testsuite><testcase classname="tests.test_example" '
        'name="test_independent"/></testsuite>'
    )
    scope = {"request": {"numerical_contracts": ["example"]}, "fingerprint": "frozen"}
    expected = contracts.retain_results(
        tmp_path, scope=scope, wheel=wheel, package=package
    )
    assert (
        contracts.verify_results(tmp_path, scope=scope, wheel=wheel, package=package)
        == expected
    )
    source.write_text("changed independent control\n")
    assert (
        json.loads((tmp_path / "numerical-contracts.json").read_text())["example"][
            "status"
        ]
        == "passed"
    )
    with pytest.raises(ValueError, match="Numerical evidence changed"):
        contracts.verify_results(tmp_path, scope=scope, wheel=wheel, package=package)
