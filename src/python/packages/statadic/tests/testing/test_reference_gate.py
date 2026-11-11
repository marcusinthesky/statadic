"""Retained reference evidence must establish completed comparator assertions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def references() -> ModuleType:
    """Load the standalone runner using the same boundary as the audit tests."""
    path = Path(__file__).parents[2] / "scripts" / "check_references.py"
    spec = importlib.util.spec_from_file_location("check_references", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.parametrize("outcome", ["", "<failure/>", "<error/>", "<skipped/>"])
def test_reference_evidence_requires_completed_comparisons(
    tmp_path: Path,
    outcome: str,
    references: ModuleType,
) -> None:
    """A report may pass only when its actual selected comparisons passed."""
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="reference" name="fit">'
        '<properties><property name="reference_dependencies" '
        'value=\'["spreg"]\'/><property name="reference_nodeid" '
        'value="tests/reference.py::test_fit[one]"/></properties>'
        + outcome
        + "</testcase></testsuite></testsuites>"
    )
    if outcome:
        with pytest.raises(ValueError, match="did not all complete"):
            references.read_results(report)
    else:
        result = references.read_results(report, ("spreg",))
        assert result["tests"] == 1
        assert result["comparators"] == {"spreg": 1}
        assert result["nodes"] == ["tests/reference.py::test_fit[one]"]
        assert result["node_comparators"] == {
            "tests/reference.py::test_fit[one]": ["spreg"]
        }
        with pytest.raises(ValueError, match="lifelines"):
            references.read_results(report, ("spreg", "lifelines"))


@pytest.mark.integration
@pytest.mark.regression
def test_empty_reference_report_cannot_pass(
    tmp_path: Path,
    references: ModuleType,
) -> None:
    """Zero selected comparisons is not successful validation."""
    report = tmp_path / "report.xml"
    report.write_text("<testsuites/>")
    with pytest.raises(ValueError, match="did not all complete"):
        references.read_results(report)


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.parametrize(
    "metadata", ["", '<property name="reference_nodeid" value="fit"/>']
)
def test_reference_nodes_require_exact_pytest_metadata(
    tmp_path: Path, references: ModuleType, metadata: str
) -> None:
    """JUnit classname reconstruction cannot establish exact parametrized identity."""
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testcase classname="tests.reference" name="test_fit[one]">'
        f"<properties>{metadata}</properties></testcase></testsuites>"
    )
    with pytest.raises(ValueError, match="exact reference_nodeid"):
        references.read_results(report)


def _mapping() -> dict[str, object]:
    return {
        "python_symbols": ["statadic.estimation.linear.LeastSquares"],
        "construction_paths": ["statadic.estimation.linear.LeastSquares"],
        "nodeids": ["tests/reference.py::test_fit[one]"],
        "oracle_kind": "external",
        "distribution": "spreg",
        "call": "spreg.OLS",
        "quantities": ["parameters"],
        "conventions": "Same retained rows",
        "limits": "Numerical comparison only",
        "lanes": ["external-float64"],
    }


@pytest.mark.integration
@pytest.mark.regression
def test_retained_report_receipt_rejects_missing_and_skipped_nodes(
    tmp_path: Path, references: ModuleType
) -> None:
    """Installed-wheel reconciliation uses exact assertions and raw-byte identity."""
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(
        json.dumps({"methods": [{"id": "linear", "reference_mappings": [_mapping()]}]})
    )
    report = tmp_path / "report.xml"
    complete = (
        '<testsuites><testcase name="fit"><properties>'
        '<property name="reference_dependencies" value=\'["spreg"]\'/>'
        '<property name="reference_nodeid" value="tests/reference.py::test_fit[one]"/>'
        "</properties></testcase></testsuites>"
    )
    report.write_text(complete)
    receipt = references.reference_receipt(catalogue, report, "external-float64")
    assert receipt["mapped_nodes"] == ["tests/reference.py::test_fit[one]"]
    report.write_text(complete.replace("fit[one]", "fit[two]"))
    with pytest.raises(ValueError, match="did not execute"):
        references.reference_receipt(catalogue, report, "external-float64")
    report.write_text(complete.replace("</testcase>", "<skipped/></testcase>"))
    with pytest.raises(ValueError, match="did not all complete"):
        references.reference_receipt(catalogue, report, "external-float64")


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.parametrize(
    ("node", "comparator", "error"),
    [
        ("tests/reference.py::test_fit[two]", "spreg", "did not execute"),
        ("tests/reference.py::test_fit[one]", "statsmodels", "comparator mismatch"),
        ("tests/reference.py::test_fit[one]", "spreg", None),
    ],
)
def test_catalogue_reconciliation_requires_exact_node_and_comparator(
    references: ModuleType, node: str, comparator: str, error: str | None
) -> None:
    """A sibling parameter or unrelated library cannot satisfy a declaration."""
    claim = {
        "owner_kind": "scientific-variant",
        "owner_id": "linear.ols",
        "mapping": _mapping(),
    }
    results = {
        "nodes": [node, "tests/unmapped.py::test_other"],
        "node_comparators": {
            node: [comparator],
            "tests/unmapped.py::test_other": ["scipy"],
        },
    }
    if error is not None:
        with pytest.raises(ValueError, match=error):
            references.reconcile_reference_mappings(
                [claim], "external-float64", results
            )
    else:
        report = references.reconcile_reference_mappings(
            [claim], "external-float64", results
        )
        assert report["mapped_nodes"] == [node]
        assert report["unmapped_executed_nodes"] == ["tests/unmapped.py::test_other"]
        assert (
            references.reconcile_reference_mappings(
                [claim], "precision-float32", results
            )["executed_mappings"]
            == []
        )


@pytest.mark.integration
@pytest.mark.regression
def test_method_mapping_does_not_inherit_scientific_variant_coverage(
    tmp_path: Path, references: ModuleType
) -> None:
    """Method declarations never inherit coverage to absent variant declarations."""
    path = tmp_path / "catalogue.json"
    mapping = _mapping()
    mapping["configuration"] = "One finite numerical problem"
    path.write_text(
        json.dumps(
            {
                "methods": [{"id": "linear", "reference_mappings": [mapping]}],
                "scientific_procedures": [
                    {"variants": [{"id": "linear.ols", "reference_mappings": []}]}
                ],
            }
        )
    )
    claims = references.load_reference_mappings(path)
    assert [claim["owner_id"] for claim in claims] == ["linear"]
    assert claims[0]["owner_kind"] == "method"


@pytest.mark.integration
@pytest.mark.regression
def test_duplicate_exact_reference_nodes_cannot_pass(
    tmp_path: Path, references: ModuleType
) -> None:
    """Repeated XML records cannot masquerade as distinct completed comparisons."""
    case = (
        '<testcase name="test_fit"><properties><property name="reference_nodeid" '
        'value="tests/reference.py::test_fit"/></properties></testcase>'
    )
    report = tmp_path / "report.xml"
    report.write_text(f"<testsuites>{case}{case}</testsuites>")
    with pytest.raises(ValueError, match="Duplicate reference node"):
        references.read_results(report)
