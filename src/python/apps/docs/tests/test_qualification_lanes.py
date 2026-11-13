"""Scoped statistical execution retains every exact selected parameter cell."""

import json
import xml.etree.ElementTree as ET

import pytest
from statadic_docs_labs import numerical_contracts
from statadic_docs_labs import qualification_lanes as lanes


def scope(identities):
    return {
        "fingerprint": "frozen-scope",
        "request": {"required_variants": list(identities)},
        "variants": {
            identity: {
                "id": identity,
                "study_mappings": [
                    {
                        "study": "exact-campaign",
                        "methods": ["actual-procedure"],
                        "scenarios": ["prespecified-null"],
                        "scope": "Exact frozen configuration, independently required.",
                    }
                ],
            }
            for identity in identities
        },
    }


def report(path, nodes, outcome=None):
    root = ET.Element("testsuite")
    for node in nodes:
        filename, name = node.split("::", 1)
        case = ET.SubElement(
            root,
            "testcase",
            classname=filename.removesuffix(".py").replace("/", "."),
            name=name,
        )
        if outcome is not None:
            ET.SubElement(case, outcome)
    ET.ElementTree(root).write(path)


def test_all_current_supplements_are_27_exact_cells_without_experimental_neighbors():
    nodes = lanes.selected_nodes(scope(lanes.SUPPLEMENTAL_NODES))
    assert len(nodes) == 27
    assert len(nodes) == len(set(nodes))
    assert sum("test_independent_aft_wald_qualification" in node for node in nodes) == 8
    assert (
        "tests/qualification/test_linear.py::test_gaussian_linear_native_inference[ols]"
        in nodes
    )
    assert (
        "tests/qualification/test_panels.py::test_dyadic_complete_node_bootstrap_coverage[n32]"
        in nodes
    )
    assert not any(
        excluded in node
        for node in nodes
        for excluded in (
            "test_returns",
            "fixed_gel",
            "gms",
            "wolak",
            "random-effects",
            "stationary_bootstrap",
        )
    )


def test_partial_aft_selection_does_not_inherit_other_families_or_entry_conventions():
    nodes = lanes.selected_nodes(
        scope(["survival.independent.weibull-delayed-observed-information-wald"])
    )
    assert nodes == [
        "tests/qualification/test_survival.py::test_independent_aft_wald_qualification[delayed-n128-weibull]",
        "tests/qualification/test_survival.py::test_independent_aft_wald_qualification[delayed-n512-weibull]",
    ]


@pytest.mark.parametrize(
    "malformation",
    ["missing", "empty-method", "empty-scenario", "empty-scope", "wrong-identity"],
)
def test_supplemental_test_cannot_replace_missing_canonical_campaign_cells(
    malformation,
):
    identity = "linear.ols-wls.ols"
    resolved = scope([identity])
    variant = resolved["variants"][identity]
    if malformation == "missing":
        variant["study_mappings"] = []
    elif malformation == "wrong-identity":
        variant["id"] = "different"
    else:
        field = {
            "empty-method": "methods",
            "empty-scenario": "scenarios",
            "empty-scope": "scope",
        }[malformation]
        variant["study_mappings"][0][field] = "" if field == "scope" else []
    with pytest.raises(ValueError, match="canonical"):
        lanes.selected_nodes(resolved)


def test_extra_or_duplicate_scope_variants_are_rejected():
    resolved = scope(["one"])
    resolved["request"]["required_variants"].append("one")
    with pytest.raises(ValueError, match="differ"):
        lanes.selected_nodes(resolved)
    resolved = scope(["one"])
    resolved["variants"]["extra"] = resolved["variants"]["one"]
    with pytest.raises(ValueError, match="differ"):
        lanes.selected_nodes(resolved)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_every_parameter_cell_must_execute_exactly_once(tmp_path, mutation):
    expected = [
        "tests/qualification/test_example.py::test_it[first]",
        "tests/qualification/test_example.py::test_it[second]",
    ]
    actual = list(expected)
    if mutation == "missing":
        actual.pop()
    elif mutation == "extra":
        actual.append("tests/qualification/test_example.py::test_unrelated")
    else:
        actual.append(expected[0])
    path = tmp_path / "report.xml"
    report(path, actual)
    with pytest.raises(ValueError, match="missing, extra or duplicated"):
        lanes.read_results(path, expected)


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_failed_or_unexecuted_cells_cannot_qualify(tmp_path, outcome):
    nodes = ["tests/qualification/test_example.py::test_it[first]"]
    path = tmp_path / "report.xml"
    report(path, nodes, outcome)
    with pytest.raises(ValueError, match="Incomplete"):
        lanes.read_results(path, nodes)


def prepare(tmp_path, monkeypatch):
    nodes = (
        "tests/qualification/test_example.py::test_it[first]",
        "tests/qualification/test_example.py::test_it[second]",
    )
    monkeypatch.setattr(lanes, "SUPPLEMENTAL_NODES", {"actual": nodes})
    package = tmp_path / "package"
    for filename in (*lanes._COMMON_SOURCES, "tests/qualification/test_example.py"):
        path = package / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {filename}\n")
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"candidate wheel bytes")
    resolved = scope(["actual"])
    report(tmp_path / "qualification-scoped.xml", nodes)
    return resolved, wheel, package


@pytest.mark.parametrize(
    "changed",
    [
        "mapping",
        "source",
        "support-source",
        "wheel",
        "scope",
        "report",
        "retained-status",
    ],
)
def test_current_identity_is_rederived_not_trusted_from_a_passing_record(
    tmp_path, monkeypatch, changed
):
    resolved, wheel, package = prepare(tmp_path, monkeypatch)
    expected = lanes.retain_results(
        tmp_path, scope=resolved, wheel=wheel, package=package
    )
    assert (
        lanes.verify_results(tmp_path, scope=resolved, wheel=wheel, package=package)
        == expected
    )
    if changed == "mapping":
        resolved["variants"]["actual"]["study_mappings"][0]["methods"] = ["different"]
    elif changed in ("source", "support-source"):
        filename = (
            "tests/qualification/test_example.py"
            if changed == "source"
            else "tests/_validation/calibration.py"
        )
        (package / filename).write_text("different source\n")
    elif changed == "wheel":
        wheel.write_bytes(b"different wheel")
    elif changed == "scope":
        resolved["fingerprint"] = "different-scope"
    elif changed == "report":
        path = tmp_path / "qualification-scoped.xml"
        path.write_text(path.read_text() + "\n")
    else:
        path = tmp_path / "qualification-scoped.json"
        altered = json.loads(path.read_text())
        altered["status"] = "claimed-passed"
        path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="evidence changed"):
        lanes.verify_results(tmp_path, scope=resolved, wheel=wheel, package=package)


def test_campaign_only_scope_is_explicit_without_running_bare_pytest(tmp_path):
    resolved = scope(["canonical.campaign-only"])
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"candidate wheel bytes")
    assert lanes.selected_nodes(resolved) == []
    expected = lanes.retain_results(
        tmp_path, scope=resolved, wheel=wheel, package=tmp_path
    )
    assert expected["supplemental_execution"] == "not-required-campaign-backed"
    assert expected["campaign_qualification"] == "independently-required"
    assert expected["expected_nodes"] == expected["executed_nodes"] == []
    assert expected["report_sha256"] is None
    assert (
        lanes.verify_results(tmp_path, scope=resolved, wheel=wheel, package=tmp_path)
        == expected
    )
    report(
        tmp_path / "qualification-scoped.xml",
        ["tests/qualification/test_unrelated.py::test_it"],
    )
    with pytest.raises(ValueError, match="Unexpected"):
        lanes.verify_results(tmp_path, scope=resolved, wheel=wheel, package=tmp_path)


def test_large_joint_gaussian_contract_still_has_no_bounded_numerical_adapter():
    assert "joint-censored-gaussian-accuracy" not in numerical_contracts.CONTRACT_FILES
    assert not any(
        "test_spatial_survival" in node
        for nodes in lanes.SUPPLEMENTAL_NODES.values()
        for node in nodes
    )
