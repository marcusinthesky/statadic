"""Independent inventory fixtures catch omitted interfaces and unresolved ownership."""

import json

import pytest
from statadic_studies.catalogue import check_catalogue


def fixture_repository(root):
    registry = root / "src/lean/registry"
    registry.mkdir(parents=True)
    (registry / "methods.json").write_text(json.dumps({"methods": [{"id": "linear"}]}))
    (registry / "public-api.json").write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "name": "statadic.estimation.linear",
                        "method_id": "linear",
                        "exports": [
                            {
                                "name": "Estimator",
                                "target": "statadic.estimation.linear.Estimator",
                                "kind": "class",
                            }
                        ],
                    }
                ]
            }
        )
    )
    performance = root / "src/python/apps/performance/src/statadic_performance"
    performance.mkdir(parents=True)
    (performance / "identities.json").write_text(
        json.dumps({"ols": {"smoke": ["full-fit"], "full": ["full-fit"]}})
    )
    page = root / "src/python/packages/statadic/docs/reference/performance-workloads.md"
    page.parent.mkdir(parents=True)
    page.write_text("(performance-ols-full-fit)=\n")
    app = root / "src/python/apps/studies"
    app.mkdir(parents=True)
    path = app / "catalogue.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "public_symbols": {"statadic.estimation.linear.Estimator": "linear"},
                "methods": [
                    {
                        "id": "linear",
                        "family": "linear",
                        "python_symbols": ["statadic.estimation.linear.Estimator"],
                        "studies": [],
                        "benchmarks": ["ols/full-fit"],
                        "tutorials": [],
                        "rationale": "Scientific ownership unresolved",
                        "disposition": "ownership-review",
                        "citation_review": "source-unresolved",
                        "citation_keys": [],
                    }
                ],
            }
        )
    )
    return path


def test_ownership_review_cannot_disappear_from_coverage(tmp_path):
    fixture_repository(tmp_path)
    report = check_catalogue(tmp_path)
    assert report["ownership_review_pending"] == ["linear"]
    assert report["coverage_status"] == "incomplete"


def test_changed_public_inventory_is_rejected(tmp_path):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    data["public_symbols"].clear()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="public inventory"):
        check_catalogue(tmp_path)


def scientific_fixture(root):
    path = fixture_repository(root)
    data = json.loads(path.read_text())
    data["methods"][0]["disposition"] = "scientific-owner"
    data["methods"][0]["studies"] = ["linear.ols-covariance"]
    data["scientific_procedures"] = [
        {
            "id": "linear-point",
            "family": "linear",
            "rationale": "Concrete point estimator",
            "python_symbols": ["statadic.estimation.linear.Estimator"],
            "variants": [
                {
                    "id": "linear-point.ols",
                    "title": "OLS",
                    "selectors": {"weights": "unit"},
                    "python_symbols": ["statadic.estimation.linear.Estimator"],
                    "adapter_status": "uncovered",
                    "source_review": "source-unresolved",
                    "study_mappings": [],
                }
            ],
        }
    ]
    path.write_text(json.dumps(data))
    return path, data


def test_module_study_link_cannot_cover_unmapped_variant(tmp_path):
    scientific_fixture(tmp_path)
    report = check_catalogue(tmp_path)
    assert report["scientific_procedures"] == 1
    assert report["scientific_variants"] == 1
    assert report["missing_study"] == ["linear-point.ols"]
    assert report["source_review_pending"] == ["linear-point.ols"]
    assert report["required_studies"] == []


def test_explicit_mapping_reports_adapter_coverage_without_source_approval(tmp_path):
    path, data = scientific_fixture(tmp_path)
    variant = data["scientific_procedures"][0]["variants"][0]
    variant["adapter_status"] = "covered"
    variant["study_mappings"] = [
        {
            "study": "linear.ols-covariance",
            "methods": ["classical"],
            "scenarios": ["homoskedastic-n64-t"],
            "scope": "Only this fixed-design cell",
        }
    ]
    path.write_text(json.dumps(data))
    report = check_catalogue(tmp_path)
    assert not report["missing_study"]
    assert report["source_review_pending"] == ["linear-point.ols"]
    assert report["coverage_status"] == "incomplete"
    assert report["required_studies"] == ["linear.ols-covariance"]
    assert report["execution_status"] == "not-inferred-from-registration"


def test_unknown_study_method_cannot_claim_variant_coverage(tmp_path):
    path, data = scientific_fixture(tmp_path)
    variant = data["scientific_procedures"][0]["variants"][0]
    variant["adapter_status"] = "covered"
    variant["study_mappings"] = [
        {
            "study": "linear.ols-covariance",
            "methods": ["HC99"],
            "scenarios": ["homoskedastic-n64-t"],
            "scope": "Invalid",
        }
    ]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="method/scenario"):
        check_catalogue(tmp_path)


def test_verified_method_source_requires_primary_review_evidence(tmp_path):
    path, data = scientific_fixture(tmp_path)
    data["scientific_procedures"][0]["variants"][0]["source_review"] = (
        "verified-method-source"
    )
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="lacks evidence"):
        check_catalogue(tmp_path)


def test_duplicate_scientific_variant_is_rejected(tmp_path):
    path, data = scientific_fixture(tmp_path)
    variants = data["scientific_procedures"][0]["variants"]
    variants.append(variants[0].copy())
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate scientific variant"):
        check_catalogue(tmp_path)


def test_unknown_and_duplicate_links_are_rejected(tmp_path):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    data["methods"][0]["benchmarks"] = ["ols/missing"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="benchmarks"):
        check_catalogue(tmp_path)
    data["methods"][0]["benchmarks"] = ["ols/full-fit", "ols/full-fit"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="benchmarks"):
        check_catalogue(tmp_path)


def test_missing_tutorial_and_benchmark_anchor_are_rejected(tmp_path):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    data["tutorials"] = {"missing": "tutorials/missing.py"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing or invalid tutorial notebook"):
        check_catalogue(tmp_path)
    data["tutorials"] = {}
    path.write_text(json.dumps(data))
    page = (
        tmp_path
        / "src/python/packages/statadic/docs/reference/performance-workloads.md"
    )
    page.write_text("# Missing anchor")
    with pytest.raises(ValueError, match="Missing benchmark anchor"):
        check_catalogue(tmp_path)


def test_removing_last_benchmark_association_is_rejected(tmp_path):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    data["methods"][0]["benchmarks"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unlinked benchmarks: ols/full-fit"):
        check_catalogue(tmp_path)


def test_valid_unlinked_tutorial_is_rejected(tmp_path):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    data["tutorials"] = {"example": "tutorials/example.py"}
    path.write_text(json.dumps(data))
    notebook = tmp_path / "src/python/packages/statadic/docs/tutorials/example.py"
    notebook.parent.mkdir(parents=True)
    notebook.write_text("# Notebook")
    with pytest.raises(ValueError, match="Unlinked tutorials: example"):
        check_catalogue(tmp_path)
    data["methods"][0]["tutorials"] = ["example"]
    path.write_text(json.dumps(data))
    check_catalogue(tmp_path)
    data["methods"][0]["tutorials"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unlinked tutorials: example"):
        check_catalogue(tmp_path)


@pytest.mark.parametrize(
    "relative", ["../../outside.py", "/outside.py", "tutorials/example.md"]
)
def test_tutorial_paths_reject_escape_and_non_notebook_sources(tmp_path, relative):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    data["tutorials"] = {"example": relative}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing or invalid tutorial notebook"):
        check_catalogue(tmp_path)


def reference_mapping(root):
    tests = root / "src/python/packages/statadic/tests"
    tests.mkdir(parents=True)
    (tests / "test_reference.py").write_text(
        "import pytest\n@pytest.mark.reference\ndef test_fit(case):\n    pass\n"
    )
    return {
        "python_symbols": ["statadic.estimation.linear.Estimator"],
        "construction_paths": ["statadic.estimation.linear.Estimator"],
        "nodeids": ["tests/test_reference.py::test_fit[ordinary]"],
        "oracle_kind": "external",
        "distribution": "statsmodels",
        "call": "statsmodels.api.OLS.fit",
        "quantities": ["parameters"],
        "conventions": "Same fixed full-rank design",
        "limits": "Numerical agreement only",
        "lanes": ["external-float64"],
        "configuration": "One fixed finite OLS problem",
    }


def test_method_reference_does_not_cover_scientific_variant(tmp_path):
    path, data = scientific_fixture(tmp_path)
    data["methods"][0]["reference_mappings"] = [reference_mapping(tmp_path)]
    path.write_text(json.dumps(data))
    index = check_catalogue(tmp_path)["reference_index"]
    assert index["mapped_method_records"] == 1
    assert index["mapped_scientific_variants"] == 0
    assert index["unmapped_scientific_variants"] == ["linear-point.ols"]
    assert index["mapped_reference_definitions"] == 1
    assert index["execution_status"] == "not-inferred-from-mappings"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("nodeids", ["tests/missing.py::test_fit"], "Missing reference test file"),
        (
            "nodeids",
            ["tests/test_reference.py::test_missing"],
            "Missing reference test definition",
        ),
        ("nodeids", ["tests/test_reference.py::test_fit[*]"], "exact reference nodeid"),
        (
            "construction_paths",
            ["statadic.missing.Estimator"],
            "Unknown reference construction",
        ),
        ("python_symbols", ["statadic.missing.Estimator"], "owner symbols"),
        ("lanes", [], "oracle/lanes"),
        ("lanes", ["made-up-lane"], "oracle/lanes"),
        ("configuration", "", "numerical reference configuration"),
        ("status", "passed", "Unknown reference mapping fields"),
    ],
)
def test_reference_declarations_fail_closed(tmp_path, field, value, error):
    path = fixture_repository(tmp_path)
    data = json.loads(path.read_text())
    mapping = reference_mapping(tmp_path)
    mapping[field] = value
    data["methods"][0]["reference_mappings"] = [mapping]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        check_catalogue(tmp_path)


def test_unmapped_reference_definitions_remain_visible(tmp_path):
    fixture_repository(tmp_path)
    reference_mapping(tmp_path)
    index = check_catalogue(tmp_path)["reference_index"]
    assert index["unmapped_reference_definitions"] == [
        "tests/test_reference.py::test_fit"
    ]
    assert index["unmapped_method_records"] == ["linear"]


def test_independent_formula_mapping_without_external_lane_is_not_execution(tmp_path):
    path, data = scientific_fixture(tmp_path)
    mapping = reference_mapping(tmp_path)
    mapping["oracle_kind"] = "independent-formula"
    mapping["lanes"] = []
    data["scientific_procedures"][0]["variants"][0]["reference_mappings"] = [mapping]
    path.write_text(json.dumps(data))
    index = check_catalogue(tmp_path)["reference_index"]
    assert index["mapped_scientific_variants"] == 1
    assert index["execution_status"] == "not-inferred-from-mappings"
