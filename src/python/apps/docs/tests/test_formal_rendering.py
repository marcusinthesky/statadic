"""The formal bridge adds scoped links at rendering time only."""

from types import SimpleNamespace

from statadic_docs_labs.formal_extension import api_evidence, claims_by_symbol


def test_family_catalogue_is_projected_from_tracked_data():
    from statadic_docs_labs import formal_extension

    dependencies = []
    app = SimpleNamespace(
        env=SimpleNamespace(note_dependency=dependencies.append),
        _statadic_formal={"claims": []},
    )
    page = ["<!-- statadic:family-sources -->"]
    formal_extension.source(app, "formal/family-sources", page)
    assert "| Registry family |" in page[0]
    assert "STATADIC-ENERGY-005" in page[0]
    assert "statadic:family-sources" not in page[0]
    assert any(path.endswith("data/family-sources.json") for path in dependencies)


def test_known_limitations_projection_preserves_table_cells(tmp_path, monkeypatch):
    import json

    from statadic_docs_labs import formal_extension

    path = tmp_path / "src/python/apps/docs/evidence/known-limitations.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "title": "Observed limits",
                        "columns": ["Method", "Limit"],
                        "rows": [["A", r"x\|y"]],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(formal_extension, "ROOT", tmp_path)
    dependencies = []
    app = SimpleNamespace(
        env=SimpleNamespace(note_dependency=dependencies.append),
        _statadic_formal={"claims": []},
    )
    page = ["<!-- statadic:known-limitations -->"]
    formal_extension.source(app, "validation/known-limitations", page)
    assert page[0].startswith("## Observed limits\n\n")
    assert r"| A | x\|y |" in page[0]
    assert str(path) in dependencies


def test_sourced_claim_preserves_independent_pending_claim_review():
    import json

    from statadic_docs_labs import formal_extension

    claims = json.loads(
        (formal_extension.ROOT / "src/lean/build/certification.json").read_text()
    )["claims"]
    claim = next(claim for claim in claims if claim.get("source"))
    claim = claim | {"review": "human semantic review pending"}
    rendered = formal_extension.claim_page(claim)
    assert "Claim review: human semantic review pending" in rendered
    assert f"Source correspondence: {claim['source']['review']}" in rendered


def test_registered_catalogue_resolves_against_public_inventory(monkeypatch):
    """The authored catalogue cannot retain renamed or removed Python targets."""
    from statadic_docs_labs import formal_extension

    dependencies = []
    catalogue = formal_extension.ROOT / "src/python/apps/studies/catalogue.json"
    app = SimpleNamespace(
        config=SimpleNamespace(
            statadic_study_exports=[], statadic_research_catalogue=str(catalogue)
        ),
        env=SimpleNamespace(note_dependency=dependencies.append),
    )
    # Certification is checked by the formal bridge; this test isolates symbol links.
    monkeypatch.setattr(
        formal_extension.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    formal_extension.load(app)
    assert app._statadic_research
    assert str(catalogue) in dependencies


def test_explicit_aliases_members_and_unregistered_symbols():
    claim = {
        "id": "STATADIC-TEST-001",
        "title": "Exact identity",
        "proof_status": "kernel-checked-and-axiom-audited",
        "scope": {"relationship": "Exact arithmetic only"},
        "python": {
            "symbols": ["statadic.owner.f", "statadic.alias.f", "statadic.C.fit"]
        },
    }
    app = SimpleNamespace(_statadic_claims_by_symbol=claims_by_symbol([claim]))
    original = "A docstring."
    obj = SimpleNamespace(__doc__=original)
    for symbol in claim["python"]["symbols"]:
        lines = [original]
        api_evidence(app, "function", symbol, obj, None, lines)
        assert "formal-STATADIC-TEST-001" in "\n".join(lines)
        assert "Exact arithmetic only" in "\n".join(lines)
        assert "do not alone verify" in "\n".join(lines)
        assert obj.__doc__ == original
    lines = [original]
    api_evidence(app, "function", "statadic.unregistered", obj, None, lines)
    assert lines == [original]


def test_generated_manuscript_macros_compile_offline(tmp_path, monkeypatch):
    """Exercise stable macros with certified native statement/proof exposition."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(str(root / "src/lean/tools"))
    import bridge

    data = json.loads((root / "src/lean/build/certification.json").read_text())
    bridge.manuscript_fragments(data["claims"], tmp_path)
    identifier = data["claims"][0]["id"]
    source = tmp_path / "manuscript.tex"
    source.write_text(
        r"\documentclass[11pt]{article}"
        r"\usepackage[a4paper,margin=30mm]{geometry}\usepackage[T1]{fontenc}"
        r"\usepackage{lmodern,amsmath,amssymb,amsthm}"
        r"\newtheorem{theorem}{Theorem}\input{statadic-mathematics.tex}"
        r"\begin{document}\begin{theorem}"
        + rf"\statadicstatement{{{identifier}}}"
        + r"\end{theorem}\begin{proof}"
        + rf"\statadicproof{{{identifier}}}"
        + r"\end{proof}\end{document}"
    )
    subprocess.run(
        [
            "tectonic",
            "--untrusted",
            "--only-cached",
            "--bundle",
            str(root / "src/latex/bundle/tlextras-2022.0r0-statadic.zip"),
            "--outdir",
            str(tmp_path),
            str(source),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "manuscript.pdf").is_file()
    assert sys.modules["bridge"].__file__.startswith(str(root))


def test_catalogue_links_target_notebooks_and_performance():
    from statadic_docs_labs.formal_extension import source

    method = {
        "id": "example",
        "title": "Example",
        "tutorials": ["regression"],
        "benchmarks": ["ols/full-fit"],
        "disposition": "ownership-review",
        "citation_review": "pending",
        "rationale": "Review",
        "studies": [],
    }
    app = SimpleNamespace(
        _statadic_tutorials={"regression": "tutorials/ols/fit.py"},
        _statadic_research=[method],
        _statadic_procedures=[],
        _statadic_formal={"claims": []},
    )
    catalogue = ["<!-- statadic:research-catalogue -->"]
    source(app, "validation/research", catalogue)
    assert "../tutorials/ols/fit.py" in catalogue[0]
    assert "performance-ols-full-fit" in catalogue[0]
    performance = ["<!-- statadic:performance-catalogue -->"]
    source(app, "reference/performance-workloads", performance)
    assert "research-method-example" in performance[0]
    assert "performance-ols-full-fit" in performance[0]


def test_index_distinguishes_closed_and_open_obligations_in_one_family():
    from statadic_docs_labs.formal_extension import index_page

    data = {
        "claims": [],
        "open_obligation_ids": ["spatial-001"],
        "methods": [
            {
                "title": "Spatial",
                "obligations": [
                    {
                        "id": "spatial-001",
                        "title": "Identification",
                        "related_claims": ["STATADIC-IV-001"],
                    },
                    {
                        "id": "spatial-002",
                        "title": "Jacobian",
                        "claims": ["STATADIC-SPATIAL-001"],
                    },
                ],
            }
        ],
    }
    page = index_page(data)
    assert (
        "| Spatial | Identification | Open · related results "
        "[STATADIC-IV-001](claims/STATADIC-IV-001.md) |" in page
    )
    assert (
        "| Spatial | Jacobian | "
        "[STATADIC-SPATIAL-001](claims/STATADIC-SPATIAL-001.md) |" in page
    )


def test_release_selection_does_not_inherit_historical_evidence():
    from statadic_docs_labs.formal_extension import source

    app = SimpleNamespace(
        config=SimpleNamespace(statadic_release_evidence=True),
        _statadic_evaluations=[],
        _statadic_research=[],
        _statadic_procedures=[],
        _statadic_formal={"claims": []},
    )
    text = ["<!-- statadic:research-catalogue -->"]
    source(app, "validation/research", text)
    assert "No study exports were selected" in text[0]
    assert "no current-release Monte Carlo qualification evidence" in text[0]


def test_selected_reports_keep_profile_and_installation_boundaries():
    from statadic_docs_labs.formal_extension import source

    evaluation = {
        "frozen": {
            "id": "frozen-id",
            "payload": {
                "study": {"id": "example"},
                "profile": "smoke",
                "installation_scope": "imported-source-only",
            },
        },
        "report": {"scenarios": {}},
    }
    app = SimpleNamespace(
        config=SimpleNamespace(statadic_release_evidence=False),
        _statadic_evaluations=[evaluation],
        _statadic_research=[],
        _statadic_procedures=[],
        _statadic_formal={"claims": []},
    )
    historical = ["<!-- statadic:research-catalogue -->"]
    source(app, "validation/research", historical)
    assert "Historical evaluations" in historical[0]
    assert "smoke" in historical[0] and "imported-source-only" in historical[0]
    assert "no retained operating-characteristic bounds" in historical[0]
    app.config.statadic_release_evidence = True
    selected = ["<!-- statadic:research-catalogue -->"]
    source(app, "validation/research", selected)
    assert "Selected release evaluations" in selected[0]
    assert "selection alone does not qualify" in selected[0]


def test_reference_index_keeps_numerical_and_scientific_scope_separate():
    """A numerical witness and a declared test confer no inherited qualification."""
    from statadic_docs_labs.formal_extension import source

    mapping = {
        "oracle_kind": "external",
        "distribution": "statsmodels",
        "call": "statsmodels.api.OLS.fit",
        "construction_paths": ["statadic.api.regression.Regression.ols"],
        "nodeids": ["tests/example.py::test_ols[HC3]"],
        "quantities": ["HC3 covariance"],
        "conventions": "Normal reference; identical inputs.",
        "limits": "Fixed fixture only; no size guarantee.",
    }
    app = SimpleNamespace(
        _statadic_research=[
            {
                "id": "geometry.embedding",
                "title": "Classical scaling",
                "family": "embedding",
                "reference_mappings": [
                    mapping
                    | {
                        "construction_paths": ["statadic.api.ordination.Ordination"],
                        "configuration": "Euclidean distances, two dimensions.",
                    }
                ],
            }
        ],
        _statadic_procedures=[
            {
                "id": "linear-point",
                "title": "Linear models",
                "family": "linear",
                "variants": [
                    {
                        "id": "linear-point.ols",
                        "title": "OLS",
                        "selectors": {"weights": "unit"},
                        "reference_mappings": [mapping],
                    },
                    {"id": "linear-point.wls", "title": "WLS"},
                ],
            }
        ],
        _statadic_formal={"claims": []},
    )
    text = ["<!-- statadic:reference-index -->"]
    source(app, "validation/references", text)
    rendered = text[0]
    assert "1 / 2 scientific configurations" in rendered
    assert "1 implementation owners have separate mappings" in rendered
    assert "Unmapped scientific configurations" in rendered
    assert "`linear-point.wls`" in rendered
    assert "Normal reference; identical inputs." in rendered
    assert "no size guarantee" in rendered
    assert "Declared; execution requires retained evidence" in rendered
    assert "../downloads/reference-tests/tests/example.py" in rendered
    assert "Euclidean distances, two dimensions." in rendered
    assert "qualified" not in rendered


def test_constructor_links_require_the_exercised_construction_path():
    """Convenient spelling cannot broaden a comparison to another construction."""
    owner = {
        "id": "linear-point.ols",
        "title": "OLS",
        "python_symbols": [],
        "reference_mappings": [
            {"construction_paths": ["statadic.api.regression.Regression.ols"]}
        ],
    }
    app = SimpleNamespace(
        _statadic_research=[owner],
        _statadic_procedures=[],
        _statadic_claims_by_symbol={},
    )
    lines = []
    api_evidence(
        app, "method", "statadic.api.regression.Regression.ols", None, None, lines
    )
    assert "reference-linear-point.ols" in "\n".join(lines)
    assert "declared comparison" in "\n".join(lines)
    lines = []
    api_evidence(
        app, "method", "statadic.api.regression.Regression.wls", None, None, lines
    )
    assert lines == []
