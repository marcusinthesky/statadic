"""Adapted consumer fixtures preserve numerical intent and wheel/source isolation."""

import ast
import json
import runpy
import sys
from pathlib import Path

import jax
import pytest
from statadic_docs_labs import consumer_acceptance as acceptance
from statadic_docs_labs import consumer_fixtures as fixtures


def test_migration_changes_only_the_selected_call_verb(monkeypatch):
    monkeypatch.setattr(acceptance, "_FUNCTIONS", {"numerical_qualification"})
    source = """
from __future__ import annotations
from pipeline.io.semantic_identity import entrypoint_code_identity
SCALE = 0.25
def numerical_qualification(n):
    return Estimator().fit(n * SCALE)
def ignored_benchmark():
    raise AssertionError('must not run')
"""
    adapted, lines = acceptance.adapt_numerical_source(source)

    class Estimator:
        def estimate(self, value):
            return value + 3

    namespace = {"Estimator": Estimator}
    exec(compile(adapted, "adapted-consumer", "exec"), namespace)  # noqa: S102 -- inspected fixture
    assert namespace["numerical_qualification"](64) == 19
    assert lines == [6]
    tree = ast.parse(adapted)
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pipeline")
        for node in ast.walk(tree)
    )
    assert "ignored_benchmark" not in namespace


def test_missing_frozen_witness_cannot_be_treated_as_acceptance():
    with pytest.raises(ValueError, match="fixture functions changed"):
        acceptance.adapt_numerical_source("def numerical_qualification(n): return {}")


def test_covariance_adaptation_uses_estimand_without_rewriting_innovations():
    import numpy as np
    from statadic.api.estimand import Covariance, Estimand
    from statadic.estimation.covariance import LedoitWolf

    source = """
from statadic.api.covariance import LedoitWolf
class SpatialCovarianceEstimate:
    pass
def spatial_ledoit_wolf_covariance(values):
    innovations = values - values.mean(axis=0)
    return LedoitWolf().fit(innovations)
def _symmetrize(values):
    return (values + values.T) / 2
def _spatialize_covariance(system, innovation):
    return np.linalg.solve(system, innovation)
"""
    adapted, lines = acceptance.adapt_spatial_covariance_source(source)
    namespace = {}
    exec(compile(adapted, "adapted-covariance", "exec"), namespace)  # noqa: S102 -- inspected fixture
    values = np.array([[1.0, 4.0], [2.0, -1.0], [4.0, 0.5], [8.0, 3.0]])
    with jax.enable_x64(new_val=True):
        result = namespace["spatial_ledoit_wolf_covariance"](values)
        expected = Estimand(Covariance(), estimator=LedoitWolf()).estimate(
            values - values.mean(axis=0)
        )
    np.testing.assert_array_equal(result.sigma, expected.sigma)
    assert result.shrinkage == expected.shrinkage
    assert lines == [7]
    assert "from statadic.api.covariance import LedoitWolf" not in adapted
    assert (
        "Estimand(Covariance(), estimator=LedoitWolf()).estimate(innovations)"
        in adapted
    )
    assert "innovations = values - values.mean(axis=0)" in adapted


def test_adapter_refuses_writes_inside_consumer_repository(tmp_path):
    with pytest.raises(ValueError, match="outside consumer application"):
        acceptance.run_acceptance(
            tmp_path, tmp_path / "output", python=Path("unused"), wheel=Path("unused")
        )


def test_worker_rejects_changed_source_before_importing_any_consumer(
    tmp_path, monkeypatch
):
    worker = tmp_path / "consumer_fixtures.py"
    worker.write_text("changed")
    (tmp_path / "request.json").write_text(
        json.dumps(
            {"schema_version": 2, "files": {worker.name: "not-the-current-digest"}}
        )
    )
    monkeypatch.setattr(fixtures, "__file__", str(worker))
    with pytest.raises(ValueError, match="Acceptance source changed"):
        fixtures.main(tmp_path / "candidate.whl")


@pytest.mark.parametrize(
    "fixture", [fixtures._dyadic_fixture, fixtures._frontier_fixture]
)
def test_domain_fixtures_use_current_native_contracts(fixture):
    import numpy as np

    def compare(actual, expected, *, label):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-8, atol=1e-8, err_msg=label
        )
        return {
            "max_absolute_error": float(np.max(np.abs(np.asarray(actual) - expected)))
        }

    with jax.enable_x64(new_val=True):
        result = fixture(compare)
    assert result["status"] == "passed"


def test_current_verifier_does_not_reinterpret_a_historical_acceptance_schema(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "passed"})
    )
    with pytest.raises(ValueError, match="Incomplete consumer"):
        acceptance.verify_acceptance(tmp_path, wheel=tmp_path / "unused.whl")


def test_legacy_operational_names_cannot_supply_current_consumer_identity(tmp_path):
    request = {
        "schema_version": 2,
        "pricing_revision": acceptance.CONSUMER_REVISION,
        "pricing_source": acceptance._QUALIFICATION,
        "sample_size": 64,
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "results.json").write_text(json.dumps({"schema_version": 2}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "passed",
                "files": acceptance.inventory(tmp_path),
            }
        )
    )
    with pytest.raises(ValueError, match="revision or design changed"):
        acceptance.verify_acceptance(tmp_path, wheel=tmp_path / "unused.whl")


def test_release_cli_exposes_the_neutral_consumer_option(monkeypatch, capsys):
    main = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/release.py")
    )["main"]
    monkeypatch.setattr(sys, "argv", ["release.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--consumer-repo" in help_text
    assert "--pricing-repo" not in help_text
    monkeypatch.setattr(sys, "argv", ["release.py", "--pricing-repo", "/unused"])
    with pytest.raises(SystemExit) as rejected:
        main()
    assert rejected.value.code == 2
