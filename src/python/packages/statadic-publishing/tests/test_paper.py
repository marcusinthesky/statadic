"""Exercise the authored example and compare retained estimates with an oracle."""

import json
from pathlib import Path

import jax
import numpy as np
import polars as pl
import pytest
from statadic.api.artifacts import load_analysis
from statadic_publishing.operations import _BIB_KEY
from statadic_publishing.project import read_project

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ols-hc3"
LEAN_CLAIMS = Path(__file__).resolve().parents[4] / "lean/registry/claims.json"


def test_example_claims_resolve_in_repository_registries() -> None:
    """Example claims cite catalogued sources and existing Lean registry claims."""
    spec = read_project(EXAMPLE / "publishing.toml")
    bibliography = (EXAMPLE / spec.manuscript.bibliography.path).read_text()
    keys = set(_BIB_KEY.findall(bibliography))
    registry = {claim["id"] for claim in json.loads(LEAN_CLAIMS.read_text())["claims"]}
    for claim in spec.claims:
        assert set(claim.citations) <= keys, claim.id
        assert set(claim.registry_claims) <= registry, claim.id


@pytest.mark.requires_optional("statsmodels")
def test_paper_analysis_and_manuscript(bundle: Path) -> None:
    import statsmodels.api as sm

    spec = read_project(EXAMPLE / "publishing.toml")
    (data,) = [EXAMPLE / item.path for item in spec.inputs if item.path is not None]
    generated = bundle / "generated"
    assert {
        p.relative_to(generated).as_posix() for p in generated.rglob("*") if p.is_file()
    } == {out.path for out in spec.outputs}
    report = load_analysis(generated / "inference-snapshot").report
    frame = pl.read_csv(data)
    # The report retains normal-reference intervals, so the oracle uses z, not t.
    reference = sm.OLS(
        frame["y"].to_numpy(), sm.add_constant(frame["x"].to_numpy())
    ).fit(cov_type="HC3", use_t=False)
    tolerance = 2e-5 if not jax.config.x64_enabled else 1e-10
    interval = reference.conf_int()
    for field, expected in (
        ("estimate", reference.params),
        ("std_error", reference.bse),
        ("ci_lower", interval[:, 0]),
        ("ci_upper", interval[:, 1]),
    ):
        np.testing.assert_allclose(
            [getattr(row, field) for row in report.parameters],
            expected,
            rtol=tolerance,
            atol=tolerance,
            err_msg=field,
        )
    assert len(report.plots) == 1
    assert (bundle / "manuscript/main.pdf").is_file()
