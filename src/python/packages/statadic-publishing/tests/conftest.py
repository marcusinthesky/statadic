"""Build one immutable example bundle; mutating tests use private copies."""

from pathlib import Path

import pytest
from statadic_publishing import build
from statadic_publishing.operations import _stage
from statadic_publishing.project import read_project

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ols-hc3"


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage the example, including unusual quoting for the rewriting contract."""
    source = tmp_path_factory.mktemp("project") / "source"
    _stage(EXAMPLE, source, read_project(EXAMPLE / "publishing.toml"))
    config = source / "publishing.toml"
    config.write_text(
        config.read_text()
        .replace('"assets/refs.bib"', "'assets/refs.bib'")
        .replace(
            'title = "A synthetic regression"',
            'title = "A synthetic \\"regression\\" assets/refs.bib"',
        )
    )
    return source


@pytest.fixture(scope="session")
def bundle(project: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Retain real analysis, offline TeX compilation and log verification."""
    return build(project, tmp_path_factory.mktemp("built") / "bundle").path
