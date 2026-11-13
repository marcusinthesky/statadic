"""Every published notebook executes into a useful static native page."""

from pathlib import Path

import pytest
from statadic_docs_pages.marimo_pages import compose, execute

APP = Path(__file__).resolve().parents[1]
DOCS = APP.parents[1] / "packages/statadic/docs"
NOTEBOOKS = sorted(
    path
    for section in ("start", "tutorials", "guides")
    for path in (DOCS / section).rglob("*.py")
    if "__marimo__" not in path.parts and path.name != "conftest.py"
)
pytestmark = pytest.mark.notebook


@pytest.mark.parametrize(
    "notebook", NOTEBOOKS, ids=lambda p: p.relative_to(DOCS).as_posix()
)
def test_notebook_snapshot(notebook):
    session = execute(notebook, APP / "build/notebook-cache")
    markdown = compose(session)
    assert "# " in markdown
    assert "## Introduction" in markdown
    assert "{marimo-output}" in markdown
    assert "{marimo-launch}" in markdown
    assert "## Appendix" not in markdown
    assert "def test_" not in markdown
