"""Mark canonical documentation notebook tests as end-to-end cases."""

from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[4] / "packages/statadic/docs"
SECTIONS = tuple(DOCS / name for name in ("start", "tutorials", "guides"))


def pytest_collection_modifyitems(items):
    """Apply only to tests collected from the canonical notebook directories."""
    for item in items:
        path = Path(str(item.path)).resolve()
        if path.suffix == ".py" and any(
            path.is_relative_to(section) for section in SECTIONS
        ):
            item.add_marker(pytest.mark.e2e)
