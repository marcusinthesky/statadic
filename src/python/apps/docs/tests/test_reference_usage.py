"""Execute the authored Python Usage example on every reference page."""

import re
from pathlib import Path

import pytest

REFERENCE = Path(__file__).resolve().parents[3] / "packages/statadic/docs/reference"
SECTION = re.compile(r"^## Usage\s*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
CODE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def examples():
    """Collect each declared Usage section and require executable Python."""
    found = []
    for page in sorted(REFERENCE.glob("*.md")):
        section = SECTION.search(page.read_text())
        if section:
            code = CODE.search(section.group(1))
            if code is None:
                raise ValueError(f"No Python Usage example in {page}")
            found.append((page, code.group(1)))
    if not found:
        raise ValueError("No reference Usage examples found")
    return found


EXAMPLES = examples()


@pytest.mark.native
@pytest.mark.parametrize("page,code", EXAMPLES, ids=[page.stem for page, _ in EXAMPLES])
def test_reference_usage_runs(page, code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(page), "exec"), {"__name__": "__main__"})
