"""Distribution qualification requires its packaged data-check driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def distribution() -> ModuleType:
    """Load the standalone distribution checker at its script boundary."""
    path = Path(__file__).parents[2] / "scripts" / "check_distribution.py"
    spec = importlib.util.spec_from_file_location("check_distribution", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_distribution_input_requires_a_file(
    tmp_path: Path, distribution: ModuleType, kind: str
) -> None:
    """A missing driver or directory reports the required relative path."""
    driver = tmp_path / "scripts" / "check_distribution_data.py"
    if kind == "directory":
        driver.mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"scripts/check_distribution_data\.py"):
        distribution._distribution_inputs(tmp_path)  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.contract
def test_distribution_input_returns_driver(
    tmp_path: Path, distribution: ModuleType
) -> None:
    """The driver is resolved relative to the extracted source root."""
    driver = tmp_path / "scripts" / "check_distribution_data.py"
    driver.parent.mkdir()
    driver.write_text("# distribution driver\n")
    assert distribution._distribution_inputs(tmp_path) == driver  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.contract
def test_standalone_development_retains_its_resolved_lock(
    tmp_path: Path, distribution: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An independently resolved sdist lock remains available for reproduction."""
    source = tmp_path / "source"
    evidence = tmp_path / "evidence"
    source.mkdir()
    evidence.mkdir()
    lock = b"version = 1\nrevision = 3\n"
    (source / "uv.lock").write_bytes(lock)

    def no_command(*_args: object) -> None:
        """Leave execution to the installed-distribution integration lane."""

    monkeypatch.setattr(distribution, "_run", no_command)
    distribution._check_development(  # noqa: SLF001
        "uv", source, {"STATADIC_DISTRIBUTION_EVIDENCE": str(evidence)}
    )
    assert (evidence / "standalone-uv.lock").read_bytes() == lock
