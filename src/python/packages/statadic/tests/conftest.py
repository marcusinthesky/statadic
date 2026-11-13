# standalone package-maintenance entry point
"""STATADIC package setup: runtime annotations, x64 and optional reference data."""

from __future__ import annotations

import gc
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import install_import_hook

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
    )
    from types import ModuleType

if any(name == "statadic" or name.startswith("statadic.") for name in sys.modules):
    message = (
        "STATADIC was imported before its annotation hook; run its suite separately."
    )
    raise RuntimeError(message)

install_import_hook("statadic", "beartype.beartype")

import jax  # noqa: E402
import pytest  # noqa: E402
from hypothesis import settings as hypothesis_settings  # noqa: E402
from jax import config as jax_config  # noqa: E402

from ._budgets import register_profiles  # noqa: E402

pytest_plugins = ["statadic.testing.fixtures"]

_SCOPE_MARKERS = {"unit", "component", "integration", "e2e"}

if "JAX_ENABLE_X64" not in os.environ:
    jax_config.update("jax_enable_x64", val=True)
# Reuse compiled XLA executables across runs and worktrees; tracing still runs.
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    _cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    jax_config.update(
        "jax_compilation_cache_dir", str(_cache_home / "statadic/jax-tests")
    )
    jax_config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
# JAX compilation makes Hypothesis wall-clock deadlines measure cache state.
register_profiles()
hypothesis_settings.load_profile("statadic")


@pytest.fixture(scope="module", autouse=True)
def _release_compiled_memory() -> Generator[None]:
    """Bound per-worker JAX caches between modules when explicitly requested."""
    yield
    if os.environ.get("STATADIC_CI_MEMORY_BOUNDED") == "1":
        jax.clear_caches()
        gc.collect()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose the mandatory live-reference gate through native pytest."""
    parser.addoption(
        "--require-external-references",
        action="store_true",
        help="Require selected external comparisons, imports and completed assertions.",
    )


def _validate_marker_policy(items: list[pytest.Item]) -> None:
    """Validate declared test scope and structured marker requirements."""
    for item in items:
        scopes = [
            mark.name for mark in item.iter_markers() if mark.name in _SCOPE_MARKERS
        ]
        if len(scopes) != 1:
            detail = ", ".join(scopes) if scopes else "none"
            message = (
                f"{item.nodeid}: expected exactly one scope marker "
                f"(unit, component, integration, e2e); found {detail}"
            )
            raise pytest.UsageError(message)

        for marker_name in ("requires_optional", "requires_program"):
            for mark in item.iter_markers(marker_name):
                if (
                    len(mark.args) != 1
                    or mark.kwargs
                    or not isinstance(mark.args[0], str)
                    or not mark.args[0].strip()
                ):
                    message = (
                        f"{item.nodeid}: {marker_name} requires exactly one "
                        "nonblank string positional argument and no keywords"
                    )
                    raise pytest.UsageError(message)

        if item.get_closest_marker("external_reference"):
            if not item.get_closest_marker("reference"):
                message = f"{item.nodeid}: external_reference requires reference"
                raise pytest.UsageError(message)
            if not list(item.iter_markers("requires_optional")):
                message = (
                    f"{item.nodeid}: external_reference requires "
                    "requires_optional(name)"
                )
                raise pytest.UsageError(message)

        if item.get_closest_marker("qualification") and not item.get_closest_marker(
            "statistical"
        ):
            message = f"{item.nodeid}: qualification requires statistical"
            raise pytest.UsageError(message)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> Generator[None]:
    """Validate all nodes, then preflight selected external comparators."""
    _validate_marker_policy(items)
    yield
    for item in items:
        if item.get_closest_marker("reference"):
            item.user_properties.append(("reference_nodeid", item.nodeid))
            required = sorted(
                {str(mark.args[0]) for mark in item.iter_markers("requires_optional")}
            )
            item.user_properties.append(
                ("reference_dependencies", json.dumps(required))
            )
    if not config.getoption("--require-external-references"):
        return
    external = [item for item in items if item.get_closest_marker("external_reference")]
    if not external:
        message = "No external_reference tests were selected."
        raise pytest.UsageError(message)
    import_names = {
        "POT": "ot",
        "PyPortfolioOpt": "pypfopt",
        "scikit-learn": "sklearn",
        "scikit-bio": "skbio",
    }
    dependencies: set[str] = set()
    for item in external:
        marks = list(item.iter_markers("requires_optional"))
        if not marks:
            message = (
                f"{item.nodeid}: external_reference requires requires_optional(name)"
            )
            raise pytest.UsageError(message)
        required = sorted({str(mark.args[0]) for mark in marks})
        dependencies.update(required)
    for distribution in sorted(dependencies):
        module = import_names.get(distribution, distribution)
        try:
            import_module(module)
        except ImportError as error:
            message = (
                f"Required external comparator {distribution!r} cannot import: {error}"
            )
            raise pytest.UsageError(message) from error


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Reject unavailable required comparisons rather than reporting skip/xfail."""
    report = yield
    if (
        item.config.getoption("--require-external-references")
        and item.get_closest_marker("external_reference")
        and (report.skipped or hasattr(report, "wasxfail"))
    ):
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid}: required external comparison did not complete "
            f"({report.when}; skip/xfail is forbidden): {report.longrepr}"
        )
        if hasattr(report, "wasxfail"):
            del report.wasxfail
    return report


@pytest.fixture(scope="session")
def optional_reference(pytestconfig: pytest.Config) -> Callable[[str], ModuleType]:
    """Skip absent optional roots; broken installed packages always fail."""

    def load(module_name: str) -> ModuleType:
        try:
            return import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name.partition(".")[0]:
                raise
            message = f"optional reference dependency {module_name!r} is not installed"
            if pytestconfig.getoption("--require-external-references"):
                pytest.fail(message)
            pytest.skip(message)

    return load


@pytest.fixture(scope="session")
def reference_clouds() -> tuple[np.ndarray, np.ndarray]:
    """Return small non-degenerate clouds shared by external distance oracles."""
    return (
        np.asarray(
            [[-1.0, 0.5], [0.25, 1.5], [1.2, -0.7], [2.1, 0.1]],
            dtype=np.float64,
        ),
        np.asarray(
            [[-0.5, 0.2], [0.7, 1.3], [1.8, -1.1]],
            dtype=np.float64,
        ),
    )
