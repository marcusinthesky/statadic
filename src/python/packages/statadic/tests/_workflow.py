"""The host-workflow packages that together form the public workflow layer."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["WORKFLOW_PACKAGES", "walk_workflow_modules"]

WORKFLOW_PACKAGES = (
    "model",
    "observation",
    "report",
    "prediction",
    "ordination",
    "hypothesis",
    "estimand",
    "regression",
    "structure",
)


def walk_workflow_modules() -> list[ModuleType]:
    """Import and return every module under the workflow packages."""
    modules: list[ModuleType] = []
    for name in WORKFLOW_PACKAGES:
        package = importlib.import_module(f"statadic.{name}")
        modules.append(package)
        modules.extend(
            importlib.import_module(info.name)
            for info in pkgutil.walk_packages(package.__path__, f"statadic.{name}.")
        )
    return modules
