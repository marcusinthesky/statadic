# standalone package-maintenance entry point
"""Public namespace imports isolate unrelated scientific capabilities."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.contract
def test_root_and_ground_imports_do_not_load_downstream_families() -> None:
    """Ordinary explicit child imports preserve the dependency direction."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import types
import statadic
import statadic.api
assert statadic.__version__
assert not {'jax', 'numpy', 'scipy', 'polars'} & sys.modules.keys()
from statadic.geometry import ground
assert callable(ground.cdist)
assert 'statadic.geometry.barycentre' not in sys.modules
assert 'statadic.inference' not in sys.modules
from statadic.discrepancy import mmd
assert isinstance(mmd, types.ModuleType)
assert callable(mmd.mmd)
""",
        ],
        check=True,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_removed_alpha_regression_names_are_not_importable() -> None:
    """Superseded distribution-per-class and composer aliases stay deleted."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import statadic.api.regression as regression
import statadic.api.ordination as ordination
for name in (
    "PoissonRegression",
    "LogisticRegression",
    "ProbitRegression",
    "GammaRegression",
    "NegativeBinomialRegression",
    "HurdleRegression",
    "PreparedHurdle",
    "conditional",
    "OLS",
    "WLS",
    "Conditional",
    "IVRegression",
):
    assert not hasattr(regression, name), name
assert callable(regression.Regression)
assert callable(ordination.Ordination)
assert not hasattr(ordination, "PCA")
assert not hasattr(ordination, "PCoA")
try:
    import statadic.api.composition
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("Retired Equation module remains importable")
""",
        ],
        check=True,
    )
