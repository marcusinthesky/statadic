"""Composable statistical computation with explicit mathematical contracts.

Start with geometry.ground for point comparisons, discrepancy for measure
comparisons, and association for group/association statistics. Model defines
statistical specifications; estimation retains fitted state; inference derives
uncertainty, tests and resampling results. Operators and optimize supply shared
numerical operations; core supplies semantic and result contracts.

Import the owning family explicitly, for example
``from statadic.estimation.linear import LeastSquares``. Compound namespaces do not load
unrelated families. See the package docs/index.md question guide and
ARCHITECTURE.md for ownership and dependency rules.
"""

from importlib.metadata import version as _distribution_version

__version__ = _distribution_version("statadic")

__all__ = ["__version__"]
