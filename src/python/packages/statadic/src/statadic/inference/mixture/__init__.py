"""Split-sample inference for selected energy mixtures.

``corrected_mixture_test`` separates weight selection from calibration or
explicitly reoptimizes inside each null draw. Its result retains the selected
weights, solver evidence, studentized null and bootstrap interval. Point
mixture discrepancy belongs to ``statadic.discrepancy.mmd.energy``; equivalence
against a declared margin belongs to ``statadic.inference.equivalence``.
"""

from statadic.inference.mixture.contracts import MixtureTestResult
from statadic.inference.mixture.test import corrected_mixture_test

__all__ = ["MixtureTestResult", "corrected_mixture_test"]
