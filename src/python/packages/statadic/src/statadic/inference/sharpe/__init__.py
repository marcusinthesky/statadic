# jaxtyping shape strings are runtime contracts.
"""Sharpe uncertainty, performance benchmarks and strategy comparisons.

``deflated`` provides probabilistic and deflated Sharpe measures with declared
trial counts. ``difference`` provides paired comparisons and stationary
bootstrap intervals; ``multiple`` provides stepdown and joint comparisons across
strategies, whose familywise control requires justified bootstrap and selection
assumptions.
All measures are per observation unless the caller explicitly annualizes.

The point sample ratio and reusable sample-moment reductions belong to
:mod:`statadic.estimation.sharpe`. These eager reporting adapters preserve their
caller-owned precision and key-driven bootstrap contracts.
"""

from __future__ import annotations

from statadic.inference.sharpe.deflated import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from statadic.inference.sharpe.difference import (
    SharpeDifferenceKernelResult,
    SharpeDifferenceTest,
    sharpe_difference_test,
    sharpe_difference_test_kernel,
    sharpe_ratio_ci,
)
from statadic.inference.sharpe.multiple import (
    MultipleSharpeKernelResult,
    MultipleSharpeResult,
    multiple_sharpe_test,
    multiple_sharpe_test_kernel,
)

__all__ = [
    "DeflatedSharpeResult",
    "MultipleSharpeKernelResult",
    "MultipleSharpeResult",
    "SharpeDifferenceKernelResult",
    "SharpeDifferenceTest",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe_ratio",
    "multiple_sharpe_test",
    "multiple_sharpe_test_kernel",
    "probabilistic_sharpe_ratio",
    "sharpe_difference_test",
    "sharpe_difference_test_kernel",
    "sharpe_ratio_ci",
]
