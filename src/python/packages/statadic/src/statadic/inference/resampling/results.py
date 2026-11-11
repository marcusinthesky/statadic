# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Host reports and explicit adapters for resampling results.

Numerical procedures return the registered pytrees in ``statadic.core.results``.
Use ``to_host_test_result`` or ``to_host_confidence_interval`` only after JIT
and vmap complete. These adapters concretize scalar report fields while
retaining numerical null arrays; tracing through them raises
``HostAdapterError``. Procedure-specific selected-mixture reports belong to
``statadic.inference.mixture.contracts``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
from jaxtyping import Float  # noqa: TC002

from statadic.core.results import (  # noqa: TC001
    EstimateResult,
    TestResult,
)
from statadic.core.typing import Array  # noqa: TC001

__all__ = [
    "ConfidenceInterval",
    "HostAdapterError",
    "HypothesisTest",
    "PermutationTestResult",
    "to_host_confidence_interval",
    "to_host_test_result",
]


class ConfidenceInterval(NamedTuple):
    """Concrete lower and upper bounds, retaining tuple unpacking."""

    lower: float
    upper: float


class HostAdapterError(RuntimeError):
    """Raised when a host-only result adapter is called during JAX tracing."""


def _reject_tracers(tree: object, *, adapter: str) -> None:
    """Fail explicitly when a host adapter is reached from compiled code.

    Args:
        tree: Result pytree whose dynamic leaves are inspected.
        adapter: Public adapter name included in the diagnostic.

    Raises:
        HostAdapterError: If any result leaf is a JAX tracer.

    """
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(tree)
    ):
        message = (
            f"{adapter} is host-only; return the JAX result pytree from compiled "
            "code and call the adapter after jit/vmap completes"
        )
        raise HostAdapterError(message)


@dataclass(frozen=True)
class HypothesisTest:
    """Results of a hypothesis test.

    This dataclass contains the results from a statistical hypothesis test,
    including the observed test statistic, p-value, number of permutations used,
    and the null distribution of test statistics generated under the null
    hypothesis of no effect.

    Attributes:
        statistic: The observed test statistic computed from the original data.
        pvalue: The finite-sample add-one permutation p-value
            ``(1 + b) / (1 + B)``, where ``b`` permuted statistics are as
            extreme as or more extreme than the observed statistic. Must be in
            [0, 1], or NaN when unavailable.
        num_permutations: Number of random permutations performed to generate
            the null distribution.
        null_distribution: Array of test statistics computed from permuted data
            under the null hypothesis. Length should equal num_permutations.

    Raises:
        ValueError: If pvalue is neither NaN nor in [0, 1], or if the length of
            null_distribution does not match num_permutations.

    Examples:
        >>> result = HypothesisTest(
        ...     statistic=2.5,
        ...     pvalue=0.03,
        ...     num_permutations=999,
        ...     null_distribution=jnp.array([...]),  # 999 values
        ... )
        >>> print(f"Reject null hypothesis: {result.pvalue < 0.05}")
        Reject null hypothesis: True

    """

    statistic: float
    pvalue: float
    num_permutations: int
    null_distribution: Array

    def __post_init__(self) -> None:
        """Validate hypothesis test result data.

        Raises:
            ValueError: If pvalue is neither NaN nor in [0, 1], or if the length
                of null_distribution does not match num_permutations.

        """
        if not math.isnan(self.pvalue) and not 0.0 <= self.pvalue <= 1.0:
            message = f"pvalue must be in [0, 1], got {self.pvalue}"
            raise ValueError(message)

        if len(self.null_distribution) != self.num_permutations:
            message = (
                f"null_distribution length ({len(self.null_distribution)}) must equal "
                f"num_permutations ({self.num_permutations})"
            )
            raise ValueError(message)


@dataclass(frozen=True)
class PermutationTestResult(HypothesisTest):
    """Extended hypothesis test result with permutation-specific metadata.

    This subclass adds no additional fields but provides semantic clarity
    for permutation test results.

    Examples:
        >>> result = PermutationTestResult(
        ...     statistic=3.2,
        ...     pvalue=0.001,
        ...     num_permutations=999,
        ...     null_distribution=jnp.array([...]),
        ... )
        >>> print(
        ...     f"Minimum achievable p-value: {1 / (result.num_permutations + 1):.4f}"
        ... )
        Minimum achievable p-value: 0.0010

    """


def to_host_test_result(
    result: TestResult[Float[Array, ""]],  # jaxtyping scalar shape
) -> HypothesisTest:
    """Convert a numerical test-result pytree to the legacy eager container.

    Scalar conversion and ``HypothesisTest`` validation happen together at this
    named boundary. The null distribution deliberately remains a JAX array to
    preserve the public eager surface; only its scalar report fields are
    concretized.

    Args:
        result: Scalar numerical test result returned by a transformable core.

    Returns:
        The validated legacy hypothesis-test container.

    Raises:
        HostAdapterError: If called while ``result`` contains JAX tracers.
        ValueError: If the concrete p-value or null length is invalid.

    """
    _reject_tracers(result, adapter="to_host_test_result")
    return HypothesisTest(
        statistic=float(jax.device_get(result.statistic)),
        pvalue=float(jax.device_get(result.pvalue)),
        num_permutations=result.n_resamples,
        null_distribution=result.null_distribution,
    )


def to_host_confidence_interval(
    result: EstimateResult[Float[Array, ""]]  # jaxtyping scalar shape
    | tuple[Float[Array, ""], Float[Array, ""]],  # jaxtyping scalar
) -> ConfidenceInterval:
    """Return JAX confidence bounds as the legacy pair of floats.

    Args:
        result: Scalar estimate result or interval pytree returned by a
            transformable bootstrap.

    Returns:
        Concrete ``(lower, upper)`` confidence bounds.

    Raises:
        HostAdapterError: If called while ``result`` contains JAX tracers.

    """
    _reject_tracers(result, adapter="to_host_confidence_interval")
    lower, upper = result.ci if isinstance(result, EstimateResult) else result
    return ConfidenceInterval(
        float(jax.device_get(lower)), float(jax.device_get(upper))
    )
