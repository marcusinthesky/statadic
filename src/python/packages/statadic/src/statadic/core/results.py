"""Generic numerical results and uncertainty summaries shared across stages.

Inference returns tests and uncertainty summaries; estimation retains actual
fitted state before an uncertainty method is selected. ``*_test`` commonly
returns :class:`TestResult` and
``*_estimate`` may return :class:`EstimateResult`. The preview linear ``*_fit``
operations instead return :class:`statadic.estimation.linear.LinearFit`, preserving
the actual sample and parameter state before any uncertainty method is chosen.

Both are pytree-registered, so a ``vmap`` over a Monte-Carlo grid returns
``TestResult[Float[Array, " cells"]]`` rather than failing to flatten. That is the
whole reason they are generic: ``S`` is the statistic's own type, scalar in the
single-run case and batched after a transform, with no second container class.

No ``__post_init__`` validation — deliberate
--------------------------------------------
Host validation of values such as ``0 <= pvalue <= 1`` does not survive
tracing: comparing a tracer raises ``TracerBoolConversionError``. These
containers therefore enforce structure only. Static configuration and shape
checks live in constructing kernels; concrete value validation lives in named
``to_host_*`` adapters after a computation leaves the transform boundary.

Procedure-specific reports remain with their inference family. Generic
``TestResult`` and ``EstimateResult`` preserve numerical fields without
claiming the assumptions required by a particular procedure. Host adapters
convert them deliberately after computation, rather than adding host objects
to the numerical pytree.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum

import jax
from jaxtyping import Bool, Float, Int  # noqa: TC002

from statadic.core.typing import Array  # noqa: TC001

__all__ = [
    "EstimateResult",
    "InferenceStatus",
    "SolverResult",
    "SolverStatus",
    "TestResult",
]

# `S` is the statistic's own type: `Float[Array, ""]` for a single run,
# `Float[Array, "cells"]` after a `vmap` over a parameter grid. It deliberately
# has no `S: Array` bound: jaxtyping's parameterized hints are runtime-generated
# classes, and beartype rejects `Float[Array, ""]` as violating that nominal
# bound while importing a specialized `TestResult`. Result-producing APIs pin
# `S` to a JAX-array hint instead; the concrete non-generic fields below do too.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TestResult[S]:
    """The result of a hypothesis test — what every ``*_test`` returns.

    Attributes:
        statistic: Observed test statistic.
        pvalue: p-value in ``[0, 1]``, or NaN when unavailable. Not validated
            here; see the module docstring.
        null_distribution: Statistics under the null, shape ``(resamples,)`` —
            or ``(*batch, resamples)`` once a transform has batched it; see
            :mod:`statadic.core.typing` for why every pytree field carries
            ``*batch``.
        n_resamples: Number of resamples (permutations / bootstrap replicates).
            A **static** pytree field: it is a shape, so it must not become a
            traced leaf, and it stays constant across a ``vmap``-ed grid.

    Examples:
        >>> r = TestResult(
        ...     statistic=jnp.asarray(2.5),
        ...     pvalue=jnp.asarray(0.03),
        ...     null_distribution=jnp.zeros(999),
        ...     n_resamples=999,
        ... )
        >>> [leaf.shape for leaf in jax.tree_util.tree_leaves(r)]
        [(), (), (999,)]

    """

    # pytest's `python_classes = "Test*"` matches this name wherever a test
    # module imports it, and warns that it cannot collect a class with
    # `__init__`. Declared once here so no downstream suite has to.
    __test__ = False

    statistic: S
    pvalue: S
    null_distribution: Float[Array, "*batch resamples"]  # noqa: F722  # jaxtyping
    n_resamples: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EstimateResult[S]:
    """A point estimate with uncertainty, rather than a complete fitted state.

    Existing estimation and resampling APIs retain this summary contract. The
    linear fitting API returns ``LinearFit`` before covariance or reference
    choices are made; uncertainty providers then consume that actual fit.

    Attributes:
        estimate: The point estimate.
        se: Standard error of ``estimate``, on the same scale.
        ci: ``(low, high)`` confidence interval. A tuple of arrays flattens as
            two pytree leaves, so it survives ``vmap`` unchanged.
        converged: Solver convergence flag as a JAX **boolean** array — a Python
            ``bool`` cannot be a traced leaf. Pass ``jnp.asarray(True)`` for a
            closed-form estimator that cannot fail to converge. The dtype is
            checked, so ``tree_map(lambda x: x * 2, ...)`` over the whole
            container is a type error rather than a silent cast to int.
        n_obs: Number of observations. A **static** pytree field, for the same
            reason as ``TestResult.n_resamples``.

    """

    estimate: S
    se: S
    ci: tuple[S, S]
    converged: Bool[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    n_obs: int = field(metadata={"static": True})


class InferenceStatus(IntEnum):
    """Availability of a specific uncertainty or calibration operation.

    These reasons are separate from numerical solver convergence and parameter
    identification. Availability never certifies scientific assumptions.
    """

    AVAILABLE = 0
    INVALID_FIT = 1
    INSUFFICIENT_DF = 2
    EXACT_FIT = 3
    INVALID_COVARIANCE = 4
    INVALID_RESTRICTION = 5
    INVALID_REFERENCE = 6
    NUMERICAL_FAILURE = 7
    TOO_FEW_CLUSTERS = 8


class SolverStatus(IntEnum):
    """Numerical termination reasons carried as integer array leaves.

    ``SUCCESS`` describes the solver's numerical criterion. It establishes
    neither statistical identification nor the validity of ordinary inference.
    ``STALLED`` means an iterate stopped changing before its residual passed.
    """

    SUCCESS = 0
    MAX_STEPS = 1
    NONFINITE = 2
    FAILED = 3
    STALLED = 4


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SolverResult[S]:
    """A numerical iterate and diagnostics that survive JIT and batching.

    All diagnostics are dynamic array leaves, including ``status`` and the
    iteration budget, so each lane of a batched solve retains its own outcome.
    ``residual_norm`` is procedure-specific and documented by its producer.
    ``rank`` is a separate matrix-rank diagnostic, or ``-1`` when unavailable;
    numerical convergence does not certify identification. Direct solvers use
    zero iterations and a zero iteration budget.
    """

    value: S
    converged: Bool[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    iterations: Int[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    status: Int[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    residual_norm: Float[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    rank: Int[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    max_steps: Int[Array, "*batch"]  # noqa: F722  # jaxtyping shape

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        """Keep every numerical field dynamic in the declared dataclass order."""
        return tuple(getattr(self, item.name) for item in fields(self)), None

    @classmethod
    def tree_unflatten(
        cls, auxiliary: None, children: tuple[object, ...]
    ) -> SolverResult[S]:
        """Reconstruct structural placeholders without weakening the constructor.

        Equinox and JAX construct Boolean/None filter trees with the same node
        structure as numerical arguments. Such trees are not scientific solver
        results, so their reconstruction must bypass the public constructor's
        array contract. Ordinary callers still use that strict constructor.
        """
        del auxiliary
        result = object.__new__(cls)
        for item, child in zip(fields(cls), children, strict=True):
            object.__setattr__(result, item.name, child)
        return result
