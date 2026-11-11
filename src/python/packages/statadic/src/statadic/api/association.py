"""Mantel association between two dissimilarity matrices over shared units.

The facade owns no mathematics. Coefficients come from
:mod:`statadic.association.mantel`; the relabeling null, its native stream and the
add-one upper tail come from :mod:`statadic.inference.mantel`. A random plan without
supplied indices therefore reproduces the native relabeling stream
(:func:`statadic.inference.mantel.mantel_permutation_key`) draw for draw;
a supplied schedule is used verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, override

import jax.numpy as jnp

from statadic.api.base import Model, Result
from statadic.api.comparison import (  # noqa: TC001 -- runtime annotations
    ExchangeabilityDesign,
    PermutationPlan,
)
from statadic.api.data import RowId
from statadic.api.hypotheses import Hypothesis, ResampledReference, TestOutcome
from statadic.api.reporting import (  # noqa: TC001 -- proposition annotation
    MetadataValue,
    Metric,
    Report,
    TestRow,
    _row_identity,
)
from statadic.association.mantel import (
    MantelMethod,  # noqa: TC001
    mantel_correlation_kernel,
    partial_mantel_correlation_kernel,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.inference.mantel import mantel_permutation_kernel, mantel_permutation_key

__all__ = [
    "Mantel",
    "MantelInference",
    "MantelInput",
    "MantelResult",
    "MantelTestEvidence",
    "MatrixExchangeability",
    "PreparedMantel",
]

_MAX_SEED = 2**32
_MATRIX_NDIM = 2


def _matrix(name: str, value: ArrayLike) -> Array:
    """Validate one finite symmetric square matrix."""
    matrix = jnp.array(value, copy=True)
    if matrix.ndim != _MATRIX_NDIM or matrix.shape[0] != matrix.shape[1]:
        message = f"{name} must be a square matrix."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        message = f"{name} must contain only finite values."
        raise ValueError(message)
    if not bool(jnp.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)):
        message = f"{name} must be symmetric."
        raise ValueError(message)
    return matrix


@dataclass(frozen=True)
class MantelInput:
    """Aligned matrices and optional partial-correlation control."""

    x: ArrayLike
    y: ArrayLike
    row_ids: tuple[RowId, ...] | None = None
    control: ArrayLike | None = None


@dataclass(frozen=True)
class PreparedMantel:
    """Checked aligned matrices for a specific Mantel model."""

    x: Array
    y: Array
    row_ids: tuple[RowId, ...]
    control: Array | None
    model_token: object


@dataclass(frozen=True)
class Mantel(Model[MantelInput, PreparedMantel, "MantelResult"]):
    """Pearson or Spearman Mantel correlation of two aligned matrices."""

    method: MantelMethod = "pearson"

    @override
    def prepare(self, data: MantelInput) -> PreparedMantel:
        """Validate and retain aligned matrices."""
        x, y, row_ids, control = data.x, data.y, data.row_ids, data.control
        if self.method not in ("pearson", "spearman"):
            message = f"Unknown method '{self.method}'. Choose 'pearson' or 'spearman'."
            raise ValueError(message)
        x_matrix, y_matrix = _matrix("x", x), _matrix("y", y)
        if x_matrix.shape != y_matrix.shape:
            message = "x and y must have equal shapes."
            raise ValueError(message)
        n = x_matrix.shape[0]
        ids = tuple(range(n)) if row_ids is None else tuple(row_ids)
        if len(ids) != n or len(set(ids)) != n:
            message = "row_ids must be unique and aligned with the matrices."
            raise ValueError(message)
        control_matrix = None
        if control is not None:
            if self.method != "pearson":
                message = "The partial coefficient is Pearson-only."
                raise ValueError(message)
            control_matrix = _matrix("control", control)
            if control_matrix.shape != x_matrix.shape:
                message = "control must match the shape of x and y."
                raise ValueError(message)
        return PreparedMantel(x_matrix, y_matrix, ids, control_matrix, self._token)

    @override
    def _estimate_prepared(self, prepared: PreparedMantel) -> MantelResult:
        """Compute the observed coefficients for matching prepared matrices."""
        triangle = jnp.triu_indices(len(prepared.row_ids), k=1)
        statistic = mantel_correlation_kernel(
            prepared.x[triangle], prepared.y[triangle], self.method
        )
        partial = (
            None
            if prepared.control is None
            else partial_mantel_correlation_kernel(
                prepared.x[triangle], prepared.y[triangle], prepared.control[triangle]
            )
        )
        return MantelResult(
            (statistic, partial), prepared.row_ids, self.method, prepared.x, prepared.y
        )


@dataclass(frozen=True)
class MantelResult(Result[tuple[Array, Array | None], tuple[RowId, ...]]):
    """Observed Mantel association; no calibration until :meth:`infer`."""

    context: tuple[RowId, ...] = field(repr=False)

    method: MantelMethod
    x: Array = field(repr=False)
    y: Array = field(repr=False)

    @property
    def valid(self) -> bool:
        """Whether the requested observed coefficients are numerically available."""
        return bool(jnp.isfinite(self.statistic)) and (
            self.partial is None or bool(jnp.isfinite(self.partial))
        )

    @property
    def row_ids(self) -> tuple[RowId, ...]:
        """Return aligned row identities."""
        return self.context

    @property
    def statistic(self) -> Array:
        """Return the observed correlation."""
        return self.core[0]

    @property
    def partial(self) -> Array | None:
        """Return the descriptive partial correlation, if requested."""
        return self.core[1]

    def infer(self, *, resampling: PermutationPlan) -> MantelInference:
        """Bind a node-relabeling design without drawing or testing a null."""
        if not isinstance(resampling, PermutationPlan):
            message = "Mantel inference requires a PermutationPlan."
            raise TypeError(message)
        return MantelInference(self, resampling)

    def _resample(self, plan: PermutationPlan) -> MantelTestEvidence:
        """Relabel the rows and columns of ``y`` under ``plan``.

        A random plan without ``indices`` uses the native Mantel stream, so the
        draws equal the native kernel on that stream for the same seed. Plans may
        declare ``blocks`` as labels aligned with ``row_ids`` (objects move only
        within a block); whole-unit moves (``unit``) are rejected.
        """
        if plan.row_ids is not None and tuple(plan.row_ids) != self.row_ids:
            message = (
                "Permutation row IDs must exactly match the retained observation order."
            )
            raise ValueError(message)
        key = None
        if plan.method == "random" and plan.indices is None:
            if isinstance(plan.n, bool) or not isinstance(plan.n, int) or plan.n < 1:
                message = "Random permutation count must be a positive integer."
                raise ValueError(message)
            if (
                isinstance(plan.seed, bool)
                or not isinstance(plan.seed, int)
                or not 0 <= plan.seed < _MAX_SEED
            ):
                message = "Permutation seed must be an integer in [0, 2**32)."
                raise ValueError(message)
            key = mantel_permutation_key(plan.seed)
        schedules, design, admissible = plan.schedules(self.row_ids, key=key)
        observed, null = mantel_permutation_kernel(
            self.x, self.y, schedules, self.method
        )
        draw_valid = jnp.isfinite(null)
        valid = bool(jnp.isfinite(observed)) and bool(jnp.all(draw_valid))
        pvalue = None
        if valid:
            extreme = jnp.sum(null >= observed)
            requested = schedules.shape[0]
            pvalue = float(
                (1 + extreme) / (1 + requested)
                if plan.method == "random"
                else extreme / requested
            )
        return MantelTestEvidence(
            self,
            schedules,
            null,
            draw_valid,
            plan.method,
            pvalue,
            valid,
            design,
            admissible,
        )

    @override
    def report(self) -> Report:
        """Render the observed coefficients only."""
        metrics = [_correlation_metric("correlation", self.statistic)]
        if self.partial is not None:
            metrics.append(_correlation_metric("partial_correlation", self.partial))
        return Report(
            f"Mantel ({self.method})",
            metrics=tuple(metrics),
            metadata={
                "valid": self.valid,
                "row_ids": self.row_ids,
                "calibration": "not requested",
                "partial": "intercept and control residualized; descriptive"
                if self.partial is not None
                else None,
            },
        )


def _correlation_metric(name: str, value: Array) -> Metric:
    """Report undefined correlations without converting them to finite values."""
    valid = bool(jnp.isfinite(value))
    return Metric(
        name,
        float(value) if valid else None,
        available=valid,
        status="available" if valid else "unavailable",
        unavailable_reason=None if valid else "Correlation is undefined.",
    )


@dataclass(frozen=True)
class MantelTestEvidence:
    """Complete relabeling evidence, including nonfinite draws."""

    observed: MantelResult
    schedules: Array = field(repr=False)
    null: Array = field(repr=False)
    draw_valid: Array = field(repr=False)
    method: Literal["random", "exact"]
    pvalue: float | None
    valid: bool
    design: ExchangeabilityDesign = field(repr=False)
    admissible_arrangements: int

    def _repr_html_(self) -> str:
        """Render the report for notebook display."""
        return self.report()._repr_html_()

    def report(self) -> Report:
        """State the tail rule and keep failure-inclusive draw counts."""
        requested = self.null.shape[0]
        failed = int(jnp.sum(~self.draw_valid))
        base = self.observed.report()
        row = (
            TestRow(
                "mantel",
                float(self.observed.statistic),
                self.pvalue,
                "resampled",
                (int(requested),),
                f"{self.method} permutation relabeling ({int(requested)} draws)",
            )
            if self.valid
            else TestRow(
                "mantel",
                None,
                available=False,
                unavailable_reason="Observed or null correlations are undefined.",
            )
        )
        return Report(
            f"Mantel ({self.observed.method}): permutation inference",
            tests=(row,),
            metrics=(
                *(metric for metric in base.metrics if metric.name != "correlation"),
                Metric("requested_permutations", int(requested)),
                Metric("failed_permutations", int(failed)),
            ),
            metadata={
                **dict(base.metadata),
                "calibration": "row/column relabeling of y requested",
                "null": "objects of y exchangeable among " + self.design.description(),
                "exchangeability": self.design.metadata(self.admissible_arrangements),
                "permutation_method": self.method,
                "tail": ">= observed (ties included)",
                "pvalue_rule": "extreme / all"
                if self.method == "exact"
                else "(extreme + 1) / (all + 1)",
            },
        )


@dataclass(frozen=True)
class MatrixExchangeability:
    """Object labels of the second matrix are exchangeable relative to the first."""

    @staticmethod
    def describe() -> str:
        """State the ordinary Mantel relabeling null, not a partial-Mantel claim."""
        return (
            "The second matrix's object labels are exchangeable relative to the first."
        )

    @staticmethod
    def metadata() -> dict[str, MetadataValue]:
        """Retain the exact hypothesis family."""
        return {"kind": "matrix-label-exchangeability"}


@dataclass(frozen=True)
class MantelInference:
    """An observed Mantel coefficient and a bound node-permutation design."""

    observed: MantelResult = field(repr=False)
    resampling: PermutationPlan

    def test(
        self, hypothesis: Hypothesis[MatrixExchangeability]
    ) -> TestOutcome[MatrixExchangeability, MantelTestEvidence]:
        """Test ordinary Mantel association; retained partial Mantel is descriptive."""
        if not isinstance(hypothesis, Hypothesis) or not isinstance(
            hypothesis.null, MatrixExchangeability
        ):
            message = "Mantel inference supports Hypothesis[MatrixExchangeability]."
            raise TypeError(message)
        if hypothesis.alternative is not None:
            message = (
                "Mantel relabeling supports the declared upper-tail complement only."
            )
            raise ValueError(message)
        evidence = self.observed._resample(self.resampling)  # noqa: SLF001 -- same-owner retained procedure
        requested = evidence.null.shape[0]
        return TestOutcome(
            "mantel",
            hypothesis,
            float(self.observed.statistic) if evidence.valid else None,
            evidence.pvalue,
            ResampledReference(
                "supplied" if self.resampling.supplied else self.resampling.method,
                requested,
                correction="add-one" if self.resampling.method == "random" else "none",
            )
            if evidence.valid
            else None,
            evidence.valid,
            unavailable_reason=None
            if evidence.valid
            else "observed-or-null-correlation-unavailable",
            details={
                "requested_draws": requested,
                "failed_draws": int(jnp.sum(~evidence.draw_valid)),
            },
            construction={
                "rows": _row_identity(self.observed.row_ids),
                "method": self.observed.method,
                "partial_correlation": "descriptive only"
                if self.observed.partial is not None
                else "not requested",
                "schedule_method": evidence.method,
                "supplied_schedules": self.resampling.supplied,
                "tail": ">= observed (ties included)",
                "exchangeability": evidence.design.metadata(
                    evidence.admissible_arrangements
                ),
            },
            assumptions=(
                "Object-label exchangeability under the retained design is required.",
            ),
            evidence=evidence,
        )

    def report(self) -> Report:
        """Describe a bound procedure without testing or generating null draws."""
        return Report(
            "Mantel: bound resampling",
            metadata={
                "test_status": "not requested",
                "schedule_method": self.resampling.method,
            },
        )
