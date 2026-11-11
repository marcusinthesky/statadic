"""Estimation and reporting for supplied, aligned fixed and affine moments.

Rows supplied in the numerical problem must already follow the observation
identities; this workflow checks lengths and never reorders or drops rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, override

import jax.numpy as jnp

from statadic.api.base import Result
from statadic.api.data import Observations
from statadic.api.diagnostics import Diagnostics, MomentsDiagnostic, _diagnostic_requests
from statadic.api.hypotheses import (
    ChiSquare,
    Hypothesis,
    OveridentifyingRestrictions,
    TestOutcome,
)
from statadic.api.inference import CovarianceSpec, InferenceResult
from statadic.api.reporting import (
    MetadataValue,  # noqa: TC001 -- public outcome metadata
    ParameterRow,
    Report,
    TestRow,
    _row_identity,
)
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.gel import AffineGelFit, FixedGelFit
from statadic.estimation.gmm import AffineGMMFit
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import AffineGelResult, GelResult, gel_test
from statadic.inference.gmm import gmm_covariance

if TYPE_CHECKING:
    from statadic.inference.gmm import (
        GMMUncertainty,
    )

__all__ = [
    "AffineGELResult",
    "FixedGELResult",
    "GELCalibration",
    "GELInference",
    "GELInferenceResult",
    "GMMResult",
    "GMMSandwich",
    "MomentContext",
    "MomentData",
    "PreparedMoments",
]


_MATRIX_NDIM = 2
_PANEL_NDIM = 3


@dataclass(frozen=True)
class MomentContext:
    """The explicitly aligned host observations for a moment fit."""

    observations: Observations = field(repr=False)
    model_name: str

    @property
    def row_ids(self) -> tuple[int | str, ...]:
        """Return the supplied observation identities in their original order."""
        return self.observations.row_ids


@dataclass(frozen=True)
class MomentData[ProblemT]:
    """An already aligned numeric problem and the corresponding observations."""

    observations: Observations
    problem: ProblemT
    row_ids: tuple[int | str, ...]

    @classmethod
    def from_arrays(
        cls, problem: ProblemT, *, row_ids: tuple[int | str, ...]
    ) -> MomentData[ProblemT]:
        """Bind supplied problem rows to identities without extra columns."""
        return cls(Observations({}, row_ids=row_ids), problem, row_ids)


@dataclass(frozen=True)
class PreparedMoments[ProblemT]:
    """Owned problem arrays and their aligned host context."""

    problem: ProblemT
    context: MomentContext
    model_token: object = field(repr=False)


def _prepare_moments[ProblemT: FixedMomentProblem | AffineMomentProblem](
    data: MomentData[ProblemT],
    name: str,
    token: object,
) -> PreparedMoments[ProblemT]:
    if data.row_ids != data.observations.row_ids:
        msg = "Moment row_ids must equal observation row_ids in the same order."
        raise ValueError(msg)
    problem = data.problem
    values = (
        problem.moments if isinstance(problem, FixedMomentProblem) else problem.moments0
    )
    if (
        jnp.asarray(values).ndim != _MATRIX_NDIM
        or jnp.asarray(values).shape[0] != data.observations.n_rows
    ):
        msg = "Moment rows must match the aligned observation count."
        raise ValueError(msg)
    if isinstance(problem, AffineMomentProblem):
        derivative = jnp.asarray(problem.jacobian)
        if (
            derivative.ndim == _PANEL_NDIM
            and derivative.shape[0] != data.observations.n_rows
        ):
            msg = "Derivative rows must match the aligned observation count."
            raise ValueError(msg)
    context = MomentContext(data.observations, name)
    if isinstance(problem, AffineMomentProblem):
        return PreparedMoments(
            replace(
                problem,
                moments0=jnp.array(problem.moments0, copy=True),
                jacobian=jnp.array(problem.jacobian, copy=True),
                weight=None
                if problem.weight is None
                else jnp.array(problem.weight, copy=True),
            ),
            context,
            token,
        )
    return PreparedMoments(
        replace(problem, moments=jnp.array(problem.moments, copy=True)), context, token
    )


@dataclass(frozen=True)
class GMMSandwich(CovarianceSpec):
    """Request the actual-fit GMM sandwich with an explicit finite-sample factor."""

    finite_sample_correction: float = 1.0


@dataclass(frozen=True)
class GELCalibration:
    """Select a GEL test construction and explicit iid reference assumption."""

    kind: Literal["lr", "lm", "wald"] = "lr"
    assume_iid: bool = False


@dataclass(frozen=True)
class GELInferenceResult[NativeT: GelResult | AffineGelResult](
    Result[NativeT, MomentContext]
):
    """A native GEL test and its aligned host context."""

    @override
    def report(self) -> Report:
        """Describe retained test quantities without recalculating them."""
        core = self.core

        statistic_available = bool(jnp.isfinite(core.stat))
        pvalue_available = (
            statistic_available
            and bool(jnp.isfinite(core.pvalue))
            and int(core.dof) > 0
        )
        if not core.converged:
            pvalue_reason = "nonconverged"
        elif not core.reference_valid:
            pvalue_reason = "iid-reference-not-requested"
        elif core.dof <= 0:
            pvalue_reason = "nonpositive-degrees-of-freedom"
        else:
            pvalue_reason = "nonfinite-native-pvalue"
        name = f"gel_{core.kind}"
        if pvalue_available:
            row = TestRow(
                name,
                float(core.stat),
                float(core.pvalue),
                "chi-square",
                (int(core.dof),),
                str(ChiSquare(int(core.dof))),
            )
        elif statistic_available:
            row = TestRow(name, float(core.stat))
        else:
            row = TestRow(
                name,
                None,
                available=False,
                unavailable_reason="nonconverged"
                if not core.converged
                else "nonfinite-native-statistic",
            )
        metadata: dict[str, MetadataValue] = {
            "criterion": core.criterion,
            "kind": core.kind,
            "degrees_of_freedom": int(core.dof),
            "converged": bool(core.converged),
            "reference_valid": bool(core.reference_valid),
            "reference_available": bool(
                core.reference_valid and jnp.isfinite(core.pvalue)
            ),
        }
        if statistic_available and not pvalue_available:
            metadata["pvalue_unavailable_reason"] = pvalue_reason
        if isinstance(core, AffineGelResult):
            metadata.update(
                inner_converged=bool(core.inner_converged),
                outer_converged=bool(core.outer_converged),
                rank_stable=bool(core.rank_stable),
            )
        return Report(
            title=f"{self.context.model_name} inference",
            tests=(row,),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GMMResult[SchemeT: PlainLongRun | MeanCenteredLongRun](
    Result[AffineGMMFit[SchemeT], MomentContext]
):
    """A retained affine GMM fit, with actual-weight inference."""

    @property
    def valid(self) -> bool:
        """Return whether the native fit is valid."""
        return bool(self.core.valid)

    @property
    def parameters(self) -> Array:
        """Return the native coefficients; invalid fits hold NaN."""
        return self.core.parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the coefficient names used by ``report``."""
        return self.core.parameter_names or tuple(
            f"parameter_{i}" for i in range(self.core.parameters.shape[0])
        )

    @override
    def report(self) -> Report:
        """Describe fit diagnostics without choosing a reference law."""
        return Report(
            title="Affine GMM fit",
            parameters=tuple(
                ParameterRow(
                    name,
                    float(self.core.parameters[index])
                    if bool(self.core.valid)
                    else None,
                    valid=bool(self.core.valid),
                    status="available" if bool(self.core.valid) else "unavailable",
                    unavailable_reason=None if bool(self.core.valid) else "invalid-fit",
                )
                for index, name in enumerate(
                    self.core.parameter_names
                    or tuple(
                        f"parameter_{i}" for i in range(self.core.parameters.shape[0])
                    )
                )
            ),
            metadata={
                "fit_valid": bool(self.core.valid),
                "solver_converged": bool(self.core.solver.converged),
                "fixed_weight": self.core.fixed_weight,
                "identified": bool(self.core.identified),
                "input_valid": bool(self.core.input_valid),
                "rank": int(self.core.rank),
                "solver_status": int(self.core.solver.status),
            },
        )

    def infer(
        self, *, covariance: GMMSandwich
    ) -> InferenceResult[GMMUncertainty[SchemeT]]:
        """Compute the full sandwich using the weight retained by the fit."""
        if not isinstance(covariance, GMMSandwich):
            message = "GMM inference requires an explicit GMMSandwich request."
            raise TypeError(message)
        uncertainty = gmm_covariance(
            self.core, finite_sample_correction=covariance.finite_sample_correction
        )
        return InferenceResult(
            uncertainty,
            self.context,
            covariance,
            calculation={
                "bandwidth": self.core.bandwidth,
                "scheme": self.core.scheme.__name__,
            },
        )

    def diagnose(self, *requests: MomentsDiagnostic) -> Diagnostics:
        """Evaluate supported moment diagnostics against the retained actual fit."""
        return Diagnostics(
            self.context.model_name,
            tuple(
                request.evaluate_moments(self)
                for request in _diagnostic_requests(requests, MomentsDiagnostic)
            ),
        )


@dataclass(frozen=True)
class FixedGELResult(Result[FixedGelFit, MomentContext]):
    """Fixed moments and multiplier evidence."""

    @property
    def valid(self) -> bool:
        """Numerical multiplier availability, independent of test calibration."""
        return self.core.converged and bool(jnp.isfinite(self.core.objective))

    @override
    def report(self) -> Report:
        """Describe the multiplier solve."""
        return Report(
            title="Fixed GEL fit",
            metadata={
                "valid": self.valid,
                "converged": self.core.converged,
                "criterion": self.core.criterion,
                "effective_rank": self.core.effective_rank,
            },
        )

    def infer(self, *, calibration: GELCalibration) -> GELInference[FixedGelFit]:
        """Bind a supported calibration; an explicit hypothesis triggers the test."""
        return GELInference(self.core, self.context, calibration)


@dataclass(frozen=True)
class AffineGELResult(Result[AffineGelFit, MomentContext]):
    """Affine GEL structural and multiplier evidence."""

    @property
    def valid(self) -> bool:
        """Numerical profile availability, without claiming rank or calibration."""
        return self.core.converged and bool(
            jnp.isfinite(self.core.objective) & jnp.all(jnp.isfinite(self.core.theta))
        )

    @override
    def report(self) -> Report:
        """Describe the retained profile fit."""
        return Report(
            title="Affine GEL fit",
            parameters=tuple(
                ParameterRow(
                    f"parameter_{index}",
                    float(value) if bool(jnp.isfinite(value)) else None,
                    valid=bool(jnp.isfinite(value)),
                    status="available" if bool(jnp.isfinite(value)) else "unavailable",
                    unavailable_reason=None
                    if bool(jnp.isfinite(value))
                    else "nonfinite-parameter",
                )
                for index, value in enumerate(self.core.theta)
            ),
            metadata={
                "converged": self.core.converged,
                "criterion": self.core.criterion,
                "rank_stable": self.core.rank_stable,
                "valid": self.valid,
                "inner_converged": self.core.inner_converged,
                "outer_converged": self.core.outer_converged,
                "effective_rank": self.core.effective_rank,
                "rank_at_parameters": self.core.rank_at_parameters,
            },
        )

    def infer(self, *, calibration: GELCalibration) -> GELInference[AffineGelFit]:
        """Bind profiled LR calibration without evaluating a hypothesis."""
        return GELInference(self.core, self.context, calibration)


@dataclass(frozen=True)
class GELInference[FitT: FixedGelFit | AffineGelFit]:
    """The actual moment fit and an explicit, unevaluated GEL calibration."""

    fit: FitT = field(repr=False)
    context: MomentContext
    calibration: GELCalibration

    def __post_init__(self) -> None:
        """Reject unsupported calibration values independently of annotation hooks."""
        if not isinstance(self.calibration, GELCalibration):
            message = "GEL inference requires a GELCalibration request."
            raise TypeError(message)
        if (
            self.calibration.kind not in ("lr", "lm", "wald")
            or type(self.calibration.assume_iid) is not bool
        ):
            message = "Invalid GEL calibration choice."
            raise ValueError(message)
        if isinstance(self.fit, AffineGelFit) and self.calibration.kind != "lr":
            message = "Affine GEL supports only the profiled likelihood-ratio test."
            raise ValueError(message)

    def test(
        self, hypothesis: Hypothesis[OveridentifyingRestrictions]
    ) -> TestOutcome[OveridentifyingRestrictions, GelResult | AffineGelResult]:
        """Test the supplied moments, preserving native ranks and reference status."""
        if not isinstance(hypothesis, Hypothesis) or not isinstance(
            hypothesis.null, OveridentifyingRestrictions
        ):
            message = "GEL supports Hypothesis[OveridentifyingRestrictions]."
            raise TypeError(message)
        if hypothesis.alternative is not None:
            message = "GEL supports the complement alternative only."
            raise ValueError(message)
        moments = self.fit.moments
        parameter_count = (
            self.fit.theta.shape[0] if isinstance(self.fit, AffineGelFit) else 0
        )
        if (hypothesis.null.moment_count, hypothesis.null.parameter_count) != (
            moments.shape[1],
            parameter_count,
        ):
            message = (
                "The hypothesis dimensions must match the retained moment problem."
            )
            raise ValueError(message)
        native = (
            gel_test(self.fit, kind="lr", assume_iid=self.calibration.assume_iid)
            if isinstance(self.fit, AffineGelFit)
            else gel_test(
                self.fit,
                kind=self.calibration.kind,
                assume_iid=self.calibration.assume_iid,
            )
        )
        available = bool(jnp.isfinite(native.stat))
        calibrated = (
            available
            and bool(native.reference_valid)
            and bool(jnp.isfinite(native.pvalue))
        )
        metadata: dict[str, MetadataValue] = {
            "rows": _row_identity(self.context.row_ids),
            "criterion": native.criterion,
            "kind": native.kind,
            "converged": native.converged,
            "reference_valid": native.reference_valid,
            "assume_iid": self.calibration.assume_iid,
        }
        if isinstance(native, AffineGelResult):
            metadata.update(
                inner_converged=native.inner_converged,
                outer_converged=native.outer_converged,
                rank_stable=native.rank_stable,
            )
        return TestOutcome(
            "gel",
            hypothesis,
            native.stat if available else None,
            native.pvalue if calibrated else None,
            ChiSquare(native.dof) if calibrated else None,
            available,
            unavailable_reason=None if available else "gel-statistic-unavailable",
            details={"reference_degrees": native.dof},
            construction=metadata,
            assumptions=(
                (
                    "A calibrated tail requires the caller's iid moment-level "
                    "reference acknowledgement."
                ),
            ),
            evidence=native,
        )

    def report(self) -> Report:
        """Describe the bound calibration without calculating a test."""
        return Report(
            "GEL: bound calibration",
            metadata={
                "kind": self.calibration.kind,
                "assume_iid": self.calibration.assume_iid,
                "test_status": "not requested",
            },
        )


def _moment_result(
    fit: AffineGMMFit[PlainLongRun | MeanCenteredLongRun] | AffineGelFit | FixedGelFit,
    context: MomentContext,
) -> GMMResult[PlainLongRun | MeanCenteredLongRun] | AffineGELResult | FixedGELResult:
    """Attach aligned context to one actual, supported native moment fit."""
    if isinstance(fit, AffineGMMFit):
        return GMMResult(fit, context)
    if isinstance(fit, AffineGelFit):
        return AffineGELResult(fit, context)
    if isinstance(fit, FixedGelFit):
        return FixedGELResult(fit, context)
    message = "The moment estimator returned an unsupported native fit."
    raise TypeError(message)
