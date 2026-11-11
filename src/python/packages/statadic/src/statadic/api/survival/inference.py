"""Host adapter for model-based survival uncertainty."""

from __future__ import annotations

from statadic.api.base import FitContext  # noqa: TC001
from statadic.api.design import FittedDesign  # noqa: TC001
from statadic.api.fit_statistics import FitStatistics
from statadic.api.inference import (  # noqa: TC001
    CovarianceSpec,
    InferenceResult,
    ModelBased,
)
from statadic.estimation.survival.fit import SurvivalFit  # noqa: TC001
from statadic.estimation.survival.spatial import SpatialSurvivalFit  # noqa: TC001
from statadic.inference.survival import observed_information


def survival_inference(
    fit: SurvivalFit | SpatialSurvivalFit,
    context: FitContext,
    covariance: CovarianceSpec,
    *,
    design: FittedDesign | None = None,
    fit_statistics: FitStatistics = FitStatistics(),  # noqa: B008
) -> InferenceResult:
    """Return observed-information uncertainty without selecting calibration."""
    if (
        not isinstance(covariance, ModelBased)
        or covariance.finite_sample_correction != 1.0  # noqa: RUF069
    ):
        msg = "Survival inference supports uncorrected ModelBased covariance only."
        raise ValueError(msg)
    return InferenceResult(
        observed_information(fit),
        context,
        covariance,
        design=design,
        fit_statistics=fit_statistics,
    )


__all__ = ["survival_inference"]
