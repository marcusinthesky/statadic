# jaxtyping shape strings are runtime contracts.
"""Estimate covariance of observed random vectors.

Use estimators for sample covariance and Ledoit-Wolf estimates, shrinkage for
blending an estimate with a declared target, and cross_validation to tune the
sample weight on held-out data. Kernel variants accept validated arrays for
JAX transformations; eager doors validate statistical inputs.

Distance-implied operator construction belongs to statadic.operators.covariance.
Sampling uncertainty of fitted parameters belongs to statadic.inference.
"""

from statadic.estimation.covariance.cross_validation import cv_lambda_grid
from statadic.estimation.covariance.estimators import (
    CovarianceProblem,
    LedoitWolf,
    SampleCovariance,
    ledoit_wolf_sample_kernel,
    sample_covariance,
    sample_covariance_kernel,
)
from statadic.estimation.covariance.shrinkage import (
    ShrinkageResult,
    optimal_lambda_dist,
    plug_in_lambda_dist,
    plug_in_lambda_dist_kernel,
    sample_cov_entry_variance,
    sample_cov_entry_variance_kernel,
    shrink_to_dist,
    shrink_to_dist_kernel,
    target_weight_dist,
)

__all__ = [
    "CovarianceProblem",
    "LedoitWolf",
    "SampleCovariance",
    "ShrinkageResult",
    "cv_lambda_grid",
    "ledoit_wolf_sample_kernel",
    "optimal_lambda_dist",
    "plug_in_lambda_dist",
    "plug_in_lambda_dist_kernel",
    "sample_cov_entry_variance",
    "sample_cov_entry_variance_kernel",
    "sample_covariance",
    "sample_covariance_kernel",
    "shrink_to_dist",
    "shrink_to_dist_kernel",
    "target_weight_dist",
]
