"""Gaussian probability integration with explicit numerical evidence."""

from statadic.optimize.integration.gaussian import (
    GaussianIntegration,
    GaussianIntegrationResult,
    GaussianProposal,
    IntegrationDesign,
    InverseNormalResult,
    gaussian_log_survival,
    gaussian_tilt_proposal,
    inverse_log_normal_result,
)

__all__ = [
    "GaussianIntegration",
    "GaussianIntegrationResult",
    "GaussianProposal",
    "IntegrationDesign",
    "InverseNormalResult",
    "gaussian_log_survival",
    "gaussian_tilt_proposal",
    "inverse_log_normal_result",
]
