# jaxtyping shape strings are runtime contracts.
"""GMM sampling uncertainty and restriction tests.

Use :mod:`covariance` for uncertainty from an actual retained GMM fit and
:mod:`hypothesis` for moment Wald and overidentification tests. Parameter
intervals and restrictions compose through :mod:`statadic.inference.parameters`.
Fitting belongs to :mod:`statadic.estimation.gmm`; an arbitrary fit weight does not
license the efficient-weight reference required by conventional Hansen J.
"""

from statadic.inference.gmm.covariance import GMMUncertainty, gmm_covariance
from statadic.inference.gmm.hypothesis import (
    GMMHansenJResult,
    HansenJKernelResult,
    HansenJResult,
    WaldKernelResult,
    WaldResult,
    gmm_wald,
    gmm_wald_kernel,
    hansen_j,
    hansen_j_from_fit,
    hansen_j_kernel,
)

__all__ = [
    "GMMHansenJResult",
    "GMMUncertainty",
    "HansenJKernelResult",
    "HansenJResult",
    "WaldKernelResult",
    "WaldResult",
    "gmm_covariance",
    "gmm_wald",
    "gmm_wald_kernel",
    "hansen_j",
    "hansen_j_from_fit",
    "hansen_j_kernel",
]
