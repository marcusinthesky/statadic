"""Point estimation and retained fitted state.

Use linear for OLS/WLS, glm for count/binomial responses, gmm and gel for
moment estimators, panel for static panel regressions, and spatial for spatial
cross-section/panel models. Covariance estimates distributional covariance;
dyadic fits endpoint models; sharpe estimates sample performance.

Fitters own the actual sample, parameter order and numerical evidence.
Post-fit uncertainty and calibration belong to statadic.inference. Import the
required family directly; this namespace deliberately loads no fitters.
"""

__all__: list[str] = []
