"""Sampling uncertainty, hypothesis tests, diagnostics and resampling.

Use linear_covariance, glm_covariance, panel_covariance, gmm and spatial for
fit-derived uncertainty; parameters composes intervals and restrictions.
GEL, GMS and Wolak provide moment tests. Resampling owns explicit schedules,
bootstrap/permutation procedures and retained draw validity. Mantel, dispersion,
DISCO, dyadic, Sharpe and equivalence provide their named statistical procedures;
multiplicity controls simultaneous rejection rules.

Estimation owns fitting. Numerical convergence does not establish inferential
validity. Import a family explicitly; this namespace loads no procedures.
"""

__all__: list[str] = []
