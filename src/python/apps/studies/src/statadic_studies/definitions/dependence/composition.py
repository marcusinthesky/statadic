"""Paired operating studies for named spatial terms and dependence."""

from statadic_studies.study_types import Study

SEASONAL = Study(
    "survival.spatial-seasonal-contrast",
    "survival",
    "Estimate a structural seasonal log-time contrast with two-way fixed effects.",
    "Eight units, three periods. Asymmetric row-standardized ring (.7 right, "
    ".3 two to the left). Independent uniform phases at every unit-period. "
    "Periodic cubic interpolation through (0, .3, 0, -.3, 0) at quarter cycles; "
    "unit effects linspace(-.2,.2,8), period effects (-.15,0,.15). Each period "
    "solves (I-.25 W)log T = .2 + unit + period + season + iid N(0,.4^2). "
    "All spells enter at zero. Either all fail, or unit zero has a fixed "
    "administrative cap of 1 per period. At most one censor per period, so "
    "all Gaussian tail probabilities have analytic evaluations. Each method "
    "uses the same data, keys, free rho and unpenalized two-way spatial ML. "
    "The legacy design uses independent SciPy interpolation columns.",
    "Structural seasonal contrast f(.25)-f(.75)=.6. The methods use different "
    "coordinates for the same periodic cubic space. A 95% normal-reference "
    "Wald interval uses the full observed-information covariance. Bias/MSE "
    "condition on a successful Wald procedure; unavailable intervals count as "
    "uncovered. Small-panel incidental-parameter effects are possible.",
    ("compositional-Wald", "explicit-basis-Wald"),
    ("exact", "administrative"),
    ("bias", "mse", "coverage", "width", "failures"),
    (),
    comparison_groups=(
        ("structural-seasonal-contrast", ("compositional-Wald", "explicit-basis-Wald")),
    ),
)

SLX = Study(
    "spatial.named-slx",
    "spatial",
    "Recover a named neighbor-exposure coefficient on an asymmetric graph.",
    "N=64 directed ring; weights .7 right and .3 two to the left. Independent "
    "standard normal x and innovations. y=.2+.4*x+theta*W*x+.6*innovation; "
    "theta is zero or .5. Both procedures use identical observations and W.",
    "Coefficient theta on first-order neighbor x. Bias and MSE are conditional "
    "on an available estimate; every attempted failure is retained. NumPy "
    "least squares on explicitly constructed columns supplies the reference.",
    ("named-SLX", "explicit-NumPy"),
    ("zero", "spillover"),
    ("bias", "mse", "failures"),
    (),
    comparison_groups=(("neighbor-coefficient", ("named-SLX", "explicit-NumPy")),),
)
