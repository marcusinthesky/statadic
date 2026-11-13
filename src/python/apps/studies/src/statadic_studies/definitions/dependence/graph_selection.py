"""Prespecified conditional size and power experiment for a calibrated graph."""

from statadic_studies.study_types import Study

GRAPH_SELECTION = Study(
    "spatial.graph-selection-power",
    "spatial",
    "Measure SAR test size and power after independent projection selection.",
    "Twenty units; four equal-mass articles in two dimensions per unit. Two fixed "
    "candidate projections select coordinate 0 or 1. Each uses 1D W2 squared, a "
    "zero-diagonal RBF and h=0.25 times the median off-diagonal squared distance. "
    "The first candidate generates returns. Four independent calibration dates and "
    "four subsequent inference dates share fixed units and graphs. The covariate "
    "is standard normal; lagged x uses a distinct cyclic forward graph. The DGP is "
    "(I-rho*W)y=.2+.5*x-.3*WX+N(0,.6^2 I). rho is 0, .25 or .5. Hold the "
    "design constant across replicates and draw the full innovation panel anew.",
    "Two-sided LR for rho=0 in the inference panel at nominal .05 using chi-square "
    "with one degree of freedom. Both models refit slopes and innovation variance. "
    "Fixed-projection knows the DGP graph; training-selected maximizes training "
    "likelihood across two candidates and freezes the graph on inference dates. "
    "This estimates rejection under the stated DGP, conditional on the article and "
    "covariate design. A failed calibration candidate invalidates selection. No "
    "inference on selected direction or continuously learned representation.",
    ("fixed-projection", "training-selected"),
    ("rho-zero", "rho-quarter", "rho-half"),
    ("size", "power", "failures"),
    ("Wilks (1938), doi:10.1214/aoms/1177732360",),
    null_scenarios=("rho-zero",),
    decision_only_methods=("fixed-projection", "training-selected"),
    comparison_groups=(
        ("conditional-spatial-LR", ("fixed-projection", "training-selected")),
    ),
)
