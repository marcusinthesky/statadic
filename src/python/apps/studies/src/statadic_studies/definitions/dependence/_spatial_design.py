"""Shared frozen spatial DGP and metric descriptions."""

_PARAMETER_METRICS = ("bias", "mse", "coverage", "width", "size", "failures")
_IMPACT_METRICS = ("bias", "mse", "coverage", "width", "failures")
_TEST_METRICS = ("size", "power", "failures")
_CROSS_DESIGN = (
    "Fixed n48 directional circular graph W=.6 forward+.4 backward, explicit "
    "intercept and two fixed Gaussian covariates, beta=(.3,-.5,.8). Independent "
    "whole-network replicates with Gaussian innovation SD.5. Exact dense graph "
    "solves; no edge-independent substitution. All numerical/rank/boundary "
    "failures retained; generic RuntimeError propagates as infrastructure failure. "
)
_OBSERVED_DESIGN = (
    "Four ordered periods,12 active nodes; fixed dense zero-diagonal normalized "
    "graph per period and full intercept/covariate design beta=(.3,-.5). Omit "
    "outcomes at graph nodes t,t+4 only, preserving full covariates/active graphs. "
    "Deterministic selection independent of iid Gaussian innovations SD.5. "
)
