"""Exact process/effect/Durbin designs for required spatial panel variants."""

from statadic_studies.study_types import Study


def panel_scenarios(observed: bool, process: str) -> tuple[str, ...]:
    effects = (
        ("pooled", "unit", "time", "two-way", "random")
        if observed
        else ("pooled", "unit", "random")
    )
    return tuple(
        f"{effect}-{'durbin' if durbin else 'base'}"
        for effect in effects
        for durbin in (False, True)
        if observed or process != "SAR" or durbin
    )


PANEL_VARIANT_STUDIES = {}
for _observed, _processes in ((False, ("SAR", "SEM", "SAC")), (True, ("SAR", "SEM"))):
    for _process in _processes:
        _layout = "observed" if _observed else "balanced"
        _identity = f"spatial.panel-{_layout}-{_process.lower()}"
        _methods = (
            ("beta",)
            + (("rho",) if _process != "SEM" else ())
            + (("lambda",) if _process != "SAR" else ())
        )
        PANEL_VARIANT_STUDIES[_identity] = Study(
            _identity,
            "spatial",
            "Measure native process/effect/Durbin coefficient risk and "
            "normal-Wald coverage.",
            "Eight periods, 24 global units. Fixed exogenous Gaussian covariates and "
            "independently generated dense asymmetric zero-diagonal "
            "row-normalized W_t; "
            "graphs do not generally commute. Structural slopes(-.5,.8), pooled/random "
            "intercept .3; Durbin adds .2 W_t times first slope column. "
            "rho=.2 when present, lambda=-.15 when present, innovation SD=.5. "
            "Unit effects N(0,.5^2) drawn once per replicate; random-effect variance "
            "ratio1, independent of covariates. Fixed time effects linspace(-.2,.2,8). "
            "Each full latent vector solves A_t y_t=X_t beta+effects+B_t^-1 epsilon_t. "
            + (
                "Active unit rosters and observed outcome subsets change "
                "deterministically "
                "by period independently of innovations. Full active-node covariates/W "
                "remain; observed outcomes are selected only after the full "
                "graph solve. "
                "Unit/time effects omit intercept; two-way drops first time indicator. "
                "ObservedPanelML uses observed information, retaining "
                "missing-node marginalization."
                if _observed
                else "All 24 units remain observed. PanelML pooled/unit/random "
                "matches the DGP; unit effects use no common intercept. "
                "spatial_covariance uses the actual "
                "native information, including random-effect nuisances."
            ),
            "First structural slope -.5 and each present spatial parameter (.2,-.15). "
            "Full-network independent replicates, not independent edges. Bias/MSE are "
            "conditional on successful fit/inference; failures count as uncovered. "
            "Finite-panel incidental-parameter bias is measured, not corrected "
            "or assumed absent. "
            "Variance-boundary, identification and Hessian failures remain failures.",
            _methods,
            panel_scenarios(_observed, _process),
            ("bias", "mse", "coverage", "width", "failures"),
            (),
            qualification="coverage-only",
        )
