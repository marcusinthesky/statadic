"""Frozen dependent-inference selectors and observation-unit contracts."""

from statadic_studies.study_types import Study

COVARIANCE_DEPENDENCE_STUDIES = {
    study.id: study
    for study in (
        Study(
            "linear.dependence-covariance",
            "linear",
            "Evaluate actual Bartlett HAC and one-way cluster coverage against "
            "dependent errors, with classical control.",
            "n=256 fixed intercept and Gaussian predictor; beta=(.2,.4). Error is "
            "stationary unit-variance AR1 rho.7 in row/time order or .8 Gaussian "
            "cluster effect plus .6 iid shock in32 clusters of8 contiguous rows. "
            "Cluster and HAC choices share each dataset. HAC bandwidths0,4,12; "
            "cluster correction1 or32/31.",
            "Slope .4; normal-Wald reference for all selectors, not exact t "
            "inference or automatic dependence-model detection.",
            (
                "classical",
                "HAC-b0",
                "HAC-b4",
                "HAC-b12",
                "cluster-uncorrected",
                "cluster-G-corrected",
            ),
            ("serial-AR1", "clustered"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("newey-west-1987", "liang-zeger-1986"),
        ),
        Study(
            "panel.dependence-covariance",
            "panel",
            "Compare actual one/two-way cluster, within-entity HAC and Driscoll- "
            "Kraay kernel covariance.",
            "Fixed complete entity-major24 by24 design, intercept and Gaussian "
            "predictor, beta=(.2,.4). Entity effects SD.7; error=.6 stationary "
            "entity-specific AR1(.6)+.5 stationary common AR1(.5). Observed actual "
            "periods consecutive or every second integer; AR processes advance on "
            "actual full integer grid before selection. Entity within fit "
            "tolerance1e-6. Covariance retains native slope/effect correction; "
            "cluster group_debias=True; autocorrelation and DK bandwidth4 in "
            "actual time units, Bartlett/Parzen/QS kernels.",
            "Within slope .4. Native normal-Wald inference; no exact finite-sample "
            "or universal dependence-robust guarantee. Two-way indefinite "
            "covariance remains unavailable rather than PSD-repaired.",
            (
                "classical",
                "heteroskedastic",
                "cluster-entity",
                "cluster-time",
                "cluster-two-way",
                "autocorrelation-bartlett",
                "autocorrelation-parzen",
                "autocorrelation-qs",
                "driscoll-kraay-bartlett",
                "driscoll-kraay-parzen",
                "driscoll-kraay-qs",
            ),
            ("consecutive-time", "gapped-time"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            (
                "arellano-1987",
                "cameron-gelbach-miller-2011",
                "driscoll-kraay-1998",
                "newey-west-1987",
            ),
        ),
        Study(
            "panel.fama-macbeth",
            "panel",
            "Evaluate actual cross-section coefficient averaging and its native "
            "time-series covariance.",
            "Fixed complete entity-major32 entities by32 periods; intercept and "
            "Gaussian predictor. Per-period slope=.4+.2 stationary Gaussian "
            "process, independent or AR1(.6); response=.2+slope_t*x+.5 iid "
            "Gaussian row shock. Actual FamaMacBeth fits every cross-section; "
            "covariance bandwidth0 or4 with Bartlett/Parzen/QS and count/(count-1) "
            "correction. Every period remains separately identified.",
            "Unconditional mean slope .4, not the realized finite-period slope "
            "average. Normal-Wald reference; period counts, covariance bandwidth "
            "and actual ordering remain explicit.",
            ("iid-b0", "HAC-bartlett", "HAC-parzen", "HAC-qs"),
            ("independent-slopes", "serial-slopes"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("fama-macbeth-1973", "newey-west-1987"),
        ),
    )
}
