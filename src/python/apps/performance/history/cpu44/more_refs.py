"""Additional external comparators using the same numerical targets as integration fixtures."""
import numpy as np

base = "src/python/packages/statadic/tests/migration/"
CASES = {
    "glum-poisson": dict(method="Poisson MLE + offset/exposure", reference="glum GeneralizedLinearRegressor", fixture="12 counts, 2 columns", work="Unpenalized Poisson fit returning coefficients", source="src/python/packages/statadic/tests/estimation/test_glm.py", note="Explicit design intercept. glum gradient tolerance 1e-10; max_iter=200.", rtol=1e-5, atol=2e-7),
    "arch-bartlett": dict(method="Centered Bartlett HAC", reference="arch.covariance.kernel.Bartlett", fixture="7 observations, 2 moments, bandwidth 2", work="Centered long-run covariance from observations", source=base+"test_t75_9_arch.py#L31-L56", note="Fixed Bartlett bandwidth=2, df_adjust=0.", rtol=2e-12, atol=2e-12),
    "skbio-mantel": dict(method="Pearson Mantel", reference="scikit-bio mantel", fixture="3 objects × 2 distance matrices, 0 permutations", work="Observed coefficient from unlabelled square matrices", source=base+"test_t75_8_scikit_bio.py#L26-L56", note="Constructs scikit-bio DistanceMatrix objects inside each call; fixture matrices precomputed using NumPy Euclidean distances.", rtol=1e-12, atol=1e-12),
    "pypfopt-sample": dict(method="Sample covariance", reference="PyPortfolioOpt risk_matrix", fixture="8 returns × 3 assets", work="Return unannualized sample covariance", source=base+"test_t75_10_pyportfolioopt.py#L25-L58", note="Explicit returns_data=True, frequency=1, ddof=1.", rtol=1e-12, atol=1e-14),
}


def prepare(case, side):
    if case not in CASES or side not in ("statadic", "reference"):
        raise ValueError((case, side))
    if case == "glum-poisson":
        x = np.column_stack([np.ones(12), np.linspace(-1.5, 1.5, 12)])
        counts = np.array([0, 1, 2, 1, 3, 2, 4, 3, 6, 4, 5, 7], dtype=np.float64)
        offset, exposure = np.linspace(-.15, .2, 12), np.linspace(.75, 1.75, 12)
        if side == "statadic":
            import jax.numpy as jnp
            from statadic.estimation.glm import PoissonMLE, PoissonProblem
            x, counts, offset, exposure = map(jnp.asarray, (x, counts, offset, exposure))
            return lambda: np.asarray(PoissonMLE().estimate(PoissonProblem(x, counts, offset=offset, exposure=exposure)).parameters)
        from glum import GeneralizedLinearRegressor
        adjusted_offset = offset + np.log(exposure)
        return lambda: np.asarray(GeneralizedLinearRegressor(family="poisson", link="log", alpha=0, fit_intercept=False, gradient_tol=1e-10, max_iter=200).fit(x, counts, offset=adjusted_offset).coef_)
    if case == "pypfopt-sample":
        base_returns = np.array([[1., 0.2, -0.1], [0., 1.2, .3], [.4, -.2, 1.1], [-.7, .5, -.4]])
        returns = np.vstack([base_returns, -base_returns]) * 0.01
        if side == "statadic":
            from statadic.estimation.covariance import sample_covariance
            return lambda: np.asarray(sample_covariance(returns, ddof=1))
        from pypfopt.risk_models import risk_matrix
        return lambda: np.asarray(risk_matrix(returns, method="sample_cov", returns_data=True, frequency=1))
    x = np.array([[-1.0, .5], [.25, 1.5], [1.2, -.7], [2.1, .1]])
    y = np.array([[-.5, .2], [.7, 1.3], [1.8, -1.1]])
    if case == "arch-bartlett":
        series = np.vstack([x, y])
        if side == "statadic":
            from statadic.operators.longrun import hall_centered_hac
            return lambda: np.asarray(hall_centered_hac(series, bandwidth=2))
        from arch.covariance.kernel import Bartlett
        return lambda: np.asarray(Bartlett(series, bandwidth=2, df_adjust=0, center=True).cov.long_run)
    a, b = (np.linalg.norm(p[:, None] - p[None, :], axis=-1) for p in (x[:3], y))
    if side == "statadic":
        from statadic.api.association import Mantel
        return lambda: np.asarray(Mantel("pearson").estimate(a, b).statistic)
    from skbio import DistanceMatrix
    from skbio.stats.distance import mantel
    ids = ["a", "b", "c"]
    return lambda: np.asarray(mantel(DistanceMatrix(a, ids=ids), DistanceMatrix(b, ids=ids), method="pearson", permutations=0, seed=75)[0])
