"""External distance and transport benchmarks matching feat-docs test operations."""
import os

import numpy as np

_TESTS = "src/python/packages/statadic/tests/"
_METHODS = {
    "energy": ("Euclidean V-energy distance", "dcor.energy_distance", "migration/test_t75_2_dcor.py#L27-L50", 2e-8, 1e-8),
    "w1": ("Univariate Wasserstein W1", "POT ot.emd2", "migration/test_t75_6_pot.py#L28-L46", 1e-12, 1e-12),
    "cdist": ("Euclidean cross-distances", "sklearn.metrics.pairwise_distances", "migration/test_t75_7_sklearn.py#L27-L41", 1e-12, 1e-12),
    "pdist": ("Square Euclidean distance matrix", "scipy.spatial.distance.pdist + squareform", "geometry/test_ground_reference.py#L74-L93", 1e-13, 1e-13),
}
CASES = {}
for name, (method, reference, path, rtol, atol) in _METHODS.items():
    for scale in ("fixture", "scaled"):
        if name == "w1":
            n = 4 if scale == "fixture" else 512
            fixture = f"{n} + {n} unweighted atoms, d=1"
            work = "Raw coordinates to scalar W1; includes POT's dense cost matrix and STATADIC cloud construction"
        elif name == "pdist":
            n, d = (4, 2) if scale == "fixture" else (512, 8)
            fixture = f"n={n}, d={d}"
            work = "Raw coordinates to full square distances; includes SciPy squareform"
        else:
            n, m, d = (4, 3, 2) if scale == "fixture" else (512, 384, 8)
            fixture = f"{n} + {m} rows, d={d}"
            work = ("Raw coordinates to V-statistic, exponent=1; includes STATADIC cloud construction" if name == "energy"
                    else "Raw coordinates to full rectangular distance matrix")
        CASES[f"{name}-{scale}"] = dict(
            method=method, reference=reference, fixture=fixture, work=work, source=_TESTS+path,
            note=("Branch fixture." if scale == "fixture" else "Larger deterministic workload using the same operation; seed 20261107."),
            rtol=rtol, atol=atol,
        )


def prepare(case, side):
    name, scale = case.split("-")
    if case not in CASES or side not in ("statadic", "reference"):
        raise ValueError((case, side))

    rng = np.random.default_rng(20261107)
    if name == "w1":
        if scale == "fixture":
            x = np.array([-2.0, 0.5, 1.25, 4.0])
            y = np.array([-1.0, 1.0, 2.5, 3.0])
        else:
            x, y = rng.normal(size=512), rng.normal(loc=0.2, size=512)
    elif name == "pdist":
        x = (np.array([[0.0, 0], [3.0, 4], [-2.0, 1], [3.0, 4]]) if scale == "fixture"
             else rng.integers(-10, 11, size=(512, 8)).astype(np.float64))
    elif scale == "fixture":
        x = np.array([[-1.0, 0.5], [0.25, 1.5], [1.2, -0.7], [2.1, 0.1]])
        y = np.array([[-0.5, 0.2], [0.7, 1.3], [1.8, -1.1]])
    else:
        x, y = rng.normal(size=(512, 8)), rng.normal(loc=0.2, size=(384, 8))

    if side == "statadic":
        import jax.numpy as jnp
        if name in ("energy", "w1"):
            from statadic.sample.packed import PackedClouds
            if name == "energy":
                from statadic.discrepancy.mmd.energy import energy_distance
                def evaluate():
                    a, b = PackedClouds.from_array(jnp.asarray(x)), PackedClouds.from_array(jnp.asarray(y))
                    return np.asarray(energy_distance(a, b, metric="euclidean", exponent=1.0))
            else:
                from statadic.discrepancy.transport import w1_1d
                def evaluate():
                    a = PackedClouds.from_array(jnp.asarray(x[:, None]))
                    b = PackedClouds.from_array(jnp.asarray(y[:, None]))
                    return np.asarray(w1_1d(a, b))
            return evaluate
        from statadic.geometry.ground.metrics import EUCLIDEAN, cdist, pdist
        if name == "pdist":
            return lambda: np.asarray(pdist(jnp.asarray(x), metric=EUCLIDEAN))
        return lambda: np.asarray(cdist(jnp.asarray(x), jnp.asarray(y), metric=EUCLIDEAN))

    if name == "energy":
        from dcor import energy_distance
        return lambda: np.asarray(energy_distance(x, y, exponent=1.0))
    if name == "w1":
        # Keep POT on NumPy; extra backends otherwise probe/import unrelated libraries.
        for framework in ("JAX", "PYTORCH", "CUPY", "TENSORFLOW"):
            os.environ[f"POT_BACKEND_DISABLE_{framework}"] = "1"
        import ot
        weights = np.full(len(x), 1 / len(x))
        return lambda: np.asarray(ot.emd2(weights, weights, np.abs(x[:, None] - y[None, :])))
    if name == "pdist":
        from scipy.spatial.distance import pdist, squareform
        return lambda: squareform(pdist(x, "euclidean"))
    from sklearn.metrics import pairwise_distances
    return lambda: pairwise_distances(x, y, metric="euclidean")
