"""Additional 1-D POT solver control outside the branch's general emd2 comparison."""

import os

import numpy as np

from other_refs import CASES as DISTANCE_CASES, prepare as prepare_distance

CASES = {}
for scale in ("fixture", "scaled"):
    CASES[f"w1-{scale}"] = dict(
        DISTANCE_CASES[f"w1-{scale}"],
        reference="POT ot.emd2_1d (specialized control)",
        work="Raw coordinates to scalar W1 using the dedicated 1-D solver",
        note="Additional comparator beyond the test's general emd2. Uses the same coordinates, absolute ground cost, uniform masses, no dense transport plan.",
    )


def prepare(case, side):
    if side == "statadic":
        return prepare_distance(case, side)
    if case not in CASES or side != "reference":
        raise ValueError((case, side))
    for fw in ("JAX", "PYTORCH", "CUPY", "TENSORFLOW"):
        os.environ[f"POT_BACKEND_DISABLE_{fw}"] = "1"
    import ot

    if case == "w1-fixture":
        x = np.array([-2.0, 0.5, 1.25, 4.0])
        y = np.array([-1.0, 1.0, 2.5, 3.0])
    else:
        rng = np.random.default_rng(20261107)
        x, y = rng.normal(size=512), rng.normal(loc=0.2, size=512)
    weights = np.full(len(x), 1 / len(x))
    return lambda: np.asarray(
        ot.emd2_1d(x, y, weights, weights, metric="cityblock", dense=False, log=False)
    )
