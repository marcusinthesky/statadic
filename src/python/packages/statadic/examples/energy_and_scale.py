"""Demonstrate raw V-energy and the angular scale-invariance boundary."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from statadic.discrepancy.mmd.energy import energy_distance
from statadic.sample.packed import PackedClouds
from statadic.geometry.ground.metrics import ANGULAR, EUCLIDEAN


def main() -> None:
    """Check an analytic Euclidean witness and a nonzero angular example."""
    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([[2.0], [3.0]])
    raw_energy = energy_distance(
        PackedClouds.from_array(x),
        PackedClouds.from_array(y),
        exponent=1.0,
        metric=EUCLIDEAN,
    )
    # Cross mean=2, each within-cloud mean=0.5: 2*2-0.5-0.5=3.
    np.testing.assert_allclose(np.asarray(raw_energy), 3.0, rtol=2e-5, atol=2e-5)

    # Angular inputs must be nonzero. Positive rescaling preserves directions.
    nonzero = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scaled = 3.0 * nonzero
    direction_only = energy_distance(
        PackedClouds.from_array(nonzero),
        PackedClouds.from_array(scaled),
        metric=ANGULAR,
    )
    scale_sensitive = energy_distance(
        PackedClouds.from_array(nonzero),
        PackedClouds.from_array(scaled),
        metric=EUCLIDEAN,
    )
    # Normalization/arccos cancellation can require a looser float32 tolerance.
    np.testing.assert_allclose(np.asarray(direction_only), 0.0, atol=2e-3, rtol=0)
    if not np.isfinite(np.asarray(scale_sensitive)) or float(scale_sensitive) <= 0:
        raise AssertionError("This Euclidean witness should detect the scale change.")

    print("Raw Euclidean V-energy:", float(raw_energy))
    print("Angular scale-change discrepancy:", float(direction_only))
    print("Euclidean scale-change discrepancy:", float(scale_sensitive))
    print("These are discrepancies, not p-values or equivalence verdicts.")


if __name__ == "__main__":
    main()
