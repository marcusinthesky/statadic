# Mathematical contract migration

STATADIC now separates serialized configuration, mathematical construction, numerical
execution, and statistical calibration. Parse dynamic configuration once, then pass the
typed object through every downstream producer.

## Ground distances and spaces

Prefer strategy singletons over repeated string selection:

```python
from statadic.geometry.ground.metrics import EUCLIDEAN, SPHERE_ANGULAR, cdist

raw_distances = cdist(raw_vectors, raw_vectors, metric=EUCLIDEAN)
sphere_distances = cdist(unit_vectors, unit_vectors, metric=SPHERE_ANGULAR)
```

Use `parse_ground_distance` or `resolve_ground_distance` only at a serialized/eager
boundary. A custom strategy is declared with `declare_ground_distance`; it must name its
law, source space, transformation, authority, and required value evidence. A bare callable
is not a mathematical declaration.

Direct sphere strategies require already checked unit values and perform no transform.
Angular and normalized-Euclidean raw-vector strategies are explicit normalization
pullbacks: they may normalize because that map is part of their contract, and they carry
only a raw-space pseudometric law. Oriented positive rays and real projective classes are
different equality relations; ordinary angular distance does not identify antipodes.

## Energy construction and permutation calibration

Replace independent ground/exponent choices in optimized inference with one geometry:

```python
from statadic.discrepancy.mmd.energy import UNIT_EXPONENT, energy_geometry
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.inference.resampling.permutation import energy_permutation_test

geometry = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
result = energy_permutation_test(x, y, geometry, num_permutations=999, seed=7)
```

`EnergyPermutationResult` preserves the ground law, exponent regime, induced energy law,
two-sample statistic origin, ungrouped design, homogeneity hypothesis, Monte-Carlo
resampling calibration, and null draws. The optimized path no longer accepts a statistic
callable separately from the metric and exponent used to hoist its distance matrix.

The scalar `energy_distance` and serialized ground names remain eager compatibility doors.
They resolve a typed strategy and validate the exponent before calling the compiled core.

## Functional, estimator, and metric distinctions

- Population energy is a nonnegative functional under negative type and moment
  hypotheses. Its square root is a measure metric only under the required strong-negative-
  type/separation hypotheses.
- An empirical V-statistic is a plug-in estimator. A diagonal-debiased U-statistic is a
  sibling estimator and may be signed.
- `discrepancy.mmd.energy.matrix.energy_functional_matrix` returns a typed
  V-statistic energy-functional matrix retaining its actual ground, exponent and
  theorem branch. `metrize_energy(matrix)` accepts that bound empirical
  construction only when it separates measures. Finite valid empirical supports
  supply the needed moments; no population moment or population functional is
  established. An unrelated ground declaration cannot upgrade angular energy.
- `mmd_squared_estimate_matrix` returns an estimate matrix, not a distance matrix;
  `rooted_mmd_matrix` is the rooted V-statistic construction.

## DISCO composition

`statadic.association.decomposition.disco_components` is the pure typed decomposition. It
accepts a compatible powered-ground matrix plus a checked `GroupDesign`. It returns no
p-value. `Regression(GeometricResponse(..., distance_power=1), terms=...)`
fits the raw-distance decomposition;
`result.infer(resampling=ResponsePermutation(plan)).test(Hypothesis(Reduced(("intercept",))))`
evaluates the nested null with Monte Carlo add-one
calibration. The native `statadic.inference.disco.disco_permutation_null` remains
available for the group-only decomposition. The former
`statadic.association.dispersion.disco_test` composite surface is removed.

## What static typing cannot prove

Structural typing checks implications among declarations; it does not prove the declared
formula satisfies a universal theorem. Checked evidence proves only a named property of
the observed finite values. Governed references and Lean theorems apply only under their
explicit hypotheses. The property battery therefore remains independent: generated draws
test declared laws and case-owned adversarial witnesses demonstrate every known loss from
generic behavior.
