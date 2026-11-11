# Ordination implementation

Implemented 2026-11-10 against the API-cohesion working tree. Numerical execution
and independent integration review remain coordinator-owned and pending here.

`api.ordination_terms.OrdinationEvidence` retains the actual native fit, frozen
map and immutable training context. A fitted design term holds that evidence;
`FittedDesign.ordination(name)` searches built-in sums and interactions without
fitting again. Extension terms remain opaque. Predictive-only decoded terms may
omit inspection evidence and report that boundary explicitly.

`Ordination.pca` and `pcoa` compose the existing mathematical components. The
linear path still uses feature SVD. Selected-axis variance arrays now have width
k; variance ratios use the complete available feature variance denominator.
The full signed observation-Gram spectrum remains separate. Negative inertia,
disconnected graphs, retained query semantics and population-law limits are
unchanged. Numerical `valid` is independent of the descriptive inertia policy.

Updated `docs/reference/ordination.md` documents named inspection, the two
factories, selected-axis variance and predictive-only artifact limits.

Added public integration controls in
`tests/integration/api/test_structural_references.py`: sklearn PCA plus OLS
held-out prediction, independent linearmodels FE/IV journeys, and both strict
and descriptive signed-spectrum consumers without external application imports.
The coordinator reported the two signed-spectrum cells passed. Its combined
quantity/reference selection reported 149 passed and three failed fixture
cells: the IV terms omitted the explicitly declared endogenous column, and the
float32 FE fixture used a sub-epsilon default stopping criterion. Both fixture
declarations are corrected; their independent assertion tolerances are unchanged
and the retry remains coordinator-owned.

Focused controls are in `tests/api/test_ordination.py`: independent covariance
and centered-kernel calculations, selected-axis fractions, nested lookup with
fitting disabled, interactions, duplicate-name explanation and map-only
inspection rejection. Existing query/reference controls remain present.

Locked scoped Ruff format/check passed for owned source/tests. No numerical
tests, build or aggregate checks were run by this agent. Coordinator selection:

```bash
just python::statadic::test-fresh tests/api/test_ordination.py tests/api/test_ordination_artifacts.py
```

Artifact source/schema integration, architecture edges, inventory projections
and final qualification are separate coordinator-owned checks; this record does
not claim those checks or Monte Carlo qualification passed.
