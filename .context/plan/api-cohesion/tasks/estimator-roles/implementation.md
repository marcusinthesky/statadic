---
type: document
title: Estimator role migration implementation
description: Authored migration boundaries and verification limits.
links:
  - .context/plan/api-cohesion/tasks/estimator-roles/README.md
---

# Estimator role migration implementation

The Regression declaration owns process, effects and named SpatialX selection. Numerical spatial problems retain static process, effects, positional lag columns and optional graph_name. ML, PanelML and ObservedPanelML retain only numerical budgets/tolerances; Within and RandomEffects consume effect roles from PanelProblem. Shared spatial IV/GMM/KKP and SLX consumers reject incompatible structural roles instead of silently ignoring them.

Bare SAR/SEM/SAC and graph-qualified generated names preserve existing numerical formulas. Native fixed-effect intercept conventions remain available for independent reference controls; Regression rejects an explicit intercept with unit, time or two-way effects.

Graph declarations and recipes now share api.spatial.graphs; learning, selection and resampling moved beneath the same owner without aliases. The final learned graph evaluation retains finite-array/projection/bandwidth availability checks separately from its perturbation screen. Neither these checks nor conditional design identification establish joint graph parameter identification.

Regression preserves concrete final estimate and estimate_prepared overloads as typing specializations delegating to the shared Model algorithms. Token ownership is checked by the shared prepared entry point; the native hook does not repeat that check. This retains precise data-dependent family typing without another lifecycle.

Authored host controls cover native lowering parity, graph-qualified names, same-live-input prediction, ordered graph identities, missing binding precedence and fixed-effect intercept rejection. Numerical execution and aggregate verification belong to the coordinator; this record makes no test-pass or current-release qualification claim.

## Bounded static shape audit

The existing command
`devenv shell -- just python::statadic::typecheck-shape-coverage src/statadic/estimation/survival/spatial.py src/statadic/geometry/ground/metrics.py`
completed with exit 0. Its temporary source-mutation audit classified 7 of 10
decorated return shapes as inferred, 0 as annotation-only and 3 as unchecked.
The unchecked functions are `normalized_euclidean_distance`,
`sphere_chord_distance` and `euclidean_distance` in the ground metrics owner.
The retained raw development log is `/tmp/statadic-cohesion-shape-coverage.log`.
This is a bounded static result, not coverage of all package functions, runtime
shape validation or numerical/JAX derivative qualification. The script has no
help option; the separate `--help` discovery call exited 2 as an invalid target.
