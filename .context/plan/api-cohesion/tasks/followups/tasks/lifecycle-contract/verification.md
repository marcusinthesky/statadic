---
type: verification
title: Lifecycle contract verification
description: Evidence that one enumerating test matches every host lifecycle verb to a capability or recorded exception.
links: []
---

# Lifecycle contract verification

Test: `src/python/packages/statadic/tests/contracts/api/test_lifecycle_contract.py`
(placed under `contracts/` because the layout enforcer rejects an `api/` test
that maps to no source module).

## Method

- Walks every class defined in any `statadic.api` module, private modules included
  (302 classes at the time of writing), skipping Protocols.
- For each public `infer`, `predict`, `transform`, `diagnose` and `test`, binds
  the method signature to the call shape of the Protocols in
  `statadic.api.capabilities`. Extra parameters must have defaults.
- Rejects any public `predict_*`, `estimate_*` or `infer_*` verb. The only listed
  verb is `estimate_prepared`, accepted on `Model` subclasses.
- Fixtures: six deliberately divergent classes (wrong predict, infer keyword,
  transform arity, diagnose shape, `predict_mean`, `estimate_refined`) fail the
  audit; one conforming class passes.
- A stale exception fails, because each exception must still match no capability.

## Recorded exceptions (production signatures unchanged)

| Method | Why it matches no declared capability |
| --- | --- |
| `InferenceResult.test` | Requires keyword `reference` (chi2 or f). |
| `AffineGELResult.infer`, `FixedGELResult.infer` | Binds `calibration=`; no capability declares it. |
| `TimeBlockPairs.infer` | A resampling plan applying itself to a result, `infer(result, /)`. |

The GEL calibration shape is a candidate for a declared capability; this task
only records it.

## Result

`devenv shell -- just python::statadic::test-fresh tests/contracts/api/test_lifecycle_contract.py`:
10 passed. `ruff format` and `ruff check` clean on the file. Not run: pyrefly on
the file, and `just check` (run by the coordinator).
