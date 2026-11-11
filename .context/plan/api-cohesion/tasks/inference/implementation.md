# Supported inference implementation

The shared host references now include a located/scaled Normal law and explicit
resampling method/count/tail/correction/approximation metadata. TestOutcome keeps
typed native evidence in a keyword-only field excluded from repr and Report
serialization. Existing numerical carriers remain independently usable.

Geometric Regression, Mantel and PERMDISP bind resampling with
`infer(resampling=...)`; hypothesis-dependent calculations execute at
`test(Hypothesis(...))`. Complete evidence retains original schedules, invalid
draws, numerical calibration carriers and tail arithmetic. Freedman-Lane and
PERMDISP retain approximate calibration. Partial Mantel remains descriptive;
its ordinary Mantel relabeling test does not calibrate the partial coefficient.
Mantel and PERMDISP use the shared single-input Model lifecycle and protected
estimation hook. Concrete Regression retains its final typed entry points.

GMM uses an explicit GMMSandwich covariance request and retains its actual weight,
long-run scheme and bandwidth. GEL binds GELCalibration before testing its
supported moment-restriction hypothesis. Affine GEL rejects non-LR choices;
unacknowledged iid calibration leaves finite statistics descriptive. The earlier
GELInferenceResult remains the report carrier for an already evaluated native test.

Fixed dense-graph OLS requests expose regression-adjusted residual Moran and five
LM choices. Spatial IV exposes AK on its actual instrument projection. SDM
common-factor inference consumes explicitly attached SpatialUncertainty and rejects
another fit's covariance. LR compares two already-fitted matched Gaussian spatial
models with declared nesting. No procedure fits, learns or silently subsets a graph.
Each selected diagnostic uses its own native availability, rather than the
aggregate spatial diagnostic bundle flag.

Structural capability Protocols describe supported operations without empty
methods or universal inference signatures. ContrastResult now reports retained
values. Existing parameter tests and linear diagnostic outcomes also retain their
native evidence.

## Verification boundary

Scoped Ruff formatting/checking completed on the owned production and test files.
The initial strict source check found one dense/sparse annotation mismatch; the
OLS request inputs were narrowed to SpatialWeights[Array]. A subsequent strict
check of the ten owned production files reported no production errors. The mixed
source/test invocation additionally exposed standalone relative-import scope and
new test optional-value narrowing issues; full-package checking is coordinator-owned.
The new spatial-IV public journey also exposed a constructor overload omission,
sent to the spatial owner for correction rather than hidden by a cast.

The coordinator subsequently reported 124 passing focused inference/host tests,
149 passing quantity checks, and zero strict package type errors with seven shape
controls passing. The runtime import hook exposed one missing runtime-checkable
spatial likelihood Protocol; its annotation was corrected consistently with the
other request protocols. Existing schedules/null assertions were preserved while
migrating the assignment-unit, Mantel, permutation-null and complete-workflow tests
through `.test(...).evidence`.

`just python::architecture` passed all dependency, interface and hidden-import
checks, its 13 rejection controls, and checker lint/format checks. Policy additions
name actual hypothesis, row-identity, spatial-diagnostic and native interval
consumers; private spatial carriers now cross families through public exports.
The legacy embedding carrier remains deliberately retained pending actual
downstream migration.

No numerical tests, reference trials, campaigns or qualification were executed by
this arm. No source fidelity or qualification status is upgraded by implementation,
typing, architecture or the coordinator's bounded regression tests.

## Formal binding repair

The coordinator's one aggregate check stopped at the retired
`api.covariance.LedoitWolf.estimate_prepared` binding. A complete binding audit
found this single failure among 153 claims. The shrinkage claim now names the
unchanged native kernel, native `LedoitWolf.estimate`, `Estimand.prepare` and
`PreparedCovariance`; its diagnostic text names the actual preparation owner.
The complete Lean module, current native covariance code, host preparation and
source dossier were inspected. The target/blend kernel body, theorem,
assumptions, literature locator, limitations and pending human review are
unchanged.

After explicit public-inventory refresh, `just lean::statement-pin` passed and
changed exactly the shrinkage pin because its diagnostic ownership prose is part
of the pin. The repeated audit resolves all 98 distinct Python bindings across
153 claims. `just lean::check` then passed as the resumed aggregate component:
lint, builds, certification, blueprint HTML/PDF, registry fixtures, correspondence
and diff checks. Its log is `/tmp/statadic-cohesion-lean-check.log`; certification
retains 153 mappings, 772 compiled declarations and 47 method families. This is
integrity and proof replay, not a new source-fidelity or implementation-verification
claim. The whole aggregate check was not repeated by this arm.

The later strict documentation crosswalk exposed three inherited Ordination
member paths: runtime resolution succeeded, but those paths were absent from
the static public inventory. Each embedding claim now links the canonical
Ordination class alongside its existing native centering/spectral producers.
Their supplied-distance, cutoff, full-axis and excluded-geometry relationships
remain unchanged. No scope, source, theorem or review change required another
statement pin.

An inventory-level audit of all 153 claims and eight comparison records now finds
no absent bindings. A registry regression checks the actual claim bindings,
complementing the synthetic alias/member controls. `just lean::certify`, the
direct evidence crosswalk and `just lean::registry-test` passed; the latter ran
42 tests. The crosswalk retains 1963 public targets and 152 claims with Python
links; STATADIC-MOMENTS-001 intentionally has an empty Python-symbol list. Logs are
`/tmp/statadic-cohesion-ordination-certify.log` and
`/tmp/statadic-cohesion-ordination-registry-tests.log`.
