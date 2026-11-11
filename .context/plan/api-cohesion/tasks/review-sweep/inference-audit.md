# Inference and public-reference audit

Baseline: `38fcc25`. This arm reread all 1,140 lines of the supplied 16:51
transcript end-to-end and checked the existing API-cohesion dispositions against
the current producers. The changes below repair bounded contracts; they create
no new source approval, statistical qualification or release gate.

## Repaired contracts

`api/inference.py` now reports contrast components separately. A finite retained
point remains available when covariance is unavailable; a finite standard error
remains available when only the requested interval fails. Confidence and the
normal reference remain in the contrast report metadata and note. Native
calculation and availability gates are unchanged.

`DyadicResult.infer` now requires the explicit keyword
`resampling=DyadicInference(...)`. Its retained `NodeBootstrapResult` supports
`test(Hypothesis(LinearRestriction.zero(name)))` only for a single unit-coefficient
zero restriction on an actual native regressor. It rejects joint restrictions,
nonzero targets, scaled coefficients, alternatives and foreign fitted designs.
The native node-multiplicity law, schedule, attempted denominator, failed draws
and smaller-sign-tail add-one p-value remain authoritative. The test retains the
complete native result as typed evidence. Any failed draw withholds the p-value
and resampled reference; this adapter supplies no joint or null-imposed bootstrap
calibration. Existing dropped-collinear terms remain unavailable.

The Hypothesis profiles now explicitly inherit a repository profile with
`suppress_health_check=()`. Hypothesis's built-in CI profile otherwise contributes
a suppressed health check when `CI` is present. The policy test exercises all four
repository budgets with CI absent and present; its empty-suppression assertion is
unchanged.

## Deliberately retained distinctions

GEL retains `infer(calibration=GELCalibration(...))`. This request selects an
asymptotic LR/LM/Wald calibration and its explicit iid acknowledgement; it does
not construct a covariance or draw a resampling schedule. Renaming it to either
of those requests would misdescribe the calculation. Affine GEL remains LR-only;
an unacknowledged reference leaves an otherwise finite statistic descriptive.

Canonical graph results already retain their complete `Regression` specification
in `RegressionContext`, in addition to the aligned `PreparedGraphRegression` data.
The bounded `infer(resampling=TimeBlockPairs(...))` route uses that existing state and
reuses the exact original point fit; each sampled date block replays the declared
graph/geometry choices and initial values. A narrow typed graph-resampling
protocol avoids a runtime import cycle. The request checks canonical context,
model/preparation identity and supported balanced-panel scope. The expert
`bootstrap_graph_regression(model, data, ...)` component remains useful for an
unfitted specification and shares the same refit implementation. Neither route
adds a null-calibrated `.test`: the native result provides percentile uncertainty
with its existing tail-resolution and failed-refit limits.

Partial Mantel remains descriptive; PERMDISP keeps its declared exchangeability
scheme; Sharpe retains its joint procedure. Boundary laws, arbitrary joint dyadic
restrictions and unsupported GLM/bootstrap diagnostics remain separate work.

## Independent public spatial comparisons

The coordinator ran the seven-node file
`tests/integration/api/test_spatial_reference_walkthroughs.py` under x64 with
external references required: **7 passed, no skips**, in 12.07 seconds. The one
warning is spreg's bounded optimizer using an absolute tolerance. Retained output:
`/tmp/statadic-sweep-spatial-references.log`.

Five newly added exact nodes establish actual point-estimation comparisons:

| Exact test suffix | Actual route and comparison | Deliberate limits |
| --- | --- | --- |
| `test_cross_section_ml_public_fit_matches_spreg[lag]` | `Regression` with `ML`, SAR: parameters, full log likelihood, innovation variance, structural fitted values and expected-information covariance against `spreg.ML_Lag`. | Fixed directed graph, constant and two regressors; no learned graph. |
| `test_cross_section_ml_public_fit_matches_spreg[error]` | The same public ML route with SEM against `spreg.ML_Error`. | Same finite fixed-graph scope. |
| `test_panel_ml_public_routes_match_spreg[observed]` | Explicit `ObservedPanelML` produces `ObservedSpatialPanelResult`; unit-effects SAR agrees with `spreg.Panel_FE_Lag`. | Complete balanced rows and constant graph; not the catalogue's pooled missing-outcome specialization. |
| `test_panel_ml_public_routes_match_spreg[complete]` | Explicit `PanelML` produces `SpatialPanelResult` for the same model. | Constant graph; not evidence for varying-period graphs. |
| `test_panel_ml_public_routes_match_spreg[graph]` | Explicit `GraphML` produces `GraphRegressionResult` for the same fixed graph. | No fitted geometry, learned projection, bandwidth or selection. |

Each panel comparison checks named x/z/rho coefficients, full log likelihood after
the documented parameter-independent spreg offset, and structural predictions.
It does not compare panel standard errors. The two existing diagnostic nodes
continue to compare OLS Moran/LM and spatial-IV AK statistics; their results do
not establish coefficient agreement.

The SAR/SEM nodes may map to their exact existing scientific variants. The three
panel nodes map only to their actual method/configuration records: their constant
graph and complete unit-effects fixture does not match the narrower existing
missing-outcome, varying-graph or joint-geometry variant selectors. No variant is
added merely to increase a coverage count.

## Verification boundary

This arm ran scoped Ruff, strict Pyrefly and whitespace checks; the scoped type
check reported zero errors. It ran no numerical tests, campaign, benchmark or
qualification. The coordinator reported 126 passing initial repair tests and a
passing architecture gate with 13 negative controls. Those results and the seven
reference comparisons are bounded regression evidence. They do not close source
fidelity, Monte Carlo, full joint-survival or resource qualification obligations.

After the retained-context graph repair, the coordinator reported eight passing
graph/dyadic tests, zero strict source/test errors, seven passing isolated shape
controls and a passing architecture gate with 13 negative controls. The graph
request's independent whole-date regression checks that the returned original is
the same object and only the requested resamples are refit. Existing failed-draw
and learned-map/bandwidth controls continue to exercise the shared procedure.

## Inventory, catalogue and formal integrity

Following the coordinator's final source-format freeze, explicit
`just lean::inventory-refresh` and `just lean::certify` passed. Certification
retained 153 literature mappings, 772 compiled declarations and 47 method families.
The complete evidence crosswalk passed with 1,968 canonical public targets and
eight implementation-comparison definitions. It links 152 claims to Python;
`STATADIC-MOMENTS-001` deliberately has no Python symbol binding. No claim statement,
assumption, scope, source, human-review status or statement pin changed.

The catalogue checker passed after migrating the five spatial result leaves to
their public owners and moving the two shared input carriers to `api.data`.
New dyadic and graph request members stay with their existing method owners;
no scientific procedure was added. Ten bounded mappings describe the five newly
executed spatial nodes, including the three native method configurations. The
catalogue now records 2,491 public symbols, 216 methods, 51 scientific procedures,
309 variants, 151 reference declarations and 173 distinct exact reference nodes.
There are 96 variants and 21 method records with at least one explicit reference
mapping. These are discoverability counts, not family-wide or release
qualification; the catalogue continues to report incomplete coverage.

Retained local outputs are `/tmp/statadic-sweep-certify.log`,
`/tmp/statadic-sweep-catalogue-check.json` and
`/tmp/statadic-sweep-evidence-crosswalk.json`. The aggregate check, documentation build
and installed-wheel evidence remain coordinator-owned.
