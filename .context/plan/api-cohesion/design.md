# Accepted design

Accepted 2026-11-10 against clean baseline `31c0cc9`. This supersedes earlier
host spelling decisions, not scientific qualification or historical evidence.

## Public workflow

Align → Specify → Estimate → Inspect → Diagnose → Predict or infer → Report.

Regression owns outcome models; Ordination owns coordinates; Estimand owns
covariance and supplied moments in this slice; Hypothesis states propositions;
Report presents evidence. Numerical primitives remain independently usable.
Unsupported capabilities remain absent; no universal result framework is added.

## Lifecycle and typing

Model.estimate(data) shares the prepare/estimate algorithm. Its Regression
specialization retains final typed overloads for results that depend on the data
layout; making the generic base final would erase that distinction or misstate
the return type. Both concrete Regression entry points retain final typed specializations that
delegate to the shared lifecycle. The common estimate_prepared implementation
validates producing-model identity once before calling _estimate_prepared.
Preparation validates concrete inputs. Preserve family input/preparation/result
types and honest data-dependent unions. Capability Protocols cover actual
prediction, transformation and inference. Invalid numerical evidence stays
reportable; require_valid remains opt-in.

Tighten closed options and normalized dispatch without closing extension
protocols. Traced statuses are array leaves; static choices remain metadata.
Intentionally eager GEL fitting stays eager, including rank/feasibility checks.
Neither a conversion removal nor a registered pytree qualifies derivatives.
Reuse existing identity/matrix/provenance types, without a parallel brand system.

## Structural and spatial contracts

Regression effects default to pooled, never estimator defaults. Process, effects
and exogenous-lag choices belong to numerical problems, with static metadata.
Within consumes problem effects too. Shared IV/GMM/SLX/KKP consumers must consume
or reject each structural role. Solver controls stay on estimators.

Host unit/time/two-way effects reject explicit Intercept; preserve native effect
parameterizations and rank checks without normalization conversion. Names are
generated once in actual fits: rho[graph], lambda[graph], W[graph]:column.
Native callers may retain generic names when optional graph_name is absent.
Inference, restrictions, reports and persistence share that same tuple.

Live cross-sectional spatial prediction consumes SpatialData; Gaussian spatial
panels consume PanelData and share a period/node prediction carrier. Preserve
structural/reduced-form meanings, full graphs, missing-outcome/roster rules and
marginal random effects. Frozen predictors consume observations against retained
graph realizations; they never silently ignore a supplied live graph registry.
Ordinary PanelResult also accepts PanelData: retained predictor transformations
consume full-node covariates and live named WX graphs, then gather the requested
outcome rows. Observations keeps the retained-graph behavior. Conditional effects
require known training keys. The native spatial-likelihood panel results retain
their period/node prediction carrier; no output-axis equivalence is implied.
PanelLayout and SpatialPanelData are neutral input carriers owned by api.data.
The spatial compound namespace is a navigation map; fitted results live in its
public cross_section, panel, observed and graph_regression owners.

Graph declarations/recipes move to api.spatial.graphs. Learning, selection and
resampling remain explicit sibling leaves; no old-path aliases remain.
Scientific numerical owners do not move or depend on host workflows.

## Survival

SpatialSurvivalData(panel, integrations) binds replayable per-period designs to
the one estimate/prepare input. GaussianIntegration remains static execution
controls. Prediction can unwrap the original bundle or accept its panel without
drawing randomness. Default survival prediction is median; other targets retain
explicit values and paired/grid semantics. Remove redundant prediction aliases.

An expert refinement operation consumes prepared state and independent validation
designs. Preserve every attempt, draw stream and diagnostic; no implicit estimator
return-type or converged-solution derivative change is allowed.

## Ordination and Estimand

Retain immutable native fit/context evidence beside fitted coordinate maps, with
named design lookup and no high-level result import into the low owner.
Prediction-only artifacts may omit inspection evidence explicitly. PCA variance
properties have selected-axis length k; ratios divide by full available variance.
Keep full signed Gram eigenvalues separately. PCA/PCoA factories return existing
compositions, not new model identities.

Estimand(Covariance(columns=...), estimator=...) consumes observations.
Estimand(MomentConditions(), estimator=...) consumes MomentData; numerical moment
arrays live only in its problem. Move covariance bricks beside their numerical
owners, preserve existing covariance/GMM/GEL result families, retire competing
host estimation doors and retain qualified core.statistics.Estimand unchanged.

## Inference, reporting and scope

Use typed covariance/resampling/calibration requests. Hypothesis-dependent work
runs at test(Hypothesis(...)). TestOutcome retains typed native evidence, without
serializing arrays into its report. Preserve schedules, failed draws and actual
reference assumptions. Add Normal/resampled references only for actual producers.
Expose bounded OLS Moran/LM, spatial-IV AK and SDM common-factor requests. LR is
an explicit matched two-fit comparison with declared nesting.

Mantel node permutation, dyadic node bootstrap, PERMDISP full-model residual
permutation and Sharpe joint stepdown retain their own laws. Partial Mantel stays
descriptive. Numerical availability, convergence, identification, calibration
and source approval remain distinct in reports.

Metric learning remains experimental. Full-corpus memory, constrained mixtures,
diagonal innovations, boundary bootstrap, differentiated barycentres, unit slopes
and expanded GAM/spatial families are follow-ups. Existing alpha scientific gates
remain open. Tests are serialized through pinned devenv/locked uv and named files.
Extend the existing reference runner/catalogue; retain independent expected
calculations and exact source/lock/artifact/node identities. Downstream production
sources/pins/data remain untouched; new prose and fixtures use neutral names.
