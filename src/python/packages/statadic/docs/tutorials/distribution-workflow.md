# One distribution per observation: compare, embed and predict

**Goal:** reuse a retained weighted-measure geometry without confusing atoms with independent observations.
**Prerequisites:** the first regression tutorial and the distinction between a sample and an empirical probability measure.
**Data:** 24 synthetic observations, each containing 3–6 atoms with explicitly normalized masses.

## Understand the two levels

A row is a whole statistical observation, identified by `unit-i`. The `distribution` column contains an empirical measure for that row. Atom locations and their probabilities live inside the row; they are not additional independent rows in the regression or the group permutation.

The script uses 18 rows for training and six for held-out prediction. The RBF bandwidth is fixed in advance for this illustration. If bandwidths, projections, landmarks or other features are selected using outcomes, the learning and inference design must account for that selection.

## Execute a complete workflow

From the repository root, with the [installed environment](../start/installation.md) active:

```bash
python src/python/packages/statadic/examples/distribution_workflow.py
```

This example is adapted from the end-to-end workflow test. It is a learning example, not a statistical-qualification study.

```{literalinclude} ../../examples/distribution_workflow.py
:language: python
```

{download}`Download the complete script <../../examples/distribution_workflow.py>`

## What each stage means

The prepared geometry retains the concrete kernel and ordered training references.
`Ordination(DistanceGram(), ...)` supplies principal coordinates. Regression
with a raw-distance `GeometricResponse` supplies the DISCO decomposition;
permutation inference is selected afterward. The 31-permutation plan is a
smoke-example budget, not recommended research precision. Synthetic grouping
does not establish a real-world exchangeability design.

The regression terms use distances to two named **training** landmarks plus the scalar `price` variable. Prediction uses the retained geometry and landmarks. It does not refit the representation using the held-out rows. Fit learned preprocessing inside each training fold in a genuine evaluation.

HC3 uncertainty belongs to the actual fitted regression. It does not incorporate arbitrary outcome-based geometry selection, and it does not turn the synthetic association into a causal effect.

## Check more than whether code runs

Inspect whole-observation counts and row order, atom-mass normalization, fit availability, failed permutation counts and prediction shape. The source tests additionally check paired schedules, persistence and training-coordinate reconstruction. Those are useful acceptance tests to preserve when extending the tutorial.

## Assumptions and evidence

The [kernel separation result](../formal/claims/STATADIC-MMD-007.md) depends on
injectivity of the mean embedding on the chosen distribution class; it does not
make every kernel identify distributions. Claim pages record their source and
human correspondence-review status. Inspect the
{ref}`MMD study mappings <research-procedure-mmd.estimation>` and
{ref}`permutation mappings <research-procedure-permutation.two-sample>` alongside
[transformation scope](../capabilities.md) and
[known limits](../validation/known-limitations.md). This small learning example
is not a Monte Carlo qualification campaign.

## Exercise

Replace the fixed RBF representation with the host Euclidean `Energy` representation, using its documented rooted convention. Before running, identify which values may change and which row-identity contracts must remain the same. Then explain why resampling atoms within a measure targets different uncertainty from permuting outer observations.
