# When a row is a whole distribution

**Learning objective:** keep the unit of analysis, atom masses, training geometry
and prediction references distinct.

## Predict first

If one unit contains twelve measurements and another contains eight, do you have
twenty independent units or two distribution-valued observations?

## Experiment

The synthetic dataset has 24 whole units with unequal support sizes. Atom masses
are uniform within each unit. The split happens before preparation: 18 training
units and six held-out units.

Choose rooted Euclidean energy, RBF MMD with fixed bandwidth one, or one-dimensional
W2. Each option is a declared native geometry supported by the reviewed workflow.
This does not license substituting an arbitrary distance into every downstream
method.

The native workflow prepares the training geometry, fits two-dimensional PCoA,
and extends the existing axes to held-out measures. It then fits OLS with an
intercept, a scalar price covariate and distances to two named training landmarks.
Held-out prediction retains those landmarks; it does not refit them on the query.

## Interpretation and limits

One point in the embedding represents a whole empirical measure. Two displayed
axes may omit distance information. Atom counts are not the number of independent
outer observations, and atom weights are not WLS precision weights.

The held-out RMSE is an exploratory workflow output, not a performance claim.
Choosing a geometry after comparing its held-out error makes that set a validation
set. Independent final evaluation requires untouched data. Prepare learned terms
inside every training fold when doing cross-validation.

No distribution-valued outcome regression, automatic statistical qualification,
or within-measure resampling is added by this demonstration.

[Next: calibration as an experiment](calibration.md)

```{raw} html
<a href="snapshots/distributional-workflow.html">View the executed static snapshot</a> · <a href="downloads/distributional-workflow.py">Download the notebook</a> · <a href="snapshots/distributional-workflow.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
