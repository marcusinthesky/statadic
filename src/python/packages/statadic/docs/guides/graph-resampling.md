# Refit uncertainty across panel dates

For a balanced graph panel with pooled or unit fixed effects, treat each date's full population as one observation of a time process. Choose a block length in date units based on the dependence horizon. One date per block assumes independent, identically distributed dates. Longer blocks require equally spaced dates, approximate stationarity and sufficiently weak temporal dependence. With fixed units and growing time span, the usual circular-block argument requires blocks long enough to capture the dependence while their fraction of the time span tends to zero. Short panels, regime changes or weakly identified geometry can make percentile coverage poor. [Politis and Romano, *A Circular Block-Resampling Procedure for Stationary Data*](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data)

```python
from statadic.api.spatial.resampling import TimeBlockPairs

fitted = model.estimate(panel)
boot = fitted.infer(
    resampling=TimeBlockPairs(block_length=4, n_boot=999, seed=42),
)
report = boot.report(confidence=0.95)
print(report.to_text())
```

The canonical fitted result retains its exact model specification and aligned
data, so the request reuses that original point fit and replays preparation,
estimation choices and initial values only for the resamples. Conditional
fixed-graph covariance remains available through `infer(covariance=...)`.
The expert `bootstrap_graph_regression(model, panel, ...)` component also remains
available when starting from an unfitted specification.

Here `model` is a graph `Regression` specification and `panel` is its `PanelData`. A single supplied graph can enter this estimator through an explicit `GraphML()`. The bootstrap draws overlapping circular blocks of complete dates, relabels repeated dates, and fits the original model from its declared initial values on each draw. Each sampled date keeps the original ordered unit roster, exogenous columns, outcomes, empirical measures and graph sources. STATADIC rebuilds mean-design transformations, graph weights and spatial covariate lags. Shared `GraphML` projection and bandwidth parameters are re-estimated jointly on the resample. Fixed recipes, external encoder training, prior feature selection and any previously chosen hyperparameters are conditional inputs. The time key itself cannot enter the response or predictors.

The report gives the original regression estimates, bootstrap standard deviations and equal-tailed percentile intervals for coefficients and learned scalar bandwidths. Projection coordinates can change sign or rotate under equivalent maps. Their original fitted values are retained in graph metadata. Coordinatewise intervals and hypothesis-test p-values are omitted. Using this procedure for a weakly identified map or a graph entering only through a dependence coefficient near zero requires a separate calibration study.

`boot.periods` maps the integer entries of the index schedule to source dates. For example, `[boot.periods[i] for i in boot.indices[0].tolist()]` reconstructs the first sample. The report records the seed, block length, requested and valid counts and failure reasons. `n_eff` counts successful numerical refits. The number of distinct source dates is available separately through `len(boot.periods)`. A failed refit withholds standard errors and intervals for the campaign. With all fits valid, a requested interval needs at least one effective draw in each nominal tail. The raw draws and statuses remain available for diagnosis.
