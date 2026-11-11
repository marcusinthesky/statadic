# Compositional spatial model pilots

These paired CPU studies used Python 3.13.5, JAX 0.11.0 and float64. The seed is 20261104 for both campaigns.

| Estimand and scenario | Available procedures / attempts | Bias (Monte Carlo SE) | RMSE | Nominal 95% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| Neighbor exposure, true coefficient 0 | 64/64 | +0.0122 (0.0136) | 0.1086 | Not evaluated |
| Neighbor exposure, true coefficient 0.5 | 64/64 | +0.0105 (0.0109) | 0.0872 | Not evaluated |
| Seasonal contrast, exact event times | 32/32 | −0.0454 (0.0516) | 0.2908 | 24/32 (75.0%) |
| Seasonal contrast, administrative censoring | 30/32 | −0.0109 (0.0618) | 0.3330 | 21/32 (65.6%) |

The SLX study compares the named spatially lagged exposure with NumPy least squares on an explicit `[1, x, W @ x]` design. Across 128 paired trials, the largest absolute coefficient difference was 1.78e-15.

The survival study generates lognormal spatial AFT data for eight units over three periods with a spatial lag coefficient of 0.25 and two-way fixed effects. Its estimand is the structural log-time difference between phases 0.25 and 0.75, with generating value 0.6. It compares the fitted STATADIC cyclic spline against a specialist fit with independently constructed SciPy periodic interpolation columns. The administrative scenario caps one prespecified unit's time at 1 in each period. Each censored period has a univariate conditional tail probability.

Both survival routes returned point estimates for all 64 panels. They lacked a valid contrast variance on the same two censored panels. The bias and RMSE rows use the available procedures. Coverage includes all attempted panels and counts unavailable intervals as uncovered. The largest absolute contrast difference across the 62 jointly available intervals was 1.15e-8. The coverage Monte Carlo standard errors were 7.8 percentage points for exact events and 8.5 for administrative censoring. The nominal intervals undercovered in both small-panel scenarios.

All 192 paired trial records passed campaign verification. Run the registered smoke campaigns from the repository root.

```bash
JAX_ENABLE_X64=1 just python::studies::run build/studies/named-slx \
  --study spatial.named-slx --profile smoke --trials 64 --seed 20261104
JAX_ENABLE_X64=1 just python::studies::run build/studies/seasonal-contrast \
  --study survival.spatial-seasonal-contrast --profile smoke --trials 32 \
  --draws 16 --seed 20261104
```
