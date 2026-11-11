# CPU performance against reference libraries

The original 44 float64 workloads at `689d10a` were replayed on the head of PR #7
(`99c1b00`) and on the optimized source (`667c713`). Each revision has its own
measurements of the same external libraries. The numbers below use the new runs.
The fixed-effects spatial lag fit went from **842.11 ms to 0.55 ms** on repeat
calls and now runs faster than `spreg.Panel_FE_Lag` at **5.59 ms**. Its first
call still took **3.00 s**, versus **6.55 ms** for spreg.

| Comparison across 44 workloads | Candidate vs PR #7 | Candidate vs external reference |
| --- | ---: | ---: |
| Faster warm in all three trials | 36 | 30 |
| Slower warm in all three trials | 1 | 11 |
| Warm ordering changed across trials | 7 | 3 |
| Lower median warm time | 40 | 33 |
| Lower median first-call time | 43 | 1 |
| Lower median process peak RSS | 43 | 2 |

All **264** STATADIC–reference output pairs passed the original case-specific
tolerances across the two revisions. So did all **132** candidate–parent
STATADIC output comparisons. The verifier checks finite float64 results and
identical shapes. Output coverage varies by row. For example, the fixed-effects
spatial lag case checks coefficients, normalized log likelihood and covariance.
The unpenalized glum Poisson case checks coefficients.

[All 44 workloads, fixture sources, tolerances and medians](results/cpu44/summary.csv) ·
[528 worker records with all 3,168 call times](results/cpu44/trials.csv) ·
[Machine-readable counts](results/cpu44/counts.json) ·
[Environment and replay instructions](history/cpu44/README.md)

## Selected workloads

Times are warm milliseconds. Each entry is the median of three worker medians,
each based on five calls. Peak RSS is the median whole-process high-water mark in
MiB. The reference columns use the external workers in the candidate run.

| Workload | Reference | PR #7 ms | Candidate ms | Reference ms | Candidate / reference peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Spatial lag ML, 40 nodes | spreg ML_Lag | 15.99 | 0.70 | 13.22 | 525 / 274 |
| Spatial lag ML, 160 nodes | spreg ML_Lag | 92.60 | 16.30 | 12.37 | 528 / 276 |
| Spatial lag IV, 40 nodes | spreg GM_Lag | 7.54 | 0.90 | 8.75 | 485 / 275 |
| Fixed-effects spatial lag, 12 units × 8 periods | spreg Panel_FE_Lag | 842.11 | 0.55 | 5.59 | 552 / 269 |
| Weibull AFT, 240 right-censored spells | lifelines WeibullAFTFitter | 67.71 | 0.66 | 85.17 | 509 / 174 |
| Log-normal AFT, delayed entry, 240 spells | lifelines LogNormalAFTFitter | 104.77 | 1.60 | 189.33 | 542 / 173 |
| Entity fixed effects, 200 rows | linearmodels PanelOLS | 681.59 | 1.96 | 19.42 | 465 / 173 |
| Fama–MacBeth, 122 unbalanced rows | linearmodels FamaMacBeth | 4.16 | 3.30 | 20.98 | 413 / 172 |
| OLS and HC3, 10,000 rows | statsmodels OLS | 26.77 | 1.76 | 1.50 | 388 / 187 |
| Same OLS and HC3, one outer JIT | statsmodels OLS | 1.00 | 0.80 | 1.43 | 376 / 187 |
| Euclidean energy, 512 × 384 points, 8 dimensions | dcor.energy_distance | 5.43 | 3.22 | 24.60 | 307 / 295 |
| 1D Wasserstein, 512 atoms per sample | POT emd2 | 3.16 | 0.32 | 13.87 | 285 / 196 |
| Same Wasserstein inputs, dedicated solver | POT emd2_1d | 2.56 | 0.31 | 0.10 | 287 / 186 |

For a single delayed-entry log-normal fit, lifelines took **0.20 s** and STATADIC
took **3.47 s**. Repeating the fit with the same shape and specification reverses
the per-call ordering. If those medians hold, STATADIC recovers its first-call cost
at about **19 fits** for this AFT model and **596 fits** for the fixed-effects
spatial lag. These crossover counts extrapolate beyond the five measured repeats.
Each lifelines fit includes formula parsing. For OLS the outer
JIT projects the fitted state to coefficients and HC3 covariance. The two POT
rows repeat the same STATADIC operation against different solvers. The 32 original
fixtures, eight size checks and four execution/solver controls are separate
rows in the counts.

## Implementation and memory

- Fixed-graph SAR and SEM reuse the graph spectrum in the likelihood search.
  SAR projects the outcome and its spatial lag once. For common-graph balanced
  unit effects, filtering commutes with time demeaning. This removes the large
  dummy-variable solve while retaining the likelihood normalization and
  nuisance-adjusted expected information. [Spatial likelihood details](../../packages/statadic/docs/spatial-models.md)
- OLS, panel, GLM, spatial IV/GMM and independent AFT fits reuse compiled
  calculations across compatible inputs. Numerical data remain dynamic.
  Panel indexing groups labels outside tracing. Bounded iterations use
  `lax.cond` to skip completed updates. AFT fit and covariance avoid redundant
  rank decompositions.
- Weighted one-dimensional transport uses sorted empirical distributions.
  Large Euclidean V-energy reductions stream rows with `lax.map` on CPU.
  On the 512 × 384 float64 case XLA's compiled temporary-buffer estimate fell
  from **4,875,020 bytes to 74,424 bytes**. The estimate comes from
  `_energy_distance_core.lower(x, y, 1.0, EUCLIDEAN).compile().memory_analysis().temp_size_in_bytes`
  with both inputs prepared as full-support `PackedClouds`.

The same energy case had the only increase in median process peak RSS against
PR #7, **304.43 to 307.06 MiB**. The process high-water mark includes imports,
compilation, buffers and all six calls. The optimized STATADIC process used less
peak memory than the reference on two workloads, the tiny energy fixture and
Bartlett long-run covariance.

The four-point square distance matrix was the only consistent warm regression
against PR #7, **0.041 to 0.047 ms**. Reference timings also varied between the
two runs. The unchanged external libraries ran faster warm on 32 of 44 rows
in the second run. Their cross-workload median ratio was **0.91**. Their peak
RSS fell on all 44 rows, with a median ratio of **0.976**. Small differences
should be read in light of this host variability. Three trials describe the
observed ordering and provide no confidence bound on future runs.
