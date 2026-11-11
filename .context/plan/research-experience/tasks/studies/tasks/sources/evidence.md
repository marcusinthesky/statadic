# Proposed bounded source correspondences — review draft

2026-11-04. Root source, catalogue and governed records were read only. This file
proposes scopes for a second reviewer; **no approval or source status was changed**.
Formula correspondence does not certify implementation semantics, assumptions,
finite-sample calibration, current campaigns or a release wheel.

## Recommended decisions

| Variant IDs | Proposed decision | Primary source key |
| --- | --- | --- |
| `longrun.estimation.plain-newey-west-fixed-bandwidth` | Approve the stated fixed-bandwidth Bartlett operator only | `newey-west-1987` |
| `longrun.estimation.hall-centered-fixed-bandwidth` | Conditional candidate after creating/reviewing Hall 2000 record and correcting false 1987 attribution | proposed `hall-2000-covariance-power` |
| `longrun.estimation.plain-newey-west-automatic-bandwidth`, `longrun.estimation.hall-centered-automatic-bandwidth` | Keep pending; native rule is not the cited plug-in selector | separate selector source required |
| `panel.covariance.cluster-entity`, `panel.covariance.cluster-time` | Approve uncorrected independent-cluster linear sandwich core, with assumptions below | `liang-zeger-1986` |
| `panel.covariance.cluster-two-way` | Approve inclusion–exclusion core; bounded pooled finite-G convention below | `cameron-gelbach-miller-2011` |
| `panel.estimation.famamacbeth`, `panel.covariance.fama-macbeth-hac` | Approve time-average cross-section coefficients and bandwidth 0/debiased covariance only | `fama-macbeth-1973` |
| `panel.covariance.driscoll-kraay` | Bounded algorithm candidate; precursor primary paper checked, final journal formula not independently read | `driscoll-kraay-1998`, with explicit 1995 precursor locator |
| `sharpe.psr-dsr.probabilistic-sharpe`, `sharpe.psr-dsr.deflated-sharpe-selection-adjustment` | Approve nondegenerate formula core under supplied independence/dispersion assumptions | `bailey-lopez-de-prado-2014` |
| `sharpe.interval-difference.paired-sharpe-difference-test` | Do not approve as Ledoit–Wolf2008's studentized procedure | `ledoit-wolf-2008` is related context only |
| `sharpe.multiple.romano-wolf-stepdown` | Bounded stepdown/fixed-studentizer candidate; no higher-order or finite-B calibration claim | `romano-wolf-2005` |
| `sharpe.multiple.reality-check-joint-test` | Keep pending: native studentized circular-bootstrap variant differs from original stated construction | `white-2000` |

## 1. Newey–West fixed-bandwidth operator

Primary article: Whitney K. Newey and Kenneth D. West, *A Simple, Positive
Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
Econometrica 55(3),703–708(1987), DOI 10.2307/1913610. [Primary paper scan](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1987.pdf).

**Locator:** p. 703 Eq(4) defines each lag product with denominator T and raw fitted
moment rows; p. 704 Eq(5) adds symmetric lag products with weight 1−j/(m+1).
Both pages were visually inspected from the paper scan. Native
`operators/longrun.py:147–209` uses exactly that denominator, weight and symmetric
sum when `center=False` and a nonnegative explicit bandwidth is supplied.

**Proposed scope:** algorithmic Bartlett HAC identity for finite moment rows,
without finite-sample multiplier or mean subtraction. Its population use requires
the paper's moment/dependence/bandwidth assumptions; fixed 4 need not consistently
estimate an infinite-lag target. Native `optimal_bandwidth()` is
floor(4(T/100)^(2/9)), a sample-size rule. Approving the operator does not approve
that rule as Newey–West1994/Andrews1991 data-driven plug-in bandwidth selection.

## 2. Hall attribution correction and centered HAC

Primary article: **Alastair R. Hall(2000)**, *Covariance Matrix Estimation and the
Power of the Overidentifying Restrictions Test*, Econometrica 68(6),1517–1527,
DOI[10.1111/1468-0262.00171](https://doi.org/10.1111/1468-0262.00171).
[Primary journal paper](https://users.ssc.wisc.edu/~behansen/718/Hall2000.pdf).

**Locator:** p. 1517 introduction; p. 1521 Eq(9) replaces each fitted moment row by
its deviation from the sample mean before forming lag products. Brief exact quote
from p. 1517: “using the sample moment in mean deviation form” (9 words).
Native `hall_centered_hac_kernel` passes `center=True` to the same Bartlett
operator, matching this bounded construction.

**Important correction:** the root module currently says Hall 1987 at lines 12 and
239; the mathematical-contracts page repeats 1987. That attribution was not
supported. Hall 2000 is the primary source located here. Proposed approval needs
a governed Hall 2000 record, not promotion of an unrelated 1987 citation. No
approval of native automatic bandwidth or complete misspecified/profiled GMM
inference follows from centering alone.

## 3. Independent-cluster sandwich

Primary article: Kung-Yee Liang and Scott L. Zeger(1986), *Longitudinal data
analysis using generalized linear models*, Biometrika 73(1),13–22,
DOI[10.1093/biomet/73.1.13](https://doi.org/10.1093/biomet/73.1.13).
[Primary article PDF](https://www.maths.usyd.edu.au/u/jchan/GLM/Liang%26Zeger1986GEE.pdf).

**Locator:** p16 Theorem 2 and the following variance-estimation paragraph replace
subject outcome covariance by residual outer products in the sandwich. For a
linear identity-link independence-working-covariance equation this becomes
sum of cluster score outer products surrounded by the regression bread.
Native `_cluster_meat`, with `group_debias=False`, computes exactly that core from
fit-owned coefficient influences.

**Bounded scope:** independent entities, or independent time clusters if that is
the actual sampling grouping; correctly specified conditional mean and ordinary
full-rank linear fit. Approve `debiased=False,count_effects=False` for an exact
uncorrected convention. Do not attribute finite-G multipliers, absorbed-effect
corrections, arbitrary cross-cluster dependence, informative missingness or exact
small-cluster t reference laws to this theorem. Within-estimator-specific
exogeneity and robust panel asymptotics need their own source review.

## 4. Two-way cluster inclusion–exclusion and finite-G convention

Primary author manuscript underlying Cameron, Gelbach and Miller(2011),
*Robust Inference With Multiway Clustering*, JBES 29(2),238–249,
DOI 10.1198/jbes.2010.07136:
[authors' 15 May 2009 manuscript](https://cameron.econ.ucdavis.edu/research/JBESpaper2009version.pdf).

**Locator:** PDF p10/printed draft p10 Eq(2.10)–(2.11): two one-way cluster
sandwiches minus their intersection sandwich. Draft p11 discusses componentwise
multipliers G/(G−1)(N−1)/(N−K), with analogous H and intersection I counts.
Native `_panel_cluster_covariance` matches the inclusion–exclusion algebra for
unique entity-time rows. `_cluster_meat` supplies G/(G−1)(N−1)/N; the outer
`panel_covariance` correction supplies N/(N−K) for pooled models with no absorbed
count. Since intersection rows are unique, I=N and its inner correction is 1.

**Proposed scope:** uncorrected core generally; stated finite-G convention for
pooled full-rank designs with `debiased=True,count_effects=False,group_debias=True`.
Do not extend approval to arbitrary fixed-effect df counting, PSD repair,
few-cluster exactness, every panel transformation or normal/t reference choice.

## 5. Fama–MacBeth baseline, not its HAC extension

Primary article: Eugene F. Fama and James D. MacBeth(1973), *Risk, Return, and
Equilibrium: Empirical Tests*, JPE 81(3),607–636,
DOI 10.1086/260061. [Primary article scan](https://people.hec.edu/rosu/wp-content/uploads/sites/43/2023/09/Fama-MacBeth-Risk-return-and-equilibrium-Empirical-tests-1973.pdf).

**Locator:** pp616–617 describe month-by-month cross-section regressions;
p. 619 describes averaging coefficients and gives the unnumbered t formula:
mean coefficient divided by its time-series SD divided by sqrt(number of months).
Page 619 was visually inspected.

Native estimation averages valid period coefficient vectors. With every period
identified, bandwidth 0 and `debiased=True`, native covariance equals
sum_t(beta_t−mean)(beta_t−mean)'/[T(T−1)], so its diagonal SE is the stated sample
SD/sqrtT. This is a bounded baseline match for `famamacbeth` and the zero-bandwidth
branch of `fama-macbeth-hac`. Do not approve positive-bandwidth Bartlett/Parzen/QS
extensions, selective missing-period behavior, generated-regressor corrections or
universal asset-pricing validity from the original source.

## 6. Driscoll–Kraay algorithm candidate

The final 1998 bibliographic record is verified separately. The primary precursor
checked here is John Driscoll and Aart Kraay, *Spatial Correlations in Panel Data*,
World Bank Policy Research Working Paper 1553(December1995):
[author/institutional primary paper](https://documents1.worldbank.org/curated/en/944831468739460842/pdf/multi0page.pdf).

**Locator:** printed pp3–5 Eqs(3)–(5), theorem on pp7–8 Eq(7), discussion after
the theorem and fixed-effects footnote 4. Cross-sectional average moment rows feed
ordinary Bartlett time-series HAC; the paper states a growing-time asymptotic
result, not an N-large fixed-T guarantee.

Native `_panel_time_covariance(method='driscoll-kraay')` sums influences by period
before applying the time-distance window; inverse-X'X normalization cancels the
paper's cross-sectional averaging normalization. This supports a narrowly bounded
balanced-period Bartlett core with optional finite-sample corrections disabled.
**Hold full approval** until the final 1998 article is checked or the reviewer
explicitly records the 1995 primary precursor as the locator. Other kernels,
automatic bandwidths, irregular period gaps and finite-effect corrections remain
separate conventions.

## 7. PSR/DSR nondegenerate formula core

Primary author paper: Bailey and López de Prado(2014), *The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality*,
JPM 40(5),94–107, DOI 10.3905/jpm.2014.40.5.094.
[author manuscript](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

**Locator:** manuscript p. 7 Eq(1), p. 8 Eq(2), appendix p. 13 Python
`getExpMaxSR` snippet. Page 8 formula was visually inspected. Native PSR uses
Phi((SRhat−benchmark)*sqrt(n−1)/sqrt(1−skew*SRhat+(kurtosis−1)*SRhat²/4)).
Native expected maximum uses the same Euler-constant weighted normal quantiles
and cross-trial variance scale; DSR evaluates PSR at that maximum benchmark.

**Proposed scope:** n>1, positive variance denominator; for DSR, N>1 and positive
supplied cross-trial dispersion, with N the independent/effective-independent
trial count. Non-excess kurtosis and per-observation Sharpe units must be retained.
The supplied-score formula is an asymptotic significance approximation, not a
posterior probability. N=1/nonpositive-variance point fallbacks, automatic effective
trial-count estimation, serial-dependence corrections and universal FWER are not
approved. The paper credits PSR's original development to their 2012 work; 2014 is
a direct formula source, not a claim that PSR originated in 2014.

## 8. Ledoit–Wolf paired procedure: substantive mismatch

Primary author working paper: Ledoit and Wolf(2008), *Robust Performance Hypothesis
Testing with the Sharpe Ratio*, January 2008, institutional working paper 320,
later JEF 15,850–859, DOI 10.1016/j.jempfin.2008.03.002.
[primary paper](https://www.econ.uzh.ch/apps/workingpapers/wp/iewwp320.pdf).

**Locator:** §3.2, printed pp6–10; p10 direct p-value uses bootstrap quantities
abs(delta*_m−delta_hat)/SE*_m against abs(delta_hat)/SE_hat, with its plus-one
finite-resample convention. Each resample's studentizer is recomputed.

Native `_sharpe_difference_kernel` instead compares
abs(delta_boot−mean(delta_boot)) to abs(delta_hat), with one common bootstrap SD
used only for the returned statistic and no plus-one p-value correction. The
common SD cancels from the decision; this is a centered, non-studentized bootstrap
comparison, not the paper's implemented studentized test. **Keep exact method
correspondence pending**. Paired block resampling is related motivation, not
sufficient source approval. No implementation change is proposed in this dossier.

## 9. Romano–Wolf stepdown candidate

Primary author-hosted journal article:
[Romano–Wolf 2005](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffa286d4d1/etca.pdf),
Econometrica 73,1237–1282, DOI 10.1111/j.1468-0262.2005.00615.x.
**Locator:** §4.1 pp1252–1254, Eq(10), Algorithms 4.1–4.2 and footnote 22.

Native code orders observed standardized statistics, computes bootstrap maxima on
successively smaller remaining subsets and applies cumulative maxima to adjusted
p-values. This matches the stepdown max-statistic architecture. It reuses one
original bootstrap-estimated SD in every bootstrap lane. Footnote22 explicitly
acknowledges fixed original studentizers as a first-order-consistent alternative,
while preferring reestimation from bootstrap samples. **Propose bounded approval
of that fixed-studentizer stepdown architecture**, conditional on consistent
SE/bootstrap laws. Exclude the paper's recommended per-resample studentization,
higher-order refinements, its block-length Algorithm 7.1, arbitrary serial laws,
finite-B exact control or plus-one calibration. Native automatic block length is
selected from the best observed return differential and needs separate analysis.

## 10. White 2000 joint procedure: keep pending

Primary article: [White 2000](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf),
*A Reality Check for Data Snooping*, Econometrica 68,1097–1126,
DOI 10.1111/1468-0262.00152.
**Locator:** §2, pp1102–1105 Proposition2.2/Corollary2.4; nonlinear Sharpe extension
pp1106–1107 Corollaries2.6–2.7. The primary treatment compares maxima of centered
performance differences and uses the stationary bootstrap in its consistency
results.

Native code instead takes maxima after dividing each strategy by its own common
bootstrap SD and resamples circular blocks. Those are plausible extensions, but
they are additional conventions, not exact reproduction of the paper's displayed
raw-max stationary-bootstrap construction. **Do not approve this whole variant
merely because it is named Reality Check.** A source-backed derivation for that
specific studentized circular-block Sharpe maximum is still required. Its retained
p-values also use raw finite-B exceedance proportions, not a plus-one convention.

## Review artifacts and next actions

Local scans inspected: `/tmp/statadic-NeweyWest1987-p703.png`,
`/tmp/statadic-NeweyWest1987-p704.png`, `/tmp/statadic-DSR2014-p8.png`,
`/tmp/statadic-FamaMacBeth1973-p619.png`. PDFs retained beside these files. The sources
were fetched from primary article/author/institutional locations; secondary
summaries were not used to authorize correspondences.

The reviewer can approve the narrow scopes above, request source-specific
revisions, or keep any candidate pending. Hall 2000 needs a governed reference and
attribution correction before promotion. Every approval should preserve its
explicit exclusions in source_evidence.review/locator; it must not imply that the
entire procedure family, all numerical branches or finite-sample performance have
been certified.

## 11. Original PSR paper (2012) and bibliography proposal

David H. Bailey and Marcos M. López de Prado, *The Sharpe Ratio Efficient
Frontier*, Journal of Risk **15(2), 3–44 (Winter/December 2012)**,
DOI [10.21314/JOR.2012.255](https://doi.org/10.21314/JOR.2012.255).
Both authors' publication lists establish the journal/pages/year; the DOI resolves
to the publisher's article. The [author manuscript](https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf)
is dated July 2012. Proposed governed key: `bailey-lopez-de-prado-2012`.

**Formula locator:** manuscript §3, printed p9 (PDF page index8), Eq(11), with
estimator standard deviation in §2.5 p8 and Python implementation in Appendix3.
The formula is Phi((SRhat−SR*)sqrt(n−1)/sqrt(1−skew*SRhat+
(kurtosis−1)*SRhat²/4)). This matches the native supplied-score PSR formula
when n>1 and its variance denominator is positive. Inputs use original-frequency
Sharpes and non-excess kurtosis. The source explicitly presents an asymptotic
construction; finite-sample exactness, posterior probability, selection control
and degenerate point fallbacks are outside this correspondence. Use the 2012
source as PSR's original paper and retain the 2014 source for DSR/expected maximum.

## 12. Native docstring findings (read only)

- `inference/sharpe/difference.py:128–138`: “following the Ledoit-Wolf (2008)
  HAC-robust/bootstrap approach” can imply the actual published test. Replace
  this with “paired circular-block bootstrap comparison inspired by Sharpe
  resampling methods” and explicitly describe mean-bootstrap recentering,
  a common bootstrap-SD reported statistic, no per-resample studentization,
  and raw exceedance proportion. Retain Ledoit–Wolf only as related context.
  The module's “serial-dependence aware” is an implementation description,
  not a validity guarantee for arbitrary dependence.
- `inference/sharpe/multiple.py:2–7,57–60,209–220`: unqualified FWER-control
  language and “is White's (2000) Reality Check” overstate correspondence.
  In particular the joint maximum is of **standardized** differences, although
  line219 describes `max_k d_k`. Call it a “studentized max-statistic joint
  bootstrap comparison inspired by Reality Check”; describe circular blocks,
  common original studentizers and raw finite-B exceedance proportions.
  Describe Romano–Wolf as fixed-studentizer stepdown architecture, with
  asymptotic control conditional on consistent bootstrap/studentizers; automatic
  best-differential block selection has not been independently justified here.
- `inference/sharpe/deflated.py:59–65,162–170`: cite 2012 for PSR; replace
  “probability that the true Sharpe exceeds” with “asymptotic normal significance
  score for comparing the per-observation Sharpe with a benchmark.” A frequentist
  score is not a posterior probability of a fixed unknown parameter.

## 13. Approval proposal semantics

The companion JSON is a **proposal**, not a registry patch. Its `approved_scope`
is the entire narrowly declared representative configuration to be reviewed.
A full-variant approval is conditional on recording that scope as authoritative;
current broad inherited `python_symbols` and textual selectors alone do not
exclude fallback branches or unrelated helpers. No finite-sample qualification,
evaluation status, adapter coverage or formal verification status changes.

For partial matches retain `source_review: source-unresolved` while recording
`correspondence_status: partial-core-reviewed` in explicit source evidence; do
not introduce a new source_review enum without updating the governed schema and
validators. The existing source-review gap must remain visible.
