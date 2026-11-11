# Primary-source correspondence decisions

Reviewed 2026-11-04. This record approves only the representative formula scope
listed below. It does not establish an implementation refinement proof,
finite-sample calibration, completed evaluation or an entire family's validity.
The catalogue is authoritative for applied statuses; the adjacent proposal remains
an unapplied historical proposal rather than a second source of live state.

| Variant | Approved scope | Primary locator |
| --- | --- | --- |
| Fixed-bandwidth Newey–West | Raw moment rows, Bartlett weights, denominator T and symmetric lag terms; no centering or finite-sample multiplier | Newey–West (1987), pp. 703–704, equations 4–5 |
| Fixed-bandwidth Hall centering | Same fixed Bartlett operator after subtracting the sample moment mean | Hall (2000), p. 1521, equation 9; Newey–West equations 4–5 |
| Probabilistic Sharpe ratio | Supplied-score formula, n > 1, positive variance denominator, non-excess kurtosis and original-frequency Sharpe | Bailey–López de Prado (2012), author manuscript section 3, printed p. 9, equation 11; variance in section 2.5 |
| Deflated Sharpe ratio | PSR evaluated at the expected maximum, N > 1, positive supplied cross-trial dispersion and supplied independent/effective-independent trial count | Bailey–López de Prado (2014), author manuscript pp. 7–8, equations 1–2; original PSR equation 11 |
| Fama–MacBeth point estimate | Arithmetic average of fully identified periodwise cross-sectional coefficient vectors | Fama–MacBeth (1973), pp. 616–619 |
| One-way DISCO/PERMANOVA label permutation | Raw Euclidean DISCO or squared-Euclidean PERMANOVA, V self-pair denominators, exchangeable pooled labels and plus-one randomized upper-tail p-value | Rizzo–Székely (2010), equations 2.5–2.6, sections 3.1–3.2; Anderson (2001), equations 1–4 |
| Unweighted OLS point estimate | Full-rank least-squares mathematical target; excludes numerical implementation equivalence | MacKinnon–White (1985), author working paper 537, section 2, equations 1–3 |
| Classical OLS covariance | All weights one, default multiplier one, RSS/(n−k) residual variance under homoskedasticity | Same paper, section 2, equation 4 |
| Conventional HC3 | Full-rank, multiplier-one OLS or fixed-positive-precision WLS via explicit whitening; leverage below one | Long–Ervin (2000), author-linked manuscript section 2, pp. 6–7; independent native whitening review |

These scopes exclude automatic bandwidth-selection claims, full GMM inference,
degenerate Sharpe fallbacks, posterior interpretations, automatic effective-trial
estimation, selective failed-period handling and Fama–MacBeth HAC inference. The
group approval excludes PERMDISP, arbitrary multifactor designs and learned
geometry. Exact catalogue symbols must point to the appropriate producer.

The HC3 source correction remains important: the original jackknife has both a
prefactor and rank-one recentering subtraction. The later Long–Ervin manuscript
explicitly defines the conventional uncentered sandwich, and independent review
matched the native fixed-precision whitening transformation. The companion source
resolves that bounded formula correspondence without changing the original paper
or claiming reference-law calibration. See [the follow-up](hc3-followup.md).

## Partial reviews that remain unresolved

- Automatic Newey–West/Hall: the sample-size-only native rule is not the cited
  data-driven plug-in selector. Only the fixed operator's core was reviewed.
- Panel entity/time/two-way clustering: independent-cluster sandwich and
  inclusion–exclusion cores were reviewed; inherited finite-G adjustments,
  absorbed effects and reference laws remain outside this review.
- Fama–MacBeth HAC: only the bandwidth-zero debiased sample-time baseline was
  reviewed. Positive-bandwidth and missing-period conventions remain open.
- Driscoll–Kraay: the balanced Bartlett core was compared with the 1995 primary
  precursor; the final 1998 paper and complete native scope remain open.
- Romano–Wolf: the fixed-original-studentizer architecture is supported by the
  2005 paper's footnote 22. Automatic block choice and full calibration remain open.
- Paired Sharpe difference: native centered circular-block differences with one
  bootstrap SD and raw finite-B exceedance differ from the published recomputed
  studentizer and plus-one construction. Related context is not exact approval.
- White reality check: native standardized maximum with circular blocks differs
  from the original raw maximum with stationary bootstrap.
- PERMDISP and Mantel: canonical bibliographic metadata was verified. Accessible
  abstracts/metadata alone do not approve modern residual, finite-sample,
  rank-correlation or node-bootstrap conventions.

Primary URLs, formula comparisons and further exclusions are retained in
[the evidence dossier](evidence.md). The Hall and Sharpe code docstrings were
corrected to match their actual implementation; numerical algorithms did not
change during this source review. Source-unresolved variants remain visible and
cannot be promoted to release qualification by a bibliography entry.
