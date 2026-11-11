# Bounded linear source proposals

Read-only audit, 2026-11-04. The reviewer made no shared source changes. The
coordinator subsequently applied the bounded decisions in
[the decision record](decisions.md). This is supporting review evidence;
the catalogue owns current status.

## Ready for coordinator review

- `linear.ols-wls.ols`: canonical `mackinnon-white-1985`, author working-paper Section2 equations(1)-(3), printed p2; independently supported by `stepniak-gauss-markov-2010`, Theorems3.2/4.1. Restrict approval to the full-rank unweighted mathematical estimator, not its floating-point implementation or deficient-rank behavior.
- `linear.covariance.classical`: same author source, equation(4), printed p2. STATADIC default unweighted covariance equals RSS/(n-p) times inverse XTX. Scope excludes arbitrary extra multipliers and separately unreviewed analytic-precision extension.

Primary full texts: [MacKinnon–White](http://qed.econ.queensu.ca/working_papers/papers/qed_wp_537.pdf), [Stępniak](https://link.springer.com/content/pdf/10.1155/2010/259672.pdf). These are source-formula reviews, not historical priority claims or numerical verification.

## HC3 correction required

Exact variant: `linear.covariance.hc3`. Existing approval mentions an omitted historical prefactor but omits the original rank-one recentering subtraction. The original1985 centered jackknife (equations12/13, pp3-4) is not identical to STATADIC's conventional uncentered squared-leverage sandwich. Preserve the reviewed leverage correction and asymptotic relationship; leave full exact-variant correspondence unresolved pending a later primary conventional-HC3 definition. Full proposed evidence wording is in the JSON.

[Long–Ervin publisher metadata](https://www.tandfonline.com/doi/abs/10.1080/00031305.2000.10474549) and [author research page](https://jslsoc.sitehost.iu.edu/research_hccm.htm) identify a promising later primary source: The American Statistician54(3),217–224(2000), DOI10.1080/00031305.2000.10474549. A retrieved manuscript PDF has garbled text; its exact formula has not yet been visually reviewed, so it is not approved.

## Explicit blockers

Analytic-precision WLS needs its own weighted formula/normalization review. The generic normal/t/F/chi-square providers accept many fitted uncertainty carriers; a Gaussian classical-OLS law cannot approve every such use. Student1908 concerns an iid-normal mean, and the retrieved Fisher1925 text only exposed the opening mean/variance discussion. This wave therefore proposes no blanket reference-family approvals. Catalogue execution, qualification and canonical-record creation remain independent steps.

[The historical proposal](linear-proposal.json) provides exact variant IDs,
bounded approval evidence, symbol restrictions and blockers.
