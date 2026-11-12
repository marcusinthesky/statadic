# HC3 source-correspondence follow-up

**Recommendation: approve a bounded correspondence for the conventional unweighted HC3 covariance formula**, using the Long–Ervin manuscript as the additional source. Keep all other guarantees separate and leave weighted HC3 and other extensions outside this approval.

## Source inspected

I visually inspected `/tmp/statadic-long-ervin-6.png` and `-7.png`, rendered from `/tmp/statadic-long-ervin-2000.pdf` (SHA-256 `19e239321efa8b6d57c9619d32e8712afeb09606be6f9a078b9fc7e4d8f44493`). Manuscript page 6 defines OLS residuals and leverage; page 7 explicitly displays

`HC3 = (X'X)^-1 X' diag(e_i^2 / (1 - h_ii)^2) X (X'X)^-1`.

It describes this as an approximation to a more complicated jackknife estimator associated with MacKinnon and White, not as the exact centered original jackknife expression. Thus it resolves the earlier concern about directly attributing the uncentered multiplier-one formula to the exact original jackknife. It does not erase that difference.

The coordinator supplied bibliographic metadata: J. Scott Long and Laurie H. Ervin, “Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model,” *The American Statistician* 54(3), 217–224 (2000), DOI `10.1080/00031305.2000.10474549`, author-hosted manuscript `https://nielsen.sites.oasis.unc.edu/soci709/odocs/00TAS.pdf`. My independent source review here concerns the displayed manuscript formula; I did not separately reverify the publication metadata. Use **manuscript page 7**, not an inferred journal-page number, as the source locator.

## Native correspondence

`src/python/packages/statadic/src/statadic/inference/linear_covariance.py:188` dispatches `hc_covariance(fit, kind="HC3")` via coefficient-influence rows, using default `finite_sample_correction=1`. For a full-column-rank unweighted OLS fit, `estimation/linear.py:184` constructs the coefficient map `R^-1 Q' = (X'X)^-1 X'` and `:186` gives the rowwise leverage `h_ii`. `operators/parameters.py:26` forms the row influence `(X'X)^-1 x_i e_i`; `:42` divides each HC3 influence by `1-h_ii` and sums its outer products. This is exactly the displayed matrix sandwich in exact arithmetic. `linear_covariance.py:218` passes the uncentered sum to the covariance carrier; `core/equations.py:486` cancels its internal observation-count scaling, leaving multiplier one for the default correction. No extra jackknife prefactor or residual-influence mean subtraction is applied.

`src/python/apps/studies/src/statadic_studies/adapters/linear.py:18` exercises the actual estimator on intercept plus fixed `linspace(-2,2)` with n=64,256 and unit relative precision; it invokes HC3 without an added correction. The design is full rank, with leverage strictly below one in both sizes. The independent NumPy formula in `tests/inference/test_linear_covariance.py:66` sums row influences divided by `(1-h)^2` and its parameterization includes unweighted HC3. I inspected this existing witness; no tests were run during this follow-up.

## Limits and catalogue handling

- Approve **unweighted OLS, full rank, `0 <= h_ii < 1`, multiplier one, HC3 covariance matrix** only, subject to successful existing checks. This is bounded source correspondence, not a Lean proof, floating-point identity or evidence of universal statistical validity.
- Preserve the current source dossier's honest distinction from the earlier centered MacKinnon–White jackknife. Cite Long–Ervin as the source for the conventional uncentered formula; do not rewrite the earlier paper as defining the same finite-sample expression.
- Do not extend this approval to non-unit/analytic WLS weights, rank-deficient fits, boundary leverage, STATADIC's near-exact-fit availability threshold, optional finite-sample correction, or behavior of any leverage floor. The native code rejects invalid/unit leverage and uses an exact-fit availability threshold; those are implementation policies outside the formula reviewed here.
- Student-t/F or normal/chi-square references in the study are independent reference choices; the manuscript formula does not establish their finite-sample exactness, nor does source review establish coverage, calibration or completed installed-wheel campaigns.
- The current `linear.covariance.hc3` catalogue row also maps `linear.weighted-inference` WLS-HC3. If `verified-method-source` is interpreted for a whole variant rather than the explicit bounded scope, do not let this approval imply weighted-formula review: record a clearly scoped sub-obligation or separately track WLS source correspondence without deleting the stable variant identity. Preserve `evaluation_status` pending unless actual retained evaluation satisfies the separate evidence gate.

## Follow-up: fixed analytic-precision WLS by explicit whitening

The native **fixed-positive-precision WLS HC3 adapter is algebraically the displayed Long–Ervin HC3 formula applied to the whitened regression**, within the assumptions below. Let `D=diag(sqrt(w_i))`, `X*=D X`, `y*=D y`, with fixed finite `w_i>0`, full-column-rank `X*` and `0<=h*_ii<1`. `estimation/linear.py:_linear_fit` constructs these exact whitened design/outcome arrays before QR, while `_assemble_linear_fit` retains raw residuals `e=y-X beta`. `_covariance_state` obtains `C=R^-1 Q'=(X*'X*)^-1 X*'` and `h*_ii=sum_j Q_ij^2`. `linear_covariance.py:_weighted_residuals` supplies `e*=D e` and `operators/parameters.py:coefficient_influence_rows` multiplies column `C_i` by `e*_i`. The HC3 reduction sums `(C_i e*_i/(1-h*_ii))(...)'`; with default correction one this is precisely

`(X*'X*)^-1 X*' diag((e*_i)^2/(1-h*_ii)^2) X* (X*'X*)^-1`.

Equivalently, writing `A=(X' W X)^-1`, each original-coordinate influence is `A x_i w_i e_i` and `h*_ii=w_i x_i' A x_i`. The generic carrier count scaling cancels as in the unweighted review. This is exact algebra over real arithmetic, not an assertion about bitwise floating-point equality.

The actual `linear.weighted-inference` study (`adapters/dependence/linear_panel.py:37–75`) uses a fixed full-rank 128-row design and predetermined positive precision `1/(.3+|x|)^2` or its reciprocal, then calls the actual `hc_covariance(wls,kind="HC3")` with the default multiplier. These are fixed design-based weights, not selected from outcomes. The existing `tests/inference/test_linear_covariance.py:66–132` independently computes the weighted influence/leverage recipe and parameterizes both weighted and unweighted HC3; I inspected it but did not rerun it.

**Updated recommendation:** the same HC3 variant may include the existing fixed-precision WLS mapping under a source scope stating *Long–Ervin formula applied to explicitly whitened X and y*, alongside the direct unweighted match. Do not claim that the paper itself establishes arbitrary weighted/survey/data-adaptive precision inference, that the generic weighted fit has no additional availability limitations, or that the matched and reversed-precision WLS scenarios have exact normal/t/F calibration. Boundary leverage, rank loss, optional correction, near-exact-fit policy, numerical validity, primary-source convention review outside this transformation, and installed-wheel evaluation remain separate. The earlier unweighted-only exclusion above is superseded only for this precisely bounded whitening correspondence.

## Import provenance

Retained from upstream revision 37e199e3e509 (upstream record `.context/plan/research-experience/tasks/studies/tasks/sources/hc3-followup.md` at `37e199e3e509`). The record describes its original review scope; the Statadic import updates namespace spellings and link destinations.
