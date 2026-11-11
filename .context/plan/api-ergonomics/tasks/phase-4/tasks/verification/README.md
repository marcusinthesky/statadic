---
type: task
title: Verify Phase 4 integration and repairs
description: Verify Phase 4 integration and repairs
links: []
status: done
---

# Verify Phase 4 integration and repairs

## Goal

Verify the implemented repairs and bounded family extensions without rerunning
the complete suite. The user requested reliance on cached selection; `just check`
and `test-fresh` are deliberately omitted. Use focused native checks in both
precisions and independent numerical witnesses for new statistical calculations.

## Acceptance

- [x] Cross-review numerical and host contracts across worker ownership.
- [x] Validate new family studies in separate float32/float64 processes.
- [x] Preserve exact study scope and pending qualification in the catalogue.
- [x] Complete package static/type/build and strict documentation checks.
- [x] Record external validation gaps without claiming hosted CI success.

## Scope and cross-review

Three GPT-6.1 Terra medium workers owned disjoint numerical repairs, artifacts
and likelihoods, and studies/IV/release provisioning. The coordinator integrated
the catalogue, performance policy, documentation and examples. Cross-review found
and corrected Gamma likelihood cancellation at tiny fixed dispersion, NB2
cancellation near its Poisson limit, and singular estimated GMM weights that
could occasionally pass in batched float32. Estimated NB2 boundary availability
remains distinct from fixed small dispersion. No automatic Poisson fallback,
universal distribution carrier or new checkpoint codecs were added.

Survival means preserve nonexistence and numerical-unavailability cases.
Log-logistic RMST uses bounded successive quadrature checks, not a certified
error bound. Existing extreme Weibull RMST underflow is documented, not claimed
fixed by this extension. IV point-fit availability and iterated convergence
remain distinct; the iterated study gates convergence explicitly. Hansen-J
requires the actual efficient-weight provenance and an explicit assumption.

## Focused evidence

- Numerical repair regressions: 14 focused panel/MMD/transport checks passed.
  Artifact checks cover typed float64/int64/uint64 input before coercion and
  held-out prediction parity for inactive nonfinite landmark coordinates.
- New IV source and host checks: 22 passed per precision, including direct
  JIT/vmap singular-cluster and full-rank controls, actual-weight covariance,
  Hansen provenance, malformed controls and injected-result schemas.
- Likelihood checks include independent SciPy fits, joint score/Hessian
  covariance, fixed/estimated dispersion, probit tails and transformation
  support. Five final Gamma checks passed against independent Stirling and
  digamma/polygamma witnesses around the normalizer branch boundary.
- New studies: 16 checks passed per precision across all 13 declared scenario
  pairs; a further guard rejects unknown study/scenario applicability. These
  smoke checks do not establish statistical qualification.
- Evidence/catalogue/IV-description regressions: 43 passed; release evidence
  checks: 11 passed. Performance identity and immutable-output checks: 21
  passed, two unrelated tests deselected.
- Eight documentation examples executed in both precision processes. The
  onboarding budget remains below its frozen cap. Package architecture and
  infrastructure document/YAML/TOML/JSON checks passed.
- Final package lint/format, strict Pyrefly, seven shape checks and focused
  layout validation passed (225 source modules, 149 test files). Both sdist and
  wheel built offline. Module permissions were narrowed to actual dependencies.
- Strict Sphinx/Lean certification/blueprint build and all executed marimo
  snapshots passed. Browser navigation/search/responsive checks passed before a
  final catalogue-only scope wording correction. The site was rebuilt for fresh
  provenance. First-use text is 1,451 words against the 1,541-word cap.

These are overlapping focused selections, not a summed full-suite result.
New catalogue variants remain `source-unresolved` and
`pending-retained-campaign`; unmapped capabilities remain uncovered.

## Limits and retained evidence

The original hosted layout-tool failures remain unexplained. A clean-cache local
wheel build succeeded but cannot establish cold x86-64/ARM64 hosted success.
Release reference provisioning is fixed; a clean hosted preflight/archive run
is still pending. Five additional findings mentioned by the supplied review were
not supplied and are not claimed resolved. No retained scientific qualification
campaign, commit or push was performed.

Analysis v1 publication exports are rejected by the new paired-analysis verifier.
For the local documentation build, original ignored exports are retained under
`build/studies/analysis-v1/`; analysis-v2 exports are made at new destinations
from unchanged frozen trials. Missing frozen comparison groups yield no paired
contrasts. This reanalysis does not upgrade source or release evidence.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
