# Finite-moment MMD integration review

## Provenance and selection

The input was `.tmp/statadic_literature_lean_52268aa.zip`, targeting repository
commit `52268aa5078d94c9bf8b092c7655fbf3ae54031a`. Archive SHA-256:
`3c9c8fb693758ff0eccc4d4dc2558fe2f5f3a4380d733dbd8e3753821dc9822a`.
The archive checksum manifest and six pinned upstream file hashes matched.
`START_HERE.html`, formal scope, integration tools, all seven mathematical
modules and native repository requirements were reviewed. The 144 bundle tests
were independently reproduced in STATADIC's locked uv environment.

The user selected published results first, an MMD-only first batch, and only
references used by accepted claims. The complete annotated bibliography and
46-family method map are retained as [research records](research/papers.json)
and [research mapping](research/method-map.json). These are imported research
evidence, not canonical bibliography or proof-status records. Their original
citation keys and review-depth notes are preserved. The reference dry-run found
60 new records and nine existing matches; 63 of 69 had search-record review
depth. No bulk reference import was performed.

The bibliography SHA-256 is
`7fae8c73d8d9889cad677529c871969018b6b2c042b2d21a67b3a78aed4b7a00`;
the mapping SHA-256 is
`cd7538fcb74d05597dd35f1146fae2cc8f30cfe96f4aee229ddc166dba632b18`.

## Source and implementation relationship

Gretton et al. (2012), Lemma 6, equation (3), p. 728, uses unequal sample
sizes, off-diagonal within-group sums and all cross-group pairs. Its proof's
empirical-estimator step, p. 729, is the precise source relationship for
`STATADIC-MMD-001`. The integrating agent compared that statement and
proof step with the finite-law algebra. Human semantic review remains pending.
The native claim registry owns the assumptions and limitations; the existing
Gretton reference now has a source dossier.

Common pair expectations are premises, rather than conclusions derived from
iid sampling. The formal target is the supplied moment expression, without
an RKHS construction or a general population-MMD interpretation. The V-bias
calculation is our supporting diagonal-counting consequence of equation (5),
p. 729, with no independent principal label or bias-sign claim.

STATADIC's scalar and packed matrix reductions average forward and reverse
cross-kernel blocks. Kernel symmetry relates that exact-real expression to
the formal single cross block. The sampling assumptions concern separate
sample groups, not the packed matrix's self-comparison diagonal. Input-derived
reference points, adaptive kernel choices, floating-point evaluation and
calibration need separate correspondence arguments.

## Deferred candidates

- Diagonal WLS is a promising next specialization, but needs exact published
  statement/proof matching and separate construction of its supplied weights.
- The intersection–union lemma is a useful generic step, not a proof of the
  current one-sided bootstrap equivalence API.
- Familywise budgeting assumes calibrated marginal bounds and a deterministic
  total budget. It does not verify group-count or adaptive corrections.
- Omitted-signal bias, scalar shrinkage and the information inequality remain
  supporting mathematics; they do not prove general misspecification, matrix
  shrinkage or Cramér–Rao results.

The other modules, research aggregator, candidate claim schema and verification
scripts were not installed. Native certification owns checked status.

The archive candidate `STATADIC-RESEARCH-MMD-U-001` maps to native claim
`STATADIC-MMD-001`: the native schema requires `STATADIC-<FAMILY>-<NNN>`. The Lean
declaration name is preserved; the registry schema was not broadened.

## Compiler corrections

The unchanged input failed compilation: finite expectation order lemmas lacked
their Mathlib import. Adding the finite-sum order import repairs that dependency.
The MMD pair-sum calculation also needed a final ring normalization; redundant
tactic combinators were removed to satisfy the pinned warning-as-error policy.
No assumptions were weakened or admissions introduced.

The deferred omitted-signal module separately uses Lean's reserved `λ` token as
a type variable. It remains uninstalled and uncorrected.

## Verification

- `just lean::build` passed with Lean 4.35.0 and warnings treated as errors.
- Native statement pinning and certification passed: seven principal mappings,
  124 first-party declarations and 46 method families. The two added modules
  account for 36 audited declarations, including private and generated names;
  they contain 15 public theorem bodies, one private theorem helper and five
  public definitions. Only `propext`, `Classical.choice` and `Quot.sound` occur
  in the audited axiom closures.
- `just lean::check` passed, including blueprint HTML/PDF rendering and all
  26 registry, integrity and inventory regression tests.
- `just python::statadic::test-fresh
  tests/discrepancy/mmd/test_independent_reference.py` passed: 13 tests, including
  unequal sample sizes, signed U reductions and rooted V comparisons.
- `just lean::programme-check` returned nonzero as intended. The new claim maps
  no completed obligations, and `programme_complete` remains false.
- `devenv shell -- just check` passed. The complete scientific fast suite
  reported 2,677 passed, 161 skipped and five warnings (688.27 seconds).
  Documentation reported 64 passed; the separate x64 native process reported
  14 passed and 50 deselected. Strict lint/type checks, wheel/sdist builds,
  executable examples, Sphinx, notebook snapshots, Lean blueprint HTML/PDF,
  LaTeX and agent-configuration checks passed.
- The six existing claim records and their statement pins are unchanged;
  the method registry and its obligations are unchanged. The retained research
  JSON files match their archive inputs byte for byte.

These checks ran on 1 November 2026 in the pinned devenv environment. Logs are
retained under `/tmp/statadic-literature-review.6f5NLn/`: `root-check.log`,
`lean-check.log`, `statement-pin.log`, `mmd-reference-tests.log` and
`programme-check.log`. Current proof status remains generated by the native
certification, rather than by this historical execution record.

The initial review encountered a missing `verification.md` link in the separate
release-experience plan. That link exists by the integration check, whose
Markdown and local-link gate now passes; this task did not change that plan.
The input HTML was read directly; a browser render could not run because
Chromium is unavailable. No all-method obligation is closed by this integration.
Dedicated statistical qualification, optional external-reference and GPU lanes
were not rerun. Human semantic review and implementation refinement remain open.
