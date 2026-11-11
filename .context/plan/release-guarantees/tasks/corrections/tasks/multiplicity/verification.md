# Multiplicity verification

## Review

**Final re-review:** The empty-family finding below is resolved. Bonferroni
returns an empty Boolean mask before division; Holm retains the same result.
Tests cover both under eager, JIT and vmap, and confirm that empty input does not
bypass refusal of BH or an unknown method. Independent rerun of `devenv shell --
just python::statadic::test-fresh tests/inference/test_multiplicity.py`: 14 passed in
3.80 seconds. No remaining finding from this re-review.

- Finding (edge case): `multiplicity.py` Bonferroni divides the Python scalar
  `alpha` by `p.shape[0]`, so an empty `(0,)` input raises `ZeroDivisionError`,
  while Holm returns an empty mask. The documented shape `(M,)` does not exclude
  zero. Decide on a consistent empty-family contract and test both methods and
  transformed execution if supported.
- Holm sorts ascending, uses `alpha/(M-i)`, stops after the first failure via
  cumulative product and scatters to the original order. Ties receive the same
  verdict; NaNs sort last and are not rejected. Bonferroni uses the full family
  count, including NaNs. The documented super-uniform true-null condition and BH
  distinction match the methods; equivalence bootstrap validity is explicitly
  unproved. The oracle, 40,000-outcome grid, ties, NaNs, JIT and vmap tests exercise
  the requested contracts. The migrated scientific and studies callers use the
  new methods; stale catalogue and documentation remain coordinator requests.
- Independent focused check: `devenv shell -- just python::statadic::test-fresh
  tests/inference/test_multiplicity.py` passed: 10 tests in 3.15 seconds. No full
  checks or shared refreshes were run in review. Publisher metadata remains for
  the independent sources lane.

## Changes and decisions

- `statadic/inference/multiplicity.py`: Holm is default; Bonferroni is available. Both guarantee familywise error at most alpha under arbitrary dependence if each true-null p-value is super-uniform. `equivalence_test` bootstrap p-value validity is unproved. NaNs are not rejected. BH remains refused because FDR control does not generally imply FWER control.
- Removed adaptive Bonferroni, k_minus_1 and n_groups. Röhmel (2011), [publisher abstract in PubMed](https://pubmed.ncbi.nlm.nih.gov/22069200/), reports the k−1 rule is too liberal for k ≥ 4; it works for three groups in the pairwise TOST setting. The generic API has no means to enforce the relevant pairwise structure. Lauzon and Caffo, [Easy Multiplicity Control in Equivalence Testing Using Two One-Sided Tests](https://pubmed.ncbi.nlm.nih.gov/20046823/), *The American Statistician* 63(2), 147–154 (2009), DOI 10.1198/tast.2009.0029, proposed the original rule. The [publisher issue](https://www.tandfonline.com/toc/utas20/67/2) lists Caffo, Lauzon and Röhmel's 2013 correction; secondary metadata gives pages 115–116. DOI landing pages for the original and Röhmel paper were inaccessible, so publisher metadata verification is incomplete.
- Holm (1979) theorem 1, page 67 and its Boole inequality proof give strong FWER under marginal validity; page 67 compares single-step and sequential Bonferroni. See [reference](../../../../../../reference/holm-1979/README.md). Article scan metadata checked; publisher metadata verification remains outstanding.
- `tests/inference/test_multiplicity.py`: exhaustive two-null grid, independent Hypothesis Holm oracle, compilation and batching parity, NaN and unsupported-method tests; removed obsolete adaptive assertions. Deleted TestMultiplicity from `test_equivalence.py`, retaining the BH guard in the dedicated tests. `tests/qualification/test_returns.py` parametrizes the two new methods.
- Studies definitions, adapter and tests now use Bonferroni and Holm and disclose bootstrap validity is unresolved; decisions are tested separately on shared p-values.

## Coordinator requests

- Catalogue: remove `equivalence.familywise.k-minus-1` and `equivalence.familywise.adaptive-bonferroni`; add `equivalence.familywise.bonferroni` and `equivalence.familywise.holm`, both citing `holm-1979`.
- CHANGELOG: replace k_minus_1 and adaptive_bonferroni with bonferroni and holm, default to holm, remove n_groups.
- `package-development.md:295` and `method-catalogue.md:65`: “Holm (default) and Bonferroni control familywise error at alpha under arbitrary dependence when every true-null p-value is valid; the equivalence bootstrap p-value has not been shown valid.”
- `known-limitations.md`: “Familywise control requires valid true-null input p-values; validity of the equivalence bootstrap p-value is not established.”
- Equivalence lane: replace multiplicity references in `equivalence.py` and `mixture/contracts.py` with “Holm or Bonferroni familywise correction requires valid true-null p-values.” Removed the now-unused multiplicity import in test_equivalence.py with coordinator-granted ownership.
- Owner flags: default changed to Holm; k_minus_1 and n_groups removed because the old rule fails at k ≥ 4 and the general API cannot establish pairwise structure.

## Commands

All scientific checks used `devenv shell --`. First sandboxed invocation of reference-new failed to connect to the Nix daemon; escalated invocation succeeded. `reference-new holm-1979` created the reference. `python::statadic::test-fresh tests/inference/test_multiplicity.py tests/inference/test_equivalence.py`: 19 passed. After final test edits, multiplicity alone: 10 passed. `python::typecheck`: passed, 0 type errors, shape suite 7 passed. `python::architecture`: passed, 13 tests. `python::lint`: initially found own style errors, fixed; subsequent run failed on concurrently edited other-lane files and the equivalence import, reported to coordinator. Task-owned package files pass scoped Ruff lint; scoped formatting was applied. Final multiplicity test rerun after refining the BH message assertion: 10 passed. `python::studies::check`: lint and formatting passed; 316 tests passed, 3 skipped, 3 warnings; catalogue validation failed on stale `equivalence.familywise.k-minus-1` as expected pending coordinator update. No full check, sync, certify, pin or inventory refresh run by this lane, per handoff. Publisher verification remains open.

## Empty-family review fix

Both supported methods now return an empty Boolean mask for an empty family; method validation still rejects BH and unknown methods on empty input. Added eager, JIT and vmap checks for both methods and validation tests. Pinned scoped Ruff format/check passed; `devenv shell -- just python::statadic::test-fresh tests/inference/test_multiplicity.py` passed 14 tests in 2.94s.
