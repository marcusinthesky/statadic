# Equivalence verification

## Review follow-up

An independent review identified that a finite percentile interval can omit a
non-finite bootstrap draw. Both the primary verdict and curve now also require
every bootstrap replicate to be finite. A regression uses 99 zeros and one
positive infinity, verifies that the 95% percentile endpoints are finite, and
checks NaN p-value and false verdicts. The result contract documents this rule.

After the fix, `devenv shell -- just python::statadic::test-fresh
tests/inference/test_equivalence.py` passed 18 tests; `just python::typecheck`
passed (7 shape tests), `just python::architecture` passed (13 tests, five
documentation dependency warnings), and `just python::lint` passed (453 files
formatted). The first lint attempt found a long line here and a concurrent
`test_dyadic.py` magic-value violation; both were resolved before the passing run.

- `equivalence.py`: removed unsupported guarantee and unrelated citations; describe
  the percentile p-value and interval verdict as approximations. Non-finite δ or
  any replicate returns NaN, including an empty sample with non-finite δ. Both
  interval endpoints and the margin must be finite for either the main verdict or
  a curve verdict to be true. Previously negative infinity could pass the upper
  endpoint comparison, and the lower endpoint was not checked.
- `mixture/contracts.py`: align the retained result documentation and multiplicity
  wording with Holm (default) and Bonferroni.
- `tests/inference/test_equivalence.py`: test NaN and both infinities as replicates
  and margins, including an empty bootstrap; test non-finite lower and upper
  interval endpoints for both the verdict and curve.
- `.context/reference/liu-gandy-2026/README.md`: source-checked bibliographic
  record. [Theorem 8 and Lemma 7](https://arxiv.org/html/2603.10886v2) concern
  the asymptotic multiplier bootstrap for bounded-kernel MMD with fixed laws.
  Theorem 8 uses an ideal conditional quantile; it does not cover selected weights
  or this energy percentile bootstrap. The other three references were removed
  because the revised text does not rely on their specific results.

Coordinator requests: update the equivalence method catalogue entries to cite
`liu-gandy-2026` as related work, without claiming it validates STATADIC's procedure;
update user-facing documentation and CHANGELOG for the approximation and NaN
behavior. Sync the new record and refresh inventories as assigned.

Commands from the root, through `devenv shell --`:

- `just reference-new liu-gandy-2026 'Kernel Tests of Equivalence'`: passed
  after the initial sandboxed attempt failed to connect to the Nix daemon socket.
- `just python::statadic::test-fresh tests/inference/test_equivalence.py`: 17 passed
  (slow tests excluded by recipe).
- `just python::typecheck`: passed; 0 errors in each type lane, 7 shape tests passed.
- `just python::architecture`: passed; 13 architecture tests passed, with five
  existing external-dependency warnings for documentation packages.
- `just python::lint`: initial failure for import order; second failure for format;
  after corrections passed, 453 files already formatted.

Full checks, sync, inventory refresh and status changes are coordinator-owned and
were not run. Baseline was interrupted and the GLM timeout remains an open follow-up.

## Review (nonfinite-pvalues lane, 2026-11-06)

Initial review found that finite percentile endpoints could coexist with a non-finite replicate and still produce a positive equivalence verdict and curve, even while the p-value was NaN. The author subsequently added an all-replicate finite guard to both verdict paths and a regression with 99 zero replicates and one positive infinity, whose 2.5% and 97.5% percentiles are both finite. Reviewed the corrected implementation: the p-value, main verdict and curve all check the complete replicate vector; non-finite margins and endpoints are also excluded. The fixed empty-reference behavior returns 1 for finite margins, and NaN for non-finite margins.

The revised guarantee accurately describes a selected-mixture percentile-bootstrap approximation without a proved size result; it distinguishes Liu and Gandy Theorem 8's bounded-kernel MMD and multiplier bootstrap from this procedure, consistent with the handoff evidence and the lane's source record. No remaining blocking finding in the reviewed scope. Review was read-only except for this section; no scientific files or shared statuses were changed. Independently ran `devenv shell -- just python::statadic::test-fresh tests/inference/test_equivalence.py`: 18 passed in 9.88s (slow tests excluded by recipe).
