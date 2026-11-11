# Verification

## Changes and decisions

- `inference/resampling/permutation/kernel.py`: finite observed statistic and all draws required, with a JAX `where`; the existing add-one arithmetic and all alternative comparisons are unchanged. Documented the condition.
- `inference/mixture/_split_runner.py`: finite guard before returning the host p-value.
- `inference/gms.py`: unavailable calibration returns `(nan, nan, False)` for critical value, p-value and rejection; GMS has no existing status field. Checking adjusted draws also catches overflow from eta. Its critical-value consistency check applies only to finite calibration inputs.
- `inference/dyadic/bootstrap.py`: unavailable p-value if coefficient or any supplied bootstrap draw is non-finite; other summary values retain their existing calculations.
- `inference/calibration.py`: coordinator extended ownership after audit found this producer; require all draws finite instead of discarding non-finite draws. Preserve empty-reference NaN and finite arithmetic.
- `inference/sharpe/multiple.py`: joint and stepdown p-values require every raw input, observed difference, bootstrap difference, studentizer and standardized statistic to be finite; the original studentizer must also be positive. Any invalid strategy or replicate invalidates the whole family, without dropping it. Existing finite calculations retain their input dtype and arithmetic.
- `inference/resampling/results.py`: legacy float results and adapter accept NaN as unavailable while retaining validation against infinities and out-of-range values.
- `core/results.py`: document unavailable NaN; no field changes.
- Mirrored tests cover NaN and infinity in observations and draws. The JAX kernels exercise each invalid case alongside a valid lane under eager, jit, vmap and jit-vmap; permutation covers all alternatives. GMS, split and dyadic functions return host values and cannot themselves be transformed. Their test cases exercise host paths. Existing finite-case tests remain in place. The shared calibration property tests now assert the settled all-draw rule rather than the old filtering rule.

The governing decisions and evidence are the task README, release-guarantees handoff and evidence (§ Non-finite p-values); this change is an implementation of that contract, not a new statistical procedure.

## Producer and consumer audit

Searched `statadic/inference` and `statadic/api` for add-one expressions, extreme counts, and count/mean comparisons.

- Fixed here: permutation kernel, mixture split runner, GMS `_finite_mc_calibration`, dyadic `bootstrap_coefficient_inference`, shared `calibration.add_one_p_kernel` and its host adapter.
- Already guarded: `api/association.py` and `api/comparison/_results.py` check observed and all draws and return host `None` when invalid.
- Owned by equivalence lane: `inference/equivalence.py` bootstrap tail.
- Fixed here after coordinator extended scope: `inference/sharpe/multiple.py` Reality Check and Romano–Wolf mean extreme comparisons.
- `inference/multiplicity.py` Holm sorts p-values and cumulates comparisons; Bonferroni compares directly. The current implementation documents that NaNs are never rejected. No sums or subtraction from one of these producer p-values were found. `inference/resampling/results.py` now preserves unavailable NaN in its host result. Sharpe's stepdown calculation ranks statistics and orders its internally computed p-values, rather than consuming these results.

## Checks

All commands run from the root in the pinned shell. Initial sandboxed devenv command could not access the Nix daemon; rerun with approved elevated execution.

- `just python::statadic::test-fresh tests/inference/resampling/test_permutation_contracts.py tests/inference/test_mixture.py tests/inference/test_gms.py tests/inference/test_dyadic_validation.py`: 54 passed in 40.70s.
- `just python::statadic::test-fresh tests/inference/test_calibration.py`: 8 passed in 3.21s.
- `just python::typecheck`: passed, 0 errors in both strict lanes; 7 shape tests passed.
- `just python::architecture`: passed, including 13 architecture unit tests; external dependency warnings for docs packages.
- `just python::lint`: passed (453 files already formatted). An earlier run encountered a concurrent unused import in the equivalence lane and subsequent formatting issue; both were resolved before the passing run.
- Follow-up focused `just python::statadic::test-fresh tests/inference/test_sharpe.py tests/inference/resampling/test_permutation_contracts.py tests/core/test_results.py`: 74 passed in 16.80s. Mixed-lane eager/jit/vmap tests exercise non-finite observed statistics, draws and zero standard errors; raw non-finite input and host NaN/infinity validation are covered. An initial run had 10 failures and 64 passes due to an undefined dtype reference introduced during extraction; corrected before the passing run. Initial lint and typecheck caught the same reference.
- Follow-up `just python::lint`, `just python::typecheck`, and `just python::architecture` passed (453 formatted files; 7 shape tests; 13 architecture tests). Architecture emitted the same docs dependency warnings.
- Full suite, sync, inventory refresh, certify and `just check` were reserved for coordinator; not run. The inherited GLM timeout remains an open follow-up.

## Requests

Coordinator: update shared CHANGELOG/docs and inventory as appropriate for the changed shared calibration, Sharpe and host-result contracts. No status field changed here.

## Review (multiplicity lane, 2026-11-06)

Two findings require fixes before acceptance:

1. **Dyadic public report discards failed draws before the new guard.** `inference/dyadic/report.py:69-76` selects `series[valid & jnp.isfinite(series)]` before calling `bootstrap_coefficient_inference`, so one failed or non-finite scheduled replicate is omitted and the p-value may be finite with a reduced denominator. The producer's new guard cannot see the omitted draw; the new direct helper test does not exercise this public path. `api/dyadic.py:163-169` exposes this report, and `api/dyadic.py:221` / `inference/dyadic/bootstrap.py:401` also expose explicitly filtered draws, whose downstream use should be assessed. Preserve failure-inclusive validity in the report and test a mixed valid/invalid schedule through its public report path without dropping the invalid replicate from calibration.
2. **Paired Sharpe's extreme-count producer is unguarded and absent from the audit.** `inference/sharpe/difference.py:54-84` computes an exceedance proportion after `sharpe_ratio_axis`, whose fallback can mask invalid raw inputs. A pinned focused reproduction called `_sharpe_difference_kernel` with `returns_a=[nan, .2, -.1, .4]`, `returns_b=[.4, -.1, .2, .5]`, and index rows `[0,1,2,3]`, `[1,2,3,0]`, `[2,2,3,1]`; it returned statistic `-3.0496507` and **p-value 0.0**. Public kernel and host callers are at lines 103 and 168. Add guards over required raw inputs and computed statistics/draws, preserve finite-case arithmetic, and test eager/jit/vmap mixed lanes plus the public host path. This is an extreme-count route rather than an add-one route, and is not covered by the multi-Sharpe tests.

Other reviewed paths: permutation kernel and shared calibration retain complete finite denominators and use JAX `where`; their mixed-lane tests cover alternatives, NaN/infinite observations and draws, eager, jit and vmap. Split and GMS guard observed and all generated draws before host reporting; host adapter keeps NaN while still rejecting infinities. Multi-Sharpe guards raw panels and derived family statistics, with mixed-lane transformation coverage. Its host path with NaN strategy input was checked with automatic and fixed block lengths: both returned NaN joint and stepdown p-values. No scientific files changed by this review. Read-only commands: `rg` caller/producer audit and diffs; pinned `devenv shell -- bash -c 'cd src/python && uv run --locked --package statadic python /tmp/statadic-review-sharpe.py'` (exit 0, two cases); same command with `/tmp/statadic-review-difference.py` (exit 0, one case, demonstrates finding 2). No broad suite run.

### Independent re-review of fixes

Both original findings are resolved within the inspected paths. The dyadic report now makes the p-value NaN when any scheduled solve or coefficient is invalid, while keeping successful-draw diagnostics; its public regression covers a mixed schedule. The paired Sharpe guard includes both raw inputs and derived calibration quantities, and its public regression covers NaN and both infinities with eager, JIT and vmap checks. Finite-case arithmetic remains unchanged. The two explicitly named filtered-draw APIs remain diagnostic interfaces, with the external recalibration limitation recorded below. No further finding in this reviewed scope.

Independent pinned checks: the original paired Sharpe reproduction now returns p-value NaN (exit 0); `devenv shell -- just python::statadic::test-fresh tests/inference/test_sharpe.py::test_paired_sharpe_public_paths_reject_nonfinite_input tests/inference/test_dyadic.py::test_public_report_preserves_failed_schedule_draw_in_pvalue` passed 4 tests in 8.28s. No broad suite was run in this re-review.

## Cross-review fixes (2026-11-06)

- `inference/dyadic/report.py`: retain the existing successful-draw SE and interval diagnostics, but replace the p-value with NaN if any scheduled solve fails or its coefficient is non-finite. The actual public report regression supplies a six-draw node schedule including a failed solve, asserts a NaN p-value, and retains the draw diagnostics. The explicitly named `api/dyadic.py:valid_draws` and `inference/dyadic/bootstrap.py:node_bootstrap_coefficient` intentionally expose filtered diagnostic arrays; no in-repository report recalibrates them, but external callers can do so and should retain the schedule/validity evidence.
- `inference/sharpe/difference.py`: guard both raw return series and all computed observed, null, centered and studentizing values, avoiding `sharpe_ratio_axis`'s degenerate-series fallback hiding NaN/inf. Public host and keyed kernel tests exercise invalid and valid mixed lanes under eager, jit and vmap; finite arithmetic is unchanged.
- `devenv shell -- just python::statadic::test-fresh tests/inference/test_sharpe.py tests/inference/test_dyadic.py tests/inference/test_dyadic_validation.py`: 78 passed in 31.97s.
- `devenv shell -- just python::typecheck`: passed; 0 errors in both strict lanes, 7 shape tests passed. `devenv shell -- just python::architecture`: passed, 13 tests, the same five docs dependency warnings.
- `devenv shell -- just python::lint`: first failed on our magic literal and a concurrent equivalence test line length, both resolved; rerun passed (453 files already formatted).

Coordinator: include the paired Sharpe and dyadic report behavior in shared release notes. Independent review requested; status left unchanged.
