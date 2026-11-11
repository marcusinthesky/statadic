# Integration verification

The eleven accepted implementation tasks and package-wide cleanup were integrated
with independent GPT-6.1 Terra medium review. Deferred tasks remain blocked,
declined tasks remain closed, and publication remains on hold.

## Final gates

The coordinator's full check did not finish against commit `af1314d`. It stopped at
Lean certification, because `reporting.py` and `inputs.py` changed after the last
inventory refresh. The integration review below established completion.

The final `devenv shell -- just python::statadic::test-artifacts` passed both wheel
installation routes and standalone source-distribution development. Its isolated
typing suite passed 50 tests; seven shape tests were skipped there because they
run separately in the pinned shape environment. The pinned shape gate passed
all seven tests.

An earlier complete main-suite run passed 2,688 tests with 166 skips, followed
by 15 evidence tests, 316 study tests with three skips, and 39 performance tests.
The first documentation run exposed an evidence-manifest assumption that every
claim used `source.review`; the kind-aware fix passed all 11 focused evidence
tests. The subsequent documentation run passed all 86 x64-off tests and the
x64-on native tests, then exposed four strict Sphinx references. Explicit
annotation exceptions and a real weighted-MMD API target corrected those links;
the complete strict site build then passed. The final full check reruns these
gates together rather than treating those partial runs as repository completion.

Independent review covered input roles and precision, live/reloaded parity,
rendering and trusted markup, cleanup callers, numerical witnesses, and formal
claim scope. Review fixes include empty intercept-only mappings, exact confidence
labels and preservation of prediction-grid precision. Focused rendering tests
passed 66 tests after the last fixes; strict typing and architecture also passed.

## Clean breaks and audit

Predictor names replace the old persistence names without aliases. Artifact schema
3 replaces schema 2 without migration. Deprecated solver, balancing, chord-cost,
axiom and covariance aliases were removed along with confirmed unused private
implementations. Public diagnostics and protocol hooks retain their supported
contracts.

The final locked Vulture audit returned its advisory exit 3 with 142 reviewed
findings. The [audit verification](dead-code-verification.md) accounts for retained
findings and removed implementations; no blanket whitelist suppresses results.

## Formal scope

The linear MMD claim is an algorithm refinement over exact reals with equal total
mass. Human mathematical review remains pending; Python, IEEE rounding and XLA
are not formally verified, and no method obligation was closed. The pre-existing
implementation correspondence drift was investigated before refreshing reviewed
pins. The explicit statement-pin gate passed eight mappings, 133 declarations
and 46 families. The public inventory was refreshed after the final source edits.

The existing Poisson manuscript listing was migrated to `save_predictor` and its
live/reloaded smoke check passed. No manuscript analysis or publication was run.

## Integration review

On 2026-11-05 a separate reviewer checked the eleven implementation sub-tasks and
the cleanup against their acceptance criteria at `af1314d`. A scripted probe covered
these areas, and all 29 checks passed:

- precision rejection for arrays, the direct constructor, mappings and frames;
- the prediction-role table on live and reloaded predictors;
- mapping and frame inputs, including lazy and pandas frames;
- predictor names, `require_valid` and LaTeX report output;
- coefficient and prediction-line plots, ribbons and overlays;
- trusted-markup snapshots.

The review made these corrections:

- It refreshed the stale public API inventory, which changed hashes only.
- It made the `require_valid` message keep each warning verbatim and list the
  numerical status in parentheses.
- It fixed two Ruff findings in `tests/api/test_inputs.py`.

The final `devenv shell -- just check` then passed every gate it runs:

| Suite | Result |
| --- | --- |
| Fast package tests | 2,691 passed, 166 skipped |
| Evidence tests | 15 passed |
| Study tests | 316 passed, 3 skipped |
| Performance tests | 39 passed |
| Documentation tests, x64 off | 86 passed |
| Documentation tests, native x64 | 17 passed |

Lean certification, the registry tests, the LaTeX workspace and the record checks
also passed. The review did not rerun the distribution tests; they rely on the
coordinator's earlier `test-artifacts` run.

## Manuscript

The draft in `src/latex/projects/01_statadic` now follows the integrated API:

- The Poisson listing fits from a mapping and predicts the rate without exposure.
  It builds its figure with `prediction_line` and persists with `save_predictor`.
- The manuscript includes the `Report.to_latex` table unchanged.
- It typesets `STATADIC-MMD-002` from the certified export.
- Its discussion no longer lists limitations that this work removed.

The script `just latex::generate 01_statadic` reran against the final source, based on
revision `af1314d`. The retained coverage, agreement and artifact values did not
change. `just latex::check` passed with no box or reference warnings, and the arXiv
bundle recompiled cleanly. Nothing was submitted.
