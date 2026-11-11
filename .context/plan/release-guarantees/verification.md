# Release-guarantees coordination and checkpoints

The coordinator owns this record. Task workers write their own `verification.md` beside their task README.

## Baseline

Command: `devenv shell -- just check`, with stdout and stderr redirected to
`/tmp/statadic-release-guarantees-baseline.log`.

**Failed and subsequently interrupted; not a completed or passing baseline.**
The main Python suite reported a failure at 37% and continued running. After
observing the failure and substantial machine slowdown, the coordinator sent
SIGINT only to the pytest controller whose stdout was this baseline log. The
run ended with exit code 2. No workers started and no scientific files changed.

The failure was
`tests/estimation/test_glm.py::test_tiny_complete_separation_is_rejected_without_the_annotation_hook`
(`src/python/packages/statadic/tests/estimation/test_glm.py:513`). Its child Python
process exceeded the existing 60-second timeout at line 542, raising
`subprocess.TimeoutExpired`; no failed numerical assertion was reported.
This does not establish the cause of the timeout or that the test is flaky.

Recorded outcomes, with line locators in the baseline log:

- Lean certification passed: 8 literature mappings, 133 compiled declarations,
  46 method families (line 44).
- Registry tests passed: 36 tests (line 236).
- Architecture regression tests passed: 13 tests (line 305); architecture,
  lint, formatting and type checks before the main suite also passed.
- Isolated shape tests passed: 7 tests (line 330).
- Timeout diagnostic: lines 416–422; failed test identity: line 455.
- Interrupted main-suite summary: **1 failed, 1,478 passed, 47 warnings** in
  1,269.33 seconds (line 459). These are partial-run counts, not full-suite counts.

Per the baseline stop rule, implementation and delegation stopped. No retry,
test relaxation, test deletion or acceptance waiver was made.

Initial `git status --short --branch`:

```text
## feat-docs...origin/feat-docs [ahead 3]
 M .context/plan/formal-literature/tasks/literature-proofs/README.md
 M .context/plan/research-release/README.md
 M .context/reference/gretton-etal-2012/README.md
 M .context/reference/hemerik-goeman-2018/README.md
 M .context/reference/long-ervin-2000/README.md
 M .context/reference/mackinnon-white-1985/README.md
 M .context/reference/phipson-smyth-2010/README.md
 M .context/reference/sejdinovic-equivalence-2013/README.md
 M .context/reference/stepniak-gauss-markov-2010/README.md
 M .context/reference/szekely-rizzo-2013/README.md
 M .context/reference/white-1980/README.md
 M ROADMAP.md
?? .context/plan/release-guarantees/
```

These changes predate implementation and are preserved. Execution uses seven
task-specific Terra workers scheduled with at most three workers active beside
the coordinator, within the session concurrency limit.

## Checkpoint 1: wave 1 frozen and integrated

The owner explicitly instructed continuation despite the failed/interrupted
baseline, with the GLM timeout retained for a follow-up. This supersedes the
baseline stop gate only; it does not turn the baseline into a pass or permit
weakening tests. Lean proof lanes are prioritized, followed by corrections,
independent reviews and integration. The original failure log remains intact.

Started Terra high-effort workers for sampled permutations, linear/WLS covariance
and Euclidean negative type. At most three workers run beside the coordinator;
weighted MMD and the three correction lanes are queued. All workers use the shared
Lean build lock and leave shared files and status changes to the coordinator.

Negative-type lane is frozen after a successful focused module build and six
declaration axiom checks (only the three permitted axioms). Provenance and the
unresolved redistribution owner flag are recorded in its verification record.
Weighted-MMD high-effort worker has started in the freed slot. Sampled and
linear/WLS workers continue; no repository-wide acceptance is claimed yet.

All four proof workers subsequently reported successful focused builds and
public-declaration axiom checks. Complete claim drafts are in their task
verification records; independent claim review and integration remain pending.
Multiplicity medium-effort implementation has started. Several attempts to
allocate or resume other worker threads hit the platform thread limit; these
are scheduling failures, not proof/test failures, and work is being queued.

Nonfinite-pvalue medium-effort worker started. Its audit found another existing
producer, `inference/calibration.py:add_one_p_kernel`, that discarded non-finite
draws. The coordinator extended that lane's explicit ownership and acceptance to
apply the already-settled all-draw finite contract there as well. No new result
fields, scientific procedure or statistical guarantee is introduced.

Multiplicity numerical implementation is frozen: default Holm, Bonferroni
alternative, both unsafe legacy names and `n_groups` removed. Its focused tests,
type/architecture checks and study tests passed; catalogue validation failed on
the intentionally stale legacy variant, and publisher metadata verification is
still pending. The coordinator authorized removal of the now-unused test import
alongside its test-class migration. Equivalence medium-effort worker has started
with that final API. Nonfinite lane is finishing additional audited Sharpe-tail
and legacy-host-adapter guards, under explicit expanded ownership in its README.

## Checkpoint 2: sources and claim review

All seven wave-1 lanes frozen. Code cross-review completed: multiplicity reviewed
nonfinite-pvalues, nonfinite-pvalues reviewed equivalence, equivalence reviewed
multiplicity. Findings repaired and independently rechecked: dyadic report
discarding failed draws, paired Sharpe hiding invalid returns, equivalence
percentiles hiding an infinite replicate, and Bonferroni empty-family division.
Final focused evidence: multiplicity 14 passed, equivalence 18 passed,
nonfinite-pvalues 78 passed. Full command history and earlier failures remain in
each lane's verification record.

Coordinator integrated nine new claims and modified LINEAR-005 (17 total),
five root imports, canonical claim documentation projections, catalogue variants,
CHANGELOG and scoped documentation. No new implementation-refinement claim or
AST binding was justified: existing implementation pins remain unchanged.
All new obligation mappings remain partial pending review, including
permutation-002; no broad obligation has been declared closed.

Inventory refresh passed. Initial sync attempts failed the unchanged 1500-token
reference-index budget as five new records enlarged its generated list (1584,
1527, 1505 tokens). Shortened only authored index prose; subsequent sync passed.
Formatting passed. Initial certification failed missing canonical claim
projections, then missing new source dossiers; projections added and independent
sources reviewer assigned dossiers. These were integration failures, not passing
certification runs. Statement pins have not been changed.

Fresh Terra high-effort claim reviewer and Terra medium-effort sources reviewer
started after scientific freeze; neither authored the work they review.
Study and architecture integration checks are running; logs:
`/tmp/statadic-release-guarantees-studies.log`,
`/tmp/statadic-release-guarantees-architecture.log`.

Architecture check passed (13 tests plus graph/lint checks). Source dossiers
were added by the independent sources lane. The next certify built successfully
(3306 jobs), then failed solely because statement/source pins are stale, as
required before review. Independent reviewer confirmed ten principal
declarations have only the three standard axioms. Review found two Blueprint
statement discrepancies and a missing LINEAR-005 moment-hypothesis limitation;
authors repaired the statements, coordinator added the limitation, and re-review
is pending. Thread-cap scheduling prevented resuming the linear author, so the
coordinator explicitly assigned the narrow OLS Blueprint edit to the MMD worker;
no OLS theorem body changed.

## Final checkpoint

Independent claim review signed off with no remaining findings, including
re-review of all fixes and ENERGY-002's supporting refactor. Coordinator updated
agent-review strings while retaining human review pending. Statement pinning
passed: 17 literature mappings, 239 compiled declarations, 46 method families.
Sync and formatting then passed.

First final full check failed in publishing after the main package suite passed:
2921 passed, 7 skipped, 48 warnings in 1085.50 seconds. Lean certification,
36 registry tests, architecture (13), shape tests (7), lint/types, wheel build
and evidence tests (15) passed first. Publishing had 6 failed, 7 passed and
5 setup errors, all caused by its authored example's old bibliography SHA256
after the five new reference records regenerated refs.bib. No tests were changed.
Coordinator refreshed only that reviewed example bibliography pin to the actual
generated file hash; its formal/coverage dispositions remain unchanged.
Focused publishing recheck is running before a fresh full rerun.

Focused publishing recheck passed: 18 tests in 30.76 seconds, including fresh-wheel
replication. Docs check then passed 86 tests and 17 float64 native tests, but
strict Sphinx build failed five unresolved Python object links from new claims.
Coordinator added explicit API documentation targets for EuclideanLaw,
hc_covariance and permutation_test_kernel. ENERGY-003 now uses the documented
public energy_distance alias, directly imported from scalar.py; independent
reviewer asked to confirm this sole claim-metadata change before re-pinning.
No theorem or test was weakened. These failures remain recorded, and the first
full-run log is preserved at `/tmp/statadic-release-guarantees-final-first.log`.

The independent reviewer approved ENERGY-003's public alias by direct-import
inspection and a locked identity assertion; pins refreshed successfully.
The first repaired site build exposed eight additional unresolved autodoc
annotation links (including JAX shape strings). Replaced the two newly added
low-level autofunction directives with explicit matching Python signatures and
their finite-input/covariance contracts. Kept strict Sphinx warnings enabled.
EuclideanLaw remains autodocumented. No Lean source or claim content changed.

Checkpoint-1 studies check passed: 316 tests passed, 3 existing skips,
3 warnings, 295.18 seconds; catalogue validation passed (exit 0).
Architecture check passed, 13 tests. The required offline LaTeX check also
passed (format/lint and both local project targets, exit 0). Logs are in
`/tmp/statadic-release-guarantees-studies.log`,
`/tmp/statadic-release-guarantees-architecture.log` and
`/tmp/statadic-release-guarantees-latex.log`.

Independent sources comparison is complete. Four new source dossiers added;
LINEAR-008 registry now cites the accessible revised author manuscript,
Section 2 equations (7)–(9). Corrected the linear lane's Long–Ervin manuscript
locator from Section 2 to Section 3. The source record retains all failed access
attempts and actual substitute evidence. DOI export remains template-owned;
the sources verification records an upstream Exordio patch proposal, untested
and unapplied locally.

## Checks omitted and owner flags

At the initial stop, the remainder of the main Python suite and the subsequent package/application,
documentation, LaTeX and AgentSync stages of this baseline did not complete.
At that stop, wave 1, cross-reviews, both integration checkpoints, source review, claim review,
statement pinning and the final check were not run because the baseline gate
failed. All task acceptance boxes and status fields remain unchanged.

Initial owner flag: investigate the GLM subprocess timeout before restarting the
baseline. The owner subsequently authorized continuation without that restart.
The multiplicity/default, `k_minus_1`, redistribution licence,
exordio DOI patch, obligation mapping and HC-formula decisions were not advanced
in the initial attempt. The owner's subsequent continuation instruction is
recorded above. Publication remains on hold; nothing was committed or submitted.

## Current owner flags

- Default correction is now Holm. Adaptive Bonferroni and `k_minus_1` are
  removed, together with `n_groups`; no restricted pairwise fallback is offered.
- The port's pricing-perspective revision is
  `66169ad5ff07f7779fe2543e24f215454bb24704`. No applicable licence was found;
  owner confirmation of reuse/redistribution remains required before publication.
- The DOI-export fix belongs upstream in Exordio. The proposed patch is in the
  sources verification, untested and unapplied locally.
- No required obligation is newly closed. Partial relationships are listed in
  [obligation-scope](obligation-scope.md); theorem proof is separate from
  numerical implementation and statistical coverage.
- No HC formula mismatch was found within the reviewed scope. HC1 needs positive
  residual degrees of freedom; HC2/HC3 need leverage below one. The conventional
  uncentered Long–Ervin HC3 differs from the original centered jackknife formula.
- The baseline GLM subprocess timeout remains open; it has not been relabeled a
  passing test or a numerical assertion failure.
