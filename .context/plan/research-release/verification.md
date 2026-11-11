# Research-release checkpoints

All commands below ran through `devenv shell --` and locked projects. Nothing was
committed, uploaded or published. Publication and the post-release compatibility
promise remain owner decisions.

## Gate 0

The baseline property correction passed independent review, nine focused tests,
one independently rerun witness, and package lint. Only the property test changed.
The full baseline `just check` passed: 34 Lean registry, 13 architecture, seven
shape, 2,691 package (166 skipped), 15 evidence, 316 studies (three skipped),
39 performance, 86 x64-off docs and 17 x64-on native tests (69 deselected).
Build, strict site, offline TeX and agent checks passed too.

## Checkpoint 1

The reviewed walkthrough, artifact, bundle and paper changes were frozen before
`just lean::inventory-refresh`, `just sync`, `just fmt` and `just check`.
The full check and the subsequent explicit `just python::statadic::test-artifacts`
both returned zero.

| Gate within `just check` | Result |
| --- | --- |
| Infrastructure and authored/generated records | Passed |
| Lean compilation, registry, blueprint and correspondence | Passed; 34 registry tests |
| Architecture and layout | Passed; 13 guard tests |
| Package lint and strict typing | Passed |
| Pinned shape environment | Seven passed |
| Fresh package suite | 2,725 passed, 143 skipped, five warnings |
| Package wheel and sdist build | Passed |
| Evidence package | 15 passed |
| Publishing companion including oracle and cached paper compile | Four passed |
| Studies | 316 passed, three skipped, three warnings |
| Performance | 39 passed |
| Examples | Passed in both precision modes |
| Documentation with x64 off | 86 passed |
| Native documentation with x64 on | 17 passed, 69 deselected |
| Strict site, native labs, TeX targets and agent checks | Passed |

The distribution gate passed both wheel routes, runtime-closure and expanded
artifact smoke checks, four installed data-extra walkthrough executions, and
standalone sdist development. Standalone typing: 50 passed, seven shape tests
skipped there and covered by the separate pinned shape gate. The new example
scripts came from the extracted sdist and imported the installed wheel outside
the checkout using isolated Python.

This checkpoint establishes acceptance for `walkthroughs`, `artifact-contract`,
companion `bundle` and companion `paper`. Operations and replication are not
established by it.

## Additional checks and corrections

- Residual helper: seven focused tests passed in each precision mode; lint,
  strict typing and seven shape tests passed.
- Artifacts: the final focused suite passed 51 tests; lint and typing passed.
  A narrow fresh-wheel check passed after correcting its explicit x64-off inputs.
- The optional whole artifact file with x64 off reported 46 passed and two
  failures in existing tests that assume x64. This is not claimed as a passing
  gate. The required fresh-process refusal test passed.
- Companion contracts: three focused tests passed in each precision mode,
  including a separate x64-off rerun after review corrections.
- Initial docs verification reported 83 passed and three failures against the
  superseded example API. The tests were updated with the independent NumPy HC3
  witness retained; the focused rerun passed five tests. The full docs check then
  passed independently and again at checkpoint 1.
- The initial publishing check reported three passed and one failed because the
  vendored TeX subset lacked article fonts. The owner authorized a refresh. The
  existing refresh command retried HTTP 429 responses and succeeded without a
  pin or log-policy change. Publishing subsequently passed four tests in each
  tested precision mode, and at checkpoint 1.
- Cross-review corrected external path validation, frozen model metadata,
  schema-before-writer diagnostics, and imports included in tutorial excerpts.
  Shipped companion analysis scripts were added to its lint/format gate.

Initial unprivileged devenv reads could not access the Nix daemon; the same
commands succeeded with the established escalation. An attempted lock from the
Python subdirectory could not find `devenv.nix`; it succeeded from the root with
`uv lock --project src/python --offline`. These were preparation failures, not
passing checks. All final locks used the pinned workspace.

## Checkpoint 2

The final source freeze followed cross-lane review of operations and replication.
`just lean::inventory-refresh`, `just sync`, `just fmt`, `just check` and the
subsequent explicit `just python::statadic::test-artifacts` all passed.

The full check passed 34 Lean registry tests, 13 architecture guard tests, seven
shape tests, 2,725 package tests (143 skipped, five warnings), 15 evidence tests,
eight publishing tests, 316 study tests (three skipped, three warnings), 39
performance tests, 86 x64-off documentation tests and 17 x64-on native tests
(69 deselected). Infrastructure, record validation, layout, lint, strict typing,
package builds, both-mode examples, strict site/native exports, offline TeX and
agent checks passed. The distribution gate again passed both wheel routes,
artifact smoke tests, all four data-extra walkthrough runs, and standalone sdist
development (50 typing tests passed, seven separately covered shape tests skipped).

| Task | Acceptance established |
| --- | --- |
| Walkthroughs | Checkpoint 1; revalidated at 2 |
| Artifact contract | Checkpoint 1; revalidated at 2 |
| Companion bundle | Checkpoint 1; revalidated at 2 |
| Companion paper | Checkpoint 1; revalidated at 2 |
| Companion operations | Checkpoint 2 |
| Companion replication | Checkpoint 2 |

The operations-focused suite passed five tests in each tested precision mode,
then seven after the effective-caller-precision regression. The final companion
suite passed eight tests in each precision mode and again inside `just check`.
Its fresh-wheel reconstruction, generated-file hashes, unavailable rows and
required failures are documented in the [replication verification](../publishing-companion/tasks/replication/verification.md).

Review replaced quoted-text path substitution with structural TOML editing,
separated verification checks so simultaneous failures are retained, and checked
declared output identity. A real child-process test caught conflicting inherited
precision settings. Architecture caught the missing direct JAX declaration; its
corrected check passed all 13 guards. Replication review required explicit
unavailable status and reason in addition to None values; corrected fixtures
passed in both modes. No statistics or approval inference was added.

The only failed premise was the prescribed TeX subset's missing resources; it
was resolved by the owner-authorized refresh of the same pinned upstream. One
HTTP HEAD probe succeeded and four download probes returned 429 while the
existing refresh retried; preparation subsequently completed. No check threshold
or upstream version was changed. The baseline studentized counterexample was an
invalid universal test property, not an established scientific implementation
defect; the scientific code was unchanged.

## Checks not run and open decisions

- `just latex::setup` was unnecessary: the vendored bundle already existed and
  was explicitly refreshed instead.
- Full slow/GPU suites, long statistical qualification campaigns and hosted CI
  were not run; they are outside this local first slice. The required local
  package, docs, reference witness, TeX, distribution and replication gates ran.
- Publication/release commands and human mathematical review were not performed.
  Publication remains on hold, and human-review statuses remain pending.
- The owner must still confirm the post-release artifact compatibility promise.
  A passing pre-release exact-schema reader is not that policy decision.
- `plot-accessor` remains blocked, companion `revision` remains backlog, and
  blocked/closed research-workflow records remain outside implementation. The
  companion parent remains in progress for its later revision slice. The
  research-release parent also remains in progress because its blocked
  plot-accessor child is still open: workflow synchronization rejects a terminal
  parent with open children. Its requested six-task slice and acceptance items
  are complete. The initial completion sync exposed this rule; the parent state
  was corrected without changing the excluded child or workflow policy.

No commits, tags, pushes, uploads or DOIs were created. Per-task verification
records contain focused evidence and helper friction; only the coordinator
marked checkpoint-established acceptance.

After completion metadata changed, `just sync` and `just fmt` passed. The focused
workflow check validated 200 views and 185 sources, `rumdl check .` passed all
292 Markdown files, and `git diff --check` passed. No implementation sources
changed after checkpoint 2; these final checks cover the completion records.
