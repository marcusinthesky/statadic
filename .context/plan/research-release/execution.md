# Research-release execution

The coordinator owns integration and shared files. Three GPT-6.1 Terra workers
at medium effort implement and cross-review; workers never complete task records.
Publication stays on hold: no commits, tags, pushes, uploads, DOIs or submissions.
Use clean breaks without aliases or deprecated paths. Keep changes minimal.

## Baseline gate

Before feature changes, investigate the saved nearly constant hedge-sample
counterexample in `test_larger_for_farther_mixture`. A statistically invalid
property may be constrained with its mathematical reason; a defective statistic
must be reported to the owner without changing it. Run one complete `just check`
after the reviewed correction to establish the green baseline.

Gate 0 passed on 2026-11-06. The generated test family now shares one nonconstant
empirical measure between target and close candidate and translates the far
candidate. Studentization does not generally preserve raw-energy ordering, since
its jackknife standard error and effective size depend on the candidate. The
strict assertion and original seeded witness remain; scientific code is unchanged.
Focused file: nine passed; independent witness: one passed; package lint passed.
The full baseline `just check` passed: 34 Lean registry tests, 13 architecture
tests, seven shape tests, 2,691 package tests (166 skipped), 15 evidence tests,
316 study tests (three skipped), 39 performance tests, 86 x64-off docs tests and
17 x64-on native tests (69 deselected), plus build, strict documentation, TeX and
agent checks. This establishes the baseline, not feature acceptance.

## Lanes and ownership

| Lane | Sequence | Sole writer |
| --- | --- | --- |
| Walkthroughs | residual helper, OLS, Poisson, tutorials, installed-wheel data lane | `api/plots.py`, mirrored tests, the two examples, tutorials, start index, composition reference, docs example runner |
| Artifacts | writer metadata and rejection message, tests, policy page, installed checks | artifact store and tests, artifacts reference and reference index |
| Companion | bundle, paper, operations, replication | `src/python/packages/statadic-publishing/` |

Artifacts owns `scripts/check_distribution.py` first. Walkthroughs takes ownership
only after independent review and integration. Companion paper waits for reviewed
bundle and integrated `residual_plot`; operations waits for paper, and replication
waits for operations. Workers report exact requested shared-file text.

The coordinator owns Tach, workspace project/Just/lock files, CHANGELOG, studies
catalogue, public inventory, synchronization, and authored plan state. Every
command uses `devenv shell --`; Python runs through its locked project. Generated
Insitu regions are changed only by `just sync`. Credentials never enter records.

## Review and checkpoints

A different lane reviews each completed batch against its task specification,
ownership, minimality, precision in both modes, persisted values, public exports,
and meaningful tests. No new statistics are authorized. Stop a task and report
any premise that no longer holds; do not widen its scope.

Pause mutations at each integrated checkpoint. If the API changed, run
`just lean::inventory-refresh`, then `just sync`, `just fmt`, `just check`, and
`just python::statadic::test-artifacts`. Never check a changing source tree. Run the
docs and companion checks for their respective tasks. Install the vendored TeX
bundle with `just latex::setup` only if missing.

Only actual checkpoint results establish acceptance. The coordinator records
per-task verification, ticks established items, and changes status. Walkthrough
verification includes helper friction. `plot-accessor`, companion `revision`,
and blocked or closed research-workflow records remain outside implementation.
The final report lists task/check results, omissions, review corrections, failed
premises, and owner decisions including the artifact compatibility promise.

## Integration findings

- Cross-review corrected companion external paths: these permit parent-relative
  references only for the specified staged dependencies, never absolute paths.
  Model metadata is frozen using STATADIC's existing metadata helpers.
- Old schema-2 artifacts lack `writer`; schema rejection therefore precedes
  schema-3 field validation. The writer version remains diagnostic only.
- Three documentation tests used the replaced example return value and CLI.
  They now check the specified output-directory journey, preserving the
  independent NumPy HC3 witness. Their x64-off focused run passed five tests.
- The companion paper's offline compile exposed missing resources in the
  vendored TeX subset. The owner explicitly authorized refreshing it on
  2026-11-06. The existing refresh recipe now warms a small article, bibliography
  and PGFplots fixture. The first upstream attempts returned HTTP 429; the
  existing retries retained the upstream pin. The refresh subsequently passed
  and wrote the expanded 6.2 MB subset; paper verification resumed. No log rule
  or threshold was weakened.

Checkpoint 1 passed `just check` and the subsequent explicit distribution gate.
The coordinator completed walkthroughs, artifact-contract, bundle and paper from
that evidence, then released operations. Operations review corrected structural
TOML staging, independent failure collection and effective caller precision in
analysis subprocesses. The replication fixture was corrected to retain explicit
unavailable status and reasons as well as None values. Its fresh-wheel rebuild
passed in both precision modes and cross-review approved the corrected test.
The final checkpoint freezes all implementation sources before repeating gates.

Checkpoint 2 passed the full repository check, including eight companion tests
with isolated reconstruction, and the subsequent explicit distribution gate.
The coordinator completed operations and replication from those results. The
research-release plan's requested first slice is complete; companion revision,
the plot accessor and publication remain outside this work.
