# Research-workflow execution

The owner approved three GPT-6.1 Terra workers at medium effort, coordinated by
one integration owner. Breaking changes are encouraged; compatibility aliases,
deprecated implementations and schema migrations are excluded. Publication
remains on hold. Blocked and closed proposals remain outside implementation.

## Assignments

| Lane | Sequence | Ownership |
| --- | --- | --- |
| Inputs | precision rejection, prediction roles, predictor naming, require-valid, mappings, frames | data, design, base, regression, prediction and embedding artifacts, input adapters and their tests |
| Rendering | PGFplots gate, LaTeX report, PlotSpec extensions, plot helpers | reporting, inference, snapshots, plot helpers, compiler templates and their tests |
| Formal | baseline correspondence investigation, linear MMD refinement, audit tooling | Lean claim and source correspondence, numerical witnesses, locked Vulture audit |

Lanes run concurrently. Plot helpers wait for integrated mappings, report
confidence metadata and PlotSpec extensions. Schema 2 becomes 3 once, with
prediction roles; snapshot work uses that version. One writer owns each shared
file. The coordinator owns Tach, changelog, catalogue, API inventory and generated
views. Package-wide dead-code deletions follow feature integration and are split
among API, scientific implementation, and test/script owners.

## Gates

1. Investigate baseline correspondence drift before accepting new pins. The
   planning baseline passed infrastructure, record, Lean compilation and 34
   registry tests, then failed implementation correspondence; later checks did
   not run.
2. A different worker reviews completed changes before dependent work starts.
   Review contracts, source ownership, minimality, precision, persisted content,
   public exports and tests. Workers record focused verification, not task
   completion.
3. At stable checkpoints pause mutations, integrate reviewed batches, explicitly
   refresh reviewed inventories/pins, run `just sync`, then `just check`. A passing
   checkpoint can establish acceptance for several completed tasks. Do not run
   concurrent full checks against changing sources.
4. After cleanup, repeat the audit, affected tests, distribution tests and full
   checks. Record retained audit findings individually. Mark acceptance only when
   established by actual results; report any failures or checks not reached.

All commands use `devenv shell --` and repository locked projects. No commits or
release actions are authorized by this execution. Human mathematical review
remains pending; exact-real proofs do not verify Python, IEEE arithmetic or XLA.
