# Research-release handoff

Read this before starting any task in this plan. Each task is written to be implemented without other context. The [evidence](evidence.md) records what the assessment found.

## Inherited rules

The [research-workflow handoff](../api-ergonomics/tasks/research-workflow/handoff.md) still governs:

- the short path forms;
- the pinned environment;
- the source rules enforced by `just python::architecture`;
- the shared change rules for public API, precision tests and scientific scope;
- the steps for finishing a task.

Read it, plus `AGENTS.md`, `src/python/packages/statadic/AGENTS.md` and `src/python/packages/statadic/ARCHITECTURE.md`.

## Settled decisions

Do not reopen these unless the owner asks:

- Library publication is on hold. Do not make the repository public, tag a release, mint a DOI, upload a package or submit a paper.
- Order: validate the researcher's journey, deliver one verified publication bundle, then freeze the contracts. Do not expand plotting, dataframe or checkpointing ecosystems first.
- `PlotSpec` and native PGFplots remain the publication path. Helpers render retained values and compute no new statistics.
- Prediction uncertainty bands remain a statistical-method project, not a plotting task.
- Publishing workflows belong in the optional `statadic-publishing` package, following its [handoff](../publishing-companion/handoff.md). Core STATADIC gains no publication object.
- Persisted-content changes extend artifact schema 3 without another bump while nothing is released.
- Renames are clean breaks with no aliases.

## Installed-wheel lanes

"Against the released wheel" means the wheel that `just python::statadic::test-artifacts` builds, because nothing is released. That command runs `src/python/packages/statadic/scripts/check_distribution.py`, which builds the sdist and wheel, installs each wheel into a fresh virtual environment outside the checkout, and runs checks with `python -I`. The command is not part of `just check`, so run it explicitly and report its result.

Keep the existing runtime-closure check unchanged. It proves that a bare wheel installs without Polars, PyArrow or reference oracles. Add new lanes beside it instead.

## Finishing a task

Follow the inherited finishing steps. Also run `just python::statadic::test-artifacts` for any task that changes `check_distribution.py`, and `just python::docs::check` for any task that changes examples or tutorials. Do not commit unless the owner asks.
