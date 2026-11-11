# Public-surface handoff

Read this before starting any task in this plan. Each task can be implemented without other context. The [evidence](evidence.md) records the measurements behind the task lists.

## Inherited rules

The [research-workflow handoff](../api-ergonomics/tasks/research-workflow/handoff.md) still governs the environment, source rules, shared change rules and finishing steps. Also read `AGENTS.md`, `src/python/packages/statadic/AGENTS.md` and `src/python/packages/statadic/ARCHITECTURE.md`; its "Dependency direction" section already says that namespace initializers are navigation maps and family initializers curate their own surface.

## Settled decisions

Do not reopen these unless the owner asks:

- There is no stability registry and no stability levels. `__all__` in a public module is the whole declaration.
- Tach keeps one uniform alphabetic-name pattern per public module. Do not copy `__all__` lists into `tach.toml`.
- Change as little as possible. Listing a name in `__all__` is cheaper than renaming it, and renaming is cheaper than moving it.
- Keep `freeze_metadata` and `metadata_to_json` public in `statadic.api.reporting`, because `statadic-publishing` imports them.
- Keep `statadic.api.inputs` public, because `RegressionInput` appears in public `Regression` signatures (`api/base.py:135`, `api/regression/_likelihood.py:37`).
- Renames are clean breaks with no aliases.
- STATADIC's own tests may import private names. Edit a test only to follow a rename.
- Change no behaviour, signature, default or numerical result.

## Deciding each name

Work through these in order for every public-looking name in your modules:

1. **Users rely on it.** It is used by the docs (`docs/reference/api.rst`, tutorials, guides or examples), by any app, by `statadic-publishing` or `statadic-evidence`, or by pricing-perspective. List it in `__all__`.
2. **Users meet it.** It appears in a public signature or return annotation, is a protocol users implement, or is an exception that public calls raise. List it. If it lives in a private module, re-export it from the public module whose calls use it, and list it there.
3. **Only its own Tach module uses it,** apart from tests. Add a leading underscore and update its call sites and tests.
4. **Another Tach module uses it, but users don't.** Prefer a leading underscore plus a visibility-scoped Tach interface, which the coordinator adds. If the name reads as a sensible user tool, list it instead.
5. **Unsure.** List it, and flag it for the owner in `verification.md`.

To check consumers, search from the repository root:

```bash
rg -n -w NAME src/python src/latex
rg -n -w NAME ~/Git/pricing-perspective/src/python --glob '!**/packages/statadic/**' --glob '!src/statadic/**'
```

pricing-perspective is read-only reference. Its untracked `src/statadic` and `src/python/packages/statadic` directories are cached STATADIC copies, not consumers. Never edit that repository.

## File ownership

Each export task owns the source modules it lists and their mirrored tests. The coordinator alone edits these shared files:

- `src/python/tach.toml`;
- `src/lean/registry/public-api.json`, refreshed only through `just lean::inventory-refresh`;
- `src/python/apps/studies/catalogue.json`, whose `public_symbols` must equal the inventory;
- `src/python/packages/statadic/CHANGELOG.md`;
- `docs/module-organization.md` and the package `ARCHITECTURE.md`.

Workers record what those files need in their `verification.md`: Tach interfaces to add or remove, symbols added to or removed from the inventory, and renamed names for the CHANGELOG.

## Recording decisions

Each export task writes `verification.md` beside its README, holding:

- a table with columns for module, names listed, names made private, and the reason or consumer evidence;
- re-exports added under rule 2;
- requested changes to the coordinator's shared files;
- names flagged for the owner;
- commands run, with counts, and anything not run.

## Checks

Export tasks run:

- focused `just python::statadic::test-fresh` runs for the touched families;
- `just python::lint`;
- `just python::typecheck`;
- `just python::architecture`;
- `just python::docs::check`, if a documented name changed.

The inventory and catalogue checks fail until the coordinator refreshes them. That is expected, so don't refresh them inside a lane. At each checkpoint the coordinator:

1. integrates the lanes;
2. runs `just lean::inventory-refresh`;
3. reconciles `catalogue.json`;
4. runs `just check`.

Do not commit unless the owner asks.
