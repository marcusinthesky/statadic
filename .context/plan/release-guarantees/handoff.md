# Release-guarantees handoff

Read this before starting any task in this plan. With this file, the [evidence](evidence.md) and its own README, each task can be done without other context.

## Inherited rules

The [research-workflow handoff](../api-ergonomics/tasks/research-workflow/handoff.md) governs the environment, source rules, shared change rules and finishing steps. Also read:

- `AGENTS.md`;
- `src/python/packages/statadic/AGENTS.md`;
- `src/lean/README.md`;
- `src/python/packages/statadic/docs/formal/literature.md`, the source-first policy.

Path shorthand in this plan:

| Prefix | Location |
| --- | --- |
| `statadic/` | `src/python/packages/statadic/src/statadic/` |
| `tests/` | `src/python/packages/statadic/tests/` |
| `studies/` | `src/python/apps/studies/` |
| `StatadicFormal/` | `src/lean/packages/StatadicFormal/StatadicFormal/` |
| `registry/` | `src/lean/registry/` |

## Settled decisions

Do not reopen these unless the owner asks.

**Python:**

- Remove or narrow a guarantee that does not hold. Do not patch it with a new procedure.
- Breaks are clean, with no aliases or deprecation shims. Nothing has been released.
- Add no result fields or result types. An unavailable p-value is NaN from JAX code and from functions that return a float. Host-API results keep returning `None`, as `statadic/api/association.py:217-226` already does.
- Any non-finite observed statistic or null draw makes the p-value unavailable. Do not drop the non-finite draws: dropping them changes B and the calibration.

**Lean:**

- New Lean goes in the existing `StatadicFormal` library as finite, exact-real statements in the style of the current claims, with honest limitations.
- Statements use StatadicFormal's existing definitions (`FiniteLaw`, `quadratic`, `KernelPSD`, `rank`, `estimate` and so on) or mathlib's. Never add a second definition of a concept that either already has.
- Not allowed in Lean files: `sorry`, `admit`, `axiom`, `native_decide`, `debug.*` options, custom `macro`/`elab`/`syntax`, `True`-valued statements and `True` premises. The certify gate rejects most of these; review rejects the rest.
- Every hypothesis must be satisfiable, and none may assume the theorem's main step. If a hypothesis carries real content, as STATADIC-MMD-001's pair moments do, list it in the claim's `scope.limitations`.
- The LaTeX `statement` describes the Lean statement exactly: no quantity appears in the LaTeX that is absent from the Lean conclusion. STATADIC-LINEAR-005 currently breaks this rule; [linear-covariance](tasks/proofs/tasks/linear-covariance/README.md) fixes it.
- Never port pricing-perspective's placeholder theorems or its `EnergyStatistics.ProbabilityMeasure`. The [evidence](evidence.md) lists them.

**Process:**

- pricing-perspective is read-only reference. Never edit it.
- Do not commit. Publication stays on hold.
- Owner continuation: proceed with the programme despite the failed/interrupted
  baseline, prioritizing Lean proofs. Retain the GLM timeout as an open follow-up;
  neither the baseline nor the unfinished checks count as passing evidence.

## File ownership

Each task lists the files it owns. The coordinator alone edits:

- `registry/claims.json`, `methods.json`, `implementation-pins.json` and `statement-pins.json`;
- `registry/public-api.json`, refreshed only through `just lean::inventory-refresh`;
- the library root `src/lean/packages/StatadicFormal/StatadicFormal.lean`;
- `studies/catalogue.json`;
- `src/python/packages/statadic/CHANGELOG.md`;
- `src/python/packages/statadic/docs/**`, including `reference/method-catalogue.md`, `reference/package-development.md` and `validation/known-limitations.md`;
- every plan status field.

`src/latex/refs.bib` is generated: never edit it.

**References.** Bibliography entries are `.context/reference/KEY/README.md` records, one directory per citation key.

- Create one with `devenv shell -- just reference-new KEY "Title"`, then fill in its CSL fields and record how the metadata was checked.
- A lane owns the records it creates.
- Only the coordinator, and the sources lane once wave 1 is frozen, runs `just sync`.

**Requests.** Workers put requests for coordinator-owned files in their `verification.md`:

- complete draft claim objects in the existing `claims.json` schema;
- obligation mappings and implementation pins;
- inventory, catalogue and CHANGELOG changes;
- documentation text.

## Recording

Each task writes `verification.md` beside its README, holding:

- what changed, file by file;
- each decision, with its source evidence (theorem, page or equation locator);
- requests for coordinator-owned files;
- items flagged for the owner;
- commands run, with counts, and anything not run.

Workers never edit status fields.

## Commands

Run everything from the repository root through `devenv shell --`.

**Python lanes run:**

- focused `just python::statadic::test-fresh PATHS` for the families they touch;
- `just python::lint`, `just python::typecheck` and `just python::architecture`;
- `just python::studies::check`, for multiplicity only.

The inventory-staleness and catalogue checks fail until the coordinator refreshes them. That is expected.

**Lean lanes** build only their own modules, and only one Lean process runs at a time across all agents:

```bash
devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Path.Module'
```

Check axioms with a scratch file outside the repository that imports the module and runs `#print axioms` on each new declaration:

```bash
devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake env lean /path/to/Scratch.lean'
```

Do not run `just lean::certify` before the coordinator merges your claim drafts. It fails while blueprint labels and registry ids differ, and while statement pins are stale.

Every Lean edit invalidates every statement pin, because pins bind the whole first-party source graph. Only the coordinator re-pins, after claim review.

**Full suites:** only the coordinator runs `just check`, `just lean::check` or a full `test-fresh`, and never two at once.

## Checkpoints

The coordinator:

1. **Baseline.** Records `git status --short --branch` and a full `just check` in `verification.md`.
2. **Wave 1 freeze.**
   - Cross-reviews the code lanes: multiplicity → nonfinite-pvalues → equivalence → multiplicity.
   - Integrates, merges the claim drafts and adds root imports.
   - Runs `just lean::inventory-refresh`, reconciles the catalogue, then runs `just sync` and `just lean::certify`. Statement-pin staleness is the only expected failure.
3. **Wave 2.** Runs sources and claim-review. Claim-review findings go back to the owning lane, which fixes them, and then the claim is reviewed again.
4. **Final.**
   - After claim review signs off: `just lean::statement-pin`, `just fmt` and `just check`.
   - Sets statuses from that run, and records omitted checks and owner flags.
