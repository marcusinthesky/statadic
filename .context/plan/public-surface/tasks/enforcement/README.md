---
type: task
title: Enforce the export rule and record consumer impact
description: "Make the inventory require __all__ and reject unlisted public-looking definitions, enable Tach exact mode, review private exceptions, document the rule and report consumer impact."
links: [".context/plan/public-surface/handoff.md", ".context/plan/public-surface/evidence.md"]
status: done
---

# Enforce the export rule and record consumer impact

## Goal

Once the export tasks are integrated, make the existing checks hold the rule, so it cannot quietly drift. This changes the existing inventory and Tach configuration rather than adding a new check.

Depends on [exports-api](../exports-api/README.md), [exports-measures](../exports-measures/README.md) and [exports-models](../exports-models/README.md).

## Acceptance

- [x] The inventory rejects a public module without a literal `__all__`.
- [x] The inventory rejects a top-level public-looking definition that is not in `__all__`.
- [x] The inventory tests cover both rejections.
- [x] `tach.toml` sets `exact = true`, and every private-helper exception still matches a real import.
- [x] The rule is documented, and the CHANGELOG lists every renamed name.
- [x] `verification.md` holds the consumer-impact list for pricing-perspective and `statadic-publishing`.
- [x] `just lean::check`, `just python::architecture`, `just python::studies::check` and `just check` pass.

## Specification

**Inventory** (`src/lean/tools/inventory.py`):

- Replace the inference fallback at lines 161-172 with the error `<path>: public module needs a literal __all__`. The `static_jaxtyping` special case goes with it.
- In a public module, a top-level function, class or `type` alias without a leading underscore must be listed in `__all__`. Otherwise the error is `<path>: <name> is not in __all__; list it or add a leading underscore`. Include the existing `TYPE_CHECKING` branch handling. Ignore imports and plain assignments.
- Report every violation in one run rather than stopping at the first.
- Update the module docstring (lines 5-6).
- In `src/lean/tools/tests/test_inventory.py`:
  - replace `test_public_class_alias_without_explicit_all` with one test per rejection;
  - keep the other tests.

**Tach** (`src/python/tach.toml`):

- Set `exact = true`, and remove the unused `statadic.core` and `statadic.sample` dependencies from `statadic.api.inputs`, or whatever `tach check --exact` reports at that point.
- Apply the interface additions and removals that the export tasks requested, including removal of the `ProjectionUnavailableError` exception.
- For each remaining private-helper exception:
  - remove it if no import uses it any more;
  - narrow its `visibility` to the modules that actually import it.
- Keep the uniform name pattern.

**Documentation:**

- Add the plan's rule as one short paragraph to `docs/module-organization.md`, under "Public modules and migration".
- Add one sentence to the "Dependency direction" section of the package `ARCHITECTURE.md`.
- In the CHANGELOG `[Unreleased]` section, add one entry listing renamed and newly listed names.

**Consumer impact**, recorded in `verification.md`:

- every name that pricing-perspective imports, checked against the final inventory. Any that became private needs owner approval, and goes to the pricing-perspective bump task as a migration item;
- the same check for `statadic-publishing`, `statadic-evidence` and the apps.

## Non-goals

- No new check script, registry or dependency.
- No per-name Tach lists.
- No changes to pricing-perspective.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
