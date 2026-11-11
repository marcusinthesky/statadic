---
type: plan
title: Tighten the public import surface before release
description: "Make every public STATADIC module declare __all__, give internal helpers a leading underscore, and align the inventory and Tach with that rule. No stability registry."
links: [".context/plan/public-surface/handoff.md", ".context/plan/public-surface/evidence.md", ".context/plan/api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md"]
status: done
---

# Tighten the public import surface before release

## Goal

Before release, make it unambiguous which STATADIC names are supported. Use Python's own conventions and the checks the repository already has: `__all__`, leading underscores, Tach interfaces, the architecture guard and the public API inventory.

This is a naming and declaration pass. It changes no behaviour, signature or numerical result. It adds no stability registry, stability levels, API-diff tool or dependency.

## The rule

1. A module is __public__ when no part of its import path starts with an underscore.
2. Every public module declares a literal `__all__`, including package `__init__` files. Navigation namespaces use `__all__ = []`.
3. A name is public only when a public module lists it in `__all__`. Anything else may change without notice.
4. In a public module, a top-level function, class or `type` alias that is not in `__all__` starts with an underscore.
5. A type used in a public signature, and an exception that public calls raise for callers to catch, is itself public.
6. Code in another Tach module imports only public names. The exception is a private helper with a visibility-scoped Tach interface; this is already how `tach.toml` works.

## Acceptance

- [x] The three export tasks and [enforcement](tasks/enforcement/README.md) are complete.
- [x] The inventory rejects any public module without `__all__`, and any unlisted public-looking definition.
- [x] No name that pricing-perspective or `statadic-publishing` imports became private without the owner's approval.
- [x] `docs/module-organization.md` and the package `ARCHITECTURE.md` state the rule.
- [x] `just check` passes. Report any check that could not run.

## Start here

Read the [handoff guide](handoff.md) first. The [evidence](evidence.md) records what was measured on 2026-11-06.

The [verification record](verification.md) records baseline, integration and final
checkpoint results.

Start only after the research-release `walkthroughs` and `artifact-contract` changes are integrated. Those tasks edit `api/plots.py`, `api/artifacts/_store.py`, the catalogue and the inventory.

| Order | Work | Depends on |
| --- | --- | --- |
| 1 | [exports-api](tasks/exports-api/README.md) | The research-release checkpoint |
| 1 | [exports-measures](tasks/exports-measures/README.md) | The research-release checkpoint |
| 1 | [exports-models](tasks/exports-models/README.md) | The research-release checkpoint |
| 2 | [enforcement](tasks/enforcement/README.md) | All three export tasks |

The three export tasks own disjoint files and can run in parallel.

## Out of scope

- A stability file, stability levels, or any per-symbol promise beyond `__all__`.
- Griffe or other API-diff tooling. Reconsider after a first release tag exists to compare against.
- Replacing Tach's uniform name pattern with per-name lists. `tach.toml:721-724` records why: it would duplicate every `__all__`.
- A separate consumer project. The wheel lanes in `check_distribution.py` already exercise installed imports.
- Moving code between modules, except where rule 5 needs a re-export.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 4 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
