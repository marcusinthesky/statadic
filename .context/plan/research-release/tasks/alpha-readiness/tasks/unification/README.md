---
type: task
title: Unify regression and ordination workflows
description: Consolidate model construction, preserve scientific contracts and implement compositional ordination and geometric regression.
links:
  - .context/plan/research-release/tasks/alpha-readiness/tasks/unification/validation.md
status: done
---

# Unify regression and ordination workflows

## Goal

Implement the accepted Ordination/Regression design from the complete unification
discussion. Regression owns predictor terms; Ordination composes directly as a
term. Native operations, actual fitted capabilities and scientific distinctions
remain available. Replace supplied-moment host wrappers with one estimation
operation. No compatibility aliases or permanent wrappers around retired models.

The coordinator owns shared lifecycle, artifact/schema integration, public
inventory, repository migration and serialized checks. Three arms own regression,
ordination and geometric regression respectively. Each completed arm receives
independent review. Existing user changes to the Lean registry are preserved.

## Acceptance

- [x] Consolidate regression families and supplied moments without losing native contracts.
- [x] Compose PCA, PCoA, kernel PCA and Isomap through retained Ordination terms.
- [x] Preserve one-factor tests and verify explicit full/reduced calculations and calibration limits.
- [x] Migrate declarative artifacts, repository consumers and progressive documentation.
- [x] Complete focused verification, independent review and final aggregate checks.

## Boundaries

Kernel PCA, Isomap and adjusted geometric inference are new implementations and
need their own evidence. Negative-spectrum approximation is explicit; disconnected
Isomap graphs fail. General Freedman–Lane calibration is approximate. Existing
source-fidelity and large spatial-survival release obligations remain open.
Performance is reported evidence, not a universal superiority gate. The downstream
repository remains read-only; current library fixtures and prose stay neutral.

[Validation](validation.md) records executed checks, independent review and the
remaining release obligations. Earlier alpha evidence retains its original
source and artifact identities.

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

- [Add reference-indexed Regression constructors](../constructors/README.md)
- [Ordination and Regression migration evidence](validation.md)

<!-- insitu:end -->
