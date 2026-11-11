---
type: task
title: Integrate finite-moment MMD reference proofs
description: Source-matched MMD U reduction and supporting V-bias algebra
links: [".context/reference/gretton-etal-2012/README.md"]
status: done
---

# Integrate finite-moment MMD reference proofs

## Goal

Integrate the finite-moment MMD U proof step and supporting V-bias calculation
with native certification and explicit sampling and implementation boundaries.
See [review and verification](verification.md) for source decisions and evidence.

## Acceptance

- [X] Compile and audit the finite-law support and MMD modules without admissions.
- [X] Bind the U proof step to the existing Gretton source and native documentation.
- [X] Run the native proof, documentation and independent MMD reference checks.

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
