---
type: plan
title: Test layout and reference validation
description: Test layout and reference validation
links: []
status: in-progress
---

# Test layout and reference validation

## Goal

Repair the layout-enforcer fork, adopt focused module/package ownership with
explicit cross-module categories, and validate STATADIC against live reference
libraries. Use an immutable Git dependency and a separate required CI reference
gate. Preserve the documented long-running qualification limitations.

## Acceptance

- [x] Verify and pin the corrected enforcer fork.
- [x] Migrate focused tests without losing cases or markers; enforce layout.
- [x] Correct invalid fit assumptions and pass fast and live-reference suites.
- [ ] Pass float32/float64 references and installed distribution checks.
- [ ] Retain evidence, synchronize views and run repository checks.

[Migration manifest](moves.json) · [Collection and reference results](collection.json) ·
[Verification and remaining gate](verification.md)

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
