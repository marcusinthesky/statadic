---
type: task
title: Measure compilation runtime and memory
description: Measure compilation runtime and memory
links: []
status: done
---

# Measure compilation runtime and memory

## Goal

Measure compilation runtime and memory

## Acceptance

- [x] Measure discrepancy, OLS and supplied-row-refit workloads in isolated processes.
- [x] Separate preparation, compilation, cold calls, warm samples and memory scope.
- [x] Validate outputs against independent NumPy references outside timing.
- [x] Execute and verify the integrated float32/float64 smoke campaign.
- [x] Pass final checks and retain review evidence.

The frozen complete `just check` passed, including all 24 performance tests.
The independent programme review records the retained CPU measurement scope and
remaining accelerator and longitudinal optimization work.

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
