---
type: document
title: Spatial survival lifecycle implementation
description: Authored migration boundaries and verification limits.
links:
  - .context/plan/api-cohesion/tasks/spatial-survival/README.md
---

# Spatial survival lifecycle implementation

SpatialSurvivalData(panel, integrations) is the single spatial survival input. Independent survival still accepts observations or PanelData. The numerical problem retains its LaggedLogTime process and graph name, while SpatialSurvivalML retains numerical search/integration settings. Preparation resolves named lag columns before fitting and preserves the full graph/observed spell axes.

The expert refine(prepared, validation_integrations=..., config=...) operation consumes the retained numerical problem and layout; estimate_refined is removed. Refinement period reductions preserve structural fields through dataclass replacement. Training/validation draw identity and integration-accuracy limits are unchanged.

All survival predictions default to marginal median duration. Other targets use predict(..., kind=...) with explicit arguments; former convenience aliases are removed. Balanced, observed and survival graph-node predictions share SpatialPanelPrediction, retaining period and unit identities. Same live named inputs reuse fitted feature/effect state and validate graph rosters rather than refitting.

Survival host tests and the seasonal/joint-variant study adapters have been migrated. Removing redundant fixed-effect intercepts preserves each existing full-effect mean span; study coefficients are still selected by their fitted names and independent scalar likelihood/mean controls remain distinct from campaign qualification. No censoring likelihood, integration tolerance, solver budget, reference law or qualification threshold is relaxed.

Numerical test execution and campaign qualification belong to the coordinator. Scoped static checks are development evidence only and are not installed-wheel or statistical qualification.
