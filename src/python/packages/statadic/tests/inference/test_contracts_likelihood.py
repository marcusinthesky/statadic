# jaxtyping dimensions are runtime contracts.
"""Every native likelihood fit satisfies one structural log-likelihood contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from statadic.estimation.glm import GLMFit
    from statadic.estimation.likelihood import LikelihoodFit as GeneralisedFit
    from statadic.estimation.spatial.graph_likelihood import GraphMLFit
    from statadic.estimation.spatial.likelihood import SpatialFit
    from statadic.estimation.spatial.observed_panel import ObservedSpatialPanelFit
    from statadic.estimation.spatial.panel import SpatialPanelFit
    from statadic.estimation.survival.fit import SurvivalFit
    from statadic.estimation.survival.spatial import SpatialSurvivalFit
    from statadic.inference.contracts import LikelihoodFit
    from statadic.model.glm import GLMEquation


@pytest.mark.unit
@pytest.mark.typing
def test_native_likelihood_fits_conform_statically() -> None:
    """Pyrefly checks each annotated parameter against the Protocol."""

    def accepts(fit: LikelihoodFit) -> None:
        del fit

    def check(
        glm: GLMFit[GLMEquation],
        generalised: GeneralisedFit,
        survival: SurvivalFit,
        cross_section: SpatialFit,
        graph: GraphMLFit,
        observed: ObservedSpatialPanelFit,
        panel: SpatialPanelFit,
        spatial_survival: SpatialSurvivalFit,
    ) -> None:
        accepts(glm)
        accepts(generalised)
        accepts(survival)
        accepts(cross_section)
        accepts(graph)
        accepts(observed)
        accepts(panel)
        accepts(spatial_survival)

    assert callable(check)
