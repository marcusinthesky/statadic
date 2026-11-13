"""Declarative prediction artifacts and rendering-only analysis snapshots."""

from ._graph_prediction import (
    FrozenGraphPredictor,
    load_graph_predictor,
    save_graph_predictor,
)
from ._graph_recipes import GraphArchive, load_graphs, save_graphs
from ._ordination import (
    OrdinationArtifact,
    load_ordination_predictor,
    save_ordination_predictor,
)
from ._prediction import (
    PredictionArtifact,
    load_predictor,
    save_predictor,
)
from ._snapshot import AnalysisSnapshot, load_analysis, save_analysis
from ._store import ArtifactError

__all__ = [
    "AnalysisSnapshot",
    "ArtifactError",
    "FrozenGraphPredictor",
    "GraphArchive",
    "OrdinationArtifact",
    "PredictionArtifact",
    "load_analysis",
    "load_graph_predictor",
    "load_graphs",
    "load_ordination_predictor",
    "load_predictor",
    "save_analysis",
    "save_graph_predictor",
    "save_graphs",
    "save_ordination_predictor",
    "save_predictor",
]
