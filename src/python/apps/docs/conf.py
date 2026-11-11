"""Sphinx configuration for the STATADIC learning site."""

import json
import os
from importlib.metadata import version as distribution_version
from pathlib import Path

project = "STATADIC"
release = distribution_version("statadic")
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "statadic_docs_labs.formal_extension",
]
bibtex_bibfiles = [str(Path(__file__).resolve().parents[4] / "src/latex/refs.bib")]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build"]
html_theme = "shibuya"
html_title = f"STATADIC {release} documentation"
templates_path = ["templates"]
html_sidebars = {"**": ["sidebars/localtoc.html"]}
html_static_path = ["static"]
html_css_files = ["statadic.css"]
html_logo = "static/brand/statadic-curve-fit-animated.svg"
html_favicon = html_logo
pygments_style = "bw"
pygments_dark_style = "bw"
html_theme_options = {
    "accent_color": "blue",
    "color_mode": "light",
    "show_ai_links": False,
    "nav_links": [
        {"title": title, "url": page}
        for title, page in (
            ("Start", "start/index"),
            ("Tutorials", "tutorials/index"),
            ("How-to", "guides/index"),
            ("Concepts", "concepts/index"),
            ("Reference", "reference/index"),
            ("Evidence", "validation/index"),
        )
    ],
}
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "description"
nitpicky = True

# Autodoc can emit unqualified names and class-role links for PEP 695 aliases.
# Resolve these annotations to real curated targets, using the type-alias role
# where appropriate rather than suppressing missing public API references.
_ANNOTATION_TARGETS = {
    "MetadataValue": ("type", "statadic.api.reporting.MetadataValue"),
    "FrozenMetadataValue": ("type", "statadic.api.reporting.FrozenMetadataValue"),
    "ReportNumber": ("type", "statadic.api.reporting.ReportNumber"),
    "Reference": ("type", "statadic.api.hypotheses.Reference"),
    "IVFit": ("class", "statadic.estimation.iv.IVFit"),
    "AffineGMMFit": ("class", "statadic.estimation.gmm.AffineGMMFit"),
    "VariabilityScheme": ("class", "statadic.core.equations.VariabilityScheme"),
}
_ANNOTATION_TARGETS.update(
    {target: (role, target) for role, target in tuple(_ANNOTATION_TARGETS.values())}
)


def _resolve_annotation(app, env, node, content):
    """Link known shortened annotations to their canonical public owner."""
    if node.get("refdomain") != "py":
        return None
    resolved = _ANNOTATION_TARGETS.get(node.get("reftarget"))
    if resolved is None:
        return None
    role, target = resolved
    return env.domains["py"].resolve_xref(
        env, node["refdoc"], app.builder, role, target, node, content
    )


def setup(app):
    """Preserve strict checking while resolving actual annotation targets."""
    app.connect("missing-reference", _resolve_annotation)


# The curated API does not yet provide pages for these annotation types and
# implementation-level references. Keep the exceptions explicit so new broken
# references still fail the strict build.
nitpick_ignore = [
    ("py:class", name)
    for name in (
        "",
        "n d",
        "m d",
        "AnovaExponent",
        "Array",
        "ArrayLike",
        "GroundDistanceSelection",
        "InputT",
        "EvidenceT",  # PEP 695 retained-evidence parameter, not a public class.
        "NullT",  # PEP 695 hypothesis parameter, not a public scientific object.
        "MeasureColumn",
        "RegressionInput",
        "ResultT",
        "RowId",
        "collections.abc.Mapping",
        "jax.Array",
        "jaxtyping.Float",
        "statadic.api.base.Estimator",
        "statadic.api.data.MeasureColumn",
        "statadic.api.design.Term",
        "statadic.core.domains.ClaimAuthority",
        "statadic.core.domains.SpaceDescriptor",
        "statadic.core.domains.ValueEvidence",
        "statadic.core.transforms.TransformDescriptor",
        "statadic.estimation.linear.LinearFit",
        "statadic.estimation.linear.LinearProblem",
        "statadic.geometry.ground._similarity_strategy.PositiveSemidefiniteKernelLaw",
        "statadic.geometry.ground._similarity_strategy.SimilarityStrategy",
    )
] + [
    ("py:data", "DECLARED_AXIOMS"),
    ("py:exc", "IncompatibleEnergyGroundExponentError"),
    ("py:func", "statadic.core.matrices.metrize_energy"),
]

# Additional types used by the tutorial API surface.
nitpick_ignore += [
    ("py:class", "''"),
    ("py:class", "'m d'"),
    ("py:class", "'n d'"),
    ("py:class", "ContextT"),
    ("py:class", "PackedClouds"),
    ("py:class", "FitT"),
    ("py:class", "GroupLabels"),
    ("py:class", "RowIds"),
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "pathlib.Path"),
    ("py:class", "Path"),  # Autodoc's shortened standard-library annotation.
    ("py:class", "statadic.api.comparison._results.GroupComparisonResult"),
    ("py:class", "statadic.api.comparison._shared.inputs.ComparisonInput"),
    ("py:class", "DesignLabels"),
    ("py:class", "InputRole"),
    ("py:class", "statadic.api.design.FittedDesign"),
    ("py:class", "statadic.api.inference.Clustered"),
    ("py:class", "statadic.api.inference.HAC"),
    ("py:class", "statadic.api.inference.InferenceResult"),
    ("py:class", "statadic.api.inference.ModelBased"),
    ("py:class", "statadic.api.representation.GeometrySpec"),
    ("py:class", "statadic.api.representation.PairwiseGeometry"),
    ("py:class", "statadic.api.representation.PreparedGeometry"),
    ("py:class", "statadic.sample.packed.PackedClouds"),
    ("py:class", "numpy.bool"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.number"),
    ("py:class", "numpy.dtype"),  # External annotation; no offline NumPy inventory.
]

# Types and helper links in the newly exposed formal-companion entry points.
# They have no dedicated API targets yet; retain exact exceptions rather than
# disabling strict reference checking for the rest of the site.
nitpick_ignore += [
    ("py:class", name)
    for name in (
        "Float",
        "SimilarityStrategy",
        "PositiveSemidefiniteKernelLaw",
        "' d'",
        "'sets sets'",
    )
] + [
    ("py:exc", "InsufficientUStatisticSampleError"),
    ("py:func", "mmd_matrix"),
    ("py:func", "statadic.sample.packed.pack"),
    ("py:func", "statadic.association.dispersion.dispersion_f_ratio"),
    ("py:func", "statadic.core.random.cell_key"),
]

# Autodoc emits class-role links for these private or PEP 695 union aliases.
# Public scientific classes have explicit targets in reference/api.rst.
nitpick_ignore += [
    ("py:class", name)
    for name in (
        "_Data",
        "_Dependence",
        "_Effects",
        "_Geometry",
        "ConditionalLaw",
        "AFTFamily",
        "RegressionEstimator",
        "RegressionFit",
    )
]

# Annotation types of the diagnostic request protocol and its report rows.
nitpick_ignore += [
    ("py:class", name)
    for name in (
        "Diagnostic",
        "FitContext",
        "FittedDesign",
        "LinearFit",
        "statadic.api.reporting.Metric",
        "statadic.api.reporting.Report",
    )
]

# Research apps remain optional for documentation consumers. When present, use
# their single authored catalogue rather than maintaining a second docs mapping.
_research_catalogue = Path(__file__).resolve().parents[1] / "studies/catalogue.json"
statadic_research_catalogue = (
    str(_research_catalogue) if _research_catalogue.is_file() else None
)
# The build/release runner selects exports explicitly. Unrelated local campaigns
# must never become current release evidence merely by existing under build/.
statadic_study_exports = json.loads(os.environ.get("STATADIC_DOCS_STUDY_EXPORTS", "[]"))
if not isinstance(statadic_study_exports, list) or not all(
    isinstance(path, str) for path in statadic_study_exports
):
    raise ValueError("STATADIC_DOCS_STUDY_EXPORTS must be a JSON list of paths")
statadic_release_evidence = os.environ.get("STATADIC_DOCS_RELEASE_EVIDENCE", "0") == "1"
