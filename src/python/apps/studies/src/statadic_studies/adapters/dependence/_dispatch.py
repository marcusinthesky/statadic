"""Scientific adapter functions grouped by computational ownership."""

from statadic_studies.adapters.dependence.dyadic import _dyadic
from statadic_studies.adapters.dependence.linear_panel import _panel, _wls
from statadic_studies.adapters.dependence.spatial import _sar, _slx


def run_dependence(study, scenario, replicate, seed):
    """Dispatch declared weighted/dependent-procedure studies."""
    adapters = {
        "linear.weighted-inference": _wls,
        "panel.static-estimation": _panel,
        "spatial.sar-inference": _sar,
        "spatial.slx-inference": _slx,
        "dyadic.point-estimation": _dyadic,
    }
    if study not in adapters:
        raise ValueError(f"No dependent-method adapter for {study}")
    return adapters[study](scenario, replicate, seed)
