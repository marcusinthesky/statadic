"""Native population operating characteristics with independent targets."""

from __future__ import annotations

import numpy as np


def _record(value, truth, draws=None):
    value, truth = float(value), float(truth)
    effort = {} if draws is None else {"inner_draws": draws}
    if not np.isfinite([value, truth]).all():
        return {"valid": False, "failure": "nonfinite-result", **effort}
    return {"valid": True, "estimate": value, "truth": truth, **effort}
