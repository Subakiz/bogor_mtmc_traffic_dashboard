"""The congestion index's levels of service: the one place the thresholds are defined.

The backend and the dataset export call :func:`congestion_label`; the dashboard shows the label it
is sent and keeps only the colours.
"""
from typing import Optional

# (upper bound, label): an index below the bound has that label; the last level has no bound.
CONGESTION_LEVELS = ((0.25, "Smooth"), (0.55, "Moderate"), (0.80, "Heavy"))
TOP_LEVEL = "Congested"


def congestion_label(index: Optional[float]) -> Optional[str]:
    """Level of service of a congestion index in [0, 1] (None for no reading)."""
    if index is None:
        return None
    for bound, label in CONGESTION_LEVELS:
        if index < bound:
            return label
    return TOP_LEVEL
