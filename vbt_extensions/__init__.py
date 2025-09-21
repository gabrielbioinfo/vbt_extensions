"""vbt_extensions package.

This package provides extensions for vectorbt, including custom indicators.
"""

from .analyzers.qindex_rank import PriceBenchmarkRequiredError, qindex_rank
from .ind.zigzag import zigzag
from .strategies.signal_golden_cross import (
    ModeMissingListError,
    ModeRequiredError,
    StyleRequiredError,
    signal_golden_cross,
)
from .strategies.signal_random import signal_random

__all__ = [
    "ModeMissingListError",
    "ModeRequiredError",
    "PriceBenchmarkRequiredError",
    "StyleRequiredError",
    "zigzag",
    "qindex_rank",
    "signal_golden_cross",
    "signal_random",
]
