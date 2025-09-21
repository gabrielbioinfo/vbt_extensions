"""Strategies package for vbt_extensions.

This package provides strategy implementations for use with vbt_extensions.
"""

from .signal_golden_cross import (
    ModeMissingListError,
    ModeRequiredError,
    StyleRequiredError,
    signal_golden_cross,
)
from .signal_random import signal_random

__all__ = [
    "ModeMissingListError",
    "ModeRequiredError",
    "StyleRequiredError",
    "signal_golden_cross",
    "signal_random",
]
