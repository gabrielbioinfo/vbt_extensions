"""Overlays e ferramentas de gestão de risco."""

from .breakeven import apply_breakeven_exit
from .fibo_stop import fibo_protective_stop
from .position_sizing import fixed_fractional_sizing
from .session_filters import session_filter
from .stop_trailing import trailing_stop
from .targets_stops_smc import smc_targets_stops
from .time_stop import time_stop

__all__ = [
    "apply_breakeven_exit",
    "fibo_protective_stop",
    "fixed_fractional_sizing",
    "session_filter",
    "trailing_stop",
    "time_stop",
    "smc_targets_stops"
]
