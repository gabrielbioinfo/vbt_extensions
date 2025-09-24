"""Overlays e ferramentas de gestão de risco."""
from .breakeven import apply_breakeven_exit
from .exit_engine import ExitConfig, ExitResult, compute_atr, simulate_exits
from .fibo_stop import fibo_protective_stop
from .position_sizing import fixed_amount, fixed_fraction, kelly_fraction, vol_target
from .session_filters import apply_session_mask, filter_sessions, session_mask
from .stop_trailing import atr_trailing, pct_trailing, rolling_extrema_trailing
from .targets_stops_smc import smc_targets_stops
from .time_stop import apply_time_stop

__all__ = [
	"apply_breakeven_exit",
	"compute_atr", "simulate_exits", "ExitConfig", "ExitResult",
	"fibo_protective_stop",
	"fixed_fraction", "fixed_amount", "kelly_fraction", "vol_target",
	"filter_sessions", "session_mask", "apply_session_mask",
	"pct_trailing", "atr_trailing", "rolling_extrema_trailing",
	"smc_targets_stops",
	"apply_time_stop",
]
