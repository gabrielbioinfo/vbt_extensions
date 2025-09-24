"""Sinais derivados de indicadores e price action."""


from .head_and_shoulders_sig import hs_exit_on_opposite_break, hs_signals_from_result
from .peaks_sig import peaks_breakout_sig, peaks_reversal_sig
from .pip_sig import pip_reconstruction_breakout_sig, pip_turning_points_sig
from .random_sig import random_sig
from .retracement_sig import fib_grid_breakout_sig, fib_mean_revert_sig, fib_trend_continuation_sig
from .smart_money_sig import bos_breakout_sig, choch_reversal_sig, combo_smart_money, fvg_fill_reversion_sig
from .sr_sig import sr_breakout_sig, sr_touch_reversal_sig
from .trend_line_sig import (
                                     classify_trend_regime,
                                     classify_trend_regime_adv,
                                     tl_breakout_sig,
                                     tl_breakout_with_regime,
                                     tl_side_aware_sig,
                                     tl_touch_reversal_sig,
                                     tl_trailing_channel_sig,
)
from .zigzag_sig import zigzag_breakout_sig, zigzag_reversal_sig

__all__ = [
    "bos_breakout_sig",
    "choch_reversal_sig",
    "classify_trend_regime",
    "classify_trend_regime_adv",
    "combo_smart_money",
    "fib_grid_breakout_sig",
    "fib_mean_revert_sig",
    "fib_trend_continuation_sig",
    "fvg_fill_reversion_sig",
    "hs_exit_on_opposite_break",
    "hs_signals_from_result",
    "peaks_breakout_sig",
    "peaks_reversal_sig",
    "pip_reconstruction_breakout_sig",
    "pip_turning_points_sig",
    "random_sig",
    "sr_breakout_sig",
    "sr_touch_reversal_sig",
    "tl_breakout_sig",
    "tl_breakout_with_regime",
    "tl_side_aware_sig",
    "tl_touch_reversal_sig",
    "tl_trailing_channel_sig",
    "zigzag_breakout_sig",
    "zigzag_reversal_sig",
]
