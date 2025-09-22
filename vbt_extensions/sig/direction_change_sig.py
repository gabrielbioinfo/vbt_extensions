# vbt_extensions/sig/direction_change_sig.py
import pandas as pd


def dc_reversal_sig(dc_event: pd.Series, *, delay: int = 1):
    """+1=topo, -1=fundo. Compra em fundo, sai em topo."""
    entries = dc_event.eq(-1).shift(delay).fillna(False).astype(bool)
    exits = dc_event.eq(1).shift(delay).fillna(False).astype(bool)
    return entries, exits


def dc_breakout_sig(close: pd.Series, swing_high: pd.Series, swing_low: pd.Series, *, confirm_shift: int = 1):
    """Compra no rompimento do swing_high DC; sai no rompimento do swing_low DC."""
    entries = (close > swing_high.shift(confirm_shift)).fillna(False).astype(bool)
    exits = (close < swing_low.shift(confirm_shift)).fillna(False).astype(bool)
    return entries, exits
