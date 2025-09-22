# vbt_extensions/sig/sr_sig.py
import pandas as pd


def sr_touch_reversal_sig(close, support_levels, resistance_levels, *, tol=0.002):
    """Compra quando toca suporte; sai quando toca resistência (com tolerância)."""
    close = close.astype(float)
    sup = support_levels.reindex_like(close)
    res = resistance_levels.reindex_like(close)

    touch_sup = (abs(close - sup) / sup) <= tol
    touch_res = (abs(close - res) / res) <= tol
    entries = touch_sup.fillna(False).astype(bool)
    exits = touch_res.fillna(False).astype(bool)
    return entries, exits


def sr_breakout_sig(close, resistance_levels, *, confirm_shift=1):
    """Compra no breakout da resistência mais recente."""
    res = resistance_levels.reindex_like(close)
    entries = (close > res.shift(confirm_shift)).fillna(False).astype(bool)
    return entries, pd.Series(False, index=close.index)
