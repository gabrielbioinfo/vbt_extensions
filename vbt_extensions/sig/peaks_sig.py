# vbt_extensions/sig/peaks_sig.py
import pandas as pd


def peaks_breakout_sig(
    close: pd.Series, swing_high: pd.Series, swing_low: pd.Series, *, confirm_shift: int = 1
):
    """Compra no rompimento do último swing_low e sai no rompimento do swing_high."""
    close = close.astype(float)
    swing_high = swing_high.reindex_like(close)
    swing_low = swing_low.reindex_like(close)

    entries = (close > swing_high.shift(confirm_shift)).fillna(False)
    exits = (close < swing_low.shift(confirm_shift)).fillna(False)
    return entries.astype(bool), exits.astype(bool)


def peaks_reversal_sig(is_top: pd.Series, is_bottom: pd.Series, *, delay: int = 1):
    """Compra em fundo detectado e sai em topo detectado (com pequeno atraso)."""
    entries = is_bottom.shift(delay).fillna(False).astype(bool)
    exits = is_top.shift(delay).fillna(False).astype(bool)
    return entries, exits
