import pandas as pd

from vbt_extensions.ind.zigzag import ZigZagResult


def zigzag_breakout_sig(
    close: pd.Series, swing_high: pd.Series, swing_low: pd.Series, *, confirm_shift: int = 1
):
    """Compra no rompimento do último swing_high, sai no rompimento do swing_low."""
    close = close.astype(float)
    swing_high = swing_high.reindex_like(close)
    swing_low = swing_low.reindex_like(close)

    entries = (close > swing_high.shift(confirm_shift)).fillna(False)
    exits = (close < swing_low.shift(confirm_shift)).fillna(False)
    return entries.astype(bool), exits.astype(bool)


def zigzag_reversal_sig(zz: ZigZagResult, *, delay: int = 1):
    """Compra em fundos detectados, sai em topos detectados (com atraso opcional)."""
    entries = zz.is_bottom.shift(delay).fillna(False).astype(bool)
    exits = zz.is_top.shift(delay).fillna(False).astype(bool)
    return entries, exits
