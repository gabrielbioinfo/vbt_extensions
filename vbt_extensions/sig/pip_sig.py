import pandas as pd


def pip_turning_points_sig(pip_mask: pd.Series, *, delay: int = 1):
    """Entra sempre que surge um PIP (toy). Melhor combinar com S/R/regime."""
    entries = pip_mask.shift(delay).fillna(False).astype(bool)
    exits = pd.Series(False, index=pip_mask.index)  # deixe para risk overlays
    return entries, exits


def pip_reconstruction_breakout_sig(close: pd.Series, reconstructed: pd.Series, *, confirm_shift: int = 1):
    """Compra quando CLOSE cruza acima da reconstrução; sai quando cruza abaixo."""
    close = close.astype(float)
    rec = reconstructed.reindex_like(close)
    cross_up = (close > rec) & (close.shift(1) <= rec.shift(1))
    cross_dn = (close < rec) & (close.shift(1) >= rec.shift(1))
    entries = cross_up.shift(confirm_shift).fillna(False).astype(bool)
    exits = cross_dn.shift(confirm_shift).fillna(False).astype(bool)
    return entries, exits
