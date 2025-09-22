import pandas as pd


def apply_time_stop(entries: pd.Series, *, max_bars: int = 20) -> pd.Series:
    """Sai após 'max_bars' desde a entrada (time stop)."""
    entries = entries.fillna(False).astype(bool)
    out = pd.Series(False, index=entries.index)

    bars_in_trade = 0
    in_pos = False

    for i, e in enumerate(entries.values):
        if e and not in_pos:
            in_pos = True
            bars_in_trade = 0
        if in_pos:
            bars_in_trade += 1
            if bars_in_trade >= max_bars:
                out.iat[i] = True
                in_pos = False
                bars_in_trade = 0
    return out
