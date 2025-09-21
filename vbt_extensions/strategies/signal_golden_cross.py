"""Signal Golden Cross Strategy Module.

This module provides the signal_golden_cross function to generate entry and exit signals
based on moving average crossovers for trend and reversion strategies.
"""

import pandas as pd
import vectorbt as vbt


class ModeRequiredError(ValueError):
    """Exception raised when a required mode is missing for signal generation."""

    def __init__(self) -> None:
        """Initialize the ModeRequiredError with a standard message."""
        super().__init__(
            "Pass 'mode=cartesian' or 'mode=pair' to specify how to combine fast and slow lists.",
        )


class ModeMissingListError(ValueError):
    """Exception raised when a required mode is missing for signal generation."""

    def __init__(self) -> None:
        """Initialize the ModeMissingListError with a standard message."""
        super().__init__("Pass fast and slow lists.")


class StyleRequiredError(ValueError):
    """Exception raised when a required style is missing for signal generation."""

    def __init__(self) -> None:
        """Initialize the StyleRequiredError with a standard message."""
        super().__init__(
            "Pass 'style=trend' or 'style=reversion' to specify how to combine fast and slow lists.",
        )


def signal_golden_cross(
    close: pd.Series,
    fast_list: list,
    slow_list: list,
    mode: str = "cartesian",
    style: str = "trend",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate entry and exit signals based on moving average crossovers for trend and reversion strategies.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices.
    fast_list : list
        List of fast moving average window sizes.
    slow_list : list
        List of slow moving average window sizes.
    mode : str, optional
        'pair' for paired fast/slow lists, 'cartesian' for all combinations (default is 'cartesian').
    style : str, optional
        'trend' for trend following, 'reversion' for mean reversion (default is 'trend').

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (entries, exits) DataFrames with MultiIndex columns for (fast, slow) pairs.

    Raises
    ------
    ValueError
        If mode or style is invalid, or if fast_list and slow_list lengths mismatch in 'pair' mode.

    """
    if mode not in ("pair", "cartesian"):
        raise ModeRequiredError
    if style not in ("trend", "reversion"):
        raise StyleRequiredError

    if mode == "pair":
        if len(fast_list) != len(slow_list):
            raise ModeMissingListError
        pairs = [(int(f), int(s)) for f, s in zip(fast_list, slow_list)]  # noqa: B905
    else:
        pairs = [(int(f), int(s)) for f in fast_list for s in slow_list if int(f) < int(s)]

    def _above(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    def _below(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    ent_cols, ex_cols = [], []
    for f, s in pairs:
        fma = vbt.MA.run(close, window=f).ma.squeeze()
        sma = vbt.MA.run(close, window=s).ma.squeeze()
        if style == "trend":
            ent = _above(fma, sma)
            exi = _below(fma, sma)
        else:
            ent = _below(fma, sma)
            exi = _above(fma, sma)
        ent_cols.append(ent.rename((f, s)))
        ex_cols.append(exi.rename((f, s)))

    entries = pd.concat(ent_cols, axis=1)
    exits = pd.concat(ex_cols, axis=1)
    cols = pd.MultiIndex.from_tuples(entries.columns, names=["fast", "slow"])
    entries.columns = cols
    exits.columns = cols
    return entries.sort_index(axis=1), exits.sort_index(axis=1)
