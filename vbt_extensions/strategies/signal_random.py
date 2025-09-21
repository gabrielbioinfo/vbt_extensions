"""Module for generating random entry/exit signals for testing risk management strategies."""

import numpy as np
import pandas as pd


def signal_random(close: pd.Series, p: float = 0.02, ncols: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate random entry and exit signals for testing risk management strategies.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices (used for index).
    p : float, optional
        Probability of entry signal at each bar (default is 0.02).
    ncols : int, optional
        Number of random signal columns to generate (default is 1).

    Returns
    -------
    tuple of pd.DataFrame
        Entries and exits DataFrames with random signals.

    """
    idx = close.index
    cols = [(i,) for i in range(ncols)]
    entries = pd.DataFrame(
        np.random.rand(len(idx), ncols) < p,  # noqa: NPY002
        index=idx,
        columns=pd.MultiIndex.from_tuples(cols, names=["rand"]),
    )
    # exit by crossing with 'shift' (closes on the next bar if entered)
    exits = entries.shift(1).fillna(value=False)
    return entries, exits
