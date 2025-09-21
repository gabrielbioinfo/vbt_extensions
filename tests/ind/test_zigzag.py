"""Unit tests for the ZigZag indicator in vbt_extensions.ind.zigzag."""

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from vbt_extensions.ind import ZIGZAG


def test_zigzag_vectorbt() -> None:
    """Test the ZigZag indicator with vectorbt integration."""
    # Create a simple DataFrame with float64 columns
    df = pd.DataFrame({
        "high": np.array([1, 2, 3, 2, 1, 2, 3, 2, 1], dtype=np.float64),
        "low": np.array([0.5, 1, 2, 1, 0.5, 1, 2, 1, 0.5], dtype=np.float64),
        "close": np.array([1, 1.8, 2.7, 2.1, 1.2, 2.1, 2.9, 2.2, 1.1], dtype=np.float64),
    })

    zz = ZIGZAG.run(
        high=df["high"].to_numpy(dtype=np.float64),
        low=df["low"].to_numpy(dtype=np.float64),
        upper=0.5,
        lower=0.5,
    )
    zig = zz.zigzag
    tops = zz.is_top.astype(bool)
    bots = zz.is_bottom.astype(bool)

    # Example signals: buy at bottoms, sell at tops
    entries = bots
    exits = tops

    pf = vbt.Portfolio.from_signals(close=df["close"], entries=entries, exits=exits)
    stats = pf.stats()
    assert isinstance(stats, pd.Series)

    zz = ZIGZAG.run(high=df["high"], low=df["low"], upper=0.5, lower=0.5)
    zig = zz.zigzag
    tops = zz.is_top.astype(bool)
    bots = zz.is_bottom.astype(bool)

    # Example signals: buy at bottoms, sell at tops
    entries = bots
    exits = tops

    pf = vbt.Portfolio.from_signals(close=df["close"], entries=entries, exits=exits)
    stats = pf.stats()
    assert isinstance(stats, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__])
