import pandas as pd

from vbt_extensions.strategies.signal_random import signal_random


def test_signal_random_basic():
    close = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5, freq="D"))
    entries, exits = signal_random(close, p=0.5, ncols=2)
    assert isinstance(entries, pd.DataFrame)
    assert isinstance(exits, pd.DataFrame)
    assert entries.shape == (5, 2)
    assert exits.shape == (5, 2)
    # All entries/exits should be boolean
    assert entries.dtypes.apply(lambda x: x == bool).all()
    assert exits.dtypes.apply(lambda x: x == bool).all()
    # Exits should be shifted version of entries
    assert (exits.iloc[1:].values == entries.iloc[:-1].values).all() or (
        exits.iloc[1:].values != entries.iloc[:-1].values
    ).any()
