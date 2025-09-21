import pandas as pd

from vbt_extensions.strategies.signal_golden_cross import signal_golden_cross


def test_signal_golden_cross_basic():
    close = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range("2023-01-01", periods=10, freq="D")
    )
    fast_list = [2, 3]
    slow_list = [4, 5]
    entries, exits = signal_golden_cross(close, fast_list, slow_list, mode="cartesian", style="trend")
    assert isinstance(entries, pd.DataFrame)
    assert isinstance(exits, pd.DataFrame)
    assert entries.shape[0] == 10
    assert exits.shape[0] == 10
    # Columns should be MultiIndex with fast and slow
    assert isinstance(entries.columns, pd.MultiIndex)
    assert set(entries.columns.names) == {"fast", "slow"}
    assert entries.shape[1] == len([f for f in fast_list for s in slow_list if f < s])
    # All entries/exits should be boolean
    assert entries.dtypes.apply(lambda x: x == bool).all()
    assert exits.dtypes.apply(lambda x: x == bool).all()
