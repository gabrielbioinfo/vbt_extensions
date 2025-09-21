"""Unit tests for qindex_rank function in vbt_extensions.analyze.qindex_rank."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from vbt_extensions.analyzers.qindex_rank import PriceBenchmarkRequiredError, qindex_rank


class DummyPortfolio:
    """A dummy portfolio class for testing qindex_rank.

    Simulates a portfolio with two strategies ("A" and "B"), providing mock returns and stats.
    """

    def __init__(self) -> None:
        """Initialize the DummyPortfolio with mock columns and returns."""
        self.wrapper = MagicMock()
        # Use a pandas Index to allow .names attribute
        self.wrapper.columns = pd.Index(["A", "B"], name="strategy")
        self._returns = pd.DataFrame(
            {
                "A": [0.01, 0.02, 0.03],
                "B": [0.02, 0.01, 0.00],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="1h"),
        )

    def returns(self) -> pd.DataFrame:
        """Return the mock returns DataFrame for the dummy portfolio.

        Returns
        -------
        pd.DataFrame
            The returns DataFrame with strategies as columns.

        """
        return self._returns

    def stats(self, column: str | None = None) -> dict:
        """Return dummy stats for the specified strategy column.

        Parameters
        ----------
        column : str or None, optional
            The strategy column name ("A" or "B"). If None, returns stats for default.

        Returns
        -------
        dict
            Dictionary containing mock statistics for the given strategy.

        """
        # Return dummy stats for each column
        return {
            "Total Return [%]": 10.0 if column == "A" else 5.0,
            "Sharpe Ratio": 1.5 if column == "A" else 1.0,
            "Max Drawdown [%]": 2.0 if column == "A" else 3.0,
            "Win Rate [%]": 60.0 if column == "A" else 50.0,
            "Total Trades": 20 if column == "A" else 15,
            "Exposure [%]": 90.0 if column == "A" else 80.0,
        }


def test_qindex_rank_success() -> None:
    """Test qindex_rank with valid inputs and check expected output structure."""
    pf = DummyPortfolio()
    price_benchmark = pd.Series([100, 101, 102], index=pf._returns.index)
    df = qindex_rank(pf, price_benchmark=price_benchmark, freq="1h")
    assert isinstance(df, pd.DataFrame)
    assert "qspot_index" in df.columns
    assert len(df) == 2
    assert set(df["strategy"]) == {"A", "B"}


def test_qindex_rank_missing_benchmark() -> None:
    """Test that qindex_rank raises PriceBenchmarkRequiredError when price_benchmark is missing."""
    pf = DummyPortfolio()
    with pytest.raises(PriceBenchmarkRequiredError):
        qindex_rank(pf)
