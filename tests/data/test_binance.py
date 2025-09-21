"""Tests for Binance data download functionality."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from vbt_extensions.data.binance import BinanceDownloadError, BinanceDownloadParams, binance_download


def test_binance_download_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful download of Binance data using binance_download.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for patching objects.

    Returns
    -------
    None

    """
    # Mock vbt.BinanceData.download to return a DataFrame
    mock_df = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0.5, 1], "close": [1.5, 2.5], "volume": [100, 200]},
        index=pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00"]),
    )

    class MockBinanceData:
        def get(self) -> pd.DataFrame:
            return mock_df

    monkeypatch.setattr("vectorbt.BinanceData.download", lambda *a, **kw: MockBinanceData())
    params = BinanceDownloadParams(
        client=MagicMock(), ticker="BTCUSDT", interval="1h", start="2023-01-01", end="2023-01-02"
    )
    df = binance_download(params)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}
    assert len(df) == 2


def test_binance_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test failure scenario for Binance data download.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for patching objects.

    Returns
    -------
    None

    Raises
    ------
    BinanceDownloadError
        If the download fails.

    """

    # Mock vbt.BinanceData.download to raise ConnectionError
    def raise_conn_error(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG001
        msg = "fail"
        raise ConnectionError(msg)

    monkeypatch.setattr("vectorbt.BinanceData.download", raise_conn_error)
    params = BinanceDownloadParams(
        client=MagicMock(),
        ticker="BTCUSDT",
        interval="1h",
        retries=2,
        sleep_sec=0.01,
    )
    with pytest.raises(BinanceDownloadError) as exc:
        binance_download(params)
    assert "Failed to load BTCUSDT" in str(exc.value)
