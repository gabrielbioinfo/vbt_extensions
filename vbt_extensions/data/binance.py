"""Functions for downloading data from Binance."""

import time
from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt
from binance.client import Client


def _ensure_tz_aware(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    if tz:
        df = df.tz_convert(tz)
    return df


def _fix_gaps(df: pd.DataFrame, interval: str, how: str = "ffill") -> pd.DataFrame:
    freq_map = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "8h": "8H",
        "12h": "12H",
        "1d": "1D",
    }
    freq = freq_map.get(interval)
    if freq is None:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_idx)
    if how == "ffill":
        df = df.ffill()
    elif how == "bfill":
        df = df.bfill()
    return df


@dataclass
class BinanceDownloadParams:
    """Parameters for downloading OHLCV data from Binance.

    Attributes
    ----------
    client : Client
        Binance API client instance.
    ticker : str
        Symbol to download (e.g., 'BTCUSDT').
    interval : str
        Data interval (e.g., '1h').
    start : str
        Start time for data download.
    end : str | None
        End time for data download.
    tz : str
        Timezone for returned data. ex: 'UTC', 'America/Sao_Paulo'.
    retries : int
        Number of download retries on failure.
    sleep_sec : float
        Seconds to sleep between retries.
    fill_gaps : bool
        Whether to fill missing time intervals in the data.

    """

    client: Client
    ticker: str
    interval: str = "1h"
    start: str = "1 year ago UTC"
    end: str | None = None
    tz: str = "UTC"
    retries: int = 3
    sleep_sec: float = 1.5
    fill_gaps: bool = True


class BinanceDownloadError(RuntimeError):
    """Custom exception raised when Binance data download fails after all retries."""

    def __init__(self, ticker: str, last_err: Exception) -> None:
        """Initialize BinanceDownloadError with ticker and last error.

        Parameters
        ----------
        ticker : str
            The ticker symbol that failed to load.
        last_err : Exception
            The last exception encountered during download.

        """
        msg = f"Failed to load {ticker}: {last_err}"
        super().__init__(msg)


def binance_download(params: BinanceDownloadParams) -> pd.DataFrame:
    """Download OHLCV data from Binance using vectorbt, with options for timezone, retries, and gap filling.

    Parameters
    ----------
    params : BinanceDownloadParams
        Parameters for the Binance data download.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns: Open, High, Low, Close, Volume.

    Raises
    ------
    BinanceDownloadError
        If data download fails after all retries.

    """
    last_err = None
    for _ in range(params.retries):
        try:
            data = vbt.BinanceData.download(
                params.ticker,
                client=params.client,
                interval=params.interval,
                start=params.start,
                end=params.end,
            ).get()
            # padroniza nomes e tipos
            cols = {c: c.capitalize() for c in data.columns}  # open->Open etc
            data = data.rename(columns=cols).sort_index()
            data = _ensure_tz_aware(data, params.tz)
            if params.fill_gaps:
                data = _fix_gaps(data, params.interval, how="ffill")
            # garante numéricos
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if c in data:
                    data[c] = pd.to_numeric(data[c], errors="coerce")
            return data[["Open", "High", "Low", "Close", "Volume"]]
        except (ConnectionError, TimeoutError, ValueError) as e:
            last_err = e
            time.sleep(params.sleep_sec)

    err_msg = BinanceDownloadError(params.ticker, last_err)
    raise err_msg
