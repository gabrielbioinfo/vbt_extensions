# vbt_extensions/data/base.py
"""
Base interfaces & helpers for OHLCV data loaders (pandas-first).

All loaders in vbt_extensions.data.* should:
- Accept a dataclass Params (ideally extending BaseDownloadParams)
- Return a pandas.DataFrame with columns: Open, High, Low, Close, Volume
- Use the helpers here to ensure consistency across sources
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

# ------------------------- Constants -------------------------

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Common mapping example; each concrete loader can extend/override
DEFAULT_TO_PD_FREQ = {
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
    "3d": "3D",
    "1w": "7D",
    "1M": "MS",
}


# ------------------------- Params -------------------------


@dataclass
class BaseDownloadParams:
    """Common parameters for OHLCV download.

    Implementations can subclass and add fields (e.g., credentials).
    """

    symbol: str
    interval: str = "1h"
    start: str | None = "1 year ago UTC"
    end: str | None = None
    tz: str = "UTC"
    retries: int = 3
    sleep_sec: float = 1.5
    fill_gaps: bool = True
    drop_partial_last: bool = True
    resample_to: str | None = None
    # freq_map allows per-loader overrides
    freq_map: dict | None = None


# ------------------------- Protocol (contract) -------------------------


class DataLoader(Protocol):
    """Contract for data loaders."""

    def download(self, params: BaseDownloadParams) -> pd.DataFrame:  # noqa: D401
        """Return OHLCV DataFrame (Open, High, Low, Close, Volume) with tz-aware index."""
        ...


# ------------------------- Shared helpers -------------------------


def ensure_tz_aware(df: pd.DataFrame, tz: str | None) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    if tz:
        df = df.tz_convert(tz)
    return df


def validate_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names/order and cast to numeric."""
    cols_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=cols_map).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for c in REQUIRED_COLS:
        if c not in df:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[REQUIRED_COLS]


def fix_gaps(
    df: pd.DataFrame, interval: str, freq_map: dict | None = None, how: str = "ffill"
) -> pd.DataFrame:
    fmap = freq_map or DEFAULT_TO_PD_FREQ
    freq = fmap.get(interval)
    if not freq or df.empty:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    out = df.reindex(full_idx)
    if how == "ffill":
        out = out.ffill()
    elif how == "bfill":
        out = out.bfill()
    return out


def drop_partial_last_candle(
    df: pd.DataFrame, interval: str, freq_map: dict | None = None
) -> pd.DataFrame:
    if df.empty:
        return df
    fmap = freq_map or DEFAULT_TO_PD_FREQ
    freq = fmap.get(interval)
    if not freq:
        return df
    last = df.index[-1]
    try:
        next_edge = last.floor(freq) + pd.tseries.frequencies.to_offset(freq)
    except Exception:
        return df
    now = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(last.tz)
    if now < next_edge:
        return df.iloc[:-1]
    return df


def maybe_resample(df: pd.DataFrame, target_freq: str | None) -> pd.DataFrame:
    if not target_freq or df.empty:
        return df
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(target_freq, label="right", closed="right").apply(ohlc).dropna(how="all")
    return out


def backoff_sleep(base: float, attempt: int) -> None:
    """Exponential backoff with jitter."""
    sleep = base * (2 ** (attempt - 1)) * (1 + 0.25 * random.random())
    time.sleep(sleep)


# ------------------------- Optional registry -------------------------

LoaderFactory = Callable[[BaseDownloadParams], pd.DataFrame]
_LOADER_REGISTRY: dict[str, LoaderFactory] = {}


def register_loader(name: str, factory: LoaderFactory) -> None:
    """Register a loader by name (e.g., 'binance', 'csv', 'b3')."""
    _LOADER_REGISTRY[name] = factory


def load_ohlcv(loader_name: str, params: BaseDownloadParams) -> pd.DataFrame:
    """Convenience wrapper using the registry."""
    if loader_name not in _LOADER_REGISTRY:
        raise KeyError(f"Loader '{loader_name}' is not registered.")
    return _LOADER_REGISTRY[loader_name](params)
