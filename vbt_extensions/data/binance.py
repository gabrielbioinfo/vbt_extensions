# vbt_extensions/data/binance.py
"""Binance OHLCV loader (pandas-first) built on top of vbt_extensions.data.base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import vectorbt as vbt

from .base import (
    BaseDownloadParams,
    backoff_sleep,
    drop_partial_last_candle,
    ensure_tz_aware,
    fix_gaps,
    maybe_resample,
    register_loader,
    validate_and_cast,
)

try:
    # Tipos/exceções do python-binance (se instalado)
    from binance.client import Client  # type: ignore
    from binance.error import BinanceAPIException, BinanceRequestException  # type: ignore
except Exception:  # pacote pode não estar disponível no ambiente
    Client = object  # fallback para typing
    BinanceAPIException = tuple()  # type: ignore
    BinanceRequestException = tuple()  # type: ignore


# --------------------- Params & Error ---------------------


@dataclass
class BinanceDownloadParams(BaseDownloadParams):
    """Params para Binance (extende BaseDownloadParams).

    Campos herdados (principais):
      - symbol: ex. "BTCUSDT"
      - interval: ex. "1h"
      - start/end: ex. "1 year ago UTC"
      - tz, retries, sleep_sec, fill_gaps, drop_partial_last, resample_to, freq_map
    """

    client: Client | None = None  # obrigatório na prática


class BinanceDownloadError(RuntimeError):
    """Erro quando falha após todas as tentativas."""

    def __init__(self, symbol: str, last_err: Exception) -> None:
        super().__init__(f"Failed to load {symbol}: {last_err!r}")


# --------------------- Loader principal ---------------------


def binance_download(params: BinanceDownloadParams) -> pd.DataFrame:
    """Baixa OHLCV da Binance via vectorbt, aplicando normalizações do 'base'."""
    if params.client is None:
        raise ValueError("BinanceDownloadParams.client não pode ser None.")

    last_err: Optional[Exception] = None

    for attempt in range(1, params.retries + 1):
        try:
            raw = vbt.BinanceData.download(
                params.symbol,
                client=params.client,
                interval=params.interval,
                start=params.start,
                end=params.end,
            ).get()

            df = validate_and_cast(raw)
            df = ensure_tz_aware(df, params.tz)

            if params.fill_gaps:
                df = fix_gaps(df, params.interval, freq_map=params.freq_map, how="ffill")

            if params.drop_partial_last:
                df = drop_partial_last_candle(df, params.interval, freq_map=params.freq_map)

            if params.resample_to:
                df = maybe_resample(df, params.resample_to)

            # metadados úteis
            df.attrs.update(
                source="binance",
                symbol=params.symbol,
                interval=params.interval,
                tz=params.tz,
            )
            return df

        except (ConnectionError, TimeoutError, ValueError) as e:
            last_err = e
        except BinanceRequestException as e:  # type: ignore[misc]
            last_err = e
        except BinanceAPIException as e:  # type: ignore[misc]
            last_err = e
        except Exception as e:
            last_err = e

        backoff_sleep(params.sleep_sec, attempt)

    raise BinanceDownloadError(params.symbol, last_err or RuntimeError("unknown error"))


# --------------------- Registry ---------------------


def _factory(p: BaseDownloadParams) -> pd.DataFrame:
    # Adaptador para o registry aceitar BaseDownloadParams
    bp = BinanceDownloadParams(**p.__dict__)  # copia campos comuns
    if getattr(p, "client", None) is None:
        raise ValueError("Para loader 'binance', é necessário fornecer 'client' em params.")
    bp.client = getattr(p, "client")
    return binance_download(bp)


register_loader("binance", _factory)


# from binance.client import Client
# from vbt_extensions.data.base import load_ohlcv, BaseDownloadParams

# client = Client(api_key="...", api_secret="...")
# df = load_ohlcv("binance", BaseDownloadParams(
#     symbol="BTCUSDT",
#     interval="1h",
#     start="1 year ago UTC",
#     tz="America/Sao_Paulo",
# ))
