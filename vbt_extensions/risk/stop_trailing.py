# vbt_extensions/risk/stop_trailing.py
"""Módulo de Trailing Stops.

Fornece funções para calcular níveis dinâmicos de stop:
- stop baseado em percentual fixo
- stop baseado em múltiplos de ATR
- stop baseado em high/low da janela

Cada função retorna uma Series/DataFrame com os níveis de stop ao longo do tempo,
para ser usado em backtests (ex.: Portfolio.from_signals com `sl_stop` dinâmico).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pct_trailing(close: pd.Series, pct: float = 0.02, long: bool = True) -> pd.Series:
    """Trailing stop baseado em percentual do preço.

    Parameters
    ----------
    close : pd.Series
        Série de preços de fechamento.
    pct : float
        Percentual de distância do preço (ex.: 0.02 = 2%).
    long : bool
        Se True, calcula trailing para posições compradas.
        Se False, para vendidas.

    Returns
    -------
    pd.Series
        Série com níveis de stop.
    """
    if long:
        trail = (close * (1 - pct)).cummax()
    else:
        trail = (close * (1 + pct)).cummin()
    return trail


def atr_trailing(high: pd.Series, low: pd.Series, close: pd.Series,
                 window: int = 14, mult: float = 3.0, long: bool = True) -> pd.Series:
    """Trailing stop baseado em ATR.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC.
    window : int
        Janela de ATR.
    mult : float
        Multiplicador do ATR.
    long : bool
        Direção (long/short).

    Returns
    -------
    pd.Series
        Níveis de stop.
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    if long:
        base = close - mult * atr
        return base.cummax()
    else:
        base = close + mult * atr
        return base.cummin()


def rolling_extrema_trailing(close: pd.Series, window: int = 20, long: bool = True) -> pd.Series:
    """Trailing stop baseado em máximas/mínimas da janela.

    Parameters
    ----------
    close : pd.Series
        Série de preços.
    window : int
        Janela de lookback.
    long : bool
        Direção.

    Returns
    -------
    pd.Series
        Série de stops trailing.
    """
    if long:
        return close.rolling(window).min().cummax()
    else:
        return close.rolling(window).max().cummin()
