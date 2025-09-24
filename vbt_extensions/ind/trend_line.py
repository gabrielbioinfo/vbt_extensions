# vbt_extensions/ind/trend_line.py
"""Trend lines (suporte/resistência) por janela deslizante — pandas-first.

Mantém o núcleo Numba do seu código, mas expõe uma API legível:
- Série da linha de suporte em PREÇO (support_level)
- Série da linha de resistência em PREÇO (resist_level)
- Slopes (úteis p/ classificar: alta/baixa/lateral)
- (opcional) flags de toque/rompimento simples

Ideia:
1) Para cada janela [i-lookback+1, i], ajustamos:
   - uma linha de tendência de baixa tocando os TOPOS (resistência)
   - uma linha de tendência de alta tocando os FUNDOS (suporte)
   usando sua rotina de otimização.

2) O wrapper transforma (slope, intercept) → valores em preço
   no candle i: level[i] = slope[i] * i + intercept[i]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        def deco(f):
            return f

        return deco


# ================= núcleo (seu código, com pequenos ajustes) ================= #


@njit(cache=True)
def _check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    """Retorna soma dos quadrados dos desvios se a linha respeita o papel;
    senão retorna -1. support=True -> todos pontos <= linha (suporte),
    support=False -> todos pontos >= linha (resistência)."""
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(y.shape[0]) + intercept
    diffs = line_vals - y
    if support and np.max(diffs) > 1e-5:
        return -1.0
    elif not support and np.min(diffs) < -1e-5:
        return -1.0
    return np.sum(diffs**2.0)


@njit(cache=True)
def _optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray) -> Tuple[float, float]:
    """Otimizador de slope respeitando a restrição (linha acima/abaixo dos pontos)."""
    slope_unit = (np.max(y) - np.min(y)) / max(1, y.shape[0])
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step
    best_slope = init_slope
    best_err = _check_trend_line(support, pivot, init_slope, y)
    if best_err < 0.0:
        return np.nan, np.nan
    get_derivative = True
    derivative = 0.0
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = _check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = _check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                break
            get_derivative = False
        test_slope = (
            best_slope - slope_unit * curr_step if derivative > 0.0 else best_slope + slope_unit * curr_step
        )
        test_err = _check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True
    return best_slope, -best_slope * pivot + y[pivot]


@njit(cache=True)
def _fit_trendlines_high_low(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Ajusta coefs (slope, intercept) para suporte (fundos) e resistência (topos) na janela."""
    x = np.arange(close.shape[0])
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = np.argmax(high - line_points)
    lower_pivot = np.argmin(low - line_points)
    support_coefs = _optimize_slope(True, lower_pivot, coefs[0], low)  # linha sob os lows
    resist_coefs = _optimize_slope(False, upper_pivot, coefs[0], high)  # linha sobre os highs
    return support_coefs, resist_coefs


@njit(cache=True)
def _trend_line_1d(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Varre a série e retorna arrays de slope/intercept ao final de cada janela."""
    n = close.shape[0]
    sup_slope = np.full(n, np.nan, dtype=np.float64)
    sup_int = np.full(n, np.nan, dtype=np.float64)
    res_slope = np.full(n, np.nan, dtype=np.float64)
    res_int = np.full(n, np.nan, dtype=np.float64)

    for i in range(lookback - 1, n):
        h = high[i - lookback + 1 : i + 1]
        l = low[i - lookback + 1 : i + 1]  # noqa: E741
        c = close[i - lookback + 1 : i + 1]
        sup_coefs, res_coefs = _fit_trendlines_high_low(h, l, c)
        sup_slope[i], sup_int[i] = sup_coefs
        res_slope[i], res_int[i] = res_coefs
    return sup_slope, sup_int, res_slope, res_int


# ================= dataclass + wrapper pandas-first ================= #


@dataclass
class TrendLineResult:
    support_level: pd.Series  # preço da linha de suporte no candle t
    resist_level: pd.Series  # preço da linha de resistência no candle t
    support_slope: pd.Series  # slope da linha de suporte (útil p/ classificar tendência)
    resist_slope: pd.Series
    touch_support: pd.Series | None = None  # True quando Low toca/abaixo (com tolerância)
    touch_resist: pd.Series | None = None  # True quando High toca/acima (com tolerância)
    break_up: pd.Series | None = None  # True quando Close cruza ACIMA da resistência
    break_down: pd.Series | None = None  # True quando Close cruza ABAIXO do suporte


class TREND_LINE:
    """Trend lines por janela: suporte (fundos) e resistência (topos).

    Parâmetros
    ----------
    high, low, close : pd.Series
    lookback : int
        Tamanho da janela (em barras).
    touch_tol : float, default 0.0
        Tolerância relativa p/ considerar 'toque' (ex.: 0.001 = 0,1%).
    compute_events : bool, default True
        Se True, computa toques e cruzamentos simples (breakouts).
    """

    @staticmethod
    def run(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        *,
        lookback: int = 30,
        touch_tol: float = 0.0,
        compute_events: bool = True,
    ) -> TrendLineResult:
        # validação leve
        if not (isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series)):
            raise TypeError("high, low, close devem ser pd.Series")

        # alinhar e tipar
        high = high.astype(float)
        low = low.reindex_like(high).astype(float)
        close = close.reindex_like(high).astype(float)

        # núcleo
        sup_slope, sup_int, res_slope, res_int = _trend_line_1d(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            int(lookback),
        )

        idx = close.index
        # níveis em preço no candle t: level[t] = slope[t]*t + intercept[t]
        t = np.arange(len(idx), dtype=np.float64)
        support_level = pd.Series(sup_slope * t + sup_int, index=idx, dtype=float)
        resist_level = pd.Series(res_slope * t + res_int, index=idx, dtype=float)

        # Series de slope p/ inspeção & classificação
        support_slope = pd.Series(sup_slope, index=idx, dtype=float)
        resist_slope = pd.Series(res_slope, index=idx, dtype=float)

        touch_support = touch_resist = break_up = break_down = None

        if compute_events:
            tol_sup = (support_level.abs() * touch_tol).fillna(0.0)
            tol_res = (resist_level.abs() * touch_tol).fillna(0.0)

            # toques: low <= sup+tol ; high >= res - tol
            touch_support = (low <= (support_level + tol_sup)) & support_level.notna()
            touch_resist = (high >= (resist_level - tol_res)) & resist_level.notna()

            # cruzamentos (usando close): crossup resistência, crossdown suporte
            above_res_now = close > resist_level
            above_res_prev = close.shift(1) > resist_level.shift(1)
            break_up = above_res_now & (~above_res_prev) & resist_level.notna()

            below_sup_now = close < support_level
            below_sup_prev = close.shift(1) < support_level.shift(1)
            break_down = below_sup_now & (~below_sup_prev) & support_level.notna()

            touch_support = touch_support.fillna(False).astype(bool)
            touch_resist = touch_resist.fillna(False).astype(bool)
            break_up = break_up.fillna(False).astype(bool)
            break_down = break_down.fillna(False).astype(bool)

        return TrendLineResult(
            support_level=support_level,
            resist_level=resist_level,
            support_slope=support_slope,
            resist_slope=resist_slope,
            touch_support=touch_support,
            touch_resist=touch_resist,
            break_up=break_up,
            break_down=break_down,
        )
