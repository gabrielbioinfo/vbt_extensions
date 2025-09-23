# vbt_extensions/risk/position_sizing.py
"""
Módulo de position sizing (dimensionamento de posição).

Fornece funções utilitárias para calcular tamanho de posição dado:
- risco fixo em % do capital
- risco fixo em valor monetário
- fração de Kelly
- sizing baseado em volatilidade (ATR ou desvio padrão)

Cada função retorna uma série/valor com a quantidade de unidades
a ser operada em cada trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_fraction(capital: float, risk_per_trade: float, entry: float, stop: float) -> int:
    """Calcula tamanho da posição baseado em % de capital arriscado.

    Parameters
    ----------
    capital : float
        Capital total disponível.
    risk_per_trade : float
        Percentual do capital a arriscar (ex.: 0.01 = 1%).
    entry : float
        Preço de entrada.
    stop : float
        Preço de stop (nível onde sai se der errado).

    Returns
    -------
    int
        Quantidade de unidades (floor).
    """
    risk_amount = capital * risk_per_trade
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0
    return int(risk_amount // risk_per_unit)


def fixed_amount(risk_amount: float, entry: float, stop: float) -> int:
    """Calcula posição com base em valor fixo arriscado (ex.: R$100 por trade)."""
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0
    return int(risk_amount // risk_per_unit)


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Calcula fração de Kelly.

    win_rate : probabilidade de ganhar (0–1)
    payoff_ratio : relação média ganho/perda

    Retorna fração ideal do capital a arriscar.
    """
    f = win_rate - (1 - win_rate) / payoff_ratio
    return max(0.0, f)


def vol_target(capital: float, target_vol: float, returns: pd.Series) -> float:
    """Dimensiona posição mirando uma volatilidade anualizada alvo.

    Parameters
    ----------
    capital : float
        Capital disponível.
    target_vol : float
        Volatilidade alvo (ex.: 0.2 = 20% anual).
    returns : pd.Series
        Série de retornos simples (diários, horários, etc.).

    Returns
    -------
    float
        Alocação fracional do capital.
    """
    realized_vol = returns.std() * np.sqrt(252)
    if realized_vol == 0:
        return 0.0
    return (target_vol / realized_vol) * capital
