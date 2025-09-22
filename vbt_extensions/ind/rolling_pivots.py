# vbt_extensions/ind/rolling_pivots.py
from dataclasses import dataclass

import pandas as pd


@dataclass
class RollingPivotsResult:
    is_top: pd.Series
    is_bottom: pd.Series
    swing_high: pd.Series
    swing_low: pd.Series


class ROLLING_PIVOTS:
    """
    Topos/fundos por janela deslizante.
    - Um topo é o valor igual ao máximo da janela centrada.
    - Um fundo é o valor igual ao mínimo da janela centrada.
    """

    @staticmethod
    def run(
        high: pd.Series,
        low: pd.Series,
        *,
        window: int = 5,  # tamanho total (ímpar recomendável)
        center: bool = True,
    ) -> RollingPivotsResult:
        if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
            raise TypeError("high e low devem ser pd.Series")

        high = high.astype(float)
        low = low.reindex_like(high).astype(float)
        mid = (high + low) / 2.0

        roll_max = mid.rolling(window=window, center=center).max()
        roll_min = mid.rolling(window=window, center=center).min()

        is_top = mid.eq(roll_max) & mid.notna()
        is_bottom = mid.eq(roll_min) & mid.notna()

        swing_high = mid.where(is_top).ffill()
        swing_low = mid.where(is_bottom).ffill()

        return RollingPivotsResult(
            is_top=is_top.astype(bool),
            is_bottom=is_bottom.astype(bool),
            swing_high=swing_high,
            swing_low=swing_low,
        )
