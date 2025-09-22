from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ZigZagResult:
    """Resultados do ZigZag.

    Atributos
    ---------
    zigzag : pd.Series
        Preço no ponto pivô (NaN fora dos pivôs).
    is_top : pd.Series
        True nos pivôs classificados como topo.
    is_bottom : pd.Series
        True nos pivôs classificados como fundo.
    swing_high : pd.Series
        Último topo conhecido (forward-filled).
    swing_low : pd.Series
        Último fundo conhecido (forward-filled).
    """

    zigzag: pd.Series
    is_top: pd.Series
    is_bottom: pd.Series
    swing_high: pd.Series
    swing_low: pd.Series


class ZIGZAG:
    """ZigZag por limiar percentual (upper/lower), pandas-first.

    Notes
    -----
    - Implementação simples e legível (pode ser otimizada com numba depois).
    - Inputs/outputs em pandas para integração direta com vectorbt (.vbt) e Portfolio.
    - Usa o preço "médio" (mid = (high+low)/2) como referência para pivôs.

    Parâmetros
    ----------
    high : pd.Series
        Série de máximas.
    low : pd.Series
        Série de mínimas.
    upper : float, default 0.03
        Variação mínima para confirmar um fundo -> topo (3%).
    lower : float, default 0.03
        Variação mínima para confirmar um topo -> fundo (3%).
    min_bars : int, default 1
        Distância mínima em barras entre pivôs consecutivos.

    Retorna
    -------
    ZigZagResult
    """

    @staticmethod
    def run(
        high: pd.Series,
        low: pd.Series,
        upper: float = 0.03,
        lower: float = 0.03,
        min_bars: int = 1,
    ) -> ZigZagResult:
        if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
            raise TypeError("high e low devem ser pd.Series")

        # Alinhar índices e garantir float
        high = high.astype(float)
        low = low.reindex_like(high).astype(float)

        mid = (high + low) / 2.0
        n = len(mid)
        if n == 0:
            empty = pd.Series(dtype=float, index=high.index)
            empty_b = pd.Series(dtype=bool, index=high.index)
            return ZigZagResult(empty, empty_b, empty_b, empty, empty)

        piv = np.zeros(n, dtype=int)  # +1 top, -1 bottom, 0 none
        last_i = 0
        last_p = mid.iloc[0]
        direction = 0  # +1 up, -1 down, 0 unk

        for i in range(1, n):
            chg = (mid.iloc[i] - last_p) / (last_p if last_p != 0 else 1e-12)

            # Confirma topo (mudança down) ou fundo (mudança up)
            if direction >= 0 and chg <= -lower and (i - last_i) >= min_bars:
                piv[last_i] = 1  # topo confirmado no pivot anterior
                direction = -1
                last_i, last_p = i, mid.iloc[i]

            elif direction <= 0 and chg >= upper and (i - last_i) >= min_bars:
                piv[last_i] = -1  # fundo confirmado no pivot anterior
                direction = 1
                last_i, last_p = i, mid.iloc[i]

            else:
                # Atualiza o candidato a pivot se o movimento se ampliou
                prev = mid.iloc[last_i]
                prev = prev if prev != 0 else 1e-12
                if abs(chg) > abs((mid.iloc[i] - prev) / prev):
                    last_i, last_p = i, mid.iloc[i]

        pivots = pd.Series(piv, index=mid.index)
        zigzag = mid.where(pivots != 0)
        is_top = pivots.eq(1)
        is_bottom = pivots.eq(-1)
        swing_high = zigzag.where(is_top).ffill()
        swing_low = zigzag.where(is_bottom).ffill()

        return ZigZagResult(
            zigzag=zigzag,
            is_top=is_top.astype(bool),
            is_bottom=is_bottom.astype(bool),
            swing_high=swing_high,
            swing_low=swing_low,
        )
