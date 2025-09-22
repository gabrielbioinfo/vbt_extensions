# vbt_extensions/ind/directional_change.py
from dataclasses import dataclass

import pandas as pd


@dataclass
class DCResult:
    dc_event: pd.Series  # +1: pivô topo, -1: pivô fundo, 0: nenhum
    dc_price: pd.Series  # preço no evento DC (NaN fora)
    swing_high: pd.Series  # último topo DC conhecido
    swing_low: pd.Series  # último fundo DC conhecido


class DIRECTIONAL_CHANGE:
    """
    Directional Change (DC) por limiar percentual delta.
    - Marca eventos quando há reversão >= delta a partir do último extremo.
    - Alterna entre regimes "subida" e "descida" (intrinsic time).
    """

    @staticmethod
    def run(
        close: pd.Series,
        *,
        delta: float = 0.01,  # 1% por padrão
        use_pct: bool = True,  # se False, trata delta em preço absoluto
    ) -> DCResult:
        if not isinstance(close, pd.Series):
            raise TypeError("close deve ser pd.Series")

        c = close.astype(float)
        n = len(c)
        if n == 0:
            empty = pd.Series(dtype=float, index=c.index)
            empty_b = pd.Series(dtype=bool, index=c.index)
            return DCResult(empty_b.astype(int), empty, empty, empty)

        # Estado
        trend = 0  # +1 up, -1 down, 0 unknown
        last_ext_i = 0
        last_ext_p = c.iloc[0]

        dc_flag = pd.Series(0, index=c.index, dtype=int)
        dc_price = pd.Series(pd.NA, index=c.index, dtype="float")

        for i in range(1, n):
            px = c.iat[i]

            def rel_change(a, b):
                return (a - b) / b if use_pct else (a - b)

            if trend >= 0:
                # acompanhando alta (ou indefinido)
                if px > last_ext_p:
                    last_ext_p = px
                    last_ext_i = i
                elif rel_change(last_ext_p, px) >= (delta if use_pct else delta):
                    # reversão para baixo >= delta -> evento topo
                    dc_flag.iat[last_ext_i] = 1
                    dc_price.iat[last_ext_i] = last_ext_p
                    trend = -1
                    last_ext_p = px
                    last_ext_i = i
            if trend <= 0:
                # acompanhando baixa (ou indefinido)
                if px < last_ext_p:
                    last_ext_p = px
                    last_ext_i = i
                elif rel_change(px, last_ext_p) >= (delta if use_pct else delta):
                    # reversão para cima >= delta -> evento fundo
                    dc_flag.iat[last_ext_i] = -1
                    dc_price.iat[last_ext_i] = last_ext_p
                    trend = 1
                    last_ext_p = px
                    last_ext_i = i

        swing_high = dc_price.where(dc_flag.eq(1)).ffill()
        swing_low = dc_price.where(dc_flag.eq(-1)).ffill()

        return DCResult(dc_event=dc_flag, dc_price=dc_price, swing_high=swing_high, swing_low=swing_low)
