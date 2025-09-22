from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StructureResult:
    is_top: pd.Series
    is_bottom: pd.Series
    swing_high: pd.Series
    swing_low: pd.Series
    hh: pd.Series  # higher high
    hl: pd.Series  # higher low
    lh: pd.Series  # lower high
    ll: pd.Series  # lower low
    bos_up: pd.Series   # rompeu último swing high (continuação de alta)
    bos_down: pd.Series # rompeu último swing low (continuação de baixa)
    choch_up: pd.Series   # mudança de baixa->alta
    choch_down: pd.Series # mudança de alta->baixa

class MARKET_STRUCTURE:
    """
    Estrutura de mercado baseada em pivôs simples (rolling window):
    - Detecta pivôs (top/bottom) por ordem k (máximo/mínimo local).
    - Mantém swing_high/low atuais (ffill).
    - Marca HH/HL/LH/LL quando um novo pivô surge.
    - BOS: close cruza acima do último swing_high (bos_up) ou abaixo do swing_low (bos_down).
    - CHOCH: após uma sequência de alta (HH/HL), romper swing_low => choch_down; análogo para alta.

    Obs: simples por design. Dá pra trocar o detector de pivôs por seu ZigZag depois.
    """

    @staticmethod
    def _pivots_rolling(high: pd.Series, low: pd.Series, order: int) -> tuple[pd.Series, pd.Series]:
        h = high.astype(float); l = low.astype(float)
        # topo local se é máximo estrito numa janela de 2*order+1
        is_top = (h == h.rolling(order*2+1, center=True).max()) & h.notna()
        is_bottom = (l == l.rolling(order*2+1, center=True).min()) & l.notna()
        # remove bordas onde a janela não completa
        is_top.iloc[:order] = False; is_top.iloc[-order:] = False
        is_bottom.iloc[:order] = False; is_bottom.iloc[-order:] = False
        return is_top.astype(bool), is_bottom.astype(bool)

    @staticmethod
    def run(high: pd.Series, low: pd.Series, close: pd.Series, *, order: int = 3, confirm_wicks: bool = False) -> StructureResult:
        high = high.astype(float); low = low.astype(float); close = close.astype(float)
        is_top, is_bottom = MARKET_STRUCTURE._pivots_rolling(high, low, order=order)

        # mantém último swing hi/lo
        swing_high = high.where(is_top).ffill()
        swing_low  = low.where(is_bottom).ffill()

        # classifica novos pivôs (apenas no candle pivot True)
        last_high = swing_high.shift(1)
        last_low  = swing_low.shift(1)

        new_top = is_top & (high > last_high.fillna(-np.inf))
        new_low = is_bottom & (low  < last_low.fillna(np.inf))

        # relacionamento HH/HL/LH/LL quando ocorre pivô
        hh = (is_top & (high > last_high))
        lh = (is_top & (high <= last_high))
        hl = (is_bottom & (low >= last_low))
        ll = (is_bottom & (low < last_low))

        # BOS: confirma rompimento no close (ou wick se preferir)
        up_line = swing_high.shift(1)  # último swing high "ativo"
        dn_line = swing_low.shift(1)

        if confirm_wicks:
            bos_up = high > up_line
            bos_down = low < dn_line
        else:
            bos_up = close > up_line
            bos_down = close < dn_line

        bos_up = bos_up.fillna(False)
        bos_down = bos_down.fillna(False)

        # Regime simples: mantemos um estado "uptrend" quando últimos eventos foram HH/HL;
        # trocamos para downtrend quando bos_down verdadeiro, e vice-versa.
        # CHOCH: marca a barra que inverte regime.
        trend_up = pd.Series(False, index=close.index)
        trend_dn = pd.Series(False, index=close.index)
        state = 0  # 1=up, -1=down, 0=unknown
        for i in range(len(close)):
            if bos_up.iat[i]:
                state = 1
            elif bos_down.iat[i]:
                state = -1
            trend_up.iat[i] = (state == 1)
            trend_dn.iat[i] = (state == -1)

        choch_up = bos_up & trend_dn.shift(1).fillna(False)   # rompe pra cima vindo de baixa
        choch_down = bos_down & trend_up.shift(1).fillna(False)

        return StructureResult(
            is_top=is_top, is_bottom=is_bottom,
            swing_high=swing_high, swing_low=swing_low,
            hh=hh.fillna(False), hl=hl.fillna(False), lh=lh.fillna(False), ll=ll.fillna(False),
            bos_up=bos_up, bos_down=bos_down,
            choch_up=choch_up, choch_down=choch_down
        )
