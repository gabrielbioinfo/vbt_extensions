from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StructureZZResult:
    is_top: pd.Series
    is_bottom: pd.Series
    swing_high: pd.Series
    swing_low: pd.Series
    bos_up: pd.Series
    bos_down: pd.Series
    choch_up: pd.Series
    choch_down: pd.Series

class MARKET_STRUCTURE_ZZ:
    """
    Market Structure (BOS/CHOCH) a partir de pivôs vindos do ZigZag:
      - is_top, is_bottom: séries booleanas (do ZigZag)
      - swing_high, swing_low: últimas âncoras (ffill) do ZigZag
      - confirmação por 'close' (ou por wick se preferir)

    Obs: por ser pandas-first, funciona com série única ou MultiIndex de colunas.
    """

    @staticmethod
    def run(
        close: pd.Series,
        *,
        is_top: pd.Series,
        is_bottom: pd.Series,
        swing_high: pd.Series,
        swing_low: pd.Series,
        confirm_wicks: bool = False,
    ) -> StructureZZResult:
        close = close.astype(float)
        is_top = is_top.reindex_like(close).fillna(False).astype(bool)
        is_bottom = is_bottom.reindex_like(close).fillna(False).astype(bool)
        swing_high = swing_high.reindex_like(close).astype(float).ffill()
        swing_low = swing_low.reindex_like(close).astype(float).ffill()

        up_line = swing_high.shift(1)
        dn_line = swing_low.shift(1)

        if confirm_wicks:
            # Se quiser, passe também high/low para uma confirmação por sombras
            raise NotImplementedError("confirm_wicks=True requer high/low; mantenha False por enquanto.")
        bos_up = (close > up_line).fillna(False)
        bos_down = (close < dn_line).fillna(False)

        # regime simples: vira up quando bos_up; vira down quando bos_down
        trend_up = pd.Series(False, index=close.index)
        trend_dn = pd.Series(False, index=close.index)
        state = 0
        for i in range(len(close)):
            if bool(bos_up.iat[i]): state = 1
            elif bool(bos_down.iat[i]): state = -1
            trend_up.iat[i] = (state == 1)
            trend_dn.iat[i] = (state == -1)

        choch_up = bos_up & trend_dn.shift(1).fillna(False)
        choch_down = bos_down & trend_up.shift(1).fillna(False)

        return StructureZZResult(
            is_top=is_top, is_bottom=is_bottom,
            swing_high=swing_high, swing_low=swing_low,
            bos_up=bos_up, bos_down=bos_down,
            choch_up=choch_up, choch_down=choch_down,
        )
