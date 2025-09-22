# vbt_extensions/ind/supports_resistances.py
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SRLevelsResult:
    support_levels: pd.Series  # último nível de suporte "forte" conhecido
    resistance_levels: pd.Series
    touches_support: pd.Series  # True quando close tocou (±tol) o suporte atual
    touches_resist: pd.Series


class MP_SUPPORTS_RESISTANCES:
    """
    Suportes/Resistências por confluência de múltiplos pivôs.
    - Usa listas de pivôs (tops/bottoms) e agrupa toques por tolerância.
    - Um nível 'forte' é aquele com >= min_touches dentro de 'tol' no histórico (rolling).
    """

    @staticmethod
    def run(
        close: pd.Series,
        *,
        pivot_highs: pd.Series,  # bool (ex.: de ZigZag, RollingPivots, DC)
        pivot_lows: pd.Series,  # bool
        lookback: int = 250,  # janelas recentes para "força"
        tol: float = 0.002,  # 0.2% de tolerância
        min_touches: int = 3,
    ) -> SRLevelsResult:
        if not isinstance(close, pd.Series):
            raise TypeError("close deve ser pd.Series")

        c = close.astype(float)
        ph = pivot_highs.reindex_like(c).fillna(False)
        pl = pivot_lows.reindex_like(c).fillna(False)

        idx = c.index
        sup = pd.Series(np.nan, index=idx)
        res = pd.Series(np.nan, index=idx)
        tsup = pd.Series(False, index=idx)
        tres = pd.Series(False, index=idx)

        # listas de níveis (preços) dentro da janela
        highs_hist = []
        lows_hist = []

        for i, t in enumerate(idx):
            px = c.iat[i]

            # atualiza listas recentes
            if ph.iat[i]:
                highs_hist.append(px)
            if pl.iat[i]:
                lows_hist.append(px)

            # corta histórico à janela
            if len(highs_hist) > lookback:
                highs_hist = highs_hist[-lookback:]
            if len(lows_hist) > lookback:
                lows_hist = lows_hist[-lookback:]

            # agrupa por tolerância (resistências a partir de highs, suportes a partir de lows)
            def strongest_level(levels):
                if not levels:
                    return np.nan
                levels = np.array(levels)
                # clusterização 1D grosseira por tolerância relativa
                clusters = []
                for lv in sorted(levels):
                    placed = False
                    for cl in clusters:
                        if abs(lv - np.mean(cl)) / np.mean(cl) <= tol:
                            cl.append(lv)
                            placed = True
                            break
                    if not placed:
                        clusters.append([lv])
                # escolhe o cluster com mais toques
                clusters.sort(key=lambda cl: (len(cl), -abs(np.mean(cl))), reverse=True)
                top = clusters[0]
                return float(np.mean(top)), len(top)

            r_lv = np.nan
            s_lv = np.nan
            r_cnt = s_cnt = 0

            if highs_hist:
                r_lv, r_cnt = strongest_level(highs_hist)
            if lows_hist:
                s_lv, s_cnt = strongest_level(lows_hist)

            # só mantém níveis "fortes"
            if r_cnt >= min_touches:
                res.iat[i] = r_lv
                tres.iat[i] = abs(px - r_lv) / r_lv <= tol
            if s_cnt >= min_touches:
                sup.iat[i] = s_lv
                tsup.iat[i] = abs(px - s_lv) / s_lv <= tol

        return SRLevelsResult(
            support_levels=sup.ffill(),
            resistance_levels=res.ffill(),
            touches_support=tsup,
            touches_resist=tres,
        )
