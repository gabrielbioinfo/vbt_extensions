from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RetracementResult:
    # Base da perna (valores)
    swing_high: pd.Series
    swing_low: pd.Series
    # Índice (pos) do último topo/fundo conhecido até t (útil p/ debugging/plots)
    last_top_pos: pd.Series
    last_bottom_pos: pd.Series
    # Direção da perna ativa: +1 = up (de low->high), -1 = down (de high->low)
    direction: pd.Series
    # Níveis de Fibo dentro de [low, high] (fib_0 .. fib_100) e extensões >100
    fib_levels: pd.DataFrame  # colunas ex.: fib_0, fib_23.6, ..., fib_100
    ext_levels: pd.DataFrame | None  # colunas ex.: ext_127.2, ext_161.8 (pode ser None)


def _last_true_pos(mask: pd.Series) -> pd.Series:
    """Retorna, a cada t, a posição (0..n-1) do último True visto até t; NaN se não houve."""
    # Convertemos True em índices inteiros, NaN caso contrário; depois ffill do 'running max'
    idx_pos = pd.Series(np.arange(len(mask), dtype=float), index=mask.index)
    pos = idx_pos.where(mask, np.nan).ffill()
    return pos


def _build_fib_levels_for_leg(
    low: pd.Series,
    high: pd.Series,
    direction_up: pd.Series,
    ratios: list[float],
    ext: list[float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Calcula níveis por barra, respeitando a direção da perna ativa."""
    # garantias
    low = low.astype(float)
    high = high.astype(float)
    direction_up = direction_up.astype(bool)

    # dentro do range [low, high]
    # - up-leg (low->high): fib_r = high - r*(high-low)
    # - down-leg (high->low): fib_r = low + r*(high-low)
    rng = (high - low).astype(float)

    fib_cols: dict[str, pd.Series] = {}
    for r in ratios:
        name = f"fib_{round(r * 100, 1):g}".replace(".0", "")
        up_val = high - r * rng
        dn_val = low + r * rng
        fib_cols[name] = np.where(direction_up, up_val, dn_val)

    # sempre incluímos 0% e 100% para referência
    fib_cols.setdefault("fib_0", np.where(direction_up, high, low))  # alvo da perna
    fib_cols.setdefault("fib_100", np.where(direction_up, low, high))  # origem da perna

    fib_df = pd.DataFrame(fib_cols, index=low.index).astype(float)

    # ordenar colunas numericamente pelo percentual (garante consistência)
    def k(c: str) -> float:
        if c.startswith("fib_"):
            return float(c.split("_")[1])
        return 9999.0

    fib_df = fib_df[sorted(fib_df.columns, key=k)]

    ext_df = None
    if ext and len(ext) > 0:
        ext_cols: dict[str, pd.Series] = {}
        # extensões:
        # - up-leg: ext_k = high + (k-1)*rng
        # - down-leg: ext_k = low - (k-1)*rng
        for kext in ext:
            name = f"ext_{kext}"
            up_val = high + (kext - 1.0) * rng
            dn_val = low - (kext - 1.0) * rng
            ext_cols[name] = np.where(direction_up, up_val, dn_val)
        ext_df = pd.DataFrame(ext_cols, index=low.index).astype(float)

    return fib_df, ext_df


class FIB_RETRACEMENT:
    """Fibonacci retracements a partir de pivôs (is_top/is_bottom) — pandas-first.

    Requer:
      - close : Série de fechamento (para alinhamento/plots)
      - swing_high, swing_low : Séries com últimos pivôs (ffilladas), vindas de ZigZag/Peaks/etc.
      - is_top, is_bottom : flags booleanas de pivô no candle (mesmo index)

    A perna ativa é determinada pela RECÊNCIA dos pivôs:
      - se o último evento recente foi 'top'  -> perna down (high->low)
      - se o último evento recente foi 'bottom' -> perna up (low->high)

    Os níveis são sempre calculados dentro do range (low, high) da perna ativa naquele t.

    Parâmetros:
      ratios: níveis de retração (0..1, exceto 0/1 que são acrescentados automaticamente)
      ext   : extensões (>1.0), ex.: [1.272, 1.618]
    """

    @staticmethod
    def run(
        close: pd.Series,
        swing_high: pd.Series,
        swing_low: pd.Series,
        is_top: pd.Series,
        is_bottom: pd.Series,
        *,
        ratios: Iterable[float] = (0.236, 0.382, 0.5, 0.618, 0.786),
    ext: Iterable[float] | None = (1.272, 1.618),
    ) -> RetracementResult:
        # validações/alinhos
        close = close.astype(float)
        swing_high = swing_high.reindex_like(close).astype(float)
        swing_low = swing_low.reindex_like(close).astype(float)
        is_top = is_top.reindex_like(close).fillna(False).astype(bool)
        is_bottom = is_bottom.reindex_like(close).fillna(False).astype(bool)

        ratios = list(ratios)
        ext = list(ext) if ext is not None else None

        # recência dos pivôs (posições numéricas cumulativas)
        last_top_pos = _last_true_pos(is_top)
        last_bottom_pos = _last_true_pos(is_bottom)

        # direção da perna ativa
        # regra: se topo é mais recente que fundo -> perna DOWN; senão UP
        direction_up = last_bottom_pos >= last_top_pos  # bool
        direction = pd.Series(np.where(direction_up, 1, -1), index=close.index)

        # níveis: usamos o par (low, high) ffill atual
        fib_df, ext_df = _build_fib_levels_for_leg(
            low=swing_low, high=swing_high, direction_up=direction_up, ratios=ratios, ext=ext
        )

        return RetracementResult(
            swing_high=swing_high,
            swing_low=swing_low,
            last_top_pos=last_top_pos,
            last_bottom_pos=last_bottom_pos,
            direction=direction,
            fib_levels=fib_df,
            ext_levels=ext_df,
        )
