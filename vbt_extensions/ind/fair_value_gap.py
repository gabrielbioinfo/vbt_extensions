from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FVGResult:
    fvg_up: pd.Series
    fvg_dn: pd.Series
    fvg_up_upper: pd.Series
    fvg_up_lower: pd.Series
    fvg_dn_upper: pd.Series
    fvg_dn_lower: pd.Series
    size: pd.Series                # tamanho (em preço)
    size_pct: pd.Series            # tamanho relativo ao preço
    filled_partial: pd.Series      # tocou dentro do intervalo
    filled_full: pd.Series         # preencheu completamente (fechou o gap)

class FAIR_VALUE_GAP:
    """
    FVG (3-velas) com filtro de tamanho e detecção de fill parcial/completo.

    Parâmetros:
      - min_size: mínimo tamanho do gap em preço ou percentual (depende de unit)
      - unit: 'pct' (default) ou 'ticks'
      - tick_size: obrigatório se unit='ticks'

    Definições:
      - FVG up: Low[t] > High[t-2]  -> gap [lower=High[t-2], upper=Low[t]]
      - FVG dn: High[t] < Low[t-2]  -> gap [lower=High[t], upper=Low[t-2]]
      - filled_partial: close entra no intervalo (entre lower e upper)
      - filled_full: close "atravessa" toda a faixa (ex.: para FVG up, close <= lower)
    """

    @staticmethod
    def run(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series | None = None,
        *,
        min_size: float = 0.0,
        unit: str = "pct",
        tick_size: float | None = None,
    ) -> FVGResult:
        high = high.astype(float); low = low.astype(float)
        if close is None: close = high  # só p/ index/forma
        close = close.reindex_like(high).astype(float)

        # regras base
        fvg_up = (low > high.shift(2))
        fvg_dn = (high < low.shift(2))

        # bordas
        fvg_up_upper = low.where(fvg_up)
        fvg_up_lower = high.shift(2).where(fvg_up)

        fvg_dn_upper = low.shift(2).where(fvg_dn)
        fvg_dn_lower = high.where(fvg_dn)

        # tamanho e filtro mínimo
        size_up = (fvg_up_upper - fvg_up_lower).where(fvg_up)  # >0
        size_dn = (fvg_dn_upper - fvg_dn_lower).where(fvg_dn)  # >0
        size = size_up.fillna(0.0) + size_dn.fillna(0.0)
        ref_price = close.abs().replace(0, np.nan)
        size_pct = (size / ref_price).fillna(0.0)

        if min_size > 0:
            if unit == "pct":
                ok = size_pct >= min_size
            elif unit == "ticks":
                if tick_size is None or tick_size <= 0:
                    raise ValueError("unit='ticks' requer tick_size > 0")
                ok = (size / tick_size) >= min_size
            else:
                raise ValueError("unit deve ser 'pct' ou 'ticks'")

            # zera onde não atende filtro
            fvg_up = fvg_up & ok
            fvg_dn = fvg_dn & ok
            fvg_up_upper = fvg_up_upper.where(fvg_up)
            fvg_up_lower = fvg_up_lower.where(fvg_up)
            fvg_dn_upper = fvg_dn_upper.where(fvg_dn)
            fvg_dn_lower = fvg_dn_lower.where(fvg_dn)
            size = size.where(ok, 0.0)
            size_pct = size_pct.where(ok, 0.0)

        # fill parcial: preço entra no intervalo (upper..lower)
        in_up_gap = (close <= fvg_up_upper) & (close >= fvg_up_lower)
        in_dn_gap = (close <= fvg_dn_upper) & (close >= fvg_dn_lower)
        filled_partial = (in_up_gap | in_dn_gap).fillna(False)

        # fill completo: close atravessa toda faixa
        # - up: close <= lower (gap totalmente coberto)
        # - dn: close >= upper
        full_up = close <= fvg_up_lower
        full_dn = close >= fvg_dn_upper
        filled_full = ((full_up & fvg_up) | (full_dn & fvg_dn)).fillna(False)

        return FVGResult(
            fvg_up=fvg_up.fillna(False), fvg_dn=fvg_dn.fillna(False),
            fvg_up_upper=fvg_up_upper, fvg_up_lower=fvg_up_lower,
            fvg_dn_upper=fvg_dn_upper, fvg_dn_lower=fvg_dn_lower,
            size=size.astype(float), size_pct=size_pct.astype(float),
            filled_partial=filled_partial.astype(bool),
            filled_full=filled_full.astype(bool),
        )
