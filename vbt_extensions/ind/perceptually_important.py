# vbt_extensions/ind/perceptually_important.py
"""
Perceptually Important Points (PIP) — pandas-first API.

- Núcleo Numba (rápido) preservado da sua versão.
- Wrapper retorna séries alinhadas ao índice para plot/sinais:
  * pip_mask        : bool (True nos índices PIP)
  * pip_values      : série esparsa com valores nos PIPs (NaN fora)
  * reconstructed   : série reconstruída por interpolação linear entre PIPs
  * pip_index (int) : série inteira com o índice "ordinal" do PIP (1..n_pips) nos pontos PIP

Distâncias suportadas:
- 0: distância vertical (absoluta) até a reta do segmento
- 1: soma de distâncias euclidianas aos dois extremos do segmento
- 2: distância perpendicular (slope-aware) à reta (DEFAULT, mais estável)

Uso:
    res = PIP.run(close, n_pips=15, dist_measure=2)
    fig = close.vbt.plot()
    fig = res.reconstructed.vbt.plot(fig=fig, trace_kwargs=dict(name="PIP Reconstructed"))
    fig = res.pip_values.vbt.scatter(fig=fig, trace_kwargs=dict(name="PIPs"))
"""

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


PIP_DIST_MEASURE_VERTICAL = 0
PIP_DIST_MEASURE_EUCLIDEAN = 1
PIP_DIST_MEASURE_SLOPE = 2  # default (perpendicular à reta)


# =================== Núcleo Numba (do seu código, ligeiramente comentado) =================== #


@njit(cache=True)
def _find_pips_1d(data: np.ndarray, n_pips: int, dist_measure: int) -> Tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    if n_pips < 2:
        n_pips = 2
    n_pips = min(n_pips, n)
    pips_x = np.empty(n_pips, dtype=np.int64)
    pips_y = np.empty(n_pips, dtype=np.float64)
    pips_x[0] = 0
    pips_x[1] = n - 1
    pips_y[0] = data[0]
    pips_y[1] = data[-1]

    for curr_point in range(2, n_pips):
        md = 0.0
        md_i = -1
        insert_index = -1
        # percorre segmentos definidos pelos PIPs atuais
        for k in range(curr_point - 1):
            left_adj = pips_x[k]
            right_adj = pips_x[k + 1]
            if right_adj - left_adj < 2:
                continue
            time_diff = right_adj - left_adj
            price_diff = pips_y[k + 1] - pips_y[k]
            slope = price_diff / time_diff
            intercept = pips_y[k] - left_adj * slope
            for i in range(left_adj + 1, right_adj):
                # mede desvio do ponto i em relação ao segmento
                if dist_measure == PIP_DIST_MEASURE_EUCLIDEAN:
                    d = ((left_adj - i) ** 2 + (pips_y[k] - data[i]) ** 2) ** 0.5
                    d += ((right_adj - i) ** 2 + (pips_y[k + 1] - data[i]) ** 2) ** 0.5
                elif dist_measure == PIP_DIST_MEASURE_SLOPE:
                    denom = (slope * slope + 1.0) ** 0.5
                    d = abs((slope * i + intercept) - data[i]) / denom
                else:  # PIP_DIST_MEASURE_VERTICAL
                    d = abs((slope * i + intercept) - data[i])
                if d > md:
                    md = d
                    md_i = i
                    insert_index = k + 1
        if md_i == -1:
            # não achou ponto melhor: repete último índice (degrada com segurança)
            md_i = pips_x[curr_point - 1]
            insert_index = curr_point
        pips_x[insert_index] = md_i
        pips_y[insert_index] = data[md_i]
    return pips_x, pips_y


# =================== Wrapper pandas-first =================== #


@dataclass
class PIPResult:
    pip_mask: pd.Series  # True nos PIPs
    pip_values: pd.Series  # valores nos PIPs (NaN fora)
    reconstructed: pd.Series  # série reconstruída por segmentos lineares
    pip_index: pd.Series  # int: 1..n_pips nos pontos PIP, NaN fora


class PIP:
    """Perceptually Important Points (PIP) com API pandas-first."""

    @staticmethod
    def run(
        close: pd.Series,
        *,
        n_pips: int = 15,
        dist_measure: int = PIP_DIST_MEASURE_SLOPE,
    ) -> PIPResult:
        if not isinstance(close, pd.Series):
            raise TypeError("close deve ser pd.Series")

        c = close.astype(float)
        x = c.values.astype(np.float64)
        idx = c.index

        pips_x, pips_y = _find_pips_1d(x, int(n_pips), int(dist_measure))

        # Ordena por índice crescente (por segurança)
        order = np.argsort(pips_x)
        pips_x = pips_x[order]
        pips_y = pips_y[order]

        # pip_mask e pip_values
        pip_mask = pd.Series(False, index=idx)
        pip_values = pd.Series(np.nan, index=idx, dtype=float)
        pip_ord = pd.Series(np.nan, index=idx, dtype=float)

        for k, xi in enumerate(pips_x, start=1):
            if 0 <= xi < len(idx):
                pip_mask.iat[xi] = True
                pip_values.iat[xi] = pips_y[k - 1]
                pip_ord.iat[xi] = float(k)

        # reconstrução por segmentos lineares entre PIPs
        reconstructed = pd.Series(np.nan, index=idx, dtype=float)
        for i in range(len(pips_x) - 1):
            a = int(pips_x[i])
            b = int(pips_x[i + 1])
            ya = float(pips_y[i])
            yb = float(pips_y[i + 1])
            run = b - a
            if run <= 0:
                continue
            for t in range(a, b + 1):
                w = (t - a) / run
                reconstructed.iat[t] = ya * (1.0 - w) + yb * w

        return PIPResult(
            pip_mask=pip_mask.astype(bool),
            pip_values=pip_values,
            reconstructed=reconstructed,
            pip_index=pip_ord.astype("Int64"),
        )


# =================== (Opcional) Wrapper IndicatorFactory =================== #
# Descomente se/quando quiser grids/combs nativos do vbt:
#
# import vectorbt as vbt
# def _apply_func(close: np.ndarray, n_pips: int, dist_measure: int):
#     close = np.asarray(close, dtype=np.float64)
#     if close.ndim > 1:
#         close = close.ravel()
#     px, py = _find_pips_1d(close, int(n_pips), int(dist_measure))
#     return px, py
#
# perceptually_important = vbt.IndicatorFactory(
#     class_name="perceptually_important",
#     short_name="pip",
#     input_names=["close"],
#     param_names=["n_pips", "dist_measure"],
#     output_names=["pips_x", "pips_y"],
# ).from_apply_func(
#     _apply_func,
#     n_pips=15,
#     dist_measure=PIP_DIST_MEASURE_SLOPE,
#     keep_pd=True,
# )
