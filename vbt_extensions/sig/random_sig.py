"""Random signals (para testar risco/sizing/pipelines)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_sig(
    close: pd.Series,
    *,
    p: float = 0.02,
    ncols: int = 1,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera sinais aleatórios (entries/exits) alinhados ao índice de `close`.

    params
    -------
    p      : probabilidade de entrada por barra (independente)
    ncols  : número de colunas (experimentos paralelos)
    seed   : semente do RNG para reproduzibilidade (opcional)

    return
    -------
    (entries_df, exits_df) com MultiIndex de colunas: ("rand", i)

    """
    rng = np.random.default_rng(seed)
    idx = close.index
    cols = pd.MultiIndex.from_tuples([(i,) for i in range(ncols)], names=["rand"])

    entries = pd.DataFrame(rng.random((len(idx), ncols)) < p, index=idx, columns=cols)
    # fecha na barra seguinte (toy). deixe os overlays de risco fazerem o resto
    exits = entries.shift(1).fillna(False)

    return entries.astype(bool), exits.astype(bool)
