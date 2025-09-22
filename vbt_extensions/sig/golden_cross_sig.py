"""Golden-cross / mean-reversion signals com médias móveis (pandas-first)."""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt


class ModeRequiredError(ValueError):
    def __init__(self) -> None:
        super().__init__("Use mode='cartesian' ou mode='pair'.")


class ModeMissingListError(ValueError):
    def __init__(self) -> None:
        super().__init__("Em mode='pair', fast_list e slow_list devem ter o mesmo tamanho.")


class StyleRequiredError(ValueError):
    def __init__(self) -> None:
        super().__init__("Use style='trend' (segue cruzamento) ou style='reversion' (opera contrário).")


def golden_cross_sig(
    close: pd.Series,
    *,
    fast_list: list[int],
    slow_list: list[int],
    mode: str = "cartesian",
    style: str = "trend",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sinais baseados em cruzamento de MAs.

    return
    -------
    (entries, exits) com MultiIndex de colunas ('fast','slow')
    """
    if mode not in ("pair", "cartesian"):
        raise ModeRequiredError
    if style not in ("trend", "reversion"):
        raise StyleRequiredError

    if mode == "pair":
        if len(fast_list) != len(slow_list):
            raise ModeMissingListError
        pairs = [(int(f), int(s)) for f, s in zip(fast_list, slow_list)]
    else:
        pairs = [(int(f), int(s)) for f in fast_list for s in slow_list if int(f) < int(s)]

    def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    def _cross_dn(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    ent_cols, ex_cols = [], []
    for f, s in pairs:
        fma = vbt.MA.run(close, window=f).ma.squeeze()
        sma = vbt.MA.run(close, window=s).ma.squeeze()
        if style == "trend":
            ent, exi = _cross_up(fma, sma), _cross_dn(fma, sma)
        else:
            ent, exi = _cross_dn(fma, sma), _cross_up(fma, sma)
        ent_cols.append(ent.rename((f, s)))
        ex_cols.append(exi.rename((f, s)))

    entries = pd.concat(ent_cols, axis=1)
    exits = pd.concat(ex_cols, axis=1)
    cols = pd.MultiIndex.from_tuples(entries.columns, names=["fast", "slow"])
    entries.columns, exits.columns = cols, cols
    return entries.sort_index(axis=1), exits.sort_index(axis=1)
