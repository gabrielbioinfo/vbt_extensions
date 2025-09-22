import pandas as pd


def apply_breakeven_exit(close: pd.Series, entries: pd.Series, *, trigger_pct: float = 0.01) -> pd.Series:
    """Sai no ponto de entrada (breakeven) depois de atingir lucro >= trigger_pct.

    Mantém estado mínimo: após uma entrada, se o preço subir pelo menos trigger_pct,
    ativa uma "linha de defesa" no preço de entrada. Se voltar até lá, gera saída.
    """
    close = close.astype(float)
    entries = entries.reindex_like(close).fillna(False).astype(bool)

    in_pos = False
    entry_px = None
    be_active = False
    out = pd.Series(False, index=close.index)

    for i, px in enumerate(close.values):
        if entries.iat[i] and not in_pos:
            in_pos = True
            entry_px = px
            be_active = False

        if in_pos:
            if not be_active and px >= entry_px * (1 + trigger_pct):
                be_active = True
            if be_active and px <= entry_px:
                out.iat[i] = True
                in_pos = False
                entry_px = None
                be_active = False
    return out
