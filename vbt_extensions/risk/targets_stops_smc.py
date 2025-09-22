from __future__ import annotations

import pandas as pd


def smc_targets_stops(
    close: pd.Series,
    entries_long: pd.Series,
    entries_short: pd.Series,
    *,
    swing_high: pd.Series,
    swing_low: pd.Series,
    # alvos alternativos (opcionais) via FVG:
    fvg_up_lower: pd.Series | None = None,
    fvg_dn_upper: pd.Series | None = None,
    rr_min: float | None = None,       # se quiser forçar R:R mínimo (ex.: 1.5)
    buffer_pct: float = 0.0,
):
    """
    Define stops/targets "naturais" para SMC:
      - Long: stop = swing_low; alvo = swing_high (ou fvg_dn_upper, se fornecido)
      - Short: stop = swing_high; alvo = swing_low (ou fvg_up_lower)
    Opcional: rr_min para evitar entradas com R:R pior que o limite.
    """
    c = close.astype(float)
    eL = entries_long.reindex_like(c).fillna(False).astype(bool)
    eS = entries_short.reindex_like(c).fillna(False).astype(bool)

    sh = swing_high.reindex_like(c).astype(float).ffill()
    sl = swing_low.reindex_like(c).astype(float).ffill()

    tgt_long = sh.copy()
    tgt_short = sl.copy()

    # se fornecer FVGs opostas para alvo "rápido"
    if fvg_dn_upper is not None:
        tgt_long = fvg_dn_upper.reindex_like(c).combine_first(tgt_long)
    if fvg_up_lower is not None:
        tgt_short = fvg_up_lower.reindex_like(c).combine_first(tgt_short)

    stop_long = sl * (1 - buffer_pct)
    stop_short = sh * (1 + buffer_pct)

    in_long = False
    in_short = False
    force_exit = pd.Series(False, index=c.index)

    for i, px in enumerate(c.values):
        if eL.iat[i]:
            # checar rr_min: (tgt - px) / (px - stop)
            if rr_min is not None:
                rr = (tgt_long.iat[i] - px) / max(px - stop_long.iat[i], 1e-12)
                if rr < rr_min:
                    in_long = False
                else:
                    in_long, in_short = True, False
            else:
                in_long, in_short = True, False

        if eS.iat[i]:
            if rr_min is not None:
                rr = (px - tgt_short.iat[i]) / max(stop_short.iat[i] - px, 1e-12)
                if rr < rr_min:
                    in_short = False
                else:
                    in_short, in_long = True, False
            else:
                in_short, in_long = True, False

        if in_long:
            if px <= stop_long.iat[i] or px >= tgt_long.iat[i]:
                force_exit.iat[i] = True
                in_long = False

        if in_short:
            if px >= stop_short.iat[i] or px <= tgt_short.iat[i]:
                force_exit.iat[i] = True
                in_short = False

    return force_exit, stop_long, stop_short, tgt_long, tgt_short
