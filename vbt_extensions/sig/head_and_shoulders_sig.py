import pandas as pd

from vbt_extensions.ind.head_and_shoulders import HeadShouldersResult


def hs_signals_from_result(
    res: HeadShouldersResult,
    *,
    confirm_shift: int = 1,
):
    """Gera sinais a partir do resultado HS/IHS.

    Regras simples:
    - HS (topo): entrar vendido (short) no rompimento da neckline (hs_break).
    - IHS (fundo): entrar comprado (long) no rompimento da neckline (ihs_break).

    As saídas ficam a seu critério (ex.: stop/trailing/time_stop/risk overlay).
    Aqui devolvemos apenas 'entries_long' e 'entries_short'.
    """
    hs_break = res.hs_break.shift(confirm_shift).fillna(False).astype(bool)
    ihs_break = res.ihs_break.shift(confirm_shift).fillna(False).astype(bool)

    entries_long = ihs_break
    entries_short = hs_break
    return entries_long, entries_short


def hs_exit_on_opposite_break(res: HeadShouldersResult, *, confirm_shift: int = 1):
    """Exemplo simples de saída: sai quando ocorrer o oposto.
    - Long sai no HS break (baixa)
    - Short sai no IHS break (alta)
    """
    exit_long = res.hs_break.shift(confirm_shift).fillna(False).astype(bool)
    exit_short = res.ihs_break.shift(confirm_shift).fillna(False).astype(bool)
    return exit_long, exit_short
