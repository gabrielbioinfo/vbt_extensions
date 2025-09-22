import pandas as pd


def fib_trend_continuation_sig(
    close: pd.Series,
    *,
    direction: pd.Series,
    fib_levels: pd.DataFrame,
    ext_levels: pd.DataFrame | None = None,
    trigger_level: str = "fib_61.8",  # rompe 61.8% a favor da perna
    confirm_shift: int = 1,
):
    """Continuação na direção da perna (breakout do nível-chave).
    - Se perna UP: entra long ao cruzar ACIMA de trigger_level
    - Se perna DOWN: entra short ao cruzar ABAIXO de trigger_level
    Saídas básicas: faz o oposto (ou deixa p/ risk overlays).
    """
    close = close.astype(float)
    dir_up = direction.reindex_like(close).astype(int).eq(1)
    lvl = fib_levels[trigger_level].reindex_like(close).astype(float)

    cross_up = (close > lvl) & (close.shift(1) <= lvl.shift(1))
    cross_dn = (close < lvl) & (close.shift(1) >= lvl.shift(1))

    entries_long = (dir_up & cross_up).shift(confirm_shift).fillna(False).astype(bool)
    exits_long = (dir_up & cross_dn).shift(confirm_shift).fillna(False).astype(bool)  # simples
    entries_short = (~dir_up & cross_dn).shift(confirm_shift).fillna(False).astype(bool)
    exits_short = (~dir_up & cross_up).shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


def fib_mean_revert_sig(
    close: pd.Series,
    *,
    fib_levels: pd.DataFrame,
    level: str = "fib_38.2",
    tol: float = 0.001,  # tolerância relativa p/ 'toque/rejeição'
    confirm_shift: int = 1,
):
    """Reversão no nível: rejeição do preço no Fibo escolhido (ex.: 38.2%).
    - Entra long se CLOSE cruza de baixo p/ cima e volta a ficar acima do nível após 'toque'
    - Entra short no análogo (de cima p/ baixo)
    """
    close = close.astype(float)
    lvl = fib_levels[level].reindex_like(close).astype(float)
    tol_abs = (lvl.abs() * tol).fillna(0.0)

    # toque (aproximação) + reversão simples via cruzamento
    near = close.sub(lvl).abs() <= tol_abs

    cross_up = (close > lvl) & (close.shift(1) <= lvl.shift(1))
    cross_dn = (close < lvl) & (close.shift(1) >= lvl.shift(1))

    entries_long = (near & cross_up).shift(confirm_shift).fillna(False).astype(bool)
    exits_long = (near & cross_dn).shift(confirm_shift).fillna(False).astype(bool)
    entries_short = (near & cross_dn).shift(confirm_shift).fillna(False).astype(bool)
    exits_short = (near & cross_up).shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


def fib_grid_breakout_sig(
    close: pd.Series,
    *,
    fib_levels: pd.DataFrame,
    use_cols: list[str] | None = None,
    confirm_shift: int = 1,
):
    """Breakouts por grade de níveis (multi-entradas).
    - Entra long a cada cruzamento para cima de um nível (exceto fib_0/fib_100)
    - Entra short a cada cruzamento para baixo
    Retorna duas Séries booleanas com múltiplos sinais ao longo do tempo.
    """
    close = close.astype(float)
    cols = [c for c in fib_levels.columns if c not in ("fib_0", "fib_100")]
    if use_cols is not None:
        cols = [c for c in cols if c in use_cols]
    cols = sorted(cols, key=lambda c: float(c.split("_")[1]))

    entries_long = pd.Series(False, index=close.index)
    entries_short = pd.Series(False, index=close.index)

    for c in cols:
        lvl = fib_levels[c].reindex_like(close).astype(float)
        cross_up = (close > lvl) & (close.shift(1) <= lvl.shift(1))
        cross_dn = (close < lvl) & (close.shift(1) >= lvl.shift(1))
        entries_long |= cross_up
        entries_short |= cross_dn

    entries_long = entries_long.shift(confirm_shift).fillna(False).astype(bool)
    entries_short = entries_short.shift(confirm_shift).fillna(False).astype(bool)

    # saídas básicas: o oposto agregado (deixe overlays fazerem o grosso)
    exits_long = entries_short
    exits_short = entries_long

    return entries_long, exits_long, entries_short, exits_short
