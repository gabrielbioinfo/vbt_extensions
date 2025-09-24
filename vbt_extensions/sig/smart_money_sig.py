import pandas as pd


def bos_breakout_sig(
    close: pd.Series,
    *,
    bos_up: pd.Series,
    bos_down: pd.Series,
    confirm_shift: int = 1,
):
    """Sinais de continuação: compra no BOS up, venda no BOS down."""
    buy = bos_up.shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)
    sell = bos_down.shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)
    return buy, sell

def choch_reversal_sig(
    close: pd.Series,
    *,
    choch_up: pd.Series,
    choch_down: pd.Series,
    confirm_shift: int = 1,
):
    """Sinais de reversão: compra quando CHOCH up, venda quando CHOCH down."""
    buy = choch_up.shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)
    sell = choch_down.shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)
    return buy, sell

def fvg_fill_reversion_sig(
    close: pd.Series,
    *,
    fvg_up_upper: pd.Series,
    fvg_up_lower: pd.Series,
    fvg_dn_upper: pd.Series,
    fvg_dn_lower: pd.Series,
    tol: float = 0.0,
    confirm_shift: int = 1,
):
    """
    Reversão ao 'fill' da FVG:
    - Em FVG de alta (gap acima): vender quando close retorna à zona [lower, upper].
    - Em FVG de baixa: comprar quando close retorna à zona [lower, upper].
    """
    c = close.astype(float)

    # FVG up (compradores): olhar fill para baixo => sinal de venda
    in_up_gap = (c <= (fvg_up_upper*(1+tol))) & (c >= (fvg_up_lower*(1-tol)))
    sell = in_up_gap.shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)

    # FVG down (vendedores): olhar fill para cima => sinal de compra
    in_dn_gap = (c <= (fvg_dn_upper*(1+tol))) & (c >= (fvg_dn_lower*(1-tol)))
    buy = in_dn_gap.shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)

    return buy, sell

def combo_smart_money(
    close: pd.Series,
    *,
    bos_up: pd.Series,
    bos_down: pd.Series,
    choch_up: pd.Series,
    choch_down: pd.Series,
    fvg_up_lower: pd.Series,
    fvg_up_upper: pd.Series,
    fvg_dn_lower: pd.Series,
    fvg_dn_upper: pd.Series,
    confirm_shift: int = 1,
):
    """
    Exemplo de regra combinada:
    - Long: CHOCH up recente OU BOS up; melhor se preço 'reacumula' numa FVG down (fill) → compra.
    - Short: CHOCH down OU BOS down; melhor se preço 'redistribui' numa FVG up (fill) → vende.
    """
    c = close.astype(float)

    # condições base
    long_base = (choch_up | bos_up)
    short_base = (choch_down | bos_down)

    # filtros por fill de FVG oposta
    in_dn_gap = (c <= fvg_dn_upper) & (c >= fvg_dn_lower)  # zona de desequilíbrio vendedor sendo "testada" por cima
    in_up_gap = (c <= fvg_up_upper) & (c >= fvg_up_lower)

    entries_long = (long_base & in_dn_gap).shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)
    entries_short = (short_base & in_up_gap).shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)

    # saídas simples: oposto do base
    exits_long = (choch_down | bos_down).shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)
    exits_short = (choch_up | bos_up).shift(confirm_shift).fillna(False).infer_objects(copy=False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short
