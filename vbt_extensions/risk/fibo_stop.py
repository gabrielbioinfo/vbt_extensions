import pandas as pd


def fibo_protective_stop(
    close: pd.Series,
    entries_long: pd.Series,
    entries_short: pd.Series,
    *,
    fib_levels: pd.DataFrame,
    behind_level: str = "fib_61.8",
    buffer_pct: float = 0.001,  # 0.1% além do nível
):
    """Gera force_exit booleana posicionando o stop no nível de Fibo escolhido
    (atrás da direção da entrada), com pequeno buffer.

    - Para long: stop = fib[behind_level] * (1 - buffer_pct)
    - Para short: stop = fib[behind_level] * (1 + buffer_pct)
    """
    close = close.astype(float)
    long_e = entries_long.reindex_like(close).fillna(False).astype(bool)
    short_e = entries_short.reindex_like(close).fillna(False).astype(bool)
    lvl = fib_levels[behind_level].reindex_like(close).astype(float)

    stop_long = lvl * (1.0 - buffer_pct)
    stop_short = lvl * (1.0 + buffer_pct)

    in_pos_long = False
    in_pos_short = False
    force_exit = pd.Series(False, index=close.index)

    for i, px in enumerate(close.values):
        if long_e.iat[i]:
            in_pos_long = True
            in_pos_short = False
        if short_e.iat[i]:
            in_pos_short = True
            in_pos_long = False

        if in_pos_long and px <= stop_long.iat[i]:
            force_exit.iat[i] = True
            in_pos_long = False

        if in_pos_short and px >= stop_short.iat[i]:
            force_exit.iat[i] = True
            in_pos_short = False

    return force_exit


# import vectorbt as vbt

# from vbt_extensions.ind.rolling_peaks_scipy import PEAKS_SCIPY
# from vbt_extensions.ind.retracement_ratios import FIB_RETRACEMENT
# from vbt_extensions.sig.retracement_sig import fib_trend_continuation_sig
# from vbt_extensions.risk.fibo_stop import fibo_protective_stop
# from vbt_extensions.plotting.helpers import plot_indicator_tearsheet

# # 1) Pivôs (usando SciPy Peaks que você já tem)
# res_peaks = PEAKS_SCIPY.run(
#     source_series=((df["High"]+df["Low"])/2),
#     smooth="ema", smooth_window=5,
#     top_prominence=0.5, bottom_prominence=0.5,  # só exemplo
# )

# # 2) Fibo por perna ativa
# fib = FIB_RETRACEMENT.run(
#     close=df["Close"],
#     swing_high=res_peaks.swing_high,
#     swing_low=res_peaks.swing_low,
#     is_top=res_peaks.is_top,
#     is_bottom=res_peaks.is_bottom,
#     ratios=(0.236, 0.382, 0.5, 0.618, 0.786),
#     ext=(1.272, 1.618),
# )

# # 3) Sinais de continuação (na direção da perna), gatilho no 61.8%
# eL, xL, eS, xS = fib_trend_continuation_sig(
#     df["Close"],
#     direction=fib.direction,
#     fib_levels=fib.fib_levels,
#     trigger_level="fib_61.8",
#     confirm_shift=1
# )

# # 4) Stop no Fibo 61.8 “atrás” da entrada
# force_out = fibo_protective_stop(
#     df["Close"], eL, eS,
#     fib_levels=fib.fib_levels,
#     behind_level="fib_61.8",
#     buffer_pct=0.001
# )
# xL = xL | force_out
# xS = xS | force_out

# # 5) Portfolio (ex.: só lado comprado)
# pf = vbt.Portfolio.from_signals(
#     close=df["Close"],
#     entries=eL,
#     exits=xL,
#     fees=0.0005, slippage=0.0005, init_cash=100_000, freq="D"
# )

# # 6) Plot (preço + níveis principais)
# fig = df["Close"].vbt.plot()
# for col in ["fib_38.2", "fib_50", "fib_61.8"]:
#     fig = fib.fib_levels[col].vbt.plot(fig=fig)
# if fib.ext_levels is not None:
#     for col in fib.ext_levels.columns:
#         fig = fib.ext_levels[col].vbt.plot(fig=fig)
# fig.show()
