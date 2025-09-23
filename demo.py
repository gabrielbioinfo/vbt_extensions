"""Demo.

Demo script for vectorbt extensions: downloads Binance data, applies market structure and fair value gap indicators,
runs a smart money strategy, performs backtesting, parameter grid search, and optional plotting.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from binance.client import Client

from vbt_extensions.data.binance import BinanceDownloadParams, binance_download
from vbt_extensions.ind.fair_value_gap import FAIR_VALUE_GAP
from vbt_extensions.ind.market_structure_zigzag import MARKET_STRUCTURE_ZZ
from vbt_extensions.ind.zigzag import ZIGZAG
from vbt_extensions.risk.targets_stops_smc import smc_targets_stops
from vbt_extensions.sig.smart_money_sig import combo_smart_money

client = Client()
params = BinanceDownloadParams(
    client=client,
    symbol="BTCUSDT",
    interval="5m",
    start="45 days ago UTC",
    tz="America/Sao_Paulo",
    fill_gaps=True,
    drop_partial_last=True,
)
df = binance_download(params)


# ==============================
# 2) SESSÃO genérica (09:00–18:00, seg–sex, America/Sao_Paulo)
# ==============================
def session_mask(index: pd.DatetimeIndex, start="09:00", end="18:00", tz=None, weekdays=None) -> pd.Series:
    idx = index
    if tz:
        idx = idx.tz_convert(tz)
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    start_t = pd.Timestamp(0).replace(hour=s_h, minute=s_m).time()
    end_t = pd.Timestamp(0).replace(hour=e_h, minute=e_m).time()
    mask = pd.Series([(t >= start_t) and (t <= end_t) for t in idx.time], index=index)
    if weekdays is not None:
        mask &= idx.weekday.isin(weekdays)
    return mask

mask = session_mask(df.index, start="09:00", end="18:00",
                    tz="America/Sao_Paulo", weekdays=[0,1,2,3,4])
df_sess = df[mask].copy()
print("Após sessão:", df_sess.shape, df_sess.index.min(), "->", df_sess.index.max())

# ==============================
# 3) INDICADORES — ZigZag -> Market Structure (BOS/CHOCH) + FVG (min size)
# ==============================
# ZigZag (usa o seu se existir; senão fallback simples)

zz = ZIGZAG.run(df_sess["High"], df_sess["Low"], upper=0.005, lower=0.005)  # ~0.5%
ms = MARKET_STRUCTURE_ZZ.run(
    df_sess["Close"],
    is_top=zz.is_top, is_bottom=zz.is_bottom,
    swing_high=zz.swing_high, swing_low=zz.swing_low
)
fvg = FAIR_VALUE_GAP.run(df_sess["High"], df_sess["Low"], df_sess["Close"],
                         min_size=0.0015, unit="pct")  # >= 0.15%

# ==============================
# 4) SINAIS combinados + Stops/Targets naturais
# ==============================
# combo_smart_money (usa seu módulo se possível; senão fallback)

eL, xL, eS, xS = combo_smart_money(
    df_sess["Close"],
    bos_up=ms.bos_up, bos_down=ms.bos_down,
    choch_up=ms.choch_up, choch_down=ms.choch_down,
    fvg_up_lower=fvg.fvg_up_lower, fvg_up_upper=fvg.fvg_up_upper,
    fvg_dn_lower=fvg.fvg_dn_lower, fvg_dn_upper=fvg.fvg_dn_upper,
    confirm_shift=1
)

# Stops/Targets naturais (usa seu módulo se possível; senão fallback)
force_exit, sl_long, sl_short, tgt_long, tgt_short = smc_targets_stops(
    df_sess["Close"], eL, eS,
    swing_high=ms.swing_high, swing_low=ms.swing_low,
    fvg_up_lower=fvg.fvg_up_lower,  # alvo p/ short
    fvg_dn_upper=fvg.fvg_dn_upper,  # alvo p/ long
    rr_min=1.5, buffer_pct=0.0005
)
xL2 = xL | force_exit
xS2 = xS | force_exit

# ==============================
# 5) BACKTEST long+short combinados
# ==============================
pf = vbt.Portfolio.from_signals(
    close=df_sess["Close"],
    entries=eL, exits=xL2,
    short_entries=eS, short_exits=xS2,
    fees=0.0005, slippage=0.0005,
    init_cash=100_000,
    freq="5min",
    cash_sharing=True,
)

print("\n=== STATS (SMC day trade BTC 5m) ===")
print(pf.stats())

# ==============================
# 6) GRID de parâmetros (rápido)
# ==============================
param_grid = {
    "zz": [0.003, 0.005, 0.008],     # 0.3%, 0.5%, 0.8%
    "fvg": [0.0010, 0.0015, 0.0020], # 0.10%, 0.15%, 0.20%
}
rows = []
for zz_thr in param_grid["zz"]:
    zz_i = ZIGZAG.run(df_sess["High"], df_sess["Low"], upper=zz_thr, lower=zz_thr)
    ms_i = MARKET_STRUCTURE_ZZ.run(
        df_sess["Close"],
        is_top=zz_i.is_top, is_bottom=zz_i.is_bottom,
        swing_high=zz_i.swing_high, swing_low=zz_i.swing_low
    )
    for fvg_thr in param_grid["fvg"]:
        fvg_i = FAIR_VALUE_GAP.run(
            df_sess["High"], df_sess["Low"], df_sess["Close"],
            min_size=fvg_thr, unit="pct"
        )
        eL_i, xL_i, eS_i, xS_i = combo_smart_money(
            df_sess["Close"],
            bos_up=ms_i.bos_up, bos_down=ms_i.bos_down,
            choch_up=ms_i.choch_up, choch_down=ms_i.choch_down,
            fvg_up_lower=fvg_i.fvg_up_lower, fvg_up_upper=fvg_i.fvg_up_upper,
            fvg_dn_lower=fvg_i.fvg_dn_lower, fvg_dn_upper=fvg_i.fvg_dn_upper,
            confirm_shift=1
        )
        force_i, _, _, _, _ = smc_targets_stops(
            df_sess["Close"], eL_i, eS_i,
            swing_high=ms_i.swing_high, swing_low=ms_i.swing_low,
            fvg_up_lower=fvg_i.fvg_up_lower, fvg_dn_upper=fvg_i.fvg_dn_upper,
            rr_min=1.5, buffer_pct=0.0005
        )
        xL2_i = xL_i | force_i
        xS2_i = xS_i | force_i

        pf_i = vbt.Portfolio.from_signals(
            close=df_sess["Close"],
            entries=eL_i, exits=xL2_i,
            short_entries=eS_i, short_exits=xS2_i,
            fees=0.0005, slippage=0.0005,
            init_cash=100_000, freq="5min", cash_sharing=True
        )
        st = pf_i.stats()
        rows.append({
            "zz_thr": zz_thr, "fvg_thr": fvg_thr,
            "Total Return [%]": float(st.get("Total Return [%]", np.nan)),
            "Sharpe Ratio": float(st.get("Sharpe Ratio", np.nan)),
            "Max Drawdown [%]": float(st.get("Max Drawdown [%]", np.nan)),
            "Win Rate [%]": float(st.get("Win Rate [%]", np.nan)),
            "Total Trades": float(st.get("Total Trades", np.nan)) if "Total Trades" in st else np.nan,
        })

grid_results = pd.DataFrame(rows).sort_values(
    ["Sharpe Ratio", "Total Return [%]"], ascending=[False, False]
)
print("\n=== TOP GRID RESULTS ===")
print(grid_results.head(10).to_string(index=False))

# ==============================
# 7) PLOTS opcionais
# ==============================
try:
    from vbt_extensions.plotting import plot_fvg, plot_price, plot_structure
    fig = plot_price(df_sess["Close"], title="BTCUSDT 5m — Day Trade (sessão)")
    fig.show()
    fig1 = plot_structure(
        df_sess["Close"],
        swing_high=ms.swing_high, swing_low=ms.swing_low,
        bos_up=ms.bos_up, bos_down=ms.bos_down,
        choch_up=ms.choch_up, choch_down=ms.choch_down,
        title="Market Structure (BOS/CHOCH)"
    )
    fig1.show()
    fig2 = plot_fvg(
        df_sess["Close"],
        fvg_up_lower=fvg.fvg_up_lower, fvg_up_upper=fvg.fvg_up_upper,
        fvg_dn_lower=fvg.fvg_dn_lower, fvg_dn_upper=fvg.fvg_dn_upper,
        title="Fair Value Gaps (min size)"
    )
    fig2.show()
except Exception as e:
    print("\n[Plot opcional] Helpers não disponíveis:", e)
    # Você ainda pode usar pf.plot()
    try:
        pf.plot().show()
    except Exception as e2:
        print("plot pf falhou:", e2)