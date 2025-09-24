"""
Demo SMC Day Trade (BTC 5m)

- Baixa dados via Binance (vbt_extensions.data.binance), com fallback sintético.
- Sessão genérica (09:00–18:00, seg–sex, America/Sao_Paulo).
- ZigZag -> Market Structure (BOS/CHOCH).
- FVG com tamanho mínimo (pct).
- Sinais combinados + stops/targets naturais.
- Backtest long+short com vectorbt.
- Grid simples de parâmetros.

Requisitos:
  pip install vectorbt python-binance pandas numpy plotly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

print("vectorbt:", vbt.__version__)

# ========= 1) Dados (Binance com fallback) =========
USE_SYNTH = False
df = None
try:
    # IMPORT CORRETO: pegue do __init__ de data (seu __init__ deve exportar!)
    from binance.client import Client

    from vbt_extensions.data import BinanceDownloadParams, binance_download

    client = Client()  # usa creds do ambiente se houver
    params = BinanceDownloadParams(
        client=client,
        ticker="BTCUSDT",              # <<== é 'ticker', não 'symbol'
        interval="5m",
        start="45 days ago UTC",
        tz="America/Sao_Paulo",
        fill_gaps=True,
        drop_partial_last=True,
    )
    df = binance_download(params)
    print("Binance OK:", df.shape, df.index.min(), "->", df.index.max())
except Exception as e:
    print("[Aviso] Falha no download Binance, usando dados sintéticos. Motivo:", e)
    USE_SYNTH = True

if USE_SYNTH:
    idx = pd.date_range(
        end=pd.Timestamp.now(tz="America/Sao_Paulo").floor("5min"),
        periods=12*45*24//(60//5),  # ~45 dias em 5m
        freq="5min",
        tz="America/Sao_Paulo",
    )
    rng = np.random.default_rng(11)
    mu, sigma = 0.00005, 0.004
    r = rng.normal(mu, sigma, size=len(idx))
    close = 30000 * np.exp(np.cumsum(r))
    high = close * (1 + rng.uniform(0, 0.002, len(idx)))
    low  = close * (1 - rng.uniform(0, 0.002, len(idx)))
    open_= pd.Series(close, index=idx).shift(1).fillna(close[0]).values
    vol  = rng.uniform(1, 20, len(idx))
    df = pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol}, index=idx, dtype=float)
    print("Sintético:", df.shape, df.index.min(), "->", df.index.max())

# ========= 2) Sessão genérica =========
def session_mask(index: pd.DatetimeIndex, start="09:00", end="18:00", tz=None, weekdays=None) -> pd.Series:
    idx = index
    if tz:
        idx = idx.tz_convert(tz)
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    start_t = pd.Timestamp(0).replace(hour=s_h, minute=s_m).time()
    end_t   = pd.Timestamp(0).replace(hour=e_h, minute=e_m).time()
    mask = pd.Series([(t >= start_t) and (t <= end_t) for t in idx.time], index=index)
    if weekdays is not None:
        mask &= idx.weekday.isin(weekdays)
    return mask

mask = session_mask(df.index, start="09:00", end="18:00", tz="America/Sao_Paulo", weekdays=[0,1,2,3,4])
df_sess = df[mask].copy()
print("Após sessão:", df_sess.shape, df_sess.index.min(), "->", df_sess.index.max())

# ========= 3) Indicadores =========
# ZigZag
try:
    from vbt_extensions.ind.zigzag import ZIGZAG
except Exception:
    # Fallback simples (rolling pivots)
    class ZIGZAG:
        @staticmethod
        def run(high, low, upper=0.005, lower=0.005):
            h = high.astype(float); l = low.astype(float)
            order = 3
            is_top = (h == h.rolling(2*order+1, center=True).max())
            is_bottom = (l == l.rolling(2*order+1, center=True).min())
            swing_high = h.where(is_top).ffill()
            swing_low  = l.where(is_bottom).ffill()
            class R: pass
            r = R()
            r.is_top = is_top.fillna(False)
            r.is_bottom = is_bottom.fillna(False)
            r.swing_high = swing_high
            r.swing_low  = swing_low
            return r

# Market Structure a partir do ZigZag
try:
    from vbt_extensions.ind.market_structure_zigzag import MARKET_STRUCTURE_ZZ
except Exception:
    class MARKET_STRUCTURE_ZZ:
        @staticmethod
        def run(close, *, is_top, is_bottom, swing_high, swing_low, confirm_wicks=False):
            close = close.astype(float)
            up_line = swing_high.shift(1); dn_line = swing_low.shift(1)
            bos_up = (close > up_line).fillna(False)
            bos_down = (close < dn_line).fillna(False)
            trend_up = pd.Series(False, index=close.index)
            trend_dn = pd.Series(False, index=close.index)
            state = 0
            for i in range(len(close)):
                if bool(bos_up.iat[i]): state = 1
                elif bool(bos_down.iat[i]): state = -1
                trend_up.iat[i] = (state == 1)
                trend_dn.iat[i] = (state == -1)
            choch_up = bos_up & trend_dn.shift(1).fillna(False)
            choch_down = bos_down & trend_up.shift(1).fillna(False)
            class R: pass
            r = R()
            r.is_top=is_top; r.is_bottom=is_bottom
            r.swing_high=swing_high; r.swing_low=swing_low
            r.bos_up=bos_up; r.bos_down=bos_down
            r.choch_up=choch_up; r.choch_down=choch_down
            return r

# FVG
try:
    from vbt_extensions.ind.fair_value_gap import FAIR_VALUE_GAP
except Exception:
    class FAIR_VALUE_GAP:
        @staticmethod
        def run(high, low, close=None, *, min_size=0.0, unit="pct", tick_size=None):
            h = high.astype(float); l = low.astype(float)
            c = h if close is None else close.astype(float).reindex_like(h)
            fvg_up = (l > h.shift(2))
            fvg_dn = (h < l.shift(2))
            fvg_up_upper = l.where(fvg_up)
            fvg_up_lower = h.shift(2).where(fvg_up)
            fvg_dn_upper = l.shift(2).where(fvg_dn)
            fvg_dn_lower = h.where(fvg_dn)
            size = (fvg_up_upper - fvg_up_lower).fillna(0) + (fvg_dn_upper - fvg_dn_lower).fillna(0)
            size_pct = (size / c.abs().replace(0, np.nan)).fillna(0)
            if min_size > 0:
                if unit == "pct":
                    ok = size_pct >= min_size
                elif unit == "ticks":
                    if not tick_size: raise ValueError("tick_size requerido")
                    ok = (size / tick_size) >= min_size
                else:
                    raise ValueError("unit inválida")
                fvg_up &= ok; fvg_dn &= ok
                fvg_up_upper = fvg_up_upper.where(fvg_up)
                fvg_up_lower = fvg_up_lower.where(fvg_up)
                fvg_dn_upper = fvg_dn_upper.where(fvg_dn)
                fvg_dn_lower = fvg_dn_lower.where(fvg_dn)
            class R: pass
            r = R()
            r.fvg_up=fvg_up.fillna(False); r.fvg_dn=fvg_dn.fillna(False)
            r.fvg_up_upper=fvg_up_upper; r.fvg_up_lower=fvg_up_lower
            r.fvg_dn_upper=fvg_dn_upper; r.fvg_dn_lower=fvg_dn_lower
            r.size=size; r.size_pct=size_pct
            # filled flags (opcional): deixe simples aqui
            r.filled_partial = pd.Series(False, index=h.index)
            r.filled_full = pd.Series(False, index=h.index)
            return r

# Executa indicadores
zz = ZIGZAG.run(df_sess["High"], df_sess["Low"], upper=0.005, lower=0.005)  # ~0.5%
ms = MARKET_STRUCTURE_ZZ.run(
    df_sess["Close"],
    is_top=zz.is_top, is_bottom=zz.is_bottom,
    swing_high=zz.swing_high, swing_low=zz.swing_low
)
fvg = FAIR_VALUE_GAP.run(
    df_sess["High"], df_sess["Low"], df_sess["Close"],
    min_size=0.0015, unit="pct"   # >= 0.15%
)

# ========= 4) Sinais + Stops/Targets =========
try:
    from vbt_extensions.sig.smart_money_sig import combo_smart_money
except Exception:
    def combo_smart_money(close, *, bos_up, bos_down, choch_up, choch_down,
                          fvg_up_lower, fvg_up_upper, fvg_dn_lower, fvg_dn_upper,
                          confirm_shift=1):
        c = close.astype(float)
        long_base  = (choch_up | bos_up)
        short_base = (choch_down | bos_down)
        in_dn_gap = (c <= fvg_dn_upper) & (c >= fvg_dn_lower)
        in_up_gap = (c <= fvg_up_upper) & (c >= fvg_up_lower)
        eL = (long_base & in_dn_gap).shift(confirm_shift).fillna(False)
        eS = (short_base & in_up_gap).shift(confirm_shift).fillna(False)
        xL = (choch_down | bos_down).shift(confirm_shift).fillna(False)
        xS = (choch_up | bos_up).shift(confirm_shift).fillna(False)
        return eL.astype(bool), xL.astype(bool), eS.astype(bool), xS.astype(bool)

eL, xL, eS, xS = combo_smart_money(
    df_sess["Close"],
    bos_up=ms.bos_up, bos_down=ms.bos_down,
    choch_up=ms.choch_up, choch_down=ms.choch_down,
    fvg_up_lower=fvg.fvg_up_lower, fvg_up_upper=fvg.fvg_up_upper,
    fvg_dn_lower=fvg.fvg_dn_lower, fvg_dn_upper=fvg.fvg_dn_upper,
    confirm_shift=1
)

try:
    from vbt_extensions.risk.targets_stops_smc import smc_targets_stops
except Exception:
    def smc_targets_stops(close, entries_long, entries_short, *,
                          swing_high, swing_low,
                          fvg_up_lower=None, fvg_dn_upper=None,
                          rr_min=None, buffer_pct=0.0):
        c = close.astype(float)
        sh = swing_high.astype(float).ffill(); sl = swing_low.astype(float).ffill()
        tgt_long  = fvg_dn_upper.combine_first(sh) if fvg_dn_upper is not None else sh
        tgt_short = fvg_up_lower.combine_first(sl) if fvg_up_lower is not None else sl
        stop_long  = sl * (1 - buffer_pct)
        stop_short = sh * (1 + buffer_pct)
        force = pd.Series(False, index=c.index)
        inL = inS = False
        for i, px in enumerate(c.values):
            if entries_long.iat[i]:
                inL, inS = True, False
                if rr_min is not None:
                    rr = (tgt_long.iat[i] - px) / max(px - stop_long.iat[i], 1e-12)
                    if rr < rr_min: inL = False
            if entries_short.iat[i]:
                inS, inL = True, False
                if rr_min is not None:
                    rr = (px - tgt_short.iat[i]) / max(stop_short.iat[i] - px, 1e-12)
                    if rr < rr_min: inS = False
            if inL and (px <= stop_long.iat[i] or px >= tgt_long.iat[i]):
                force.iat[i] = True; inL = False
            if inS and (px >= stop_short.iat[i] or px <= tgt_short.iat[i]):
                force.iat[i] = True; inS = False
        return force, stop_long, stop_short, tgt_long, tgt_short

force_exit, sl_long, sl_short, tgt_long, tgt_short = smc_targets_stops(
    df_sess["Close"], eL, eS,
    swing_high=ms.swing_high, swing_low=ms.swing_low,
    fvg_up_lower=fvg.fvg_up_lower,   # alvo p/ short
    fvg_dn_upper=fvg.fvg_dn_upper,   # alvo p/ long
    rr_min=1.5, buffer_pct=0.0005
)
xL2 = xL | force_exit
xS2 = xS | force_exit

# ========= 5) Backtest =========
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

# ========= 6) Grid de parâmetros rápido =========
param_grid = {
    "zz":  [0.003, 0.005, 0.008],     # 0.3%, 0.5%, 0.8%
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
        fvg_i = FAIR_VALUE_GAP.run(df_sess["High"], df_sess["Low"], df_sess["Close"], min_size=fvg_thr, unit="pct")
        eL_i, xL_i, eS_i, xS_i = combo_smart_money(
            df_sess["Close"],
            bos_up=ms_i.bos_up, bos_down=ms_i.bos_down,
            choch_up=ms_i.choch_up, choch_down=ms_i.choch_down,
            fvg_up_lower=fvg_i.fvg_up_lower, fvg_up_upper=fvg_i.fvg_up_upper,
            fvg_dn_lower=fvg_i.fvg_dn_lower, fvg_dn_upper=fvg_i.fvg_dn_upper,
            confirm_shift=1
        )
        force_i, *_ = smc_targets_stops(
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

grid_results = pd.DataFrame(rows).sort_values(["Sharpe Ratio", "Total Return [%]"], ascending=[False, False])
print("\n=== TOP GRID RESULTS ===")
print(grid_results.head(10).to_string(index=False))

# ========= 7) Plots (opcional) =========
try:
    # dependendo do seu __init__.py, os helpers podem estar em plotting.helpers
    try:
        from vbt_extensions.plotting import plot_fvg, plot_price, plot_structure
    except Exception:
        from vbt_extensions.plotting.helpers import plot_fvg, plot_price, plot_structure

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
    try:
        pf.plot().show()
    except Exception as e2:
        print("plot pf falhou:", e2)
