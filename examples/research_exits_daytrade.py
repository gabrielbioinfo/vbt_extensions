# research_exits_daytrade.py
# ------------------------------------------------------------
# Pesquisa de saídas (SL / TP / TS e combinações) em BTC 5m (day trade),
# inspirada no estudo "2M backtests" (Oleg Polakow).
# Mede: expectancy, win rate, avg win/avg loss, duração e motivo do exit.
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

print("vectorbt:", vbt.__version__)

# ==============================
# 0) Utils
# ==============================
def session_mask(index: pd.DatetimeIndex, start="09:00", end="18:00", tz=None, weekdays=None) -> pd.Series:
    idx = index
    if tz:
        idx = idx.tz_convert(tz)
    s_h, s_m = map(int, start.split(":")); e_h, e_m = map(int, end.split(":"))
    start_t = pd.Timestamp(0).replace(hour=s_h, minute=s_m).time()
    end_t   = pd.Timestamp(0).replace(hour=e_h, minute=e_m).time()
    mask = pd.Series([(t >= start_t) and (t <= end_t) for t in idx.time], index=index)
    if weekdays is not None:
        mask &= idx.weekday.isin(weekdays)
    return mask

def _ensure_tz(df: pd.DataFrame, tz="America/Sao_Paulo") -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if tz:
        df = df.tz_convert(tz)
    return df

# ==============================
# 1) Dados — Binance 5m (fallback sintético)
# ==============================
USE_SYNTH = False
df = None
try:
    from binance.client import Client

    from vbt_extensions.data import BinanceDownloadParams, binance_download
    client = Client()  # usa credenciais de env se houver
    params = BinanceDownloadParams(
        client=client,
        ticker="BTCUSDT",
        interval="5m",
        start="45 days ago UTC",
        tz="America/Sao_Paulo",
        fill_gaps=True,
        drop_partial_last=True,
    )
    df = binance_download(params)
    print("Binance OK:", df.shape, df.index.min(), "->", df.index.max())
except Exception as e:
    print("[Aviso] Binance falhou, usando sintético. Motivo:", e)
    USE_SYNTH = True

if USE_SYNTH:
    idx = pd.date_range(
        end=pd.Timestamp.now(tz="America/Sao_Paulo").floor("5min"),
        periods=12*45*24//(60//5), freq="5min", tz="America/Sao_Paulo"
    )
    rng = np.random.default_rng(7)
    mu, sigma = 0.00005, 0.004
    r = rng.normal(mu, sigma, size=len(idx))
    close = 30000 * np.exp(np.cumsum(r))
    high = close * (1 + rng.uniform(0, 0.002, len(idx)))
    low  = close * (1 - rng.uniform(0, 0.002, len(idx)))
    open_= pd.Series(close, index=idx).shift(1).fillna(close[0]).values
    vol  = rng.uniform(1, 20, len(idx))
    df = pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":vol}, index=idx, dtype=float)
    print("Sintético:", df.shape, df.index.min(), "->", df.index.max())

df = _ensure_tz(df, "America/Sao_Paulo")
mask = session_mask(df.index, start="09:00", end="18:00", tz="America/Sao_Paulo", weekdays=[0,1,2,3,4])
df = df[mask].copy()
print("Sessão:", df.shape, df.index.min(), "->", df.index.max())

# ==============================
# 2) Entradas simples (placeholder)
# ------------------------------
# Para isolar o efeito de SAÍDAS, vamos usar entradas "commodities":
# - um cruzamento de MAs para ter trades reais
# Você pode trocar por seus sinais SMC depois.
# ==============================
fast, slow = 10, 30
fma = vbt.MA.run(df["Close"], window=fast).ma
sma = vbt.MA.run(df["Close"], window=slow).ma

entries_long  = (fma > sma) & (fma.shift(1) <= sma.shift(1))
exits_long    = (fma < sma) & (fma.shift(1) >= sma.shift(1))
entries_short = (fma < sma) & (fma.shift(1) >= sma.shift(1))
exits_short   = (fma > sma) & (fma.shift(1) <= sma.shift(1))

# ==============================
# 3) Motor de exits (SL / TP / TS) custom
# ------------------------------
# Regras:
# - LONG: abre em Close[i]; monitora a partir de i+1:
#   * SL: Low <= entry*(1 - sl_pct)
#   * TP: High >= entry*(1 + tp_pct)
#   * TS: trailing a partir de max_price desde entrada
#         stop = max_price*(1 - ts_pct)
#   prioridade de checagem no mesmo candle: TP antes de SL (ajuste viés como quiser)
# - SHORT simétrico
# - se não bater nada, fecha no exit_signal ou no último bar (force close)
# - coleta motivo ('TP','SL','TS','EXIT','END') e PnL por trade
# ==============================

from dataclasses import dataclass


@dataclass
class ExitResult:
    exits_long: pd.Series
    exits_short: pd.Series
    trade_pnl: pd.Series       # por trade (em retorno simples)
    trade_dur: pd.Series       # em barras
    trade_reason: pd.Series    # motivo do exit
    summary: dict              # métricas agregadas

def simulate_exits_ohlc(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    entries_long: pd.Series, exits_long_signal: pd.Series,
    entries_short: pd.Series, exits_short_signal: pd.Series,
    *,
    sl_pct: float | None = None,
    tp_pct: float | None = None,
    ts_pct: float | None = None,
    prefer_tp_over_sl: bool = True,
) -> ExitResult:
    idx = close.index
    eL = entries_long.reindex(idx).fillna(False).astype(bool)
    xLsig = exits_long_signal.reindex(idx).fillna(False).astype(bool)
    eS = entries_short.reindex(idx).fillna(False).astype(bool)
    xSsig = exits_short_signal.reindex(idx).fillna(False).astype(bool)

    # saídas geradas
    xL = pd.Series(False, index=idx)
    xS = pd.Series(False, index=idx)

    # tracking
    in_long = False; in_short = False
    entry_px = np.nan; entry_i = -1
    max_px = -np.inf; min_px = np.inf

    reasons = []   # por trade
    pnls = []
    durs = []

    def close_trade(i_exit: int, px_exit: float, reason: str):
        nonlocal in_long, in_short, entry_px, entry_i, max_px, min_px
        ret = (px_exit / entry_px - 1.0) if in_long else (entry_px / px_exit - 1.0)
        duration = i_exit - entry_i
        pnls.append(ret)
        reasons.append(reason)
        durs.append(duration)
        in_long = False; in_short = False
        entry_px = np.nan; entry_i = -1
        max_px = -np.inf; min_px = np.inf

    # loop (robusto e simples)
    for k, t in enumerate(idx):
        op = float(open_.iat[k]); hi = float(high.iat[k]); lo = float(low.iat[k]); cl = float(close.iat[k])

        # abre posição se sinal no open da barra (e não estiver posicionado)
        if not in_long and not in_short:
            if eL.iat[k]:
                in_long = True; entry_px = cl; entry_i = k; max_px = cl; min_px = cl
            elif eS.iat[k]:
                in_short = True; entry_px = cl; entry_i = k; max_px = cl; min_px = cl

        if in_long:
            # atualiza trailing info
            if hi > max_px: max_px = hi
            # calcula níveis
            tp_px = entry_px * (1 + tp_pct) if tp_pct else None
            sl_px = entry_px * (1 - sl_pct) if sl_pct else None
            ts_px = max_px * (1 - ts_pct) if ts_pct else None

            # prioridade de evento dentro do candle
            did_close = False
            if prefer_tp_over_sl:
                if tp_pct and hi >= tp_px:
                    xL.iat[k] = True; close_trade(k, tp_px, "TP"); did_close = True
                if not did_close and sl_pct and lo <= sl_px:
                    xL.iat[k] = True; close_trade(k, sl_px, "SL"); did_close = True
            else:
                if sl_pct and lo <= sl_px:
                    xL.iat[k] = True; close_trade(k, sl_px, "SL"); did_close = True
                if not did_close and tp_pct and hi >= tp_px:
                    xL.iat[k] = True; close_trade(k, tp_px, "TP"); did_close = True

            if not did_close and ts_pct:
                if lo <= ts_px:  # atingiu trailing
                    xL.iat[k] = True; close_trade(k, ts_px, "TS"); did_close = True

            if not did_close and xLsig.iat[k]:
                xL.iat[k] = True; close_trade(k, cl, "EXIT"); did_close = True

            # force close na última barra
            if not did_close and (k == len(idx)-1):
                xL.iat[k] = True; close_trade(k, cl, "END")

        elif in_short:
            if lo < min_px: min_px = lo
            tp_px = entry_px * (1 - tp_pct) if tp_pct else None
            sl_px = entry_px * (1 + sl_pct) if sl_pct else None
            ts_px = min_px * (1 + ts_pct) if ts_pct else None

            did_close = False
            if prefer_tp_over_sl:
                if tp_pct and lo <= tp_px:
                    xS.iat[k] = True; close_trade(k, tp_px, "TP"); did_close = True
                if not did_close and sl_pct and hi >= sl_px:
                    xS.iat[k] = True; close_trade(k, sl_px, "SL"); did_close = True
            else:
                if sl_pct and hi >= sl_px:
                    xS.iat[k] = True; close_trade(k, sl_px, "SL"); did_close = True
                if not did_close and tp_pct and lo <= tp_px:
                    xS.iat[k] = True; close_trade(k, tp_px, "TP"); did_close = True

            if not did_close and ts_pct:
                if hi >= ts_px:
                    xS.iat[k] = True; close_trade(k, ts_px, "TS"); did_close = True

            if not did_close and xSsig.iat[k]:
                xS.iat[k] = True; close_trade(k, cl, "EXIT"); did_close = True

            if not did_close and (k == len(idx)-1):
                xS.iat[k] = True; close_trade(k, cl, "END")

    # métricas
    trade_pnl = pd.Series(pnls, name="ret")
    trade_dur = pd.Series(durs, name="dur")
    trade_reason = pd.Series(reasons, name="reason")

    if len(trade_pnl) == 0:
        summary = dict(trades=0, expectancy=np.nan, win_rate=np.nan,
                       avg_win=np.nan, avg_loss=np.nan,
                       avg_dur=np.nan, reason_counts={})
    else:
        pos = trade_pnl[trade_pnl > 0]; neg = trade_pnl[trade_pnl <= 0]
        expectancy = trade_pnl.mean()
        win_rate = (trade_pnl > 0).mean() * 100
        avg_win = pos.mean() if not pos.empty else 0.0
        avg_loss = neg.mean() if not neg.empty else 0.0
        avg_dur = trade_dur.mean()
        reason_counts = trade_reason.value_counts().to_dict()
        summary = dict(
            trades=int(len(trade_pnl)),
            expectancy=float(expectancy),
            win_rate=float(win_rate),
            avg_win=float(avg_win) if not np.isnan(avg_win) else 0.0,
            avg_loss=float(avg_loss) if not np.isnan(avg_loss) else 0.0,
            avg_dur=float(avg_dur),
            reason_counts=reason_counts
        )

    return ExitResult(xL, xS, trade_pnl, trade_dur, trade_reason, summary)

# ==============================
# 4) Experimentos — tipos de exits e grids
# ==============================
exit_setups = [
    ("none",   dict(sl_pct=None,  tp_pct=None,  ts_pct=None)),
    ("sl",     dict(sl_pct=0.01,  tp_pct=None,  ts_pct=None)),
    ("tp",     dict(sl_pct=None,  tp_pct=0.02,  ts_pct=None)),
    ("sl_tp",  dict(sl_pct=0.01,  tp_pct=0.02,  ts_pct=None)),
    ("ts",     dict(sl_pct=None,  tp_pct=None,  ts_pct=0.01)),
    ("ts_tp",  dict(sl_pct=None,  tp_pct=0.02,  ts_pct=0.01)),
]

# grids (você pode expandir)
sl_grid = [0.005, 0.01, 0.02]   # 0.5%/1%/2%
tp_grid = [0.01, 0.02, 0.03]    # 1%/2%/3%
ts_grid = [0.005, 0.01, 0.02]   # trailing 0.5%/1%/2%

rows = []
for name, base in exit_setups:
    # para cada setup, varrer o grid adequado
    sls = sl_grid if base["sl_pct"] is not None else [None]
    tps = tp_grid if base["tp_pct"] is not None else [None]
    tss = ts_grid if base["ts_pct"] is not None else [None]

    for sl in sls:
        for tp in tps:
            for ts in tss:
                res = simulate_exits_ohlc(
                    df["Open"], df["High"], df["Low"], df["Close"],
                    entries_long=entries_long, exits_long_signal=exits_long,
                    entries_short=entries_short, exits_short_signal=exits_short,
                    sl_pct=sl, tp_pct=tp, ts_pct=ts, prefer_tp_over_sl=True
                )

                # equity (1 contrato, sem alavancagem) — usando retornos por trade acumulados
                # aqui só avaliamos métricas de trades (estilo estudo do artigo)
                s = res.summary
                rows.append({
                    "setup": name,
                    "sl_pct": sl, "tp_pct": tp, "ts_pct": ts,
                    "trades": s["trades"],
                    "expectancy": s["expectancy"],
                    "win_rate_%": s["win_rate"],
                    "avg_win": s["avg_win"],
                    "avg_loss": s["avg_loss"],
                    "avg_dur_bars": s["avg_dur"],
                    "reason_counts": s["reason_counts"],
                })

results = pd.DataFrame(rows)
# ranking por expectancy e win_rate
ranked = results.sort_values(["expectancy", "win_rate_%"], ascending=[False, False])
print("\n=== TOP EXITS (por expectancy, desempate win_rate) ===")
print(ranked.head(12).to_string(index=False))

# breakdown de motivos de saída no melhor setup
if not ranked.empty:
    best = ranked.iloc[0]
    print("\nMelhor setup:", dict(best.drop("reason_counts")))
    print("Motivos:", best["reason_counts"])

# ==============================
# 5) (Opcional) Backtest com vectorbt para o melhor setup
# ------------------------------
# Transformamos as saídas simuladas em sinais e rodamos um pf de referência.
# ==============================
try:
    best = ranked.iloc[0]
    res_best = simulate_exits_ohlc(
        df["Open"], df["High"], df["Low"], df["Close"],
        entries_long=entries_long, exits_long_signal=exits_long,
        entries_short=entries_short, exits_short_signal=exits_short,
        sl_pct=None if pd.isna(best["sl_pct"]) else float(best["sl_pct"]),
        tp_pct=None if pd.isna(best["tp_pct"]) else float(best["tp_pct"]),
        ts_pct=None if pd.isna(best["ts_pct"]) else float(best["ts_pct"]),
        prefer_tp_over_sl=True
    )
    pf = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries_long, exits=res_best.exits_long,
        short_entries=entries_short, short_exits=res_best.exits_short,
        fees=0.0005, slippage=0.0005, init_cash=100_000,
        freq="5min", cash_sharing=True
    )
    print("\n=== PF STATS (melhor setup de exit) ===")
    print(pf.stats())
except Exception as e:
    print("[Aviso] Backtest de referência falhou:", e)
