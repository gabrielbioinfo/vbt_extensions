# vbt_extensions/risk/exit_engine.py
"""
Exit Engine: motor unificado de saídas (TP/SL/Trailing) para estratégias.

Modos suportados:
- TP: 'none' | 'percent' | 'atr' | 'rr' | 'level'
- SL: 'none' | 'percent' | 'atr' | 'level'
- Trailing: 'none' | 'percent' | 'atr' (Chandelier-like)

Notas:
- 'percent': usa percentual do preço de entrada (ex.: tp_pct=0.02 => +2%).
- 'atr': usa ATR da barra de entrada (ou dinâmica, se preferir) com multiplicador.
- 'rr': take-profit usando múltiplo do risco inicial: TP = entry + rr_mult*(entry - stop) (long).
- 'level': usa uma série externa por barra (ex.: nível de Fibonacci/SR), passada via tp_level_long / tp_level_short / sl_level_long / sl_level_short.

Garantias:
- Sempre fecha trade: por TP/SL/Trailing/EXIT-SIGNAL ou no "END" (bar final). 
- Checagem intra-bar: para LONG, ordem típica de prioridade: TP -> SL -> Trailing -> sinal de exit.
  (ajustável via prefer_tp_over_sl)

Retornos:
- Exits long/short (Series booleanas), níveis de TP/SL/Trailing por barra (séries com o último nível aplicável),
- Log de trades (retorno por trade, duração, motivo), resumo agregado.

Integração:
- use as saídas com vectorbt.Portfolio.from_signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# =========================
# Helpers de ATR (simples)
# =========================
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # SMA para simplicidade; troque por EMA se quiser
    atr = tr.rolling(window, min_periods=1).mean()
    return atr


# =========================
# Configurações (dataclasses)
# =========================
TPMode = Literal["none", "percent", "atr", "rr", "level"]
SLMode = Literal["none", "percent", "atr", "level"]
TrailMode = Literal["none", "percent", "atr"]

@dataclass
class ExitConfig:
    # TP
    tp_mode: TPMode = "none"
    tp_pct: float | None = None               # para 'percent'
    tp_atr_mult: float | None = None          # para 'atr'
    tp_rr_mult: float | None = None           # para 'rr'
    tp_level_long: pd.Series | None = None    # para 'level' (alvos long)
    tp_level_short: pd.Series | None = None   # para 'level' (alvos short)

    # SL
    sl_mode: SLMode = "none"
    sl_pct: float | None = None               # para 'percent'
    sl_atr_mult: float | None = None          # para 'atr'
    sl_level_long: pd.Series | None = None    # para 'level' (stops long)
    sl_level_short: pd.Series | None = None   # para 'level' (stops short)

    # Trailing
    trail_mode: TrailMode = "none"
    trail_pct: float | None = None            # para 'percent'
    trail_atr_mult: float | None = None       # para 'atr' (Chandelier)

    # ATR base (se usar 'atr' em TP/SL/Trailing)
    atr_series: pd.Series | None = None
    atr_window: int = 14

    # Prioridade intra-bar (para LONG e SHORT)
    prefer_tp_over_sl: bool = True

    # Outras opções
    force_exit_on_signal: bool = True  # respeitar sinal de exit da estratégia
    # Se quiser recalcular ATR dinamicamente ao longo do trade, troque a lógica no laço.


@dataclass
class ExitResult:
    exits_long: pd.Series
    exits_short: pd.Series
    tp_level: pd.Series
    sl_level: pd.Series
    trail_level: pd.Series
    trade_pnl: pd.Series
    trade_dur: pd.Series
    trade_reason: pd.Series
    summary: dict


# =========================
# Motor principal
# =========================
def simulate_exits(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    entries_long: pd.Series,
    exits_long_signal: pd.Series,
    entries_short: pd.Series,
    exits_short_signal: pd.Series,
    cfg: ExitConfig,
) -> ExitResult:
    # sanity e alinhamento
    idx = close.index
    open_, high, low, close = [s.reindex(idx) for s in (open_, high, low, close)]
    eL = entries_long.reindex(idx).fillna(False).astype(bool)
    xLsig = exits_long_signal.reindex(idx).fillna(False).astype(bool)
    eS = entries_short.reindex(idx).fillna(False).astype(bool)
    xSsig = exits_short_signal.reindex(idx).fillna(False).astype(bool)

    # ATR se necessário
    atr = cfg.atr_series
    if atr is None and (
        cfg.tp_mode == "atr"
        or cfg.sl_mode == "atr"
        or (cfg.trail_mode == "atr")
    ):
        atr = compute_atr(high, low, close, window=cfg.atr_window)

    # níveis (Series) – úteis para plot/diagnóstico
    tp_level_series   = pd.Series(np.nan, index=idx, dtype=float)
    sl_level_series   = pd.Series(np.nan, index=idx, dtype=float)
    trail_level_series= pd.Series(np.nan, index=idx, dtype=float)

    exitsL = pd.Series(False, index=idx)
    exitsS = pd.Series(False, index=idx)

    # estado
    in_long = False
    in_short = False
    entry_px = np.nan
    entry_i = -1

    # para trailing
    max_since_entry = -np.inf
    min_since_entry = np.inf
    trail_level = np.nan

    tp_level = np.nan
    sl_level = np.nan

    # logs por trade
    reasons, pnls, durs = [], [], []

    def _close_trade(i_exit: int, px_exit: float, reason: str):
        nonlocal in_long, in_short, entry_px, entry_i, max_since_entry, min_since_entry, trail_level, tp_level, sl_level
        # retorno simples
        pnl = (px_exit / entry_px - 1.0) if in_long else (entry_px / px_exit - 1.0)
        dur = i_exit - entry_i
        reasons.append(reason); pnls.append(pnl); durs.append(dur)

        in_long = False; in_short = False
        entry_px = np.nan; entry_i = -1
        max_since_entry = -np.inf; min_since_entry = np.inf
        trail_level = np.nan; tp_level = np.nan; sl_level = np.nan

    # helpers p/ níveis iniciais
    def _init_levels_long(k: int, ep: float) -> tuple[float, float]:
        _tp = np.nan; _sl = np.nan
        if cfg.tp_mode == "percent" and cfg.tp_pct:
            _tp = ep * (1 + cfg.tp_pct)
        elif cfg.tp_mode == "atr" and cfg.tp_atr_mult and atr is not None:
            _tp = ep + cfg.tp_atr_mult * float(atr.iat[k])
        elif cfg.tp_mode == "rr" and cfg.tp_rr_mult:
            # requer SL já definido. Se SL não for percent/atr/level, não faz sentido usar RR
            pass
        elif cfg.tp_mode == "level" and cfg.tp_level_long is not None:
            _tp = float(cfg.tp_level_long.reindex(idx).iat[k])

        if cfg.sl_mode == "percent" and cfg.sl_pct:
            _sl = ep * (1 - cfg.sl_pct)
        elif cfg.sl_mode == "atr" and cfg.sl_atr_mult and atr is not None:
            _sl = ep - cfg.sl_atr_mult * float(atr.iat[k])
        elif cfg.sl_mode == "level" and cfg.sl_level_long is not None:
            _sl = float(cfg.sl_level_long.reindex(idx).iat[k])

        # se TP for RR, usa a distância do SL
        if cfg.tp_mode == "rr" and cfg.tp_rr_mult:
            if not np.isnan(_sl):
                risk = ep - _sl
                if risk > 0:
                    _tp = ep + cfg.tp_rr_mult * risk
        return _tp, _sl

    def _init_levels_short(k: int, ep: float) -> tuple[float, float]:
        _tp = np.nan; _sl = np.nan
        if cfg.tp_mode == "percent" and cfg.tp_pct:
            _tp = ep * (1 - cfg.tp_pct)
        elif cfg.tp_mode == "atr" and cfg.tp_atr_mult and atr is not None:
            _tp = ep - cfg.tp_atr_mult * float(atr.iat[k])
        elif cfg.tp_mode == "rr" and cfg.tp_rr_mult:
            pass
        elif cfg.tp_mode == "level" and cfg.tp_level_short is not None:
            _tp = float(cfg.tp_level_short.reindex(idx).iat[k])

        if cfg.sl_mode == "percent" and cfg.sl_pct:
            _sl = ep * (1 + cfg.sl_pct)
        elif cfg.sl_mode == "atr" and cfg.sl_atr_mult and atr is not None:
            _sl = ep + cfg.sl_atr_mult * float(atr.iat[k])
        elif cfg.sl_mode == "level" and cfg.sl_level_short is not None:
            _sl = float(cfg.sl_level_short.reindex(idx).iat[k])

        if cfg.tp_mode == "rr" and cfg.tp_rr_mult:
            if not np.isnan(_sl):
                risk = _sl - ep
                if risk > 0:
                    _tp = ep - cfg.tp_rr_mult * risk
        return _tp, _sl

    for k, t in enumerate(idx):
        op = float(open_.iat[k]); hi = float(high.iat[k]); lo = float(low.iat[k]); cl = float(close.iat[k])

        # abrir posição
        if not in_long and not in_short:
            if eL.iat[k]:
                in_long = True; entry_px = cl; entry_i = k
                max_since_entry = cl; min_since_entry = cl
                tp_level, sl_level = _init_levels_long(k, entry_px)
                # trailing inicial
                if cfg.trail_mode == "percent" and cfg.trail_pct:
                    trail_level = max_since_entry * (1 - cfg.trail_pct)
                elif cfg.trail_mode == "atr" and cfg.trail_atr_mult and atr is not None:
                    trail_level = max_since_entry - cfg.trail_atr_mult * float(atr.iat[k])
                else:
                    trail_level = np.nan

            elif eS.iat[k]:
                in_short = True; entry_px = cl; entry_i = k
                max_since_entry = cl; min_since_entry = cl
                tp_level, sl_level = _init_levels_short(k, entry_px)
                if cfg.trail_mode == "percent" and cfg.trail_pct:
                    trail_level = min_since_entry * (1 + cfg.trail_pct)
                elif cfg.trail_mode == "atr" and cfg.trail_atr_mult and atr is not None:
                    trail_level = min_since_entry + cfg.trail_atr_mult * float(atr.iat[k])
                else:
                    trail_level = np.nan

        # registrar níveis visíveis
        tp_level_series.iat[k] = tp_level
        sl_level_series.iat[k] = sl_level
        trail_level_series.iat[k] = trail_level

        # atualização de trailing durante o trade
        if in_long:
            if hi > max_since_entry:
                max_since_entry = hi
            if cfg.trail_mode == "percent" and cfg.trail_pct:
                trail_level = max(trail_level, max_since_entry * (1 - cfg.trail_pct)) if not np.isnan(trail_level) else max_since_entry * (1 - cfg.trail_pct)
            elif cfg.trail_mode == "atr" and cfg.trail_atr_mult and atr is not None:
                cand = max_since_entry - cfg.trail_atr_mult * float(atr.iat[k])
                trail_level = max(trail_level, cand) if not np.isnan(trail_level) else cand

            # prioridade intra-bar
            did_close = False
            if cfg.prefer_tp_over_sl:
                if (cfg.tp_mode != "none") and (not np.isnan(tp_level)) and (hi >= tp_level):
                    exitsL.iat[k] = True; _close_trade(k, tp_level, "TP"); did_close = True
                if not did_close and (cfg.sl_mode != "none") and (not np.isnan(sl_level)) and (lo <= sl_level):
                    exitsL.iat[k] = True; _close_trade(k, sl_level, "SL"); did_close = True
            else:
                if (cfg.sl_mode != "none") and (not np.isnan(sl_level)) and (lo <= sl_level):
                    exitsL.iat[k] = True; _close_trade(k, sl_level, "SL"); did_close = True
                if not did_close and (cfg.tp_mode != "none") and (not np.isnan(tp_level)) and (hi >= tp_level):
                    exitsL.iat[k] = True; _close_trade(k, tp_level, "TP"); did_close = True

            if not did_close and cfg.trail_mode != "none" and (not np.isnan(trail_level)) and (lo <= trail_level):
                exitsL.iat[k] = True; _close_trade(k, trail_level, "TS"); did_close = True

            if not did_close and cfg.force_exit_on_signal and xLsig.iat[k]:
                exitsL.iat[k] = True; _close_trade(k, cl, "EXIT"); did_close = True

            if not did_close and (k == len(idx) - 1):
                exitsL.iat[k] = True; _close_trade(k, cl, "END")

        elif in_short:
            if lo < min_since_entry:
                min_since_entry = lo
            if cfg.trail_mode == "percent" and cfg.trail_pct:
                trail_level = min(trail_level, min_since_entry * (1 + cfg.trail_pct)) if not np.isnan(trail_level) else min_since_entry * (1 + cfg.trail_pct)
            elif cfg.trail_mode == "atr" and cfg.trail_atr_mult and atr is not None:
                cand = min_since_entry + cfg.trail_atr_mult * float(atr.iat[k])
                trail_level = min(trail_level, cand) if not np.isnan(trail_level) else cand

            did_close = False
            if cfg.prefer_tp_over_sl:
                if (cfg.tp_mode != "none") and (not np.isnan(tp_level)) and (lo <= tp_level):
                    exitsS.iat[k] = True; _close_trade(k, tp_level, "TP"); did_close = True
                if not did_close and (cfg.sl_mode != "none") and (not np.isnan(sl_level)) and (hi >= sl_level):
                    exitsS.iat[k] = True; _close_trade(k, sl_level, "SL"); did_close = True
            else:
                if (cfg.sl_mode != "none") and (not np.isnan(sl_level)) and (hi >= sl_level):
                    exitsS.iat[k] = True; _close_trade(k, sl_level, "SL"); did_close = True
                if not did_close and (cfg.tp_mode != "none") and (not np.isnan(tp_level)) and (lo <= tp_level):
                    exitsS.iat[k] = True; _close_trade(k, tp_level, "TP"); did_close = True

            if not did_close and cfg.trail_mode != "none" and (not np.isnan(trail_level)) and (hi >= trail_level):
                exitsS.iat[k] = True; _close_trade(k, trail_level, "TS"); did_close = True

            if not did_close and cfg.force_exit_on_signal and xSsig.iat[k]:
                exitsS.iat[k] = True; _close_trade(k, cl, "EXIT"); did_close = True

            if not did_close and (k == len(idx) - 1):
                exitsS.iat[k] = True; _close_trade(k, cl, "END")

    trade_pnl = pd.Series(pnls, name="ret")
    trade_dur = pd.Series(durs, name="dur")
    trade_reason = pd.Series(reasons, name="reason")

    if len(trade_pnl) == 0:
        summary = dict(trades=0, expectancy=np.nan, win_rate=np.nan,
                       avg_win=np.nan, avg_loss=np.nan, avg_dur=np.nan,
                       reason_counts={})
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

    return ExitResult(
        exits_long=exitsL,
        exits_short=exitsS,
        tp_level=tp_level_series,
        sl_level=sl_level_series,
        trail_level=trail_level_series,
        trade_pnl=trade_pnl,
        trade_dur=trade_dur,
        trade_reason=trade_reason,
        summary=summary
    )
