# vbt_extensions/ind/smart_money.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # Se existir teu FVG nativo, usamos; senão caímos no fallback
    from vbt_extensions.ind.fair_value_gap import find_fvg as _find_fvg_native  # type: ignore
except Exception:
    _find_fvg_native = None


@dataclass
class FVG:
    start_idx: int
    end_idx: Optional[int]
    low: float
    high: float
    direction: Literal["bull", "bear"]
    open_ts: pd.Timestamp
    close_ts: Optional[pd.Timestamp]
    expired: bool = False
    touches: int = 0


@dataclass
class OB:
    idx: int
    low: float
    high: float
    direction: Literal["bull", "bear"]
    open_ts: pd.Timestamp
    mitigated_ts: Optional[pd.Timestamp] = None
    expired: bool = False


@dataclass
class SMCResult:
    pivots_high: pd.Series
    pivots_low: pd.Series
    structure: pd.Series  # "HH","HL","LH","LL","neutral"
    bos_up: pd.Series
    bos_down: pd.Series
    choch_up: pd.Series
    choch_down: pd.Series
    shock_strength: pd.Series
    regime: pd.Series  # "inactive","long","short"
    last_hh: pd.Series
    last_ll: pd.Series
    fvgs: List[FVG]
    obs: List[OB]


class SMC:
    """
    Indicador SMC completo: pivôs/estrutura -> BOS/CHoCH -> FVG/OB -> regime (estado).
    Regras:
      - BOS: rompimento com fechamento (ou wick) além do último HH/LL em tendência vigente (continuação). :contentReference[oaicite:3]{index=3}
      - CHoCH: rompimento significativo no sentido oposto (potencial reversão). :contentReference[oaicite:4]{index=4}
      - FVG: padrão ICT de 3 velas (gap entre vela 1 e 3). :contentReference[oaicite:5]{index=5}
      - OB: última vela oposta pré-impulso de rompimento válido. :contentReference[oaicite:6]{index=6}
    """

    @staticmethod
    def _pivots(high: pd.Series, low: pd.Series, left: int, right: int, min_swing_pct: float) -> Tuple[pd.Series, pd.Series]:
        n = len(high)
        ph = np.zeros(n, dtype=bool)
        pl = np.zeros(n, dtype=bool)

        # pivô: high[i] maior que 'left' à esquerda e 'right' à direita; análogo p/ low
        for i in range(left, n - right):
            if all(high[i] > high[i - j - 1] for j in range(left)) and all(high[i] > high[i + j + 1] for j in range(right)):
                # filtra swings muito pequenos vs último pivô relevante
                ph[i] = True
            if all(low[i] < low[i - j - 1] for j in range(left)) and all(low[i] < low[i + j + 1] for j in range(right)):
                pl[i] = True

        # filtro por amplitude mínima entre pivôs consecutivos
        def _filter(series_vals: pd.Series, is_high=True):
            idxs = np.flatnonzero(series_vals)
            keep = []
            last_val = None
            for idx in idxs:
                v = high[idx] if is_high else low[idx]
                if last_val is None:
                    keep.append(idx)
                    last_val = v
                else:
                    pct = abs((v - last_val) / last_val) * 100 if last_val != 0 else 0.0
                    if pct >= min_swing_pct:
                        keep.append(idx)
                        last_val = v
            out = np.zeros(n, dtype=bool)
            out[keep] = True
            return out

        ph = _filter(ph, True)
        pl = _filter(pl, False)
        return pd.Series(ph, index=high.index), pd.Series(pl, index=high.index)

    @staticmethod
    def _structure_from_pivots(high: pd.Series, low: pd.Series, ph: pd.Series, pl: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Retorna rótulo HH/HL/LH/LL/neutral + níveis correntes last_hh / last_ll."""
        idx = high.index
        last_hh = np.full(len(idx), np.nan)
        last_ll = np.full(len(idx), np.nan)
        labels = np.array(["neutral"] * len(idx), dtype=object)

        last_high = None
        last_low = None

        for i, t in enumerate(idx):
            if ph.iat[i]:
                last_high = high.iat[i]
            if pl.iat[i]:
                last_low = low.iat[i]

            if last_high is not None:
                last_hh[i] = last_high
            if last_low is not None:
                last_ll[i] = last_low

            # Classificação básica olhando variação dos últimos pivôs confirmados
            if last_high is None or last_low is None:
                labels[i] = "neutral"
            else:
                # se preço atual acima do último HH e acima do último LL, tende a HL/HH
                close_i = (high.iat[i] + low.iat[i]) / 2.0
                if close_i >= last_hh[i] and close_i >= last_ll[i]:
                    labels[i] = "HH"
                elif close_i >= last_ll[i] and close_i < last_hh[i]:
                    labels[i] = "HL"
                elif close_i < last_ll[i] and close_i <= last_hh[i]:
                    labels[i] = "LL"
                else:
                    labels[i] = "LH"

        return (
            pd.Series(labels, index=idx, dtype="object"),
            pd.Series(last_hh, index=idx),
            pd.Series(last_ll, index=idx),
        )

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    @staticmethod
    def _bos_choch(close: pd.Series,
                   last_hh: pd.Series,
                   last_ll: pd.Series,
                   confirm: Literal["close", "wick"] = "close") -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Detecta BOS (up/down) e CHoCH (up/down) pela comparação com níveis de estrutura."""
        # BOS: rompimento com confirmação
        if confirm == "close":
            bos_up = close > last_hh.shift(1)
            bos_down = close < last_ll.shift(1)
        else:
            # 'wick' – mais permissivo (usa highs/lows via caller se quiser)
            bos_up = close >= last_hh.shift(1)
            bos_down = close <= last_ll.shift(1)

        # CHoCH: inversão — quando ocorre rompimento contrário ao lado “vigente”
        # Aproximação: se houve bos_up após período onde close <= last_ll (pressão de baixa), chama CHoCH_up; simetricamente p/ down.
        choch_up = bos_up & (close.shift(2) <= last_ll.shift(3))
        choch_down = bos_down & (close.shift(2) >= last_hh.shift(3))

        return bos_up.astype(bool), bos_down.astype(bool), choch_up.astype(bool), choch_down.astype(bool)

    @staticmethod
    def _compute_fvg(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                     expire_bars: int = 600, wick_logic: bool = True) -> List[FVG]:
        """Fallback simples de FVG (3 velas ICT). Usa wick_logic padrão; expira por idade."""
        idx = open_.index
        fvgs: List[FVG] = []
        open_ids: List[int] = []

        def _is_bullish_fvg(i: int) -> bool:
            # vela i é a terceira (i>=2): low[i] > high[i-2]
            return low.iat[i] > high.iat[i-2]

        def _is_bearish_fvg(i: int) -> bool:
            # vela i é a terceira: high[i] < low[i-2]
            return high.iat[i] < low.iat[i-2]

        for i in range(2, len(idx)):
            # cria
            if _is_bullish_fvg(i):
                fvgs.append(FVG(start_idx=i-2, end_idx=None,
                                low=high.iat[i-2], high=low.iat[i],
                                direction="bull", open_ts=idx[i-2], close_ts=None))
                open_ids.append(len(fvgs)-1)
            elif _is_bearish_fvg(i):
                fvgs.append(FVG(start_idx=i-2, end_idx=None,
                                low=high.iat[i], high=low.iat[i-2],
                                direction="bear", open_ts=idx[i-2], close_ts=None))
                open_ids.append(len(fvgs)-1)

            # atualiza estado (mitigação/expiração)
            for f_id in list(open_ids):
                f = fvgs[f_id]
                age = i - f.start_idx
                # Mitigação: preço entra na faixa
                if (low.iat[i] <= f.high and high.iat[i] >= f.low):
                    fvgs[f_id].end_idx = i
                    fvgs[f_id].close_ts = idx[i]
                    fvgs[f_id].touches += 1
                    # Mantemos o registro, apenas não marcamos expired
                    open_ids.remove(f_id)
                elif age >= expire_bars:
                    fvgs[f_id].expired = True
                    open_ids.remove(f_id)

        return fvgs

    @staticmethod
    def _order_blocks(close: pd.Series,
                      open_: pd.Series,
                      high: pd.Series,
                      low: pd.Series,
                      bos_up: pd.Series,
                      bos_down: pd.Series,
                      max_age: int = 300) -> List[OB]:
        """OB: última vela oposta imediatamente antes do rompimento válido."""
        idx = close.index
        obs: List[OB] = []
        open_active: List[int] = []

        n = len(idx)
        for i in range(n):
            # abrir OB quando ocorre um rompimento confirmado
            if bos_up.iat[i]:
                # vela oposta anterior (candle bear): open > close
                j = i - 1
                while j >= 0 and not (open_.iat[j] > close.iat[j]):
                    j -= 1
                if j >= 0:
                    ob = OB(idx=j, low=low.iat[j], high=high.iat[j], direction="bull", open_ts=idx[j])
                    obs.append(ob); open_active.append(len(obs)-1)

            if bos_down.iat[i]:
                # vela oposta anterior (candle bull): close > open
                j = i - 1
                while j >= 0 and not (close.iat[j] > open_.iat[j]):
                    j -= 1
                if j >= 0:
                    ob = OB(idx=j, low=low.iat[j], high=high.iat[j], direction="bear", open_ts=idx[j])
                    obs.append(ob); open_active.append(len(obs)-1)

            # atualizar mitigação/expiração
            to_remove = []
            for k in open_active:
                ob = obs[k]
                age = i - ob.idx
                # toque/mitigação
                if (low.iat[i] <= ob.high and high.iat[i] >= ob.low):
                    obs[k].mitigated_ts = idx[i]
                    to_remove.append(k)
                elif age >= max_age:
                    obs[k].expired = True
                    to_remove.append(k)
            open_active = [k for k in open_active if k not in to_remove]

        return obs

    @classmethod
    def run(cls,
            close: pd.Series,
            high: pd.Series,
            low: pd.Series,
            open_: Optional[pd.Series] = None,
            *,
            pivot_left: int = 2,
            pivot_right: int = 2,
            min_swing_pct: float = 0.2,  # em %
            bos_confirm: Literal["close", "wick"] = "close",
            choch_confirm: Literal["close", "wick"] = "close",
            atr_window: int = 14,
            min_displacement_atr: float = 0.5,
            fvg_expire_bars: int = 600,
            fvg_wick_logic: bool = True,
            ob_max_age: int = 300,
            regime_grace_bars: int = 3
            ) -> SMCResult:

        if open_ is None:
            # se abrir sem open, usa proxy (pouco ideal, mas quebra-galho)
            open_ = close.shift(1).fillna(close)

        # 1) Pivôs
        pivots_high, pivots_low = cls._pivots(high, low, pivot_left, pivot_right, min_swing_pct)

        # 2) Estrutura + níveis
        structure, last_hh, last_ll = cls._structure_from_pivots(high, low, pivots_high, pivots_low)

        # 3) BOS / CHoCH
        bos_up, bos_down, choch_up, choch_down = cls._bos_choch(close, last_hh, last_ll, bos_confirm)

        # 4) Choque (força) via ATR
        atr = cls._atr(high, low, close, atr_window)
        displacement = (close - last_hh.shift(1)).where(bos_up, (last_ll.shift(1) - close).where(bos_down, 0.0))
        shock_strength = (displacement.abs() / (atr.replace(0, np.nan))).fillna(0.0)

        # 5) FVG (usa nativo se existir)
        if _find_fvg_native is not None:
            try:
                fvgs = _find_fvg_native(open_=open_, high=high, low=low, close=close,
                                        expire_bars=fvg_expire_bars, wick_logic=fvg_wick_logic)
            except Exception:
                fvgs = cls._compute_fvg(open_, high, low, close, fvg_expire_bars, fvg_wick_logic)
        else:
            fvgs = cls._compute_fvg(open_, high, low, close, fvg_expire_bars, fvg_wick_logic)

        # 6) Order Blocks
        obs = cls._order_blocks(close, open_, high, low, bos_up, bos_down, ob_max_age)

        # 7) Regime (estado) — ativa após CHoCH com choque mínimo por algumas barras
        idx = close.index
        regime_vals = np.array(["inactive"] * len(idx), dtype=object)
        arming = 0
        active_side: Optional[str] = None

        for i in range(len(idx)):
            if choch_up.iat[i] and shock_strength.iat[i] >= min_displacement_atr:
                active_side = "long"; arming = regime_grace_bars
            elif choch_down.iat[i] and shock_strength.iat[i] >= min_displacement_atr:
                active_side = "short"; arming = regime_grace_bars

            if arming > 0:
                regime_vals[i] = active_side or "inactive"
                arming -= 1
            else:
                # se ocorrer BOS contrário forte, desarma
                if active_side == "long" and bos_down.iat[i]:
                    active_side = None
                elif active_side == "short" and bos_up.iat[i]:
                    active_side = None
                regime_vals[i] = active_side or "inactive"

        regime = pd.Series(regime_vals, index=idx, dtype="object")

        return SMCResult(
            pivots_high=pivots_high,
            pivots_low=pivots_low,
            structure=structure,
            bos_up=bos_up,
            bos_down=bos_down,
            choch_up=choch_up,
            choch_down=choch_down,
            shock_strength=shock_strength,
            regime=regime,
            last_hh=last_hh,
            last_ll=last_ll,
            fvgs=fvgs,
            obs=obs
        )
