# vbt_extensions/sig/smart_money_sig.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from vbt_extensions.ind.smart_money import SMC, SMCResult


@dataclass
class SmartMoneySignalConfig:
    side: Literal["both", "long", "short"] = "both"
    wait_pullback_to_ob: bool = True
    enter_on_fvg_fill: bool = False
    risk_rr: float = 2.0  # take-profit = 2R por padrão
    sl_padding_pct: float = 0.05  # 0.05% além do OB
    min_shock_strength: float = 0.5
    regime_only_after_choch: bool = True


class SmartMoneySignal:
    """
    Gera entries/exits a partir do indicador SMC:
      - Ativa direção após CHoCH com shock_strength mínimo (regime).
      - Entrada preferida: pullback na OB do rompimento; opcional: fill de FVG.
      - SL: fora da OB (padding). TP: múltiplo de R (risk_rr).
    """

    def __init__(self, cfg: Optional[SmartMoneySignalConfig] = None):
        self.cfg = cfg or SmartMoneySignalConfig()

    def generate(self,
                 open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                 smc: Optional[SMCResult] = None) -> dict:
        if smc is None:
            smc = SMC.run(close=close, high=high, low=low, open_=open_)

        idx = close.index
        entries = pd.Series(False, index=idx)
        exits = pd.Series(False, index=idx)
        sides = pd.Series("", index=idx)  # "long"|"short"|""

        active_side: Optional[str] = None
        current_sl = None
        current_tp = None
        in_position = False

        # index rápidos por tempo p/ OB e FVG
        ob_by_ts = {ob.open_ts: ob for ob in smc.obs}

        for i, t in enumerate(idx):
            regime = smc.regime.iat[i]

            # ativa direção se pedido "apenas após CHoCH"
            if self.cfg.regime_only_after_choch:
                if regime not in ("long", "short"):
                    # sem direção -> sem entrada
                    pass
                else:
                    active_side = regime
            else:
                # usa BOS como proxy de direção, se preferir (não usado por padrão)
                if smc.bos_up.iat[i]:
                    active_side = "long"
                elif smc.bos_down.iat[i]:
                    active_side = "short"

            if in_position:
                # saída “básica” por SL/TP
                if active_side == "long":
                    if low.iat[i] <= (current_sl or -1e12):
                        exits.iat[i] = True; in_position = False; active_side = None
                    elif high.iat[i] >= (current_tp or 1e12):
                        exits.iat[i] = True; in_position = False; active_side = None
                elif active_side == "short":
                    if high.iat[i] >= (current_sl or 1e12):
                        exits.iat[i] = True; in_position = False; active_side = None
                    elif low.iat[i] <= (current_tp or -1e12):
                        exits.iat[i] = True; in_position = False; active_side = None
                continue

            # entradas (não estamos em posição)
            if active_side and (self.cfg.side in ("both", active_side)):
                # pullback na OB mais recente do lado ativo
                if self.cfg.wait_pullback_to_ob:
                    # encontre OB válida mais próxima (não mitigada/expirada)
                    candidate = None
                    for ob in reversed(smc.obs):
                        if ob.direction == ("bull" if active_side == "long" else "bear") and not ob.expired:
                            candidate = ob
                            break
                    if candidate:
                        # entrou na zona?
                        in_zone = (low.iat[i] <= candidate.high and high.iat[i] >= candidate.low)
                        if in_zone:
                            entries.iat[i] = True
                            sides.iat[i] = active_side
                            in_position = True
                            # SL/TP básicos
                            if active_side == "long":
                                sl = candidate.low * (1 - self.cfg.sl_padding_pct / 100)
                                r = close.iat[i] - sl
                                tp = close.iat[i] + self.cfg.risk_rr * r
                                current_sl, current_tp = sl, tp
                            else:
                                sl = candidate.high * (1 + self.cfg.sl_padding_pct / 100)
                                r = sl - close.iat[i]
                                tp = close.iat[i] - self.cfg.risk_rr * r
                                current_sl, current_tp = sl, tp

                # alternativa: entrada no "fill" de FVG
                elif self.cfg.enter_on_fvg_fill and smc.fvgs:
                    for f in reversed(smc.fvgs):
                        if f.expired or (f.end_idx is not None):
                            continue
                        # se preço entrou na faixa, “fill”
                        in_gap = (low.iat[i] <= f.high and high.iat[i] >= f.low)
                        if in_gap and ((active_side == "long" and f.direction == "bull") or
                                       (active_side == "short" and f.direction == "bear")):
                            entries.iat[i] = True
                            sides.iat[i] = active_side
                            in_position = True
                            # SL/TP: extremos da faixa
                            if active_side == "long":
                                sl = f.low * (1 - self.cfg.sl_padding_pct / 100)
                                r = close.iat[i] - sl
                                tp = close.iat[i] + self.cfg.risk_rr * r
                                current_sl, current_tp = sl, tp
                            else:
                                sl = f.high * (1 + self.cfg.sl_padding_pct / 100)
                                r = sl - close.iat[i]
                                tp = close.iat[i] - self.cfg.risk_rr * r
                                current_sl, current_tp = sl, tp
                            break

        return dict(entries=entries, exits=exits, sides=sides, sl=pd.Series(current_sl, index=idx), tp=pd.Series(current_tp, index=idx))
