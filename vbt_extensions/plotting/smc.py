# vbt_extensions/plotting/smc.py
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from vbt_extensions.ind.smart_money import SMCResult


def plot_smc_overlay(
    fig: Optional[go.Figure],
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    smc: SMCResult,
    *,
    show_pivots: bool = True,
    show_structure_levels: bool = True,
    show_bos: bool = True,
    show_choch: bool = True,
    show_fvgs: bool = True,
    show_obs: bool = True,
    show_regime_bg: bool = True,
    name: str = "BTCUSDT"
) -> go.Figure:
    idx = close.index

    if fig is None:
        fig = go.Figure()

    # Candlestick base
    fig.add_trace(go.Candlestick(
        x=idx, open=open_.values, high=high.values, low=low.values, close=close.values,
        name=name, showlegend=False
    ))

    # Pivots
    if show_pivots:
        ph_idx = idx[smc.pivots_high.values]
        pl_idx = idx[smc.pivots_low.values]
        fig.add_trace(go.Scatter(x=ph_idx, y=high[smc.pivots_high].values, mode='markers',
                                 marker=dict(symbol='triangle-up', size=8),
                                 name='Pivot High'))
        fig.add_trace(go.Scatter(x=pl_idx, y=low[smc.pivots_low].values, mode='markers',
                                 marker=dict(symbol='triangle-down', size=8),
                                 name='Pivot Low'))

    # Estrutura: níveis correntes
    if show_structure_levels:
        fig.add_trace(go.Scatter(x=idx, y=smc.last_hh.values, mode='lines',
                                 line=dict(dash='dot'), name='Last HH'))
        fig.add_trace(go.Scatter(x=idx, y=smc.last_ll.values, mode='lines',
                                 line=dict(dash='dot'), name='Last LL'))

    # Eventos BOS / CHoCH
    def _add_vlines(mask: pd.Series, label: str):
        xs = idx[mask.values]
        for x in xs:
            fig.add_vline(x=x, line_width=1, line_dash="dash", opacity=0.25)
        fig.add_trace(go.Scatter(x=xs, y=[None]*len(xs), mode='markers', name=label, marker=dict(size=1)))

    if show_bos:
        _add_vlines(smc.bos_up, "BOS↑")
        _add_vlines(smc.bos_down, "BOS↓")

    if show_choch:
        _add_vlines(smc.choch_up, "CHoCH↑")
        _add_vlines(smc.choch_down, "CHoCH↓")

    # FVGs
    if show_fvgs:
        for f in smc.fvgs:
            x0 = f.open_ts
            x1 = f.close_ts if f.close_ts is not None else idx[-1]
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=f.low, y1=f.high,
                opacity=0.15,
                line=dict(width=0),
                fillcolor="LightGreen" if f.direction == "bull" else "LightCoral",
            )

    # OBs
    if show_obs:
        for ob in smc.obs:
            x0 = ob.open_ts
            x1 = ob.mitigated_ts if ob.mitigated_ts is not None else idx[-1]
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=ob.low, y1=ob.high,
                opacity=0.18,
                line=dict(width=1),
                fillcolor="PaleTurquoise" if ob.direction == "bull" else "Thistle",
            )

    # Regime background
    if show_regime_bg:
        active = smc.regime
        for i, t in enumerate(idx):
            if active.iat[i] == "long":
                fig.add_vrect(x0=t, x1=t, opacity=0.04, fillcolor="green", line_width=0)
            elif active.iat[i] == "short":
                fig.add_vrect(x0=t, x1=t, opacity=0.04, fillcolor="red", line_width=0)

    fig.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig
