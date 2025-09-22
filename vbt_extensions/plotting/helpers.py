# vbt_extensions/plotting/helpers.py
from typing import Optional

import pandas as pd


def ensure_series(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    raise TypeError(f"esperado pd.Series para {name or 'obj'}")


def plot_price(
    close: pd.Series,
    *,
    title: Optional[str] = None,
    fig=None,
):
    """Plota o preço base e retorna a figura (plotly Figure) do vectorbt."""
    close = ensure_series(close, "close").astype(float)
    fig = close.vbt.plot(fig=fig)
    if title is not None:
        fig.update_layout(title=title)
    return fig


def plot_entries_exits(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    *,
    fig=None,
    entry_marker_size: int = 6,
    exit_marker_size: int = 6,
):
    """Marca entradas (triângulos para cima) e saídas (triângulos para baixo)."""
    close = ensure_series(close, "close").astype(float)
    entries = ensure_series(entries, "entries").reindex_like(close).fillna(False)
    exits = ensure_series(exits, "exits").reindex_like(close).fillna(False)

    fig = close.vbt.plot(fig=fig)

    # entradas
    idx_e = entries[entries].index
    if len(idx_e) > 0:
        fig.add_scatter(
            x=idx_e,
            y=close.loc[idx_e],
            mode="markers",
            marker_symbol="triangle-up",
            marker_size=entry_marker_size,
            name="Entry",
            hovertemplate="%{x}<br>Entry: %{y}<extra></extra>",
        )

    # saídas
    idx_x = exits[exits].index
    if len(idx_x) > 0:
        fig.add_scatter(
            x=idx_x,
            y=close.loc[idx_x],
            mode="markers",
            marker_symbol="triangle-down",
            marker_size=exit_marker_size,
            name="Exit",
            hovertemplate="%{x}<br>Exit: %{y}<extra></extra>",
        )
    return fig


def plot_pivots(
    close: pd.Series,
    *,
    is_top: Optional[pd.Series] = None,
    is_bottom: Optional[pd.Series] = None,
    swing_high: Optional[pd.Series] = None,
    swing_low: Optional[pd.Series] = None,
    fig=None,
    marker_size: int = 6,
):
    """Plota pivôs (top/bottom) como marcadores e swings como linhas."""
    close = ensure_series(close, "close").astype(float)
    fig = close.vbt.plot(fig=fig)

    if swing_high is not None:
        sh = ensure_series(swing_high, "swing_high").reindex_like(close)
        fig = sh.vbt.plot(fig=fig, trace_kwargs=dict(name="Swing High"))

    if swing_low is not None:
        sl = ensure_series(swing_low, "swing_low").reindex_like(close)
        fig = sl.vbt.plot(fig=fig, trace_kwargs=dict(name="Swing Low"))

    if is_top is not None:
        it = ensure_series(is_top, "is_top").reindex_like(close).fillna(False)
        idx_t = it[it].index
        if len(idx_t) > 0:
            fig.add_scatter(
                x=idx_t,
                y=close.loc[idx_t],
                mode="markers",
                marker_symbol="x",
                marker_size=marker_size,
                name="Top",
                hovertemplate="%{x}<br>Top: %{y}<extra></extra>",
            )

    if is_bottom is not None:
        ib = ensure_series(is_bottom, "is_bottom").reindex_like(close).fillna(False)
        idx_b = ib[ib].index
        if len(idx_b) > 0:
            fig.add_scatter(
                x=idx_b,
                y=close.loc[idx_b],
                mode="markers",
                marker_symbol="circle",
                marker_size=marker_size,
                name="Bottom",
                hovertemplate="%{x}<br>Bottom: %{y}<extra></extra>",
            )
    return fig


def plot_sr_levels(
    close: pd.Series,
    *,
    support_levels: Optional[pd.Series] = None,
    resistance_levels: Optional[pd.Series] = None,
    touches_support: Optional[pd.Series] = None,
    touches_resist: Optional[pd.Series] = None,
    fig=None,
    touch_marker_size: int = 6,
):
    """Plota suportes/resistências (linhas) e marca toques (pontos)."""
    close = ensure_series(close, "close").astype(float)
    fig = close.vbt.plot(fig=fig)

    if support_levels is not None:
        sup = ensure_series(support_levels, "support_levels").reindex_like(close)
        fig = sup.vbt.plot(fig=fig, trace_kwargs=dict(name="Support"))

    if resistance_levels is not None:
        res = ensure_series(resistance_levels, "resistance_levels").reindex_like(close)
        fig = res.vbt.plot(fig=fig, trace_kwargs=dict(name="Resistance"))

    if touches_support is not None:
        ts = ensure_series(touches_support, "touches_support").reindex_like(close).fillna(False)
        idx_ts = ts[ts].index
        if len(idx_ts) > 0:
            fig.add_scatter(
                x=idx_ts,
                y=close.loc[idx_ts],
                mode="markers",
                marker_symbol="square",
                marker_size=touch_marker_size,
                name="Touch Support",
                hovertemplate="%{x}<br>Touch S: %{y}<extra></extra>",
            )

    if touches_resist is not None:
        tr = ensure_series(touches_resist, "touches_resist").reindex_like(close).fillna(False)
        idx_tr = tr[tr].index
        if len(idx_tr) > 0:
            fig.add_scatter(
                x=idx_tr,
                y=close.loc[idx_tr],
                mode="markers",
                marker_symbol="diamond",
                marker_size=touch_marker_size,
                name="Touch Resistance",
                hovertemplate="%{x}<br>Touch R: %{y}<extra></extra>",
            )
    return fig


def plot_band(
    upper: pd.Series,
    lower: pd.Series,
    *,
    fig=None,
    name: str = "Band",
    showlegend: bool = True,
):
    """Plota uma banda (faixa) área entre duas séries (upper/lower)."""
    upper = ensure_series(upper, "upper").astype(float)
    lower = ensure_series(lower, "lower").reindex_like(upper).astype(float)

    # Desenha duas traces e preenche a área
    fig = upper.vbt.plot(fig=fig, trace_kwargs=dict(name=f"{name} Upper", showlegend=showlegend))
    fig.add_scatter(
        x=lower.index,
        y=lower.values,
        mode="lines",
        name=f"{name} Lower",
        showlegend=showlegend,
        fill="tonexty",
        hovertemplate="%{x}<br>Lower: %{y}<extra></extra>",
    )
    return fig


def plot_indicator_tearsheet(
    close: pd.Series,
    *,
    # pivôs
    is_top: Optional[pd.Series] = None,
    is_bottom: Optional[pd.Series] = None,
    swing_high: Optional[pd.Series] = None,
    swing_low: Optional[pd.Series] = None,
    # S/R
    support_levels: Optional[pd.Series] = None,
    resistance_levels: Optional[pd.Series] = None,
    touches_support: Optional[pd.Series] = None,
    touches_resist: Optional[pd.Series] = None,
    # sinais
    entries: Optional[pd.Series] = None,
    exits: Optional[pd.Series] = None,
    title: Optional[str] = None,
):
    """Compositor: monta um gráfico completo com preço, pivôs, S/R e entradas/saídas."""
    fig = plot_price(close, title=title)

    # pivôs / swings
    if any(x is not None for x in (is_top, is_bottom, swing_high, swing_low)):
        fig = plot_pivots(
            close,
            is_top=is_top,
            is_bottom=is_bottom,
            swing_high=swing_high,
            swing_low=swing_low,
            fig=fig,
        )

    # suportes/resistências
    if any(x is not None for x in (support_levels, resistance_levels, touches_support, touches_resist)):
        fig = plot_sr_levels(
            close,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            touches_support=touches_support,
            touches_resist=touches_resist,
            fig=fig,
        )

    # entradas/saídas
    if entries is not None or exits is not None:
        fig = plot_entries_exits(
            close,
            entries=entries if entries is not None else pd.Series(False, index=close.index),
            exits=exits if exits is not None else pd.Series(False, index=close.index),
            fig=fig,
        )
    return fig


def plot_trend_lines(
    close: pd.Series,
    *,
    support_level: pd.Series,
    resist_level: pd.Series,
    touch_support: pd.Series | None = None,
    touch_resist: pd.Series | None = None,
    break_up: pd.Series | None = None,
    break_down: pd.Series | None = None,
    title: str | None = "Trend Lines",
    fig=None,
):
    """Overlay de linhas de tendência (suporte/resistência) com marcadores opcionais."""
    close = ensure_series(close, "close").astype(float)
    sup = ensure_series(support_level, "support_level").reindex_like(close)
    res = ensure_series(resist_level, "resist_level").reindex_like(close)

    fig = plot_price(close, title=title, fig=fig)
    fig = sup.vbt.plot(fig=fig, trace_kwargs=dict(name="Support TL"))
    fig = res.vbt.plot(fig=fig, trace_kwargs=dict(name="Resistance TL"))

    if touch_support is not None:
        ts = ensure_series(touch_support, "touch_support").reindex_like(close).fillna(False)
        idx_ts = ts[ts].index
        if len(idx_ts) > 0:
            fig.add_scatter(
                x=idx_ts,
                y=close.loc[idx_ts],
                mode="markers",
                marker_symbol="square",
                name="Touch Support",
            )
    if touch_resist is not None:
        tr = ensure_series(touch_resist, "touch_resist").reindex_like(close).fillna(False)
        idx_tr = tr[tr].index
        if len(idx_tr) > 0:
            fig.add_scatter(
                x=idx_tr,
                y=close.loc[idx_tr],
                mode="markers",
                marker_symbol="diamond",
                name="Touch Resistance",
            )
    if break_up is not None:
        bu = ensure_series(break_up, "break_up").reindex_like(close).fillna(False)
        idx_bu = bu[bu].index
        if len(idx_bu) > 0:
            fig.add_scatter(
                x=idx_bu,
                y=close.loc[idx_bu],
                mode="markers",
                marker_symbol="triangle-up",
                name="Break Up",
            )
    if break_down is not None:
        bd = ensure_series(break_down, "break_down").reindex_like(close).fillna(False)
        idx_bd = bd[bd].index
        if len(idx_bd) > 0:
            fig.add_scatter(
                x=idx_bd,
                y=close.loc[idx_bd],
                mode="markers",
                marker_symbol="triangle-down",
                name="Break Down",
            )
    return fig


def plot_regime_background(
    close: pd.Series,
    regime: pd.Series,
    *,
    colors: dict | None = None,
    opacity: float = 0.10,
    title: str | None = None,
    fig=None,
):
    """
    Colore o fundo por regime categórico ('up'/'down'/'flat') ao longo do tempo.

    Params
    ------
    close : Série de preços (para base do gráfico)
    regime: Série categórica alinhada (valores como 'up', 'down', 'flat')
    colors: dict opcional, ex.: {'up':'green','down':'red','flat':'gray'}
    opacity: 0..1 para transparência das faixas
    title  : Título do gráfico
    fig    : Figure existente (opcional)

    Retorna
    -------
    plotly.graph_objects.Figure
    """
    import numpy as np

    close = ensure_series(close, "close").astype(float)
    regime = ensure_series(regime, "regime").reindex_like(close)

    # defaults de cores
    colors = colors or {"up": "green", "down": "red", "flat": "gray"}

    # plota o preço base
    fig = plot_price(close, title=title, fig=fig)

    # achamos "blocos" contínuos de mesmo regime
    reg_vals = regime.fillna("flat").astype(str).values
    idx = close.index

    if len(idx) == 0:
        return fig

    start = 0
    for i in range(1, len(reg_vals) + 1):
        end_block = (i == len(reg_vals)) or (reg_vals[i] != reg_vals[i - 1])
        if end_block:
            reg = reg_vals[i - 1]
            color = colors.get(reg, "gray")
            x0 = idx[start]
            x1 = idx[i - 1]
            # adiciona faixa vertical
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=color,
                opacity=opacity,
                layer="below",
                line_width=0,
            )
            start = i

    return fig


# from vbt_extensions.ind.trend_line import TREND_LINE
# from vbt_extensions.sig.trend_line_sig import classify_trend_regime
# from vbt_extensions.plotting.helpers import plot_regime_background

# # 1) calcula trend lines
# res = TREND_LINE.run(
#     high=df['High'], low=df['Low'], close=df['Close'],
#     lookback=30, touch_tol=0.001
# )

# # 2) classifica regimes a partir dos slopes
# reg = classify_trend_regime(
#     res.support_slope, res.resist_slope,
#     up_thresh=0.0001, down_thresh=0.0001
# )

# # 3) plota com fundo colorido por regime
# fig = plot_regime_background(
#     df['Close'],
#     reg,
#     colors={'up':'#1a9850','down':'#d73027','flat':'#aaaaaa'},
#     opacity=0.12,
#     title="Close + Regime (Trend Line)"
# )
# # opcional: sobrepor as linhas de tendência
# fig = res.support_level.vbt.plot(fig=fig, trace_kwargs=dict(name="Support TL"))
# fig = res.resist_level.vbt.plot(fig=fig, trace_kwargs=dict(name="Resistance TL"))
# fig.show()


def _series_pair_to_band(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Gera (upper, lower) entre duas séries quaisquer, ponto a ponto."""
    a = ensure_series(a).astype(float)
    b = ensure_series(b).reindex_like(a).astype(float)
    upper = pd.concat([a, b], axis=1).max(axis=1)
    lower = pd.concat([a, b], axis=1).min(axis=1)
    return upper, lower


def plot_fib_zones(
    close: pd.Series,
    fib_levels: "pd.DataFrame",
    *,
    bands: list[tuple[str, str]] = (("fib_61.8", "fib_38.2"), ("fib_38.2", "fib_23.6")),
    colors: dict[str, str] | None = None,
    opacity: float = 0.15,
    show_level_lines: bool = True,
    title: str | None = "Fibonacci Zones",
    fig=None,
):
    """
    Sombreia faixas entre pares de níveis de Fibonacci ao longo do tempo.

    Params
    ------
    close : Série de preços (base do gráfico)
    fib_levels : DataFrame com colunas tipo 'fib_23.6', 'fib_38.2', 'fib_50', 'fib_61.8', 'fib_100', etc.
    bands : lista de tuplas (nivel_a, nivel_b) definindo as faixas a sombrear
    colors : opcional, mapeia o nome "nivel_a-nivel_b" para cor; ex.: {'fib_61.8-fib_38.2': '#66bd63'}
    opacity : transparência das faixas
    show_level_lines : se True, também plota as linhas dos níveis usados
    title : título do gráfico
    fig : Figure existente (opcional)

    Retorna
    -------
    plotly Figure
    """
    import re

    close = ensure_series(close, "close").astype(float)
    fig = plot_price(close, title=title, fig=fig)

    colors = colors or {}
    used_level_names: set[str] = set()

    # helper pra nome amigável da faixa
    def band_name(a: str, b: str) -> str:
        # normaliza 61.8-38.2 independentemente da ordem
        def pct(x: str) -> float:
            m = re.search(r"([\d\.]+)", x)
            return float(m.group(1)) if m else 9999.0

        aa, bb = (a, b) if pct(a) >= pct(b) else (b, a)
        return f"{aa}-{bb}"

    for a, b in bands:
        if a not in fib_levels.columns or b not in fib_levels.columns:
            continue
        upper, lower = _series_pair_to_band(fib_levels[a], fib_levels[b])
        nm = band_name(a, b)
        color = colors.get(nm, None)

        # desenha banda (sem cor fixa caso não fornecida → usa default do plotly)
        fig = plot_band(upper, lower, fig=fig, name=nm, showlegend=True)
        if color:
            # ajusta última duas traces (upper & lower) pra aplicar fillcolor
            # (upper já foi adicionado por plot_band como linha; lower com fill='tonexty')
            # pegamos os dois últimos traces:
            tr_upper = fig.data[-2]
            tr_lower = fig.data[-1]
            tr_upper.update(line=dict(color=color))
            tr_lower.update(line=dict(color=color), fillcolor=color, opacity=opacity)

        used_level_names.update([a, b])

    if show_level_lines:
        for lvl_name in sorted(used_level_names, key=lambda c: float(c.split("_")[1])):
            if lvl_name in fib_levels.columns:
                fig = fib_levels[lvl_name].vbt.plot(fig=fig, trace_kwargs=dict(name=lvl_name))

    return fig


# from vbt_extensions.ind.retracement_ratios import FIB_RETRACEMENT
# from vbt_extensions.plotting.helpers import plot_fib_zones

# # Após calcular 'fib' com FIB_RETRACEMENT.run(...)
# bands = [("fib_61.8", "fib_38.2"), ("fib_38.2", "fib_23.6")]  # você escolhe
# colors = {
#     "fib_61.8-fib_38.2": "#66bd63",
#     "fib_38.2-fib_23.6": "#fdae61",
# }

# fig = plot_fib_zones(
#     df["Close"],
#     fib.fib_levels,
#     bands=bands,
#     colors=colors,
#     opacity=0.18,
#     show_level_lines=True,
#     title="Fibonacci Zones (38.2–61.8, 23.6–38.2)"
# )
# fig.show()
