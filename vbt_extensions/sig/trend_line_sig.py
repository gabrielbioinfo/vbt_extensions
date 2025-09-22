import pandas as pd


def tl_breakout_sig(
    close: pd.Series,
    *,
    resist_level: pd.Series,
    support_level: pd.Series,
    confirm_shift: int = 1,
):
    """Sinais de breakout: compra no rompimento de resistência, vende no rompimento de suporte."""
    close = close.astype(float)
    r = resist_level.reindex_like(close)
    s = support_level.reindex_like(close)

    cross_up = (close > r) & (close.shift(1) <= r.shift(1)) & r.notna()
    cross_dn = (close < s) & (close.shift(1) >= s.shift(1)) & s.notna()

    entries_long = cross_up.shift(confirm_shift).fillna(False).astype(bool)
    exits_long = cross_dn.shift(confirm_shift).fillna(False).astype(bool)

    # Para shorts, o oposto:
    entries_short = cross_dn.shift(confirm_shift).fillna(False).astype(bool)
    exits_short = cross_up.shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


def tl_touch_reversal_sig(
    close: pd.Series,
    *,
    high: pd.Series,
    low: pd.Series,
    resist_level: pd.Series,
    support_level: pd.Series,
    tol: float = 0.0,
    confirm_shift: int = 1,
):
    """Sinais de reversão em toques:
    - Long quando tocar suporte (low <= suporte + tol*|suporte|) e fechar acima do suporte.
    - Short quando tocar resistência (high >= resistência - tol*|resist|) e fechar abaixo da resistência.
    """
    close = close.astype(float)
    high = high.reindex_like(close).astype(float)
    low = low.reindex_like(close).astype(float)
    r = resist_level.reindex_like(close).astype(float)
    s = support_level.reindex_like(close).astype(float)

    tol_r = (r.abs() * tol).fillna(0.0)
    tol_s = (s.abs() * tol).fillna(0.0)

    touch_sup = (low <= (s + tol_s)) & s.notna()
    touch_res = (high >= (r - tol_r)) & r.notna()

    # Confirmação simples: candle fecha do "lado certo" após o toque
    long_conf = close > s
    short_conf = close < r

    entries_long = (touch_sup & long_conf).shift(confirm_shift).fillna(False).astype(bool)
    exits_long = (touch_res & short_conf).shift(confirm_shift).fillna(False).astype(bool)

    entries_short = (touch_res & short_conf).shift(confirm_shift).fillna(False).astype(bool)
    exits_short = (touch_sup & long_conf).shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


def tl_trailing_channel_sig(
    close: pd.Series,
    *,
    resist_level: pd.Series,
    support_level: pd.Series,
    band_pad: float = 0.0,
    confirm_shift: int = 1,
):
    """Modo 'canal': entra long quando entrar na zona acima do suporte e sai se perder o suporte;
    entra short quando entrar na zona abaixo da resistência e sai se romper a resistência.
    band_pad cria um colchão: suporte*(1+pad) / resistência*(1-pad)."""
    close = close.astype(float)
    r = resist_level.reindex_like(close).astype(float)
    s = support_level.reindex_like(close).astype(float)

    sup_band = s * (1.0 + band_pad)
    res_band = r * (1.0 - band_pad)

    in_long_zone = close > sup_band
    exit_long = close < s
    in_short_zone = close < res_band
    exit_short = close > r

    entries_long = (
        (in_long_zone & ~in_long_zone.shift(1).fillna(False)).shift(confirm_shift).fillna(False).astype(bool)
    )
    entries_short = (
        (in_short_zone & ~in_short_zone.shift(1).fillna(False))
        .shift(confirm_shift)
        .fillna(False)
        .astype(bool)
    )
    exits_long = exit_long.shift(confirm_shift).fillna(False).astype(bool)
    exits_short = exit_short.shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


def tl_side_aware_sig(
    close: pd.Series,
    *,
    resist_level: pd.Series,
    support_level: pd.Series,
    support_slope: pd.Series,
    resist_slope: pd.Series,
    confirm_shift: int = 1,
    slope_thresh: float = 0.0,
):
    """Sinais de breakout respeitando o 'lado' da tendência.

    Regras:
    - Só entra comprado se houver breakout de resistência E suporte_slope > slope_thresh
    - Só entra vendido se houver breakout de suporte E resist_slope < -slope_thresh
    """
    close = close.astype(float)
    r = resist_level.reindex_like(close)
    s = support_level.reindex_like(close)
    sup_slope = support_slope.reindex_like(close).astype(float)
    res_slope = resist_slope.reindex_like(close).astype(float)

    cross_up = (close > r) & (close.shift(1) <= r.shift(1)) & r.notna()
    cross_dn = (close < s) & (close.shift(1) >= s.shift(1)) & s.notna()

    entries_long = (cross_up & (sup_slope > slope_thresh)).shift(confirm_shift).fillna(False).astype(bool)
    exits_long = (cross_dn).shift(confirm_shift).fillna(False).astype(bool)

    entries_short = (cross_dn & (res_slope < -slope_thresh)).shift(confirm_shift).fillna(False).astype(bool)
    exits_short = (cross_up).shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


def classify_trend_regime(
    support_slope: pd.Series,
    resist_slope: pd.Series,
    *,
    up_thresh: float = 0.0,
    down_thresh: float = 0.0,
) -> pd.Series:
    """Classifica regime por slopes: 'up', 'down', 'flat'.

    - 'up' se suporte em alta OU resistência em alta (sup_slope > up_thresh)
    - 'down' se resistência em baixa OU suporte em baixa (res_slope < -down_thresh)
    - 'flat' caso contrário
    """
    sup = support_slope.astype(float)
    res = resist_slope.astype(float)

    cond_up = sup > up_thresh
    cond_down = res < -down_thresh

    regime = pd.Series("flat", index=sup.index)
    regime[cond_up] = "up"
    regime[cond_down] = "down"
    return regime


# from vbt_extensions.ind.trend_line import TREND_LINE
# from vbt_extensions.sig.trend_line_sig import tl_side_aware_sig, classify_trend_regime

# res = TREND_LINE.run(
#     high=df['High'], low=df['Low'], close=df['Close'],
#     lookback=30, touch_tol=0.001
# )

# # sinais respeitando slope
# eL, xL, eS, xS = tl_side_aware_sig(
#     df['Close'],
#     resist_level=res.resist_level,
#     support_level=res.support_level,
#     support_slope=res.support_slope,
#     resist_slope=res.resist_slope,
#     slope_thresh=0.0001,  # exige inclinação mínima
# )

# # classificar regimes
# regimes = classify_trend_regime(
#     res.support_slope,
#     res.resist_slope,
#     up_thresh=0.0001,
#     down_thresh=0.0001,
# )

# print(regimes.value_counts())


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    r = returns.fillna(0.0).astype(float)
    return r.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def classify_trend_regime_adv(
    *,
    support_slope: pd.Series,
    resist_slope: pd.Series,
    close: pd.Series,
    vol_window: int = 20,
    up_thresh: float = 0.0,
    down_thresh: float = 0.0,
    vol_thresh: float | None = None,  # se None, usa percentil (ver abaixo)
    vol_percentile: float = 0.7,  # 70% como default p/ definir “alto”
    smooth_span: int = 5,  # suaviza slopes para evitar zigue-zague
    min_persist: int = 3,  # exige persistência mínima do rótulo
) -> pd.Series:
    """
    Classificação de regime multi-estado, combinando slope & volatilidade:

    Labels possíveis:
      - 'volatile_up'   (slope de alta + vol alta)
      - 'calm_up'       (slope de alta + vol baixa)
      - 'volatile_down' (slope de baixa + vol alta)
      - 'calm_down'     (slope de baixa + vol baixa)
      - 'range'         (demais casos / lateral)

    Parâmetros:
      up_thresh / down_thresh: limites para considerar slope relevante.
      vol_window: janela da vol (std dos retornos).
      vol_thresh: se None, usa quantil 'vol_percentile' da vol histórica como limite.
      smooth_span: EMA nos slopes antes da classificação.
      min_persist: número mínimo de barras para manter o rótulo (histerese).
    """
    sup = support_slope.astype(float).reindex_like(close)
    res = resist_slope.astype(float).reindex_like(close)
    sup_s = _ema(sup, smooth_span)
    res_s = _ema(res, smooth_span)

    # proxy de direção: suporte em alta favorece “up”; resistência em queda favorece “down”
    is_up = sup_s > up_thresh
    is_down = res_s < -down_thresh

    # volatilidade
    ret = close.astype(float).pct_change()
    vol = _rolling_vol(ret, vol_window)
    if vol_thresh is None:
        vt = vol.quantile(vol_percentile)
    else:
        vt = float(vol_thresh)
    vol_high = vol > vt

    regime = pd.Series("range", index=close.index, dtype=object)
    regime[is_up & vol_high] = "volatile_up"
    regime[is_up & ~vol_high] = "calm_up"
    regime[is_down & vol_high] = "volatile_down"
    regime[is_down & ~vol_high] = "calm_down"

    # persistência mínima (suaviza rótulos)
    if min_persist > 1 and len(regime) > 0:
        vals = regime.values.copy()
        run_label = vals[0]
        run_len = 1
        for i in range(1, len(vals)):
            if vals[i] == run_label:
                run_len += 1
            else:
                # se o novo rótulo durou menos que min_persist, força manter o anterior
                # retroativamente
                j = i
                while j < len(vals) and vals[j] != run_label and (j - i + 1) < min_persist:
                    j += 1
                if (j - i + 1) < min_persist:
                    vals[i:j] = run_label
                    run_len += j - i
                else:
                    run_label = vals[i]
                    run_len = 1
        regime = pd.Series(vals, index=regime.index, dtype=object)

    return regime


def tl_breakout_with_regime(
    close: pd.Series,
    *,
    resist_level: pd.Series,
    support_level: pd.Series,
    regime: pd.Series,
    valid_long: tuple[str, ...] = ("calm_up", "volatile_up"),
    valid_short: tuple[str, ...] = ("calm_down", "volatile_down"),
    confirm_shift: int = 1,
):
    close = close.astype(float)
    r = resist_level.reindex_like(close)
    s = support_level.reindex_like(close)
    regime = regime.reindex_like(close).astype(str)

    cross_up = (close > r) & (close.shift(1) <= r.shift(1)) & r.notna()
    cross_dn = (close < s) & (close.shift(1) >= s.shift(1)) & s.notna()

    entries_long = (cross_up & regime.isin(valid_long)).shift(confirm_shift).fillna(False).astype(bool)
    exits_long = (cross_dn).shift(confirm_shift).fillna(False).astype(bool)

    entries_short = (cross_dn & regime.isin(valid_short)).shift(confirm_shift).fillna(False).astype(bool)
    exits_short = (cross_up).shift(confirm_shift).fillna(False).astype(bool)

    return entries_long, exits_long, entries_short, exits_short


# from vbt_extensions.sig.trend_line_sig import classify_trend_regime_adv

# reg_adv = classify_trend_regime_adv(
#     support_slope=res.support_slope,
#     resist_slope=res.resist_slope,
#     close=df["Close"],
#     vol_window=20,
#     up_thresh=0.0001,
#     down_thresh=0.0001,
#     vol_percentile=0.7,
#     smooth_span=5,
#     min_persist=3,
# )
