import numpy as np
import pandas as pd


def estimate_prominence_from_vol(
    close: pd.Series,
    *,
    method: str = "atr",
    window: int = 14,
    factor: float = 1.0,
) -> float:
    """Heurística simples para calibrar 'prominence' a partir da vol.

    - method='atr': usa ATR aproximado (True Range com High/Low sintéticos).
    - method='stdev': usa desvio padrão de retornos.
    - Retorna um escalar para usar como prominence (em unidades do 'source_series').

    Obs.: se seu 'source_series' for CLOSE, a prominence natural está em preço.
    """
    c = close.astype(float)

    if method == "stdev":
        r = c.pct_change().dropna()
        vol = r.rolling(window).std(ddof=1).iloc[-1]  # vol em %
        return float(c.iloc[-1] * vol * factor)

    # ATR aproximado usando close como proxy (se tiver H/L reais, melhor usar eles)
    tr = c.diff().abs()
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr * factor)
