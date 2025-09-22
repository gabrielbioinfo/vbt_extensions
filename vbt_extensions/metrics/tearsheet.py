import json

import pandas as pd


def basic_metrics(equity_or_returns: pd.Series) -> dict:
    """Calcula Sharpe, MaxDD e CAGR (aprox)."""
    s = equity_or_returns.dropna()

    # heurística: se média abs < 0.2 → provavelmente são retornos
    is_returns = s.abs().mean() < 0.2
    if is_returns:
        r = s
        equity = (1.0 + r).cumprod()
    else:
        equity = s
        r = equity.pct_change().fillna(0.0)

    ann_factor = 252.0
    sharpe = (r.mean() / (r.std(ddof=1) + 1e-12)) * ann_factor**0.5

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    maxdd = dd.min()

    if len(equity) > 1:
        years = len(equity) / ann_factor
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(years, 1e-12)) - 1.0
    else:
        cagr = 0.0

    return {"Sharpe": float(sharpe), "MaxDD": float(maxdd), "CAGR": float(cagr)}


def save_metrics_json(metrics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
