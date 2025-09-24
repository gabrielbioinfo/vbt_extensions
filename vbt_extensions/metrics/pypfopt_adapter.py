# vbt_extensions/metrics/pypfopt_adapter.py
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

def pypfopt_max_sharpe(close: pd.DataFrame) -> pd.DataFrame:
    mu = expected_returns.mean_historical_return(close)
    S = risk_models.sample_cov(close)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()
    w = pd.Series(cleaned).reindex(close.columns).fillna(0)
    return pd.DataFrame([w], index=[close.index[-1]])
