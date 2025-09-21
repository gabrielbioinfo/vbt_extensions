"""Module for ranking backtest results and calculating performance metrics.

Provides functions to compute annualized alpha, beta, and a composite ranking index for strategy evaluation.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


class PriceBenchmarkRequiredError(ValueError):
    """Exception raised when a required price benchmark is missing for alpha calculation."""

    def __init__(self) -> None:
        """Initialize the PriceBenchmarkRequiredError with a standard message."""
        super().__init__("Pass 'price_benchmark=close' (price series) to safely calculate alpha.")


def _annualize(x_per_period: float, periods_per_year: int) -> float:
    return x_per_period * periods_per_year


def _ols_alpha_beta(r: pd.Series, m: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, m], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan
    y = df.iloc[:, 0].to_numpy()
    x = df.iloc[:, 1].to_numpy()
    x = np.c_[np.ones_like(x), x]  # [const, mkt]

    beta_hat = np.linalg.lstsq(x, y, rcond=None)[0]
    alpha, beta = beta_hat[0], beta_hat[1]
    return alpha, beta


def _tanh_scale(x: float, scale: float) -> float:
    # x/scale ~ 1 → tanh(1)=0.76
    return np.tanh(np.nan_to_num(x / scale))


def compute_alpha_series(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    periods_per_year: int = 8760,
) -> tuple[float, float]:
    """Return (alpha_annual, beta) using per-period returns (same frequency)."""
    a, b = _ols_alpha_beta(strategy_ret, benchmark_ret)
    if np.isnan(a):
        return np.nan, np.nan
    return _annualize(a, periods_per_year), b


def qindex_rank(
    pf: vbt.Portfolio,
    price_benchmark: pd.Series = None,  # default: HODL of the backtest's own 'close'
    freq: str = "1h",
    weights: dict | None = None,  # weights for the composite index
) -> pd.DataFrame:
    """Rank portfolio backtest results using a q-index.

    Parameters
    ----------
    pf : vbt.Portfolio
        The portfolio object containing backtest results.
    price_benchmark : pd.Series, optional
        Benchmark price series for alpha/beta calculation. Default is None.
    freq : str, optional
        Frequency string (e.g., '1h', '1D') to determine periods per year. Default is '1h'.
    weights : dict, optional
        Weights for the composite index components. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with ranking metrics and composite index for each portfolio column.

    Raises
    ------
    ValueError
        If price_benchmark is not provided.

    """
    # periods/year according to frequency
    periods_per_year = {"1h": 24 * 365, "1D": 252, "D": 252, "1min": 60 * 24 * 252}.get(freq, 24 * 365)

    # benchmark: by default, HODL of the portfolio's own close
    # (if you want to use another, pass price_benchmark with the same frequency)
    if price_benchmark is None:
        # pf.returns is already aligned; we use the same 'close' estimated by the portfolio
        # If you have the 'close' series used: pass it here. Otherwise, we approximate:
        # benchmark_ret = buy&hold of the asset (close) if available; otherwise,
        # we use the mean of pf.returns columns as a proxy (not recommended).
        # Better: keep the original 'close' and pass via price_benchmark.
        raise PriceBenchmarkRequiredError

    benchmark_ret = price_benchmark.pct_change().rename("mkt_ret")

    # default weights for the index (adjust as desired)
    if weights is None:
        weights = {
            "ret": 0.30,  # total return (signal of money made/lost)
            "sharpe": 0.30,  # risk/return ratio
            "alpha": 0.20,  # annual alpha vs benchmark
            "dd": 0.20,  # penalty for drawdown
        }

    rows = []
    for col in pf.wrapper.columns:
        s = pf.stats(column=col)

        # basic metrics (with fallback)
        ret = s.get("Total Return [%]", np.nan)
        sharpe = s.get("Sharpe Ratio", np.nan)
        maxdd = s.get("Max Drawdown [%]", s.get("Worst Drawdown [%]", np.nan))
        winrate = s.get("Win Rate [%]", np.nan)
        trades = s.get("Total Trades", s.get("Total Closed Trades", np.nan))
        exposure = s.get("Exposure [%]", s.get("Exposure Time [%]", np.nan))

        # per-period returns of the strategy (same frequency as pf)
        rets = pf.returns() if callable(getattr(pf, "returns", None)) else pf.returns
        strat_ret = rets[col]

        # annual alpha vs benchmark (OLS)
        alpha_ann, beta = compute_alpha_series(strat_ret, benchmark_ret, periods_per_year=periods_per_year)

        # --------- Normalizations for the composite index ---------
        # intuitive signals:
        #  - positive return and sharpe help
        #  - positive annual alpha helps
        #  - high drawdown should hurt (so it enters with negative sign)

        # scales (fine-tune as you prefer):
        ret_n = _tanh_scale(ret, 50.0)  # 50% ≈ soft saturation point
        sh_n = _tanh_scale(sharpe, 2.0)  # Sharpe 2 ~ excellent
        alpha_n = _tanh_scale(alpha_ann, 20.0)  # 20% p.a. ~ good
        dd_n = -_tanh_scale(abs(maxdd), 30.0)  # the higher the DD, the more negative

        # composite index:  based on return, Sharpe ratio, annualized alpha, and drawdown
        q_index = (
            weights["ret"] * ret_n
            + weights["sharpe"] * sh_n
            + weights["alpha"] * alpha_n
            + weights["dd"] * dd_n
        )
        # clip to [-1, 1] (optional)
        q_index = float(np.clip(q_index, -1.0, 1.0))

        row = {
            **{  # noqa: C416
                name: val
                for name, val in zip(  # noqa: B905
                    pf.wrapper.columns.names,
                    col if isinstance(col, tuple) else (col,),
                )
            },
            "Return_%": ret,
            "Sharpe": sharpe,
            "MaxDD_%": maxdd,
            "WinRate_%": winrate,
            "Trades": trades,
            "Exposure_%": exposure,
            "Alpha_ann_%": alpha_ann,  # annualized alpha in %
            "Beta": beta,
            "qspot_index": q_index,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # sort by index; in case of tie, use Sharpe and Return
    return df.sort_values(["qspot_index", "Sharpe", "Return_%"], ascending=[False, False, False]).reset_index(
        drop=True,
    )
