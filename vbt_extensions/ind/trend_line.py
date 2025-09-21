"""Trend Line indicator for use with vectorbt."""

import numpy as np
import vectorbt as vbt

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001, D103
        def deco(f):  # noqa: ANN001, ANN202
            return f

        return deco


@njit(cache=True)
def _check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:  # noqa: FBT001
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(y.shape[0]) + intercept
    diffs = line_vals - y
    if support and np.max(diffs) > 1e-5:  # noqa: PLR2004, SIM114
        return -1.0
    elif not support and np.min(diffs) < -1e-5:  # noqa: PLR2004, RET505
        return -1.0
    return np.sum(diffs**2.0)


@njit(cache=True)
def _optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray) -> tuple[float, float]:  # noqa: FBT001
    slope_unit = (np.max(y) - np.min(y)) / y.shape[0]
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step
    best_slope = init_slope
    best_err = _check_trend_line(support, pivot, init_slope, y)
    if best_err < 0.0:
        return np.nan, np.nan
    get_derivative = True
    derivative = 0.0
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = _check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = _check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                break
            get_derivative = False
        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step
        test_err = _check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True
    return best_slope, -best_slope * pivot + y[pivot]


@njit(cache=True)
def _fit_trendlines_high_low(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.arange(close.shape[0])
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = np.argmax(high - line_points)
    lower_pivot = np.argmin(low - line_points)
    support_coefs = _optimize_slope(True, lower_pivot, coefs[0], low)  # noqa: FBT003
    resist_coefs = _optimize_slope(False, upper_pivot, coefs[0], high)  # noqa: FBT003
    return support_coefs, resist_coefs


@njit(cache=True)
def _trend_line_1d(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    support_slope = np.full(n, np.nan, dtype=np.float64)
    resist_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback - 1, n):
        h = high[i - lookback + 1 : i + 1]
        l = low[i - lookback + 1 : i + 1]  # noqa: E741
        c = close[i - lookback + 1 : i + 1]
        support_coefs, resist_coefs = _fit_trendlines_high_low(h, l, c)
        support_slope[i] = support_coefs[0]
        resist_slope[i] = resist_coefs[0]
    return support_slope, resist_slope


def _apply_func(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if high.ndim > 1:
        high = high.ravel()
    if low.ndim > 1:
        low = low.ravel()
    if close.ndim > 1:
        close = close.ravel()
    return _trend_line_1d(high, low, close, int(lookback))


trend_line = vbt.IndicatorFactory(
    class_name="trend_line",
    short_name="tl",
    input_names=["high", "low", "close"],
    param_names=["lookback"],
    output_names=["support_slope", "resist_slope"],
).from_apply_func(
    _apply_func,
    lookback=30,
    keep_pd=True,
)
