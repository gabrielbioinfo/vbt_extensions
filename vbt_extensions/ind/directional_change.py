"""Directional Change indicator for use with vectorbt."""

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
def _directional_change_1d(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    is_top = np.zeros(n, dtype=np.float64)
    is_bottom = np.zeros(n, dtype=np.float64)
    up_zig = True
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0
    for i in range(n):
        if up_zig:
            if high[i] > tmp_max:
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma:
                is_top[tmp_max_i] = 1.0
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else:  # noqa: PLR5501
            if low[i] < tmp_min:
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma:
                is_bottom[tmp_min_i] = 1.0
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i
    return is_top, is_bottom


def _apply_func(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if close.ndim > 1:
        close = close.ravel()
    if high.ndim > 1:
        high = high.ravel()
    if low.ndim > 1:
        low = low.ravel()
    return _directional_change_1d(close, high, low, float(sigma))


directional_change = vbt.IndicatorFactory(
    class_name="directional_change",
    short_name="dc",
    input_names=["close", "high", "low"],
    param_names=["sigma"],
    output_names=["is_top", "is_bottom"],
).from_apply_func(
    _apply_func,
    sigma=0.02,
    keep_pd=True,
)
