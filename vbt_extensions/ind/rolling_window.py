"""Rolling window local extremes indicator for use with vectorbt."""

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
def _rw_extremes_1d(data: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    is_top = np.zeros(n, dtype=np.float64)
    is_bottom = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < order * 2 + 1:
            continue
        k = i - order
        v = data[k]
        # Check top
        top = True
        for j in range(1, order + 1):
            if data[k + j] > v or data[k - j] > v:
                top = False
                break
        if top:
            is_top[k] = 1.0
        # Check bottom
        bottom = True
        for j in range(1, order + 1):
            if data[k + j] < v or data[k - j] < v:
                bottom = False
                break
        if bottom:
            is_bottom[k] = 1.0
    return is_top, is_bottom


def _apply_func(close: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    if close.ndim > 1:
        close = close.ravel()
    return _rw_extremes_1d(close, int(order))


rolling_window = vbt.IndicatorFactory(
    class_name="rolling_window",
    short_name="rw",
    input_names=["close"],
    param_names=["order"],
    output_names=["is_top", "is_bottom"],
).from_apply_func(
    _apply_func,
    order=10,
    keep_pd=True,
)
