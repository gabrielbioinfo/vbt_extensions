"""Perceptually Important Points (PIP) indicator for use with vectorbt."""

import numpy as np
import vectorbt as vbt

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001, D103
        def deco(f):  # noqa: ANN001, ANN202
            return f

        return deco


PIP_DIST_MEASURE_SLOPE = 2


@njit(cache=True)
def _find_pips_1d(data: np.ndarray, n_pips: int, dist_measure: int) -> tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    pips_x = np.empty(n_pips, dtype=np.int64)
    pips_y = np.empty(n_pips, dtype=np.float64)
    pips_x[0] = 0
    pips_x[1] = n - 1
    pips_y[0] = data[0]
    pips_y[1] = data[-1]
    for curr_point in range(2, n_pips):
        md = 0.0
        md_i = -1
        insert_index = -1
        for k in range(curr_point - 1):
            left_adj = pips_x[k]
            right_adj = pips_x[k + 1]
            time_diff = right_adj - left_adj
            price_diff = pips_y[k + 1] - pips_y[k]
            slope = price_diff / time_diff
            intercept = pips_y[k] - left_adj * slope
            for i in range(left_adj + 1, right_adj):
                d = 0.0
                if dist_measure == 1:
                    d = ((left_adj - i) ** 2 + (pips_y[k] - data[i]) ** 2) ** 0.5
                    d += ((right_adj - i) ** 2 + (pips_y[k + 1] - data[i]) ** 2) ** 0.5
                elif dist_measure == PIP_DIST_MEASURE_SLOPE:
                    d = abs((slope * i + intercept) - data[i]) / (slope**2 + 1) ** 0.5
                else:
                    d = abs((slope * i + intercept) - data[i])
                if d > md:
                    md = d
                    md_i = i
                    insert_index = k + 1
        pips_x[insert_index] = md_i
        pips_y[insert_index] = data[md_i]
    return pips_x, pips_y


def _apply_func(close: np.ndarray, n_pips: int, dist_measure: int) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    if close.ndim > 1:
        close = close.ravel()
    return _find_pips_1d(close, int(n_pips), int(dist_measure))


perceptually_important = vbt.IndicatorFactory(
    class_name="perceptually_important",
    short_name="pip",
    input_names=["close"],
    param_names=["n_pips", "dist_measure"],
    output_names=["pips_x", "pips_y"],
).from_apply_func(
    _apply_func,
    n_pips=5,
    dist_measure=2,
    keep_pd=True,
)
