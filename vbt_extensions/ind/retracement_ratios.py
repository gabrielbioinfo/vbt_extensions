"""Retracement ratios indicator for use with vectorbt."""

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
def _retracement_ratios_1d(ext_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = ext_p.shape[0]
    seg_height = np.empty(n, dtype=np.float64)
    seg_height[:] = np.nan
    retrace_ratio = np.empty(n, dtype=np.float64)
    retrace_ratio[:] = np.nan
    log_retrace_ratio = np.empty(n, dtype=np.float64)
    log_retrace_ratio[:] = np.nan
    for i in range(1, n):
        seg_height[i] = np.abs(ext_p[i] - ext_p[i - 1])
        if i > 1 and seg_height[i - 1] != 0:
            retrace_ratio[i] = seg_height[i] / seg_height[i - 1]
            if retrace_ratio[i] > 0:
                log_retrace_ratio[i] = np.log(retrace_ratio[i])
    return seg_height, retrace_ratio, log_retrace_ratio


def _apply_func(ext_p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext_p = np.asarray(ext_p, dtype=np.float64)
    if ext_p.ndim > 1:
        ext_p = ext_p.ravel()
    return _retracement_ratios_1d(ext_p)


retracement_ratios = vbt.IndicatorFactory(
    class_name="retracement_ratios",
    short_name="rr",
    input_names=["ext_p"],
    param_names=[],
    output_names=["seg_height", "retrace_ratio", "log_retrace_ratio"],
).from_apply_func(
    _apply_func,
    keep_pd=True,
)
