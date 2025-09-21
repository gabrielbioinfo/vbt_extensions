"""Zigzag indicator implementation for use with vectorbt."""

import numpy as np
import vectorbt as vbt

# Optional: numba acceleration (can remove if you don't want numba)
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001, D103
        def deco(f):  # noqa: ANN001, ANN202
            return f

        return deco


@njit(cache=True)
def _zigzag_1d(
    high: np.ndarray,
    low: np.ndarray,
    upper: float,
    lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = high.shape[0]
    zig = np.empty(n, dtype=np.float64)
    zig[:] = np.nan
    is_top = np.zeros(n, dtype=np.float64)
    is_bot = np.zeros(n, dtype=np.float64)

    up_thr = 1.0 + upper
    down_thr = 1.0 - lower

    last_price = float(high[0])
    last_kind = 0  # 0=none, 1=top, -1=bottom
    last_idx = 0

    for i in range(1, n):
        hi = float(high[i])
        lo = float(low[i])

        # top trigger
        if (hi / last_price) >= up_thr and last_kind != 1:
            last_kind = 1
            last_price = hi
            last_idx = i
            zig[i] = hi
            is_top[i] = 1.0
            continue

        # bottom trigger
        if (lo / last_price) <= down_thr and last_kind != -1:
            last_kind = -1
            last_price = lo
            last_idx = i
            zig[i] = lo
            is_bot[i] = 1.0
            continue

        # trailing (higher-high at top / lower-low at bottom)
        if last_kind == 1 and hi > zig[last_idx] if not np.isnan(zig[last_idx]) else False:
            zig[last_idx] = hi
            last_price = hi
        elif last_kind == -1 and lo < zig[last_idx] if not np.isnan(zig[last_idx]) else False:
            zig[last_idx] = lo
            last_price = lo

    return zig, is_top, is_bot


def _apply_func(
    high: np.ndarray,
    low: np.ndarray,
    upper: float,
    lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if high.ndim > 1:
        high = high.ravel()
    if low.ndim > 1:
        low = low.ravel()
    return _zigzag_1d(high, low, float(upper), float(lower))


ZIGZAG = vbt.IndicatorFactory(
    class_name="ZIGZAG",
    short_name="ZZ",
    input_names=["high", "low"],
    param_names=["upper", "lower"],
    output_names=["zigzag", "is_top", "is_bottom"],
).from_apply_func(
    _apply_func,
    upper=0.02,  # 2%
    lower=0.02,  # 2%
    keep_pd=True,
)
