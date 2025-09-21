"""Head and Shoulders pattern indicator for use with vectorbt.

Implements full pattern detection logic, outputs arrays of pattern attributes.

References:
- https://www.youtube.com/watch?v=6iFqjd5BOHw
- https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/head_shoulders.py

"""

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
def rw_top(data: np.ndarray, i: int, order: int) -> bool:
    if i < order or i + order >= data.shape[0]:
        return False
    center = data[i]
    for j in range(i - order, i + order + 1):
        if data[j] > center:
            return False
    return True


@njit(cache=True)
def rw_bottom(data: np.ndarray, i: int, order: int) -> bool:
    if i < order or i + order >= data.shape[0]:
        return False
    center = data[i]
    for j in range(i - order, i + order + 1):
        if data[j] < center:
            return False
    return True


@njit(cache=True)
def _detect_hs_patterns(data: np.ndarray, order: int):
    n = data.shape[0]
    max_patterns = n // 10 + 1
    # Output arrays for pattern attributes
    hs_start_i = np.full(max_patterns, -1, dtype=np.int32)
    hs_break_i = np.full(max_patterns, -1, dtype=np.int32)
    hs_head_height = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_neck_slope = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_head_width = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_pattern_r2 = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_type = np.full(max_patterns, 0, dtype=np.int32)  # 0=HS, 1=IHS

    pat_count = 0
    # Deques for recent extrema and types
    recent_extrema = np.full(5, -1, dtype=np.int32)
    recent_types = np.full(5, 0, dtype=np.int32)
    hs_lock = False
    ihs_lock = False
    last_is_top = False

    for i in range(n):
        # Detect rolling window extrema
        if rw_top(data, i, order):
            # Shift left
            for k in range(4):
                recent_extrema[k] = recent_extrema[k + 1]
                recent_types[k] = recent_types[k + 1]
            recent_extrema[4] = i
            recent_types[4] = 1
            ihs_lock = False
            last_is_top = True
        if rw_bottom(data, i, order):
            for k in range(4):
                recent_extrema[k] = recent_extrema[k + 1]
                recent_types[k] = recent_types[k + 1]
            recent_extrema[4] = i
            recent_types[4] = -1
            hs_lock = False
            last_is_top = False
        if recent_extrema[0] == -1:
            continue
        # Check for pattern
        if last_is_top:
            ihs_extrema = recent_extrema[1:5]
            hs_extrema = recent_extrema[0:4]
        else:
            ihs_extrema = recent_extrema[0:4]
            hs_extrema = recent_extrema[1:5]
        # HS pattern
        if not hs_lock:
            pat = _check_hs_pattern(hs_extrema, data, i)
            if pat[0] != -1:
                hs_start_i[pat_count] = pat[0]
                hs_break_i[pat_count] = pat[1]
                hs_head_height[pat_count] = pat[2]
                hs_neck_slope[pat_count] = pat[3]
                hs_head_width[pat_count] = pat[4]
                hs_pattern_r2[pat_count] = pat[5]
                hs_type[pat_count] = 0
                pat_count += 1
                hs_lock = True
        # IHS pattern
        if not ihs_lock:
            pat = _check_ihs_pattern(ihs_extrema, data, i)
            if pat[0] != -1:
                hs_start_i[pat_count] = pat[0]
                hs_break_i[pat_count] = pat[1]
                hs_head_height[pat_count] = pat[2]
                hs_neck_slope[pat_count] = pat[3]
                hs_head_width[pat_count] = pat[4]
                hs_pattern_r2[pat_count] = pat[5]
                hs_type[pat_count] = 1
                pat_count += 1
                ihs_lock = True
    # Truncate outputs to found patterns
    return (
        hs_start_i[:pat_count],
        hs_break_i[:pat_count],
        hs_head_height[:pat_count],
        hs_neck_slope[:pat_count],
        hs_head_width[:pat_count],
        hs_pattern_r2[:pat_count],
        hs_type[:pat_count],
    )


@njit(cache=True)
def _check_hs_pattern(extrema_indices, data, i):
    # Returns tuple (-1, ...) if not found
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]
    if i - r_armpit < 2:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    r_shoulder = r_armpit + np.argmax(data[r_armpit + 1 : i]) + 1 if i > r_armpit + 1 else r_armpit
    if data[head] <= max(data[l_shoulder], data[r_shoulder]):
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] < r_midpoint or data[r_shoulder] < l_midpoint:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run if neck_run != 0 else 0.0
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    if data[i] > neck_val:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1.0
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0:
            return (-1, -1, np.nan, np.nan, np.nan, np.nan)
        if data[l_shoulder - j] < neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break
    if pat_start == -1:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    pattern_r2 = 1.0  # Placeholder for R2
    return (pat_start, i, head_height, neck_slope, head_width, pattern_r2)


@njit(cache=True)
def _check_ihs_pattern(extrema_indices, data, i):
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]
    if i - r_armpit < 2:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    r_shoulder = r_armpit + np.argmin(data[r_armpit + 1 : i]) + 1 if i > r_armpit + 1 else r_armpit
    if data[head] >= min(data[l_shoulder], data[r_shoulder]):
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] > r_midpoint or data[r_shoulder] > l_midpoint:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run if neck_run != 0 else 0.0
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    if data[i] < neck_val:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1.0
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0:
            return (-1, -1, np.nan, np.nan, np.nan, np.nan)
        if data[l_shoulder - j] > neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break
    if pat_start == -1:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan)
    head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    pattern_r2 = 1.0  # Placeholder for R2
    return (pat_start, i, head_height, neck_slope, head_width, pattern_r2)


def _apply_func(close: np.ndarray, order: int):
    close = np.asarray(close, dtype=np.float64)
    if close.ndim > 1:
        close = close.ravel()
    return _detect_hs_patterns(close, int(order))


head_and_shoulders = vbt.IndicatorFactory(
    class_name="head_and_shoulders",
    short_name="hs",
    input_names=["close"],
    param_names=["order"],
    output_names=[
        "start_i",
        "break_i",
        "head_height",
        "neck_slope",
        "head_width",
        "pattern_r2",
        "pattern_type",
    ],
).from_apply_func(
    _apply_func,
    order=6,
    keep_pd=True,
)
