"""
Head and Shoulders (HS / IHS) detector — pandas-first API.

- Mantém núcleo em Numba (rápido).
- Retorna séries alinhadas ao índice para plot/sinais:
  * hs_start, ihs_start            -> marcadores de início do padrão
  * hs_break, ihs_break            -> marcadores no candle de rompimento
  * hs_neckline, ihs_neckline      -> linha da neckline ao longo do trecho do padrão
  * hs_head_height, ihs_head_height-> altura da cabeça (série esparsa)
  * hs_type / ihs_type             -> flags bool por tipo (redundante, mas útil)

Observações:
- Usamos uma “neckline” linear definida pelos pontos axilares (armpits).
- O núcleo retorna também índices dos armpits e o valor da neckline no 1º axilar,
  para que possamos reconstruir a linha no wrapper pandas.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        def deco(f):
            return f

        return deco


# ========= Núcleo Numba (adaptado do seu original) ========= #


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
def _check_hs_pattern(extrema_indices, data, i):
    # Saída: (start_i, break_i, head_h, neck_slope, head_w, pattern_r2,
    #         l_armpit_i, r_armpit_i, neck_val_at_l_armpit)
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]

    if i - r_armpit < 2:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    r_shoulder = r_armpit + np.argmax(data[r_armpit + 1 : i]) + 1 if i > r_armpit + 1 else r_armpit
    if data[head] <= max(data[l_shoulder], data[r_shoulder]):
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] < r_midpoint or data[r_shoulder] < l_midpoint:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run if neck_run != 0 else 0.0
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    if data[i] > neck_val:  # para HS (top), rompimento é para baixo
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    head_width = r_armpit - l_armpit
    pat_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0:
            return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)
        if data[l_shoulder - j] < neck:
            pat_start = l_shoulder - j
            break
    if pat_start == -1:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    pattern_r2 = 1.0  # placeholder
    # adicionamos armpits e valor da neckline no l_armpit para reconstrução
    return (pat_start, i, head_height, neck_slope, head_width, pattern_r2, l_armpit, r_armpit, data[l_armpit])


@njit(cache=True)
def _check_ihs_pattern(extrema_indices, data, i):
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]

    if i - r_armpit < 2:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    r_shoulder = r_armpit + np.argmin(data[r_armpit + 1 : i]) + 1 if i > r_armpit + 1 else r_armpit
    if data[head] >= min(data[l_shoulder], data[r_shoulder]):
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] > r_midpoint or data[r_shoulder] > l_midpoint:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run if neck_run != 0 else 0.0
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    if data[i] < neck_val:  # para IHS (fundo), rompimento é para cima
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    head_width = r_armpit - l_armpit
    pat_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0:
            return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)
        if data[l_shoulder - j] > neck:
            pat_start = l_shoulder - j
            break
    if pat_start == -1:
        return (-1, -1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan)

    head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    pattern_r2 = 1.0  # placeholder
    return (pat_start, i, head_height, neck_slope, head_width, pattern_r2, l_armpit, r_armpit, data[l_armpit])


@njit(cache=True)
def _detect_hs_patterns(data: np.ndarray, order: int):
    n = data.shape[0]
    max_patterns = n // 10 + 1

    # Saídas (alocamos e truncamos no final)
    hs_start_i = np.full(max_patterns, -1, dtype=np.int32)
    hs_break_i = np.full(max_patterns, -1, dtype=np.int32)
    hs_head_height = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_neck_slope = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_head_width = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_pattern_r2 = np.full(max_patterns, np.nan, dtype=np.float64)
    hs_type = np.full(max_patterns, 0, dtype=np.int32)  # 0=HS, 1=IHS
    hs_l_ap_i = np.full(max_patterns, -1, dtype=np.int32)
    hs_r_ap_i = np.full(max_patterns, -1, dtype=np.int32)
    hs_neck_base = np.full(max_patterns, np.nan, dtype=np.float64)

    pat_count = 0
    recent_extrema = np.full(5, -1, dtype=np.int32)
    recent_types = np.full(5, 0, dtype=np.int32)
    hs_lock = False
    ihs_lock = False
    last_is_top = False

    for i in range(n):
        # Extremos por janela
        if rw_top(data, i, order):
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

        if last_is_top:
            ihs_extrema = recent_extrema[1:5]
            hs_extrema = recent_extrema[0:4]
        else:
            ihs_extrema = recent_extrema[0:4]
            hs_extrema = recent_extrema[1:5]

        # HS (top) — rompe para baixo
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
                hs_l_ap_i[pat_count] = pat[6]
                hs_r_ap_i[pat_count] = pat[7]
                hs_neck_base[pat_count] = pat[8]
                pat_count += 1
                hs_lock = True

        # IHS (fundo) — rompe para cima
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
                hs_l_ap_i[pat_count] = pat[6]
                hs_r_ap_i[pat_count] = pat[7]
                hs_neck_base[pat_count] = pat[8]
                pat_count += 1
                ihs_lock = True

    # Trunca
    return (
        hs_start_i[:pat_count],
        hs_break_i[:pat_count],
        hs_head_height[:pat_count],
        hs_neck_slope[:pat_count],
        hs_head_width[:pat_count],
        hs_pattern_r2[:pat_count],
        hs_type[:pat_count],
        hs_l_ap_i[:pat_count],
        hs_r_ap_i[:pat_count],
        hs_neck_base[:pat_count],
    )


# ========= Dataclass (pandas-first) ========= #


@dataclass
class HeadShouldersResult:
    hs_start: pd.Series  # bool: início do padrão HS
    hs_break: pd.Series  # bool: candle de rompimento da neckline (HS)
    ihs_start: pd.Series  # bool: início do padrão IHS
    ihs_break: pd.Series  # bool: candle de rompimento da neckline (IHS)
    hs_neckline: pd.Series  # linha da neckline (HS), NaN fora
    ihs_neckline: pd.Series  # linha da neckline (IHS), NaN fora
    hs_head_height: pd.Series  # altura da cabeça (série esparsa no break)
    ihs_head_height: pd.Series  # idem para IHS


class HEAD_AND_SHOULDERS:
    """Detector de Head & Shoulders com API pandas-first.

    Parâmetros
    ----------
    close : pd.Series
    order : int
        Tamanho da meia-janela (mesmo conceito do seu RW).
    """

    @staticmethod
    def run(close: pd.Series, order: int = 6) -> HeadShouldersResult:
        if not isinstance(close, pd.Series):
            raise TypeError("close deve ser pd.Series")

        c = close.astype(float)
        x = c.values.astype(np.float64)
        (start_i, break_i, head_h, neck_slope, head_w, pat_r2, pat_type, l_ap_i, r_ap_i, neck_base) = (
            _detect_hs_patterns(x, int(order))
        )

        # Séries vazias
        idx = c.index
        hs_start = pd.Series(False, index=idx)
        hs_break = pd.Series(False, index=idx)
        ihs_start = pd.Series(False, index=idx)
        ihs_break = pd.Series(False, index=idx)
        hs_neckline = pd.Series(np.nan, index=idx, dtype=float)
        ihs_neckline = pd.Series(np.nan, index=idx, dtype=float)
        hs_head_h_s = pd.Series(np.nan, index=idx, dtype=float)
        ihs_head_h_s = pd.Series(np.nan, index=idx, dtype=float)

        # Reconstrói necklines e marcações
        for k in range(len(pat_type)):
            s_i = int(start_i[k])
            b_i = int(break_i[k])
            lap = int(l_ap_i[k])
            rap = int(r_ap_i[k])
            slope = float(neck_slope[k])
            base = float(neck_base[k])
            h = float(head_h[k])
            if s_i < 0 or b_i < 0 or lap < 0 or rap < 0:
                continue

            # linha da neckline do l_armpit ao r_armpit
            # neckline(t) = base + (t - lap)*slope
            for t in range(lap, rap + 1):
                val = base + (t - lap) * slope
                if pat_type[k] == 0:
                    hs_neckline.iat[t] = val
                else:
                    ihs_neckline.iat[t] = val

            # marcações de início e rompimento
            if pat_type[k] == 0:
                hs_start.iat[s_i] = True
                hs_break.iat[b_i] = True
                hs_head_h_s.iat[b_i] = h
            else:
                ihs_start.iat[s_i] = True
                ihs_break.iat[b_i] = True
                ihs_head_h_s.iat[b_i] = h

        return HeadShouldersResult(
            hs_start=hs_start,
            hs_break=hs_break,
            ihs_start=ihs_start,
            ihs_break=ihs_break,
            hs_neckline=hs_neckline,
            ihs_neckline=ihs_neckline,
            hs_head_height=hs_head_h_s,
            ihs_head_height=ihs_head_h_s,
        )
