
"""Detecção de topos e fundos em séries temporais usando scipy.signal.find_peaks.

Este módulo fornece:
- Classe PeaksResult para encapsular resultados de detecção de picos.
- Funções auxiliares para suavização e detecção de picos.
- Classe PEAKS_SCIPY para interface pandas-first e parâmetros flexíveis.
"""

from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_widths
except Exception as e:
    raise ImportError("Este módulo requer scipy. Instale com: pip install scipy") from e


@dataclass
class PeaksResult:
    """Saída padronizada para detecção de topos e fundos."""

    is_top: pd.Series
    is_bottom: pd.Series
    swing_high: pd.Series
    swing_low: pd.Series
    peak_prominence: pd.Series | None = None
    peak_width: pd.Series | None = None
    peak_height: pd.Series | None = None


def _smooth_series(
    s: pd.Series,
    smooth: Literal["ema", "sma"] | None = None,
    smooth_window: int = 5,
    ema_alpha: float | None = None,
) -> pd.Series:
    s = s.astype(float)
    if smooth is None:
        return s
    if smooth == "sma":
        return s.rolling(window=smooth_window, min_periods=1).mean()
    if smooth == "ema":
        alpha = ema_alpha if ema_alpha is not None else (2.0 / (smooth_window + 1.0))
        return s.ewm(alpha=alpha, adjust=False).mean()
    raise ValueError("smooth deve ser None, 'ema' ou 'sma'.")


def _run_find_peaks(
    y: pd.Series,
    *,
    prominence: Union[float, Tuple[float, float]] | None = None,
    distance: int | None = None,
    width: Union[float, Tuple[float, float]] | None = None,
    height: Union[float, Tuple[float, float]] | None = None,
) -> Tuple[np.ndarray, dict]:
    """Wrapper leve para find_peaks trabalhando com Series."""
    x = y.values.astype(float)
    idx, props = find_peaks(
        x,
        prominence=prominence,
        distance=distance,
        width=width,
        height=height,
    )
    return idx, props


class PEAKS_SCIPY:
    """Topos/Fundos com SciPy find_peaks (pandas-first, legível).

    - Detecta topos em `source_series` (p. ex., High, Close, ou mid).
    - Detecta fundos aplicando find_peaks em `-source_series`.
    - Permite suavização (EMA/SMA) antes da detecção para reduzir ruído.
    - Exposição de parâmetros clássicos do find_peaks:
      prominence, distance, width, height.

    Notas:
    - Para séries OHLC, costumo usar `mid = (High+Low)/2` como `source_series`.
    - Para fundos, usamos `-series` (picos em -y == vales em y).
    """

    @staticmethod
    def run(
        source_series: pd.Series,
        *,
        # pré-processamento (opcional)
        smooth: Literal["ema", "sma"] | None = None,
        smooth_window: int = 5,
        ema_alpha: float | None = None,
        # parâmetros find_peaks topos
        top_prominence: Union[float, Tuple[float, float]] | None = None,
        top_distance: int | None = None,
        top_width: Union[float, Tuple[float, float]] | None = None,
        top_height: Union[float, Tuple[float, float]] | None = None,
        # parâmetros find_peaks fundos (em -series)
        bottom_prominence: Union[float, Tuple[float, float]] | None = None,
        bottom_distance: int | None = None,
        bottom_width: Union[float, Tuple[float, float]] | None = None,
        bottom_height: Union[float, Tuple[float, float]] | None = None,
    ) -> PeaksResult:
        if not isinstance(source_series, pd.Series):
            raise TypeError("source_series deve ser pd.Series")

        s = _smooth_series(source_series, smooth, smooth_window, ema_alpha)
        idx = s.index

        # --- TOPOS ---
        top_i, top_props = _run_find_peaks(
            s,
            prominence=top_prominence,
            distance=top_distance,
            width=top_width,
            height=top_height,
        )
        is_top = pd.Series(False, index=idx)
        if len(top_i) > 0:
            is_top.iloc[top_i] = True

        # --- FUNDOS (picos de -s) ---
        inv = -s
        bot_i, bot_props = _run_find_peaks(
            inv,
            prominence=bottom_prominence,
            distance=bottom_distance,
            width=bottom_width,
            height=bottom_height,
        )
        is_bottom = pd.Series(False, index=idx)
        if len(bot_i) > 0:
            is_bottom.iloc[bot_i] = True

        # swing highs / lows: últimos pivôs conhecidos (preenche pra frente)
        swing_high = s.where(is_top).ffill()
        swing_low = s.where(is_bottom).ffill()

        # métricas opcionalmente expostas (prominence/width/height)
        def _to_series(arr: np.ndarray | None, default_nan=True) -> pd.Series | None:
            if arr is None:
                return None
            out = pd.Series(np.nan if default_nan else 0.0, index=idx, dtype=float)
            if len(top_i) and arr is top_props.get("prominences", None):
                out.iloc[top_i] = arr
                return out
            if len(top_i) and arr is top_props.get("widths", None):
                out.iloc[top_i] = arr
                return out
            if len(top_i) and arr is top_props.get("peak_heights", None):
                out.iloc[top_i] = arr
                return out
            # fundos (aplicar nos bot_i)
            if len(bot_i) and arr is bot_props.get("prominences", None):
                # guardamos só uma das métricas; se quiser separar, duplique com nomes *_bottom
                out.iloc[bot_i] = arr
                return out
            if len(bot_i) and arr is bot_props.get("widths", None):
                out.iloc[bot_i] = arr
                return out
            if len(bot_i) and arr is bot_props.get("peak_heights", None):
                out.iloc[bot_i] = arr
                return out
            return None

        # Preferimos expor métricas de topo; se quiser métricas separadas para fundos,
        # você pode duplicar os campos (ex.: peak_prominence_top / peak_prominence_bottom).
        peak_prominence = _to_series(top_props.get("prominences", None))
        peak_width = _to_series(top_props.get("widths", None))
        peak_height = _to_series(top_props.get("peak_heights", None))

        return PeaksResult(
            is_top=is_top.astype(bool),
            is_bottom=is_bottom.astype(bool),
            swing_high=swing_high,
            swing_low=swing_low,
            peak_prominence=peak_prominence,
            peak_width=peak_width,
            peak_height=peak_height,
        )
