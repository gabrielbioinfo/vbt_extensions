"""
vbt_extensions: extensões pandas-first para vectorbt no processo quant.

Subpackages:
- data: loaders padronizados de OHLCV (ex.: Binance)
- ind: indicadores custom (trend lines, fibo, PIP, H&S, etc.)
- sig: geração de sinais (trend, reversão, regime…)
- risk: overlays de risco (stops, sizing, filtros)
- plotting: helpers de plot (tearsheets, zonas fibo, regimes)
- analyzers: analisadores e rankings
- metrics: relatórios/tearsheets

Exemplo:
    from vbt_extensions.data import load_ohlcv, BaseDownloadParams
    from vbt_extensions.sig import golden_cross_sig
    from vbt_extensions.ind import TREND_LINE
"""

__version__ = "0.2.0"

from . import analyzers, data, ind, metrics, plotting, risk, sig


# from vbt_extensions.data import load_ohlcv, BaseDownloadParams
# from vbt_extensions.ind import TREND_LINE, FIB_RETRACEMENT
# from vbt_extensions.sig import golden_cross_sig, tl_breakout_with_regime
# from vbt_extensions.risk import fibo_protective_stop
# from vbt_extensions.plotting import plot_trend_lines, plot_fib_zones
