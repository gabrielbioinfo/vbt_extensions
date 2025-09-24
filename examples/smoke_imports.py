import vbt_extensions as vext

# data
from vbt_extensions.data import BinanceDownloadParams, binance_download

# ind
from vbt_extensions.ind.zigzag import ZIGZAG
from vbt_extensions.ind.market_structure_zigzag import MARKET_STRUCTURE_ZZ
from vbt_extensions.ind.fair_value_gap import FAIR_VALUE_GAP

# sig
from vbt_extensions.sig.smart_money_sig import combo_smart_money

# risk
from vbt_extensions.risk.targets_stops_smc import smc_targets_stops
from vbt_extensions.risk.stop_trailing import atr_trailing
from vbt_extensions.risk.position_sizing import fixed_fraction

# plotting (dependendo do __init__, pode ser .plotting.helpers)
try:
    from vbt_extensions.plotting import plot_price, plot_structure, plot_fvg
except ImportError:
    from vbt_extensions.plotting.helpers import plot_price, plot_structure, plot_fvg

print("OK: imports básicos")
