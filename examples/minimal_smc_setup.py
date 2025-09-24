import pandas as pd
import vectorbt as vbt
from binance.client import Client

from vbt_extensions.data import BinanceDownloadParams, binance_download
from vbt_extensions.ind.fair_value_gap import FAIR_VALUE_GAP
from vbt_extensions.ind.market_structure_zigzag import MARKET_STRUCTURE_ZZ
from vbt_extensions.ind.zigzag import ZIGZAG
from vbt_extensions.risk.targets_stops_smc import smc_targets_stops
from vbt_extensions.sig.smart_money_sig import combo_smart_money

client = Client()  # usa env vars se setadas
params = BinanceDownloadParams(
    client=client, ticker="BTCUSDT",
    interval="5m", start="5 days ago UTC",
    tz="America/Sao_Paulo",
    fill_gaps=True, drop_partial_last=True
)
df = binance_download(params)
assert {"Open","High","Low","Close","Volume"} <= set(df.columns)
print("DF:", df.shape, df.index.min(), "->", df.index.max())


# use df do download (ou crie um sintético se preferir)
def session_mask(index, start="09:00", end="18:00", tz="America/Sao_Paulo", weekdays=[0,1,2,3,4]):
    idx = index.tz_convert(tz) if index.tz is not None and tz else index
    s_h, s_m = map(int, start.split(":")); e_h, e_m = map(int, end.split(":"))
    st = pd.Timestamp(0).replace(hour=s_h, minute=s_m).time()
    et = pd.Timestamp(0).replace(hour=e_h, minute=e_m).time()
    mask = pd.Series([(t >= st) and (t <= et) for t in (idx.tz_convert(tz).time if tz else idx.time)], index=index)
    return mask & index.weekday.isin(weekdays)

mask = session_mask(df.index)
df_sess = df[mask].copy()

zz = ZIGZAG.run(df_sess["High"], df_sess["Low"], upper=0.005, lower=0.005)
ms = MARKET_STRUCTURE_ZZ.run(
    df_sess["Close"],
    is_top=zz.is_top, is_bottom=zz.is_bottom,
    swing_high=zz.swing_high, swing_low=zz.swing_low
)
fvg = FAIR_VALUE_GAP.run(df_sess["High"], df_sess["Low"], df_sess["Close"], min_size=0.0015, unit="pct")

eL, xL, eS, xS = combo_smart_money(
    df_sess["Close"],
    bos_up=ms.bos_up, bos_down=ms.bos_down,
    choch_up=ms.choch_up, choch_down=ms.choch_down,
    fvg_up_lower=fvg.fvg_up_lower, fvg_up_upper=fvg.fvg_up_upper,
    fvg_dn_lower=fvg.fvg_dn_lower, fvg_dn_upper=fvg.fvg_dn_upper,
    confirm_shift=1
)

force_exit, *_ = smc_targets_stops(
    df_sess["Close"], eL, eS,
    swing_high=ms.swing_high, swing_low=ms.swing_low,
    fvg_up_lower=fvg.fvg_up_lower, fvg_dn_upper=fvg.fvg_dn_upper,
    rr_min=1.5, buffer_pct=0.0005
)

pf = vbt.Portfolio.from_signals(
    close=df_sess["Close"],
    entries=eL, exits=(xL | force_exit),
    short_entries=eS, short_exits=(xS | force_exit),
    fees=0.0005, slippage=0.0005,
    init_cash=100_000, freq="5min",
    cash_sharing=True,
)
print(pf.stats())
