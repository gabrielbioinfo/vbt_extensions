
import numpy as np
import pandas as pd
import vectorbt as vbt
from binance.client import Client

from vbt_extensions.data.binance import BinanceDownloadParams, binance_download

client = Client()
params = BinanceDownloadParams(
    client=client, symbol="BTCUSDT",
    interval="1h", start="90 days ago UTC", tz="UTC",
    fill_gaps=True, drop_partial_last=True,
)
df = binance_download(params)
print("Dados Binance carregados:", df.shape, df.index.min(), "->", df.index.max())
print(binance_download)
