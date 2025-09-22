# vbt-extensions

**Extensões para [vectorbt](https://github.com/polakowo/vectorbt)** em Python, com foco em **indicadores custom**, **sinais derivados**, **gestão de risco** e **visualização**.  
Tudo é **pandas-first** (Series/DataFrames), para encaixar direto no ecossistema do `vectorbt`.

---

## 📑 Sumário

- [✨ Features principais](#-features-principais)  
- [📂 Estrutura do pacote](#-estrutura-do-pacote)  
- [🚀 Instalação](#-instalação)  
- [🔧 Exemplos rápidos](#-exemplos-rápidos)  
- [🧩 Exemplo completo](#-exemplo-completo)  
- [📊 Roadmap](#-roadmap)  
- [📚 Referências & Inspiração](#-referências--inspiração)  
- [📝 License](#-license)

---

## ✨ Features principais

- Indicadores não cobertos por TA-Lib (ex.: **ZigZag**, **Trend Lines**, **Head & Shoulders**, **Perceptually Important Points**).
- Sinais de trading derivados de indicadores (**Golden Cross**, **Breakout de linhas de tendência**, **Fibo retracements**).
- Overlays de **gestão de risco** (stops, sizing, filtros de sessão, etc.).
- Helpers de **plotting** para price + overlays (zonas fibo, regimes, trend lines).
- Estrutura modular para expandir (`data/`, `ind/`, `sig/`, `risk/`, `plotting/`, `metrics/`, `analyzers/`).

---

## 📂 Estrutura do pacote

```
vbt_extensions/
├── data/          # fontes de dados padronizadas (ex.: binance)
├── ind/           # indicadores pandas-first
├── sig/           # geração de sinais (a partir de indicadores)
├── risk/          # overlays de gestão de risco
├── plotting/      # helpers de visualização
├── metrics/       # métricas e relatórios (tearsheet, stats)
├── analyzers/     # analisadores customizados
└── __init__.py
```

### `data/`
- `base.py` → classe base para loaders, utilitários (gap fill, resample, casting).
- `binance.py` → loader OHLCV via `vectorbt.BinanceData` + ajustes (timezone, gaps).

### `ind/`
- `zigzag.py` → pivôs rápidos (tops/bottoms).
- `trend_line.py` → suporte/resistência por regressão ou pivôs.
- `head_and_shoulders.py` → padrão H&S / inverted H&S.
- `perceptually_important.py` → PIP simplificado (feature extraction).
- `retracement_ratios.py` → níveis de Fibonacci por swing.

### `sig/`
- `golden_cross_sig.py` → sinais de cruzamento de médias (trend/reversion).
- `random_sig.py` → sinais aleatórios (testes de risco).
- `trend_line_sig.py` → breakout/touch/reversal com linhas de tendência.
- `retracement_sig.py` → sinais de breakout/reversão em níveis fibo.

### `risk/`
- `breakeven.py` → stop automático em break-even.
- `fibo_stop.py` → stop baseado em níveis fibo.
- `stop_trailing.py` → trailing stops.
- `time_stop.py` → stop temporal.
- `session_filters.py` → filtros por horário/sessão.
- `position_sizing.py` → fixed fractional sizing.

### `plotting/`
- `helpers.py` → funções para plotar preço, bandas, trend lines, regimes e zonas fibo.

### `metrics/`
- `tearsheet.py` → geração de relatórios de performance (stats + plots).

### `analyzers/`
- `qindex_rank.py` → ranking customizado de portfólios/estratégias.

---

## 🚀 Instalação

```bash
pip install vbt-extensions
# ou direto do GitHub
pip install git+https://github.com/gabrielbioinfo/vbt_extensions.git
# via uv
uv pip install vbt-extensions
```

Requisitos: Python ≥3.12, `numpy`, `pandas`, `vectorbt`.  
`numba` é opcional (mas recomendado).

---

## 🔧 Exemplos rápidos

### 📊 Data (Binance OHLCV)
```python
from vbt_extensions.data import BinanceDownloadParams, binance_download
from binance.client import Client

client = Client()
params = BinanceDownloadParams(client=client, ticker="BTCUSDT", interval="1h", start="1 month ago UTC")
df = binance_download(params)
print(df.head())
```

### 📈 Indicadores (ind)

#### ZigZag
```python
from vbt_extensions.ind import ZIGZAG
zz = ZIGZAG.run(df["High"], df["Low"], upper=0.03, lower=0.03)
print(zz.zigzag.dropna().tail())
```

#### Trend Line
```python
from vbt_extensions.ind import TREND_LINE
tl = TREND_LINE.run(df["High"], df["Low"], df["Close"], lookback=50)
print(tl.support_slope.tail())
```

#### Fibo Retracements
```python
from vbt_extensions.ind import FIB_RETRACEMENT
fib = FIB_RETRACEMENT.run(df["High"], df["Low"])
print(fib.levels.tail())
```

#### Perceptually Important Points (PIP)
```python
from vbt_extensions.ind import PIP
pip = PIP.run(df["Close"], n_pips=7, dist_measure=2)
print(pip.pips_x, pip.pips_y)
```

#### Head & Shoulders
```python
from vbt_extensions.ind import HEAD_AND_SHOULDERS
hs = HEAD_AND_SHOULDERS.run(df["Close"], order=6)
print(hs.pattern_type.value_counts())
```

### 🚦 Sinais (sig)

#### Golden Cross
```python
from vbt_extensions.sig import golden_cross_sig
entries, exits = golden_cross_sig(df["Close"], fast_list=[5, 10], slow_list=[20, 50])
```

#### Random Signals
```python
from vbt_extensions.sig import random_sig
entries, exits = random_sig(df["Close"], p=0.02, ncols=3, seed=42)
```

#### Trend Line Signals
```python
from vbt_extensions.sig import tl_breakout_sig
entries, exits = tl_breakout_sig(df["Close"], tl)
```

#### Fibo Signals
```python
from vbt_extensions.sig import fib_trend_continuation_sig
entries, exits = fib_trend_continuation_sig(df["Close"], fib)
```

### 🛡️ Risco (risk)

```python
from vbt_extensions.risk import trailing_stop, breakeven_stop, time_stop, fibo_protective_stop

# trailing stop de 5%
exits_ts = trailing_stop(df["Close"], entries, stop_pct=0.05)

# stop em breakeven
exits_be = breakeven_stop(df["Close"], entries, exits_ts)

# stop temporal (encerra após 10 barras)
exits_time = time_stop(df["Close"], entries, max_bars=10)

# stop em nível de fibo
exits_fibo = fibo_protective_stop(df["Close"], entries, fib)
```

### 🎨 Plotting (plotting)
```python
from vbt_extensions.plotting import plot_price, plot_trend_lines, plot_fib_zones

plot_price(df["Close"])
plot_trend_lines(df["Close"], tl)
plot_fib_zones(df["Close"], fib)
```

### 📑 Métricas (metrics)
```python
from vbt_extensions.metrics import full_tearsheet
from vectorbt import Portfolio

pf = Portfolio.from_signals(close=df["Close"], entries=entries, exits=exits, freq="1h")
full_tearsheet(pf)
```

### 🔍 Analisadores (analyzers)
```python
from vbt_extensions.analyzers import qindex_rank
ranks = qindex_rank([pf])  # recebe lista de portfólios
print(ranks)
```

---

## 🧩 Exemplo completo

Um fluxo mínimo de ponta a ponta:

```python
from binance.client import Client
from vectorbt import Portfolio

# 1. Data
from vbt_extensions.data import BinanceDownloadParams, binance_download
client = Client()
params = BinanceDownloadParams(client=client, ticker="BTCUSDT", interval="1h", start="3 months ago UTC")
df = binance_download(params)

# 2. Indicador
from vbt_extensions.ind import TREND_LINE
tl = TREND_LINE.run(df["High"], df["Low"], df["Close"], lookback=100)

# 3. Sinais
from vbt_extensions.sig import tl_breakout_sig
entries, exits = tl_breakout_sig(df["Close"], tl)

# 4. Backtest
pf = Portfolio.from_signals(close=df["Close"], entries=entries, exits=exits, freq="1h")

# 5. Relatório
from vbt_extensions.metrics import full_tearsheet
full_tearsheet(pf)
```

---

## 📊 Roadmap

- Expandir biblioteca de sinais (`sig/`) para padrões clássicos (H&S, triângulos, etc.).
- Adicionar módulos de otimização (grid, walk-forward, permutation tests).
- Melhorar docs com notebooks de exemplo (`/examples`).
- Integração com **Quant Lab** (ambiente maior para backtests, análises e portfólios).

---

## 📚 Referências & Inspiração

- Neurotrader YouTube: [@neurotrader888](https://www.youtube.com/@neurotrader888/videos)  
- Artigo: *Flexible Time Series Pattern Matching Based on Perceptually Important Points*  
- Clássicos: *Systematic Trading* (R. Carver), *Testing and Tuning Trading Systems* (T. Bandy), *Permutation and Randomization Tests for Trading System Development* (A. Aronson)

---

## 📝 License
MIT
