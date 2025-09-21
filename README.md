# vbt-extensions

A collection of custom indicators and tools for [vectorbt](https://github.com/polakowo/vectorbt), including a fast ZigZag indicator with optional numba acceleration.

## Features

- **ZigZag Indicator**: Efficient implementation for trend detection, with optional numba JIT for speed.
- Designed for easy integration with vectorbt’s `IndicatorFactory`.
- Extensible structure for adding more indicators.

## Installation

```bash
pip install vbt-extensions
```
Or, to install directly from GitHub:
```bash
pip install git+https://github.com/gabrielbioinfo/vbt_extensions.git
```
Or, if using `uv`:
```bash
uv pip install vbt-extensions
```

## Requirements

- Python >= 3.12
- numpy
- pandas
- vectorbt
- numba (optional, for acceleration)

## Usage

### Basic Example
```python
import numpy as np
from vbt_extensions.ind.zigzag import ZIGZAG

high = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1], dtype=np.float64)
low = np.array([0.5, 1, 2, 1, 0.5, 1, 2, 1, 0.5], dtype=np.float64)

zigzag, is_top, is_bottom = ZIGZAG.run(high, low, upper=0.02, lower=0.02).values
print(zigzag)
```

### Integration with vectorbt
```python
import vectorbt as vbt
from vbt_extensions.ind import ZIGZAG

# Assuming df is a DataFrame with lower-case column names
high = df['high']; low = df['low']

zz = ZIGZAG.run(high=high, low=low, upper=0.03, lower=0.03)
zig  = zz.zigzag      # prices at pivots (Series/DataFrame)
tops = zz.is_top.astype(bool)
bots = zz.is_bottom.astype(bool)

# Example signals (for illustration):
entries = bots  # buy at bottoms
exits   = tops  # sell at tops

pf = vbt.Portfolio.from_signals(close=df['close'], entries=entries, exits=exits, freq='1h')
pf.stats()
```

## Testing

Unit tests are provided in the `tests/` directory. To run tests:

```bash
pytest
```

## Contributing

Feel free to open issues or submit pull requests for new indicators or improvements!

## References & Inspiration

- Neurotrader YouTube channel: [https://www.youtube.com/@neurotrader888/videos](https://www.youtube.com/@neurotrader888/videos)
- Neurotrader Tops and Bottoms video: [https://www.youtube.com/watch?v=X31hyMhB-3s](https://www.youtube.com/watch?v=X31hyMhB-3s)
- Neurotrader Market Profile Supports and Resistances video: [https://www.youtube.com/watch?v=mNWPSFOVoYA](https://www.youtube.com/watch?v=mNWPSFOVoYA)
- Academic article: "Flexible Time Series Pattern Matching Based on Perceptually Important Points" by Fu Lai Korris Chung, T.C. Fu, Wing Pong Robert Luk, Vincent To Yee Ng ([link](https://research.polyu.edu.hk/en/publications/flexible-time-series-pattern-matching-based-on-perceptually-impor))

## License

MIT
