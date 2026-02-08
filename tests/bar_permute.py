"""
Bar permutation for Monte Carlo Permutation Testing.
Adapted from neurotrader888/mcpt (MIT License).

Shuffles OHLC bars in log-space, preserving single-bar statistics
(volatility, skew, kurtosis) but destroying temporal patterns.
"""

import numpy as np
import pandas as pd
from typing import Optional


def get_permutation(
    ohlc: pd.DataFrame,
    start_index: int = 0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a permuted version of an OHLC DataFrame.

    Shuffles bars after start_index in log-space:
      - Intra-bar relative values (high, low, close relative to open) are shuffled together
      - Inter-bar gaps (open relative to previous close) are shuffled separately
    This preserves single-bar statistics but destroys temporal dependencies.

    Args:
        ohlc: DataFrame with columns 'Close', 'Open', 'High', 'Low' (yfinance format)
              OR 'close', 'open', 'high', 'low' (lowercase format). Both accepted.
        start_index: Bars before this index are kept unchanged. Bars from
                     start_index+1 onward are permuted.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with same index and columns as input, containing permuted OHLC data.
    """
    assert start_index >= 0

    np.random.seed(seed)

    # Detect and normalize column names
    if 'Close' in ohlc.columns:
        col_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
        reverse_map = {v: k for k, v in col_map.items()}
        working_df = ohlc.rename(columns=col_map)
        capitalize_output = True
    else:
        working_df = ohlc
        capitalize_output = False

    n_bars = len(working_df)
    time_index = working_df.index

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    if perm_n <= 0:
        # Nothing to permute
        return ohlc.copy()

    log_bars = np.log(working_df[['open', 'high', 'low', 'close']])

    # Get start bar
    start_bar = log_bars.iloc[start_index].to_numpy()

    # Open relative to last close (gap between bars)
    r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()

    # Prices relative to this bar's open (intra-bar structure)
    r_h = (log_bars['high'] - log_bars['open']).to_numpy()
    r_l = (log_bars['low'] - log_bars['open']).to_numpy()
    r_c = (log_bars['close'] - log_bars['open']).to_numpy()

    # Slice to only the permutable portion
    relative_open = r_o[perm_index:]
    relative_high = r_h[perm_index:]
    relative_low = r_l[perm_index:]
    relative_close = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intra-bar relative values (high/low/close) together
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[perm1]
    relative_low = relative_low[perm1]
    relative_close = relative_close[perm1]

    # Shuffle inter-bar gaps separately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[perm2]

    # Reconstruct permuted OHLC from relative prices
    perm_bars = np.zeros((n_bars, 4))

    # Copy real data before start_index
    perm_bars[:start_index] = log_bars.iloc[:start_index].to_numpy()

    # Copy start bar
    perm_bars[start_index] = start_bar

    for i in range(perm_index, n_bars):
        k = i - perm_index
        perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[k]   # open
        perm_bars[i, 1] = perm_bars[i, 0] + relative_high[k]       # high
        perm_bars[i, 2] = perm_bars[i, 0] + relative_low[k]        # low
        perm_bars[i, 3] = perm_bars[i, 0] + relative_close[k]      # close

    # Convert back from log-space
    perm_bars = np.exp(perm_bars)

    perm_df = pd.DataFrame(
        perm_bars,
        index=time_index,
        columns=['open', 'high', 'low', 'close'],
    )

    if capitalize_output:
        perm_df = perm_df.rename(columns=reverse_map)

    return perm_df
