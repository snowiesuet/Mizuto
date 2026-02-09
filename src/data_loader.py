"""Load NQ futures CSV data from the data/ directory.

CSV format: semicolon-delimited, no header.
Columns: date;time;open;high;low;close;volume

Date formats:
  - 1d / 5m files: YYYY-MM-DD  (e.g. 2024-01-02)
  - 1h file:       DD/MM/YYYY  (e.g. 01/01/2024)
"""

import os
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

COLUMN_NAMES = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]


def load_nq_csv(filepath, date_format=None):
    """Load a single NQ CSV file into a DataFrame.

    Args:
        filepath: Path to the CSV file.
        date_format: strptime format for the date column.
                     Auto-detected if None (DD/MM/YYYY vs YYYY-MM-DD).

    Returns:
        DataFrame with DatetimeIndex and columns Open/High/Low/Close/Volume.
    """
    df = pd.read_csv(filepath, sep=";", header=None, names=COLUMN_NAMES)

    # Auto-detect date format from first row if not specified
    if date_format is None:
        sample = str(df["Date"].iloc[0])
        if "/" in sample:
            date_format = "%d/%m/%Y"
        else:
            date_format = "%Y-%m-%d"

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format=f"{date_format} %H:%M:%S"
    )
    df = df.set_index("Datetime").drop(columns=["Date", "Time"])
    df = df.sort_index()

    # Ensure numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_nq(timeframe="1d", years=None):
    """Load NQ data for a given timeframe, optionally filtering by year(s).

    Args:
        timeframe: One of '1d', '1h', '5m'.
        years: List of year ints to include (e.g. [2023, 2024]).
               None loads all available files. Ignored for '1h' (single file).

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    tf_dir = os.path.join(DATA_DIR, "nq", timeframe)

    if not os.path.isdir(tf_dir):
        raise FileNotFoundError(f"Data directory not found: {tf_dir}")

    if timeframe == "1h":
        # Single file for 1h
        files = glob.glob(os.path.join(tf_dir, "*.csv"))
    else:
        # Multiple yearly files for 1d / 5m
        files = sorted(glob.glob(os.path.join(tf_dir, "*.csv")))
        if years:
            files = [f for f in files if any(str(y) in os.path.basename(f) for y in years)]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {tf_dir}")

    dfs = [load_nq_csv(f) for f in files]
    df = pd.concat(dfs).sort_index()

    # Drop any duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    return df
