# data/alignment.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List

from .binance_loader import load_binance_data


def load_deribit_npz_as_df(directory: str) -> pd.DataFrame:
    """
    Charge tous les .npz de surfaces IV dans un DataFrame indexé par timestamp (datetime).
    """
    files = sorted(
        f for f in os.listdir(directory) if f.endswith(".npz")
    )
    rows = []
    for f in files:
        full = os.path.join(directory, f)
        d = np.load(full)
        ts = int(d["timestamp"])
        dt = pd.to_datetime(ts, unit="s")
        rows.append({"timestamp": dt, "file": full})
    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


def align_binance_deribit(
    binance_df: pd.DataFrame,
    deribit_index_df: pd.DataFrame,
    tolerance: str = "5min",
) -> pd.DataFrame:
    """
    Alignement temporel via merge_asof:
      - binance_df: OHLCV Binance indexé par datetime
      - deribit_index_df: DataFrame avec colonne "file", index datetime
    Retourne un DataFrame avec colonne 'iv_file' alignée.
    """
    binance_df = binance_df.sort_index()
    deribit_index_df = deribit_index_df.sort_index()
    deribit_index_df = deribit_index_df.rename(columns={"file": "iv_file"})

    aligned = pd.merge_asof(
        binance_df,
        deribit_index_df,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    return aligned
