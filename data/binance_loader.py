# data/binance_loader.py
from __future__ import annotations
import ccxt
import pandas as pd
import numpy as np
from typing import Tuple


def _to_ts_ms(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def fetch_binance_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    limit: int = 10_000,
    futures: bool = False,
) -> pd.DataFrame:
    """
    Récupère les OHLCV sur Binance (spot ou futures perp).
    """
    if futures:
        exchange = ccxt.binanceusdm()
    else:
        exchange = ccxt.binance()

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = _to_ts_ms(df)
    return df


def fetch_binance_funding(
    symbol: str = "BTCUSDT",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Funding rate Binance (USDT-margined futures).
    ccxt: fetch_funding_rate_history
    """
    exchange = ccxt.binanceusdm()
    res = exchange.fetch_funding_rate_history(
        symbol=symbol, limit=limit
    )
    df = pd.DataFrame(res)
    df = _to_ts_ms(df.rename(columns={"fundingRate": "funding_rate"}))
    return df[["funding_rate"]]


def align_series(
    *dfs: pd.DataFrame, how: str = "inner"
) -> pd.DataFrame:
    """
    Aligne plusieurs DataFrames sur la même timeline.
    Suppose un index datetime.
    """
    base = dfs[0]
    for d in dfs[1:]:
        base = base.join(d, how=how)
    return base


def load_binance_data(
    spot_symbol: str = "BTC/USDT",
    futures_symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    limit: int = 10_000,
) -> pd.DataFrame:
    """
    Retourne un DataFrame unique avec:
      - close_spot
      - close_fut
      - volume_spot
      - funding_rate
    """
    spot = fetch_binance_ohlcv(spot_symbol, timeframe=timeframe, limit=limit, futures=False)
    fut = fetch_binance_ohlcv(spot_symbol, timeframe=timeframe, limit=limit, futures=True)
    fut = fut.rename(columns={"close": "close_fut", "volume": "volume_fut"})

    funding = fetch_binance_funding(futures_symbol, limit=limit)

    df = align_series(spot, fut[["close_fut"]], funding)
    df = df.rename(columns={"close": "close_spot", "volume": "volume_spot"})
    return df
