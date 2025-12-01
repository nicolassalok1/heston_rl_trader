# data/real_market_loader.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .binance_loader import load_binance_data
from .deribit_loader import build_iv_surface


@dataclass
class RealMarketData:
    btc_prices: np.ndarray
    btc_fut_prices: np.ndarray
    btc_funding: np.ndarray
    btc_oi: np.ndarray            # placeholder (tu peux le remplir plus tard)
    btc_iv_surface: np.ndarray
    k_grid: np.ndarray
    t_grid: np.ndarray

    shit_prices: np.ndarray
    shit_volumes: np.ndarray
    shit_funding: np.ndarray


def load_real_market_data(
    btc_spot_symbol: str = "BTC/USDT",
    btc_fut_symbol: str = "BTCUSDT",
    shit_spot_symbol: str = "DOGE/USDT",
    shit_fut_symbol: str = "DOGEUSDT",
    timeframe: str = "1m",
    limit: int = 2000,
) -> RealMarketData:
    # BTC
    btc_df = load_binance_data(
        spot_symbol=btc_spot_symbol,
        futures_symbol=btc_fut_symbol,
        timeframe=timeframe,
        limit=limit,
    )

    btc_prices = btc_df["close_spot"].values.astype(np.float32)
    btc_fut_prices = btc_df["close_fut"].fillna(btc_df["close_spot"]).values.astype(np.float32)
    btc_funding = btc_df["funding_rate"].fillna(0.0).values.astype(np.float32)
    btc_oi = np.ones_like(btc_prices, dtype=np.float32)  # placeholder

    # Shitcoin
    shit_df = load_binance_data(
        spot_symbol=shit_spot_symbol,
        futures_symbol=shit_fut_symbol,
        timeframe=timeframe,
        limit=limit,
    )

    shit_prices = shit_df["close_spot"].values.astype(np.float32)
    shit_volumes = shit_df["volume_spot"].values.astype(np.float32)
    shit_funding = shit_df["funding_rate"].fillna(0.0).values.astype(np.float32)

    # IV surface BTC: on prend UNE surface statique pour simplifier
    iv_surface, k_grid, t_grid = build_iv_surface(currency="BTC")
    # on la répète sur toute la série (à toi de la rendre time-varying après)
    btc_iv_surface = np.repeat(iv_surface[None, :, :], len(btc_prices), axis=0)

    return RealMarketData(
        btc_prices=btc_prices,
        btc_fut_prices=btc_fut_prices,
        btc_funding=btc_funding,
        btc_oi=btc_oi,
        btc_iv_surface=btc_iv_surface,
        k_grid=k_grid,
        t_grid=t_grid,
        shit_prices=shit_prices,
        shit_volumes=shit_volumes,
        shit_funding=shit_funding,
    )
