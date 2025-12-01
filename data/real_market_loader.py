# data/real_market_loader.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import pandas as pd

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

    if "iv_file" in btc_df.columns:
        btc_iv_surface, k_grid, t_grid = _load_surfaces_from_iv_files(btc_df["iv_file"])
    else:
        iv_surface, k_grid, t_grid = build_iv_surface(currency="BTC")
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


def _load_surfaces_from_iv_files(iv_files: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge une surface IV par timestamp en lisant la colonne iv_file (chemins .npz).
    Tous les fichiers doivent partager le même k_grid / t_grid. Si un fichier est
    manquant ou vide, on réutilise la dernière surface chargée.
    """
    surfaces = []
    k_grid = None
    t_grid = None
    last_surface = None

    for f in iv_files:
        if pd.isna(f) or not f:
            if last_surface is None:
                raise ValueError("iv_file manquant et aucune surface précédente disponible.")
            surfaces.append(last_surface.copy())
            continue

        path = Path(str(f))
        if not path.exists():
            raise FileNotFoundError(f"Fichier IV introuvable: {path}")

        with np.load(path) as data:
            iv_surface = data["iv_surface"].astype(np.float32)
            k = data["k_grid"].astype(np.float32)
            t = data["t_grid"].astype(np.float32)

        if k_grid is None:
            k_grid = k
            t_grid = t
        else:
            if k.shape != k_grid.shape or t.shape != t_grid.shape:
                raise ValueError(f"k_grid/t_grid incompatibles dans {path}")

        last_surface = iv_surface
        surfaces.append(iv_surface)

    btc_iv_surface = np.stack(surfaces, axis=0)
    return btc_iv_surface, k_grid, t_grid
