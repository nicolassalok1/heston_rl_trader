# data/deribit_loader.py
from __future__ import annotations
import requests
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path
from datetime import datetime, timezone


DERIBIT_API = "https://www.deribit.com/api/v2"


def _get_json(endpoint: str, params: Dict) -> Dict:
    url = f"{DERIBIT_API}{endpoint}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["result"]


def fetch_deribit_instruments(currency: str = "BTC", kind: str = "option") -> List[Dict]:
    res = _get_json("/public/get_instruments", {
        "currency": currency,
        "kind": kind,
        "expired": False,
    })
    return res


def fetch_orderbook(instrument_name: str) -> Dict:
    return _get_json("/public/get_order_book", {
        "instrument_name": instrument_name
    })


def build_iv_surface(
    currency: str = "BTC",
    maturities_max: int = 4,
    strikes_per_maturity: int = 5,
    save: bool = True,
    output_dir: str = "data/deribit_surfaces",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit une surface IV brutale (exemple) :
      - sélectionne quelques maturités
      - quelques strikes autour de l'ATM
    Retourne:
      iv_surface [NK, NT]
      k_grid (log-moneyness approximatif)
      t_grid (en années)
    """
    instruments = fetch_deribit_instruments(currency=currency, kind="option")
    df = pd.DataFrame(instruments)

    df = df[df["option_type"] == "call"]
    df["expiration"] = pd.to_datetime(df["expiration_timestamp"], unit="ms")
    now = df["creation_timestamp"].max()
    now = pd.to_datetime(now, unit="ms")

    df["T"] = (df["expiration"] - now).dt.total_seconds() / (365.0 * 24 * 3600)

    maturities = sorted(df["T"].unique())[:maturities_max]
    t_grid = np.array(maturities, dtype=np.float32)
    rows = []

    for T in t_grid:
        df_t = df[np.isclose(df["T"], T)]
        if df_t.empty:
            continue
        atm_idx = (df_t["strike"] - df_t["strike"].median()).abs().idxmin()
        atm_strike = df_t.loc[atm_idx, "strike"]
        df_t["k"] = np.log(df_t["strike"] / atm_strike)
        df_t = df_t.sort_values("k")
        rows.append(df_t)

    k_vals = sorted(set(np.round(r["k"], 4).tolist() for r in [x["k"] for x in [row for row in rows]]))
    # simplification: on prendra un set fixe de strikes autour de 0

    # pour garder simple: on se contente de strikes_per_maturity centrés sur k=0
    k_grid = np.linspace(-0.4, 0.4, strikes_per_maturity, dtype=np.float32)

    NK = len(k_grid)
    NT = len(t_grid)
    iv_surface = np.zeros((NK, NT), dtype=np.float32)

    spot_price = None
    for j, T in enumerate(t_grid):
        df_t = rows[j]
        if df_t.empty:
            continue
        for i, k in enumerate(k_grid):
            idx = (df_t["k"] - k).abs().idxmin()
            inst = df_t.loc[idx, "instrument_name"]
            ob = fetch_orderbook(inst)
            if spot_price is None:
                spot_price = ob.get("underlying_price") or ob.get("index_price") or ob.get("mark_price")
            iv = ob.get("mark_iv", None)
            if iv is None:
                iv = 0.0
            iv_surface[i, j] = float(iv) / 100.0  # Deribit donne en %

    if spot_price is None:
        spot_price = 0.0
    spot_price = float(spot_price)

    if save:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        _save_surface_npz(
            iv_surface=iv_surface,
            k_grid=k_grid,
            t_grid=t_grid,
            spot=spot_price,
            timestamp=ts,
            output_dir=output_dir,
        )
    return iv_surface, k_grid, t_grid


#C’est volontairement simplifié. Pour de la prod, tu feras un cache local ou un loader batch.


def _save_surface_npz(
    iv_surface: np.ndarray,
    k_grid: np.ndarray,
    t_grid: np.ndarray,
    spot: float,
    timestamp: str,
    output_dir: str,
) -> str:
    """
    Sauvegarde une surface Deribit dans data/deribit_surfaces/ avec la structure demandée.
    Le timestamp est conservé tel quel dans le fichier; on remplace les ':' du nom de fichier
    pour rester compatible Windows.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{timestamp.replace(':', '-')}.npz"
    path = out_dir / fname
    np.savez(
        path,
        iv_surface=iv_surface,
        k_grid=k_grid,
        t_grid=t_grid,
        spot=float(spot),
        timestamp=timestamp,
    )
    return str(path)
