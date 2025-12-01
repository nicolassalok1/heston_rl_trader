# data/deribit_scraper.py
from __future__ import annotations
import os
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict

DERIBIT_API = "https://www.deribit.com/api/v2"


def _get_json(endpoint: str, params: Dict) -> Dict:
    url = f"{DERIBIT_API}{endpoint}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["result"]


def fetch_instruments(currency: str = "BTC", kind: str = "option") -> pd.DataFrame:
    res = _get_json("/public/get_instruments", {
        "currency": currency,
        "kind": kind,
        "expired": False,
    })
    df = pd.DataFrame(res)
    return df


def fetch_orderbook(instrument_name: str) -> Dict:
    return _get_json("/public/get_order_book", {
        "instrument_name": instrument_name
    })


def snapshot_iv_surface(
    currency: str = "BTC",
    strikes_per_maturity: int = 5,
    max_maturities: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Construit une surface IV Deribit simplifiée à un instant t:
      - quelques maturités
      - quelques strikes autour de l'ATM
    Retourne:
      iv_surface [NK,NT], k_grid [NK], t_grid [NT], spot, timestamp
    """
    instruments = fetch_instruments(currency=currency, kind="option")
    df = instruments[instruments["option_type"] == "call"].copy()

    # timestamps
    now_ms = max(df["creation_timestamp"])
    now = pd.to_datetime(now_ms, unit="ms")

    df["expiration"] = pd.to_datetime(df["expiration_timestamp"], unit="ms")
    df["T"] = (df["expiration"] - now).dt.total_seconds() / (365.0 * 24 * 3600)

    # on garde quelques maturités
    maturities = sorted(df["T"].unique())[:max_maturities]
    if len(maturities) == 0:
        raise RuntimeError("Aucune maturité trouvée.")

    rows = []
    t_list = []
    for T in maturities:
        df_t = df[np.isclose(df["T"], T)]
        if df_t.empty:
            continue
        # approximatif : median strike comme proxy ATM
        atm_idx = (df_t["strike"] - df_t["strike"].median()).abs().idxmin()
        atm_strike = df_t.loc[atm_idx, "strike"]
        df_t["k"] = np.log(df_t["strike"] / atm_strike)
        df_t = df_t.sort_values("k")
        rows.append(df_t)
        t_list.append(T)

    if len(rows) == 0:
        raise RuntimeError("Impossible de construire les lignes par maturité.")

    t_grid = np.array(t_list, dtype=np.float32)
    # grille k : symétrique autour de 0
    NK = strikes_per_maturity
    k_grid = np.linspace(-0.4, 0.4, NK, dtype=np.float32)
    NT = len(t_grid)
    iv_surface = np.zeros((NK, NT), dtype=np.float32)

    # spot approximé = underlying_price d'un instrument
    spot = float(rows[0]["underlying_price"].iloc[0])

    for j, df_t in enumerate(rows):
        for i, k in enumerate(k_grid):
            idx = (df_t["k"] - k).abs().idxmin()
            inst = df_t.loc[idx, "instrument_name"]
            ob = fetch_orderbook(inst)
            iv = ob.get("mark_iv", None)
            if iv is None:
                iv = 0.0
            iv_surface[i, j] = float(iv) / 100.0  # en fraction

    ts_unix = int(time.time())
    return {
        "iv_surface": iv_surface,
        "k_grid": k_grid,
        "t_grid": t_grid,
        "spot": np.array(spot, dtype=np.float32),
        "timestamp": np.array(ts_unix, dtype=np.int64),
    }


def save_snapshot_npz(
    output_dir: str = "data/deribit_surfaces",
    currency: str = "BTC",
):
    os.makedirs(output_dir, exist_ok=True)
    snap = snapshot_iv_surface(currency=currency)
    ts = int(snap["timestamp"])
    fname = os.path.join(output_dir, f"{currency}_iv_{ts}.npz")
    np.savez(fname, **snap)
    print(f"[INFO] Saved IV snapshot to {fname}")


if __name__ == "__main__":
    save_snapshot_npz()
