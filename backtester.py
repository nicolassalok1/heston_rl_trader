from __future__ import annotations
import numpy as np


def compute_pnl_stats(equity_curve: np.ndarray, rf: float = 0.0, dt: float = 1.0):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    pnl = equity_curve[-1] - equity_curve[0]

    avg_ret = returns.mean()
    vol_ret = returns.std() + 1e-8
    sharpe = (avg_ret - rf * dt) / vol_ret

    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax
    max_dd = drawdowns.min()

    return {
        "pnl": float(pnl),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_ret": float(avg_ret),
        "vol_ret": float(vol_ret),
    }
