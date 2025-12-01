# rl/reward.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    pnl_scale: float = 1.0
    turnover_penalty: float = 0.001     # pénalité sur le notional échangé
    drawdown_penalty: float = 0.1       # poids sur drawdown instantané
    leverage_penalty: float = 0.05      # pénalité sur |exposure| élevé
    target_vol: float = 0.02            # vol cible par step (approx)
    vol_penalty: float = 0.05


class RewardEngine:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.equity_history = []

    def reset(self):
        self.equity_history = []

    def compute_reward(
        self,
        equity_prev: float,
        equity_now: float,
        position_prev: float,
        position_now: float,
        price: float,
    ) -> float:
        pnl = equity_now - equity_prev
        rel_pnl = pnl / max(equity_prev, 1e-6)

        self.equity_history.append(equity_now)
        if len(self.equity_history) < 5:
            inst_dd = 0.0
        else:
            eq = np.array(self.equity_history, dtype=np.float32)
            cummax = np.maximum.accumulate(eq)
            dd = (eq - cummax) / cummax
            inst_dd = float(dd[-1])

        turnover = abs(position_now - position_prev) * price
        lev = abs(position_now * price / max(equity_now, 1e-6))

        reward = 0.0
        reward += self.cfg.pnl_scale * rel_pnl
        reward -= self.cfg.turnover_penalty * turnover
        reward -= self.cfg.drawdown_penalty * max(0.0, -inst_dd)
        reward -= self.cfg.leverage_penalty * lev

        return float(reward)
