from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

import gymnasium as gym
from gymnasium import spaces

from features.feature_engine import FeatureEngine
from features.state_builder import StateBuilder
from data.simulated_data import SimMarketData
from rl.reward import RewardEngine, RewardConfig
from sentiment.sentiment_module import SentimentProvider


@dataclass
class TradingEnvConfig:
    window_shitcoin: int = 60
    step_start: int = 100
    max_steps: int = 5000
    initial_capital: float = 1000.0
    transaction_cost: float = 0.0005


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        market: SimMarketData,
        feature_engine: FeatureEngine,
        state_builder: StateBuilder,
        config: TradingEnvConfig,
    ):
        super().__init__()
        self.market = market
        self.fe = feature_engine
        self.sb = state_builder
        self.cfg = config
        self.sentiment_provider = SentimentProvider()
        self.reward_engine = RewardEngine(RewardConfig())

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.obs_dim = None
        self.observation_space = None

        self.t = None
        self.done = False
        self.position = 0.0
        self.cash = self.cfg.initial_capital
        self.last_price = None

    def _build_context(self, t: int) -> Dict[str, Any]:
        w = self.cfg.window_shitcoin
        start = max(0, t - w + 1)
        end = t + 1

        shit_prices = self.market.shit_prices[start:end]
        shit_volumes = self.market.shit_volumes[start:end]
        shit_funding = self.market.shit_funding[start:end]

        btc_prices_hist = self.market.btc_prices[: t + 1]
        fut_price = self.market.btc_fut_prices[t]
        funding_rate = self.market.btc_funding[t]
        oi = self.market.btc_oi[t]
        iv_surface = self.market.btc_iv_surface[t]
        k_grid = self.market.k_grid
        t_grid = self.market.t_grid

        snap = self.sentiment_provider.fetch_snapshot()
        sent_ctx = self.sentiment_provider.to_feature_dict(snap)

        close_price = self.market.btc_prices[t]
        high_price = close_price
        low_price = close_price
        volume_last = 1.0

        ctx = {
            "shitcoin": {
                "prices": shit_prices,
                "volumes": shit_volumes,
                "funding": shit_funding,
            },
            "btc": {
                "prices": btc_prices_hist,
                "idx": t,
                "future_price": fut_price,
                "funding_rate": funding_rate,
                "open_interest": oi,
                "iv_surface": iv_surface,
                "k_grid": k_grid,
                "t_grid": t_grid,
            },
            "sentiment": sent_ctx,
            "generic": {
                "close": close_price,
                "high": high_price,
                "low": low_price,
                "volume": volume_last,
            },
        }
        return ctx

    def _get_state(self) -> np.ndarray:
        ctx = self._build_context(self.t)
        feat_vec, _ = self.fe.compute_features(ctx)
        state = self.sb.build_state(feat_vec)
        flat = state.reshape(-1).astype(np.float32)

        if self.obs_dim is None:
            self.obs_dim = flat.shape[0]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
        return flat

    def reset(self, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.sb.reset()
        self.done = False

        self.t = self.cfg.step_start
        self.position = 0.0
        self.cash = self.cfg.initial_capital
        self.last_price = float(self.market.btc_prices[self.t])
        self.reward_engine.reset()

        obs = self._get_state()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Env is done. Call reset().")

        a = float(np.clip(action[0], -1.0, 1.0))

        price_t = float(self.market.btc_prices[self.t])
        self.t += 1
        if self.t >= min(self.cfg.max_steps, len(self.market.btc_prices) - 1):
            self.done = True
        price_tp1 = float(self.market.btc_prices[self.t])

        position_prev = self.position
        equity = self.cash + self.position * price_t

        target_position_value = a * equity
        target_position = target_position_value / price_t

        trade_size = target_position - self.position
        cost = self.cfg.transaction_cost * abs(trade_size) * price_t

        self.cash = equity - target_position_value - cost
        self.position = target_position

        new_equity = self.cash + self.position * price_tp1
        reward = self.reward_engine.compute_reward(
            equity_prev=equity,
            equity_now=new_equity,
            position_prev=position_prev,
            position_now=target_position,
            price=price_t,
        )

        obs = self._get_state()
        info = {
            "equity": new_equity,
            "price": price_tp1,
            "position": self.position,
        }
        terminated = self.done
        truncated = False

        return obs, reward, terminated, truncated, info
