from __future__ import annotations
import time
from typing import Dict, Any

import numpy as np
import torch

from models.heston_inverse_model import load_heston_inverse_model
from features.feature_engine import create_default_feature_engine
from features.state_builder import StateBuilder
from rl.ppo_agent import PPOAgent, PPOConfig

# Ce module est un SQUELETTE.
# TODO: remplacer les parties "fetch_*" et "send_order" par des appels réels aux APIs.


class LiveDataFeed:
    def __init__(self):
        pass

    def fetch_latest_context(self) -> Dict[str, Any]:
        # TODO: implémenter:
        #  - récupération spot/futures/options/sentiment etc.
        raise NotImplementedError


class LiveExecutionEngine:
    def __init__(self):
        pass

    def send_order(self, side: str, size: float, price: float | None = None):
        # TODO: implémenter l'envoi d'ordres (Binance/Bybit/Deribit)
        raise NotImplementedError


def run_live_trading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    btc_model = load_heston_inverse_model(nk=5, nt=4, ckpt_path=None, device=device)
    shit_model = load_heston_inverse_model(nk=3, nt=4, ckpt_path=None, device=device)

    fe = create_default_feature_engine(
        shitcoin_heston_inverse=shit_model,
        btc_heston_inverse=btc_model,
        device=device,
    )

    dummy_ctx = {
        "shitcoin": {"prices": np.ones(60), "volumes": np.ones(60), "funding": np.zeros(60)},
        "btc": {
            "prices": np.ones(100),
            "idx": 99,
            "future_price": 1.0,
            "funding_rate": 0.0,
            "open_interest": 1.0,
            "iv_surface": np.ones((5, 4)),
            "k_grid": np.linspace(-0.4, 0.4, 5),
            "t_grid": np.array([0.05, 0.25, 0.5, 1.0]),
        },
        "sentiment": {
            "sentiment_score": 0.0,
            "msg_rate": 0.0,
            "fear_greed": 0.0,
        },
        "generic": {
            "close": 1.0,
            "high": 1.0,
            "low": 1.0,
            "volume": 1.0,
        },
    }
    dummy_vec, _ = fe.compute_features(dummy_ctx)
    dim = dummy_vec.shape[0]

    sb = StateBuilder(dim=dim, window=16, clip_value=5.0, training=False)

    obs_dim = dim * 16
    ppo_cfg = PPOConfig()
    agent = PPOAgent(obs_dim=obs_dim, cfg=ppo_cfg)
    agent.pi_net.eval()

    feed = LiveDataFeed()
    exec_engine = LiveExecutionEngine()

    position = 0.0

    while True:
        ctx = feed.fetch_latest_context()
        feat_vec, _ = fe.compute_features(ctx)
        state = sb.build_state(feat_vec)
        obs = state.reshape(-1).astype(np.float32)

        action, logp, mu = agent.act(obs)
        target_exposure = float(np.clip(action[0], -1.0, 1.0))

        print(f"Target exposure: {target_exposure:.3f}")

        # TODO: calculer taille d'ordre, envoyer via exec_engine.send_order(...)
        time.sleep(1.0)
