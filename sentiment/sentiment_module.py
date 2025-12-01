# sentiment/sentiment_module.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class SentimentSnapshot:
    sentiment_score: float
    msg_rate: float
    fear_greed: float


class SentimentProvider:
    """
    Squelette:
      - à connecter à tes vrais flux:
         - Telegram (nombre de messages, mots clés)
         - Twitter (vader/BERT score)
         - Fear & Greed index (crypto)
    """

    def __init__(self):
        pass

    def fetch_snapshot(self) -> SentimentSnapshot:
        # TODO: remplacer par du vrai
        return SentimentSnapshot(
            sentiment_score=0.0,
            msg_rate=0.0,
            fear_greed=0.5,
        )

    def to_feature_dict(self, snap: SentimentSnapshot) -> Dict[str, float]:
        return {
            "sentiment_score": float(snap.sentiment_score),
            "msg_rate": float(snap.msg_rate),
            "fear_greed": float(snap.fear_greed),
        }
