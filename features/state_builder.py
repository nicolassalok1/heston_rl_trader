from __future__ import annotations
import numpy as np


class RunningStats:
    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8


class Normalizer:
    def __init__(self, dim: int, clip_value: float = 5.0):
        self.stats = RunningStats(dim)
        self.clip_value = clip_value

    def normalize(self, x: np.ndarray, update_stats: bool = True) -> np.ndarray:
        if update_stats:
            self.stats.update(x)
        normed = (x - self.stats.mean) / self.stats.std
        np.clip(normed, -self.clip_value, self.clip_value, out=normed)
        return normed.astype(np.float32)


class StateBuffer:
    def __init__(self, window: int, dim: int):
        self.window = window
        self.dim = dim
        self.buffer = np.zeros((window, dim), dtype=np.float32)
        self.ptr = 0
        self.initialized = False

    def reset(self):
        self.buffer.fill(0.0)
        self.ptr = 0
        self.initialized = False

    def push(self, x: np.ndarray):
        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.window
        if not self.initialized and self.ptr == 0:
            self.initialized = True

    def get_state(self) -> np.ndarray:
        if not self.initialized:
            return self.buffer.copy()
        indices = np.arange(self.ptr, self.ptr + self.window) % self.window
        return self.buffer[indices].copy()


class StateBuilder:
    def __init__(
        self,
        dim: int,
        window: int = 16,
        clip_value: float = 5.0,
        training: bool = True,
    ):
        self.dim = dim
        self.window = window
        self.training = training
        self.normalizer = Normalizer(dim=dim, clip_value=clip_value)
        self.buffer = StateBuffer(window=window, dim=dim)

    def reset(self):
        self.buffer.reset()

    def build_state(self, feature_vec: np.ndarray) -> np.ndarray:
        feature_vec = feature_vec.astype(np.float32)
        normed = self.normalizer.normalize(feature_vec, update_stats=self.training)
        self.buffer.push(normed)
        state = self.buffer.get_state()
        return state
