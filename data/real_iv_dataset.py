# data/real_iv_dataset.py

from __future__ import annotations
import os
import numpy as np
from typing import List, Dict
import torch
from torch.utils.data import Dataset


class RealIvSurfaceDataset(Dataset):
    """
    Dataset pour surfaces IV Deribit déjà sauvegardées sur disque sous forme .npz
    Chaque fichier doit contenir:
        iv_surface: [NK, NT]
        k_grid: [NK]
        t_grid: [NT]
        spot: float
        timestamp: float ou int
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.files: List[str] = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".npz")
            ]
        )
        if len(self.files) == 0:
            raise ValueError(f"Aucun fichier .npz trouvé dans {directory}")

        # 1er fichier pour déterminer NK et NT
        sample = np.load(self.files[0])
        self.NK = sample["iv_surface"].shape[0]
        self.NT = sample["iv_surface"].shape[1]

        # Normalisation globale de la surface
        # (on effectue un pass préliminaire)
        all_vals = []
        for fp in self.files:
            d = np.load(fp)
            surf = d["iv_surface"].astype(np.float32)
            all_vals.append(surf.reshape(-1))
        all_vals = np.concatenate(all_vals, axis=0)

        self.iv_mean = float(all_vals.mean())
        self.iv_std = float(all_vals.std() + 1e-8)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        d = np.load(fp)

        iv = d["iv_surface"].astype(np.float32)             # [NK, NT]
        k_grid = d["k_grid"].astype(np.float32)
        t_grid = d["t_grid"].astype(np.float32)
        spot = float(d["spot"])

        # Normalisation de la surface
        iv_norm = (iv - self.iv_mean) / self.iv_std

        x = torch.from_numpy(iv_norm).unsqueeze(0)  # [1,NK,NT]
        y_dummy = torch.zeros(5, dtype=torch.float32)  # pas de label vrai → self-supervised
        return x, y_dummy
