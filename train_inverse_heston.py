from __future__ import annotations
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.heston_inverse_model import InverseHestonModel


# ============================================================
# 1) Grille (k, T) fixe
# ============================================================

# Log-moneyness
K_POINTS = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float32)
# Maturités (années)
T_POINTS = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0], dtype=np.float32)

NK = len(K_POINTS)
NT = len(T_POINTS)


# ============================================================
# 2) Génération synthétique de paramètres Heston
# ============================================================

def sample_heston_params_batch(batch_size: int) -> np.ndarray:
    """
    Tire des paramètres Heston plausibles:
      kappa, theta, sigma, rho, v0
    """
    # kappa ~ LogU[0.5, 8]
    kappa = np.exp(np.random.uniform(np.log(0.5), np.log(8.0), size=batch_size))
    # theta ~ LogU[0.01, 0.08]
    theta = np.exp(np.random.uniform(np.log(0.01), np.log(0.08), size=batch_size))
    # sigma ~ LogU[0.1, 1.0]
    sigma = np.exp(np.random.uniform(np.log(0.1), np.log(1.0), size=batch_size))
    # rho ~ in [-0.95, 0.2]
    rho = np.random.beta(2.0, 2.0, size=batch_size)  # [0,1]
    rho = rho * (0.2 + 0.95) - 0.95                  # → [-0.95,0.2]
    # v0 ~ LogU[0.01, 0.08]
    v0 = np.exp(np.random.uniform(np.log(0.01), np.log(0.08), size=batch_size))

    params = np.stack([kappa, theta, sigma, rho, v0], axis=1).astype(np.float32)
    return params


# ============================================================
# 3) Générateur de surface Heston-like (cheap)
# ============================================================

def synthetic_heston_like_surface(
    params: np.ndarray,
    k_points: np.ndarray,
    t_points: np.ndarray,
) -> np.ndarray:
    """
    Générateur "Heston-like" rapide pour entraîner l'inverseur.
    Ce n'est PAS un vrai pricer Heston, mais une approximation qui
    reproduit la géométrie (term-structure + skew + smile).

    params = (kappa, theta, sigma, rho, v0)
    Retourne w(k,T) = iv^2 * T  (variance totale).
    """
    kappa, theta, sigma, rho, v0 = params

    KK, TT = np.meshgrid(k_points, t_points, indexing="ij")  # [NK,NT]

    # term-structure de variance instantanée (style CIR)
    var_t = np.maximum(theta + (v0 - theta) * np.exp(-kappa * TT), 1e-6)
    iv_level = np.sqrt(var_t)  # [NK,NT]

    # skew contrôlé par rho : plus négatif → plus de skew
    skew = rho * KK / np.sqrt(1.0 + TT)

    # smile quadratique contrôlé par sigma
    curvature = sigma * (KK**2)

    iv = np.maximum(iv_level + skew + curvature, 1e-4)
    w = iv**2 * TT  # variance totale
    return w.astype(np.float32)


# ============================================================
# 4) Dataset PyTorch
# ============================================================

class HestonInverseDataset(Dataset):
    """
    Dataset synthétique pour entraîner l'inverseur Heston.

    Chaque sample:
      - X: surface normalisée w_norm [1,NK,NT]
      - y: paramètres SCALÉS (z-score) [5]
      - w_true_norm: surface normalisée (pour reconstruction loss)
    """

    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples = n_samples

        # 1) Params
        params = sample_heston_params_batch(n_samples)  # [N,5]
        self.params = params

        # 2) Surfaces w(k,T)
        surfaces = []
        for i in range(n_samples):
            w = synthetic_heston_like_surface(
                self.params[i],
                K_POINTS,
                T_POINTS,
            )  # [NK,NT]
            surfaces.append(w)
        self.surfaces = np.stack(surfaces, axis=0)  # [N,NK,NT]

        # 3) Normalisation des surfaces globalement
        self.w_mean = self.surfaces.mean()
        self.w_std = self.surfaces.std() + 1e-8
        self.surfaces_norm = (self.surfaces - self.w_mean) / self.w_std

        # 4) Normalisation des paramètres (z-score)
        self.param_mean = self.params.mean(axis=0)
        self.param_std = self.params.std(axis=0) + 1e-8
        self.params_scaled = (self.params - self.param_mean) / self.param_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        w_norm = self.surfaces_norm[idx]          # [NK,NT]
        y_scaled = self.params_scaled[idx]        # [5]
        x = torch.from_numpy(w_norm).unsqueeze(0) # [1,NK,NT]
        y = torch.from_numpy(y_scaled)            # [5]
        w_true_norm = torch.from_numpy(w_norm)    # [NK,NT]
        return x, y, w_true_norm


# ============================================================
# 5) Boucle d'entraînement
# ============================================================

def train_inverse_heston(
    n_train: int = 50000,
    n_val: int = 5000,
    batch_size: int = 256,
    lr: float = 3e-4,
    n_epochs: int = 30,
    w_recon: float = 0.3,
    ckpt_path: str = "models/heston_inverse_synth.pt",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    # Dataset
    train_ds = HestonInverseDataset(n_train)
    val_ds = HestonInverseDataset(n_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Modèle
    model = InverseHestonModel(NK, NT, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_param = nn.SmoothL1Loss()
    loss_recon = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)  # [B,NK,NT]

            opt.zero_grad(set_to_none=True)
            y_pred, w_pred = model(xb)       # y_pred [B,5], w_pred [B,1,NK,NT]
            w_pred = w_pred.squeeze(1)       # [B,NK,NT]

            Lp = loss_param(y_pred, yb)
            Lr = loss_recon(w_pred, wb)

            loss = Lp + w_recon * Lr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                y_pred, w_pred = model(xb)
                w_pred = w_pred.squeeze(1)
                Lp = loss_param(y_pred, yb)
                Lr = loss_recon(w_pred, wb)
                loss = Lp + w_recon * Lr
                val_loss_sum += loss.item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(n_val_batches, 1)
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "param_mean": train_ds.param_mean,
                    "param_std": train_ds.param_std,
                    "w_mean": train_ds.w_mean,
                    "w_std": train_ds.w_std,
                    "K_POINTS": K_POINTS,
                    "T_POINTS": T_POINTS,
                },
                ckpt_path,
            )
            print(f"  -> New best model saved to {ckpt_path}")


if __name__ == "__main__":
    train_inverse_heston()
