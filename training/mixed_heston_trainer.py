# training/mixed_heston_trainer.py
from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.heston_inverse_model import InverseHestonModel
from train_inverse_heston import HestonInverseDataset, K_POINTS, T_POINTS
from data.real_iv_dataset import RealIvSurfaceDataset


def train_mixed_inverse_heston(
    synth_n_train: int = 50000,
    real_dir: str = "data/deribit_surfaces",
    batch_synth: int = 256,
    batch_real: int = 64,
    lr: float = 3e-4,
    epochs: int = 20,
    w_recon_synth: float = 0.3,
    w_recon_real: float = 1.0,
    ckpt_out: str = "models/heston_inverse_mixed.pt",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # 1) Datasets
    synth_ds = HestonInverseDataset(synth_n_train)
    real_ds = RealIvSurfaceDataset(real_dir)

    NK, NT = K_POINTS.shape[0], T_POINTS.shape[0]
    assert real_ds.NK == NK and real_ds.NT == NT, "Grilles (NK,NT) incohérentes."

    synth_loader = DataLoader(
        synth_ds, batch_size=batch_synth, shuffle=True, num_workers=4, pin_memory=True
    )
    real_loader = DataLoader(
        real_ds, batch_size=batch_real, shuffle=True, num_workers=4, pin_memory=True
    )

    real_iter = iter(real_loader)

    # 2) Modèle
    model = InverseHestonModel(NK, NT, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_param = nn.SmoothL1Loss()
    loss_recon = nn.MSELoss()

    best_loss = float("inf")
    os.makedirs(os.path.dirname(ckpt_out), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb_s, yb_s, wb_s in synth_loader:
            xb_s = xb_s.to(device)  # [B_s,1,NK,NT]
            yb_s = yb_s.to(device)  # [B_s,5]
            wb_s = wb_s.to(device)  # [B_s,NK,NT]

            # batch real (reconstruction only)
            try:
                xb_r, _ = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                xb_r, _ = next(real_iter)
            xb_r = xb_r.to(device)  # [B_r,1,NK,NT]

            opt.zero_grad(set_to_none=True)

            # Synthétique: params + recon
            params_s, surf_s = model(xb_s)
            surf_s = surf_s.squeeze(1)
            Lp = loss_param(params_s, yb_s)
            Lr_s = loss_recon(surf_s, wb_s)

            # Réel: recon only
            _, surf_r = model(xb_r)
            surf_r = surf_r.squeeze(1)
            Lr_r = loss_recon(surf_r, xb_r.squeeze(1))

            loss = Lp + w_recon_synth * Lr_s + w_recon_real * Lr_r
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[Epoch {epoch}] mixed_loss={avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "param_mean": synth_ds.param_mean,
                    "param_std": synth_ds.param_std,
                    "w_mean": synth_ds.w_mean,
                    "w_std": synth_ds.w_std,
                    "iv_mean": real_ds.iv_mean,
                    "iv_std": real_ds.iv_std,
                    "K_POINTS": K_POINTS,
                    "T_POINTS": T_POINTS,
                },
                ckpt_out,
            )
            print(f"  -> New best model saved to {ckpt_out}")


if __name__ == "__main__":
    train_mixed_inverse_heston()
