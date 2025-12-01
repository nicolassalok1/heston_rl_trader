# train_heston_real.py

from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.heston_inverse_model import InverseHestonModel, load_heston_inverse_model
from data.real_iv_dataset import RealIvSurfaceDataset


def train_heston_real(
    data_dir: str = "data/deribit_surfaces",
    ckpt_pretrain: str = "models/heston_inverse_synth.pt",
    lr: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 10,
    save_path: str = "models/heston_inverse_real.pt",
    train_param_heads: bool = False,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # 1. Dataset réel
    ds = RealIvSurfaceDataset(data_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    NK, NT = ds.NK, ds.NT

    # 2. Charger modèle pré-entraîné synthétique
    model = InverseHestonModel(NK, NT, hidden_dim=128).to(device)
    synth = torch.load(ckpt_pretrain, map_location=device)
    model.load_state_dict(synth["model_state_dict"])
    print("[INFO] Pretrained synthetic model loaded.")

    # 3. Geler têtes de paramètres ?
    if not train_param_heads:
        for name, p in model.named_parameters():
            if "head_" in name:     # head_kappa, head_theta, ...
                p.requires_grad = False
        print("[INFO] Parameter heads frozen (only surface reconstruction trained).")

    # Optimiseur
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )

    loss_recon = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        sum_loss = 0.0
        n_batches = 0

        for xb, _ in loader:
            xb = xb.to(device)

            opt.zero_grad()

            params_pred, surf_pred = model(xb)  # surf_pred = [B,1,NK,NT]
            surf_pred = surf_pred.squeeze(1)

            # Loss de reconstruction sur iv normalisée
            loss = loss_recon(surf_pred, xb.squeeze(0).squeeze(0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            sum_loss += loss.item()
            n_batches += 1

        print(f"[Epoch {epoch}] recon_loss={sum_loss/n_batches:.5f}")

    # Sauvegarde du modèle affiné
    ckpt = {
        "model_state_dict": model.state_dict(),
        "iv_mean": ds.iv_mean,
        "iv_std": ds.iv_std,
        "k_grid": ds.files[0],  # on ne stocke pas correctement ici, à toi d'ajouter
        "NK": NK,
        "NT": NT,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"[INFO] Fine-tuned model saved to {save_path}")
    

if __name__ == "__main__":
    train_heston_real()
