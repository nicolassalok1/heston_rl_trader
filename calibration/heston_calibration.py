# calibration/heston_calibration.py
from __future__ import annotations
import torch
from typing import Tuple

from models.heston_pricer import heston_call_price_torch
from models.black_scholes import bs_call_price_torch


def calibrate_heston_torch(
    S0: float,
    K: torch.Tensor,
    T: torch.Tensor,
    market_iv: torch.Tensor,
    r: float = 0.0,
    q: float = 0.0,
    n_steps: int = 500,
    lr: float = 1e-2,
    verbose: bool = True,
) -> Tuple[float, float, float, float, float]:
    """
    Calibration Heston simple via gradient descent torch:
      - Market IV -> Market prices
      - Minimise MSE prix_model vs prix_market

    market_iv: [NK,NT]
    """

    device = market_iv.device
    dtype = torch.float64

    K = K.to(device=device, dtype=dtype)
    T = T.to(device=device, dtype=dtype)
    market_iv = market_iv.to(device=device, dtype=dtype)

    # convertit IV → prix via Black-Scholes
    market_price = bs_call_price_torch(
        S0=S0,
        K=K,
        T=T,
        iv=market_iv,
        r=r,
        q=q,
    ).to(dtype)

    # paramètres init (log-space sauf rho)
    kappa = torch.tensor(1.5, device=device, dtype=dtype, requires_grad=True)
    theta = torch.tensor(0.04, device=device, dtype=dtype, requires_grad=True)
    sigma = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
    rho = torch.tensor(-0.5, device=device, dtype=dtype, requires_grad=True)
    v0 = torch.tensor(0.04, device=device, dtype=dtype, requires_grad=True)

    opt = torch.optim.Adam([kappa, theta, sigma, rho, v0], lr=lr)

    for step in range(1, n_steps + 1):
        opt.zero_grad()

        model_price = heston_call_price_torch(
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            params=(kappa, theta, sigma, rho, v0),
            n_integration=128,
            u_max=80.0,
        ).to(dtype)

        loss = ((model_price - market_price)**2).mean()

        loss.backward()
        opt.step()

        # projeter les paramètres dans un domaine raisonnable
        with torch.no_grad():
            theta.clamp_(1e-4, 0.5)
            sigma.clamp_(1e-3, 3.0)
            v0.clamp_(1e-4, 0.5)
            kappa.clamp_(1e-3, 10.0)
            rho.clamp_(-0.99, 0.99)

        if verbose and step % 50 == 0:
            print(f"[step {step}] loss={float(loss):.6f} "
                  f"kappa={float(kappa):.3f} theta={float(theta):.4f} "
                  f"sigma={float(sigma):.3f} rho={float(rho):.3f} v0={float(v0):.4f}")

    return float(kappa), float(theta), float(sigma), float(rho), float(v0)
