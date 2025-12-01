# models/black_scholes.py
from __future__ import annotations
import math
import torch


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative normal distribution N(x) implemented via erf."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_call_price_torch(
    S0: float,
    K: torch.Tensor,
    T: torch.Tensor,
    iv: torch.Tensor,
    r: float = 0.0,
    q: float = 0.0,
) -> torch.Tensor:
    """
    Black-Scholes call price in torch with broadcasting over strikes/maturities.

    Args:
        S0: spot price (float)
        K: strike tensor [NK]
        T: maturity tensor [NT]
        iv: implied vol tensor [NK, NT]
        r: risk-free rate
        q: dividend/borrow rate
    Returns:
        call prices [NK, NT] (same device/dtype as iv/K/T)
    """
    device = iv.device
    dtype = iv.dtype

    S0_t = torch.tensor(S0, device=device, dtype=dtype)
    r_t = torch.tensor(r, device=device, dtype=dtype)
    q_t = torch.tensor(q, device=device, dtype=dtype)

    K_mat = K.to(device=device, dtype=dtype).view(-1, 1)  # [NK,1]
    T_mat = T.to(device=device, dtype=dtype).view(1, -1)  # [1,NT]
    vol = iv.to(device=device, dtype=dtype)

    vol = torch.clamp(vol, min=1e-8)
    T_safe = torch.clamp(T_mat, min=1e-8)
    sqrtT = torch.sqrt(T_safe)

    log_m = torch.log(S0_t / K_mat)
    d1 = (log_m + (r_t - q_t + 0.5 * vol * vol) * T_mat) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    disc_q = torch.exp(-q_t * T_mat)
    disc_r = torch.exp(-r_t * T_mat)

    call = S0_t * disc_q * Nd1 - K_mat * disc_r * Nd2
    return call
