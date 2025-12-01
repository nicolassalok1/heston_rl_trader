# models/heston_pricer.py
from __future__ import annotations
import torch
import math
from typing import Tuple


def heston_charfunc_torch(
    u: torch.Tensor,
    T: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    v0: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    logS0: torch.Tensor,
) -> torch.Tensor:
    """
    Char-fonction Heston φ(u) en torch (complex128 si possible).
    u: [...,]
    Retourne: φ(u) de même shape.
    """
    dtype = torch.complex128
    u = u.to(dtype)
    kappa = kappa.to(dtype)
    theta = theta.to(dtype)
    sigma = sigma.to(dtype)
    rho = rho.to(dtype)
    v0 = v0.to(dtype)
    r = r.to(dtype)
    q = q.to(dtype)
    T = T.to(dtype)
    logS0 = logS0.to(dtype)

    i = 1j

    # paramètres auxiliaires
    alpha = -u * u * 0.5 - 0.5 * u * i
    beta = kappa - rho * sigma * i * u
    gamma = 0.5 * sigma * sigma

    d = torch.sqrt(beta * beta - 4.0 * alpha * gamma)
    g = (beta - d) / (beta + d + 1e-14)

    exp_neg_dT = torch.exp(-d * T)
    one_minus_gexp = 1.0 - g * exp_neg_dT
    one_minus_g = 1.0 - g

    # C(T,u)
    C = (kappa * theta / (sigma * sigma)) * (
        (beta - d) * T - 2.0 * torch.log(one_minus_gexp / one_minus_g)
    )

    # D(T,u)
    D = ((beta - d) / (sigma * sigma)) * ((1.0 - exp_neg_dT) / one_minus_gexp)

    # char-fonction
    phi = torch.exp(C + D * v0 + i * u * (logS0 + (r - q) * T))
    return phi


def heston_call_price_torch(
    S0: float,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    q: float,
    params: Tuple[float, float, float, float, float],
    n_integration: int = 256,
    u_max: float = 100.0,
) -> torch.Tensor:
    """
    Prix de call Heston via intégrale numérique différentiable.

    S0: spot
    K: tensor [NK]
    T: tensor [NT]
    params: (kappa, theta, sigma, rho, v0)
    Retourne: price [NK, NT]
    """
    device = K.device
    dtype = torch.float64

    S0_t = torch.tensor(S0, dtype=dtype, device=device)
    r_t = torch.tensor(r, dtype=dtype, device=device)
    q_t = torch.tensor(q, dtype=dtype, device=device)
    logS0 = torch.log(S0_t)

    kappa, theta, sigma, rho, v0 = [torch.tensor(p, dtype=dtype, device=device) for p in params]

    # grille d'intégration u in [0, u_max]
    u = torch.linspace(1e-6, u_max, n_integration, dtype=dtype, device=device)  # [U]

    # reshape pour broadcast
    K_mat = K.view(-1, 1)         # [NK,1]
    T_mat = T.view(1, -1)         # [1,NT]
    logK = torch.log(K_mat)       # [NK,1]

    # on veut P1 et P2
    i = torch.complex(torch.tensor(0., device=device, dtype=dtype),
                      torch.tensor(1., device=device, dtype=dtype))

    # P1: φ(u - i)
    u1 = u - i
    phi1 = heston_charfunc_torch(
        u1.view(-1, 1, 1),
        T_mat,
        kappa, theta, sigma, rho, v0,
        r_t, q_t,
        logS0,
    )  # [U,NK,NT]
    # P2: φ(u)
    phi2 = heston_charfunc_torch(
        u.view(-1, 1, 1),
        T_mat,
        kappa, theta, sigma, rho, v0,
        r_t, q_t,
        logS0,
    )

    # facteurs exp(-i u ln K)
    exp_iu_logK = torch.exp(-i * u.view(-1, 1, 1) * logK)  # [U,NK,1]

    # formule type Heston
    # Pj = 1/2 + 1/pi ∫ Re( e^{-i u lnK} * f_j(u) / (i u) ) du
    # où f1 = φ(u - i) / (S0 * e^{(r-q)T}), f2 = φ(u) / (S0 * e^{(r-q)T})

    disc = torch.exp(-r_t * T_mat)  # [1,NT]
    forward = S0_t * torch.exp((r_t - q_t) * T_mat)  # [1,NT]

    f1 = phi1 / forward  # [U,NK,NT]
    f2 = phi2 / forward

    integrand1 = (exp_iu_logK * f1 / (i * u.view(-1, 1, 1))).real  # [U,NK,NT]
    integrand2 = (exp_iu_logK * f2 / (i * u.view(-1, 1, 1))).real

    # intégration numérique trapz sur u
    du = u[1] - u[0]
    P1 = 0.5 + (1.0 / math.pi) * torch.trapz(integrand1, dx=du, dim=0)  # [NK,NT]
    P2 = 0.5 + (1.0 / math.pi) * torch.trapz(integrand2, dx=du, dim=0)

    # Call = S0 e^{-qT} P1 - K e^{-rT} P2
    S0_disc = S0_t * torch.exp(-q_t * T_mat)  # [1,NT]
    K_disc = K_mat * disc  # [NK,NT]

    call = S0_disc * P1 - K_disc * P2
    return call.to(torch.float32)
