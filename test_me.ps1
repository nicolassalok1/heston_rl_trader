$ErrorActionPreference = 'Stop'

if ($PSScriptRoot) {
    Set-Location $PSScriptRoot
}

# Ensure the virtual environment exists (falls back to set_me.ps1 the first time)
$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtualenv not found. Bootstrapping it with set_me.ps1..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'set_me.ps1')
}

Write-Host "Using Python at $venvPython" -ForegroundColor Cyan

Write-Host "`n[1/3] Syntax check (compileall)..." -ForegroundColor Cyan
$pathsToCheck = @(
    'backtester.py',
    'live_trading.py',
    'train_ppo.py',
    'train_inverse_heston.py',
    'train_heston_real.py',
    'env',
    'features',
    'models',
    'rl',
    'data',
    'calibration',
    'training',
    'sentiment'
)
& $venvPython -m compileall -q @pathsToCheck

Write-Host "`n[2/3] Smoke test: Heston pricer..." -ForegroundColor Cyan
@'
import torch
from models.heston_pricer import heston_call_price_torch

K = torch.tensor([28000.0, 30000.0, 32000.0], dtype=torch.float64)
T = torch.tensor([0.1, 0.25], dtype=torch.float64)
params = (1.2, 0.04, 0.5, -0.5, 0.04)

prices = heston_call_price_torch(
    S0=30000.0,
    K=K,
    T=T,
    r=0.0,
    q=0.0,
    params=params,
    n_integration=64,
    u_max=50.0,
)
if torch.isnan(prices).any() or torch.isinf(prices).any():
    raise SystemExit("Heston pricer produced invalid values")
print(f"Avg call price: {prices.mean().item():.4f}")
'@ | & $venvPython -

Write-Host "`n[3/3] Smoke test: TradingEnv with simulated data..." -ForegroundColor Cyan
@'
import numpy as np
import torch
from pathlib import Path
from data.simulated_data import simulate_market, SimMarketConfig
from models.heston_inverse_model import load_heston_inverse_model
from features.feature_engine import create_default_feature_engine
from features.state_builder import StateBuilder
from env.trading_env import TradingEnv, TradingEnvConfig

device = torch.device("cpu")

market = simulate_market(SimMarketConfig(n_steps=200))

# Use real checkpoints when available (btc uses the synthetic-trained inverse here).
btc_ckpt = Path("models/heston_inverse_synth.pt")
if not btc_ckpt.exists():
    raise SystemExit(f"Missing BTC inverse checkpoint: {btc_ckpt}")
btc_model = load_heston_inverse_model(
    nk=5,
    nt=6,  # checkpoint trained on 5x6 grid
    ckpt_path=str(btc_ckpt),
    device=device,
)

# Keep shitcoin on the dummy inverse unless you have a matching checkpoint (nk=3, nt=4).
shit_ckpt = None
shit_model = load_heston_inverse_model(nk=3, nt=4, ckpt_path=shit_ckpt, device=device)
fe = create_default_feature_engine(
    shitcoin_heston_inverse=shit_model,
    btc_heston_inverse=btc_model,
    device=device,
)

t0 = 50
ctx = {
    "shitcoin": {
        "prices": market.shit_prices[max(0, t0-29):t0+1],
        "volumes": market.shit_volumes[max(0, t0-29):t0+1],
        "funding": market.shit_funding[max(0, t0-29):t0+1],
    },
    "btc": {
        "prices": market.btc_prices[:t0+1],
        "idx": t0,
        "future_price": market.btc_fut_prices[t0],
        "funding_rate": market.btc_funding[t0],
        "open_interest": market.btc_oi[t0],
        "iv_surface": market.btc_iv_surface[t0],
        "k_grid": market.k_grid,
        "t_grid": market.t_grid,
    },
    "sentiment": {
        "sentiment_score": 0.0,
        "msg_rate": 0.0,
        "fear_greed": 0.5,
    },
    "generic": {
        "close": float(market.btc_prices[t0]),
        "high": float(market.btc_prices[t0]),
        "low": float(market.btc_prices[t0]),
        "volume": 1.0,
    },
}
feature_vec, _ = fe.compute_features(ctx)
sb = StateBuilder(dim=feature_vec.shape[0], window=8, clip_value=5.0, training=True)

env_cfg = TradingEnvConfig(
    window_shitcoin=30,
    step_start=t0,
    max_steps=120,
    initial_capital=1000.0,
    transaction_cost=0.0,
)
env = TradingEnv(market=market, feature_engine=fe, state_builder=sb, config=env_cfg)

obs, _ = env.reset()
for i in range(20):
    action = np.array([0.1 * (-1) ** i], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"Env smoke completed at step {env.t} with equity {info['equity']:.2f}")
'@ | & $venvPython -

Write-Host "`nAll smoke tests completed." -ForegroundColor Green
