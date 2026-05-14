# EE290 Final Project — Coordination-Aware Ergodic Coverage
Extension of Mendoza et al. (2026) decentralized ergodic coverage with a coordination term that reduces spatial redundancy across UAV agents.

## Setup
```bash
pip install numpy matplotlib scikit-learn scipy
```

## Files
- `core.py` — environment, GP belief updates, REMC policy, and simulation loop shared across all scripts
- `simulation_main.py` — main experiment comparing baseline vs V1 vs V2
- `experiment_tau_sweep.py` — varies belief/policy update frequency τ
- `experiment_scalability.py` — varies team size M

Outputs saved to `./outputs/`.

## Running
```bash
python simulation_main.py        # main results
python experiment_tau_sweep.py   # tau experiment
python experiment_scalability.py # scalability
```

## Methods
**Baseline:** Mendoza et al. Algorithm 1 unmodified (α=0)

**V1:** subtracts average neighbor empirical coverage from each UAV's belief target before computing its policy

**V2:** same as V1 but weighted by `(1 - rho_bar(r))` so high-importance regions are protected from being down-weighted

Default parameters: 8×8 grid, M=3 UAVs, T=600 steps, τ=20, α=0.2, R_comm=2.5, R_sense=1.5, 5 seeds
