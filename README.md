# Decentralized Ergodic Coverage with Coordination

**EE290 Final Project — Spring 2026**
**Inayyah Don Nazwim**

Extends Mendoza et al. (2026) decentralized ergodic coverage framework with a coordination-aware target distribution modification that reduces spatial redundancy across UAV agents.

---

## Setup

```bash
pip install numpy matplotlib scikit-learn scipy
```

---

## Files

| File | Description |
|---|---|
| `core.py` | Shared environment, GP, REMC, and simulation loop. Import by all other files. |
| `simulation_main.py` | Main comparison: Baseline vs. V1 vs. V2. Produces regret, overlap, heatmap, and ROI discovery figures. |
| `experiment_tau_sweep.py` | Varies belief/policy update period τ across coordination conditions. |
| `experiment_scalability.py` | Varies team size M ∈ {2, 3, 4, 6} for baseline and V2. |

All outputs saved to `./outputs/`.

---

## How to run

**Main comparison (start here):**
```bash
python simulation_main.py
```

**Tau sweep (Mendoza feedback experiment):**
```bash
python experiment_tau_sweep.py
```

**Scalability study:**
```bash
python experiment_scalability.py
```

---

## Methods summary

**Baseline:** Mendoza et al. (2026) Algorithm 1, unmodified.

**V1 — Original coordination term:**
```
phi_coord(r) = phi_bar(r) - alpha * avg_j[rho_hat_j(r)] * ||phi_bar||_1
rho_bar = normalize(clip(phi_coord, 1e-8))
```

**V2 — Importance-weighted coordination term:**
```
phi_coord(r) = phi_bar(r) - alpha * (1 - rho_bar(r)) * avg_j[rho_hat_j(r)] * ||phi_bar||_1
rho_bar = normalize(clip(phi_coord, 1e-8))
```
The `(1 - rho_bar(r))` weight protects high-importance regions from being down-weighted, removing the coverage artifact present in V1.

---

## Parameters

| Parameter | Value |
|---|---|
| Grid size | 8×8 (64 regions) |
| UAVs (default) | 3 |
| ROIs | 4 Gaussian centers at (2,2), (2,5), (5,2), (5,5) |
| No-fly zones | 2×2 center block |
| Timesteps | 600 |
| τGP = τP | 20 (default) |
| R_comm | 2.5 grid cells |
| R_sense | 1.5 grid cells |
| α (coordinated) | 0.4 (default) |
| Seeds per condition | 5 |
