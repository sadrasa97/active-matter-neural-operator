# Physics-Constrained Fourier Neural Operator for Active Matter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A physics-constrained Fourier Neural Operator (FNO) for learning macroscopic closure models of Active Brownian Particle (ABP) systems beyond the Toner-Tu validity regime.

## Overview

This project implements a data-driven approach to discover macroscopic evolution equations for active matter systems directly from microscopic simulations, without relying on phenomenological closure approximations.

Key idea: a physics-constrained neural operator can learn a stable and generalizable macroscopic closure model beyond the regime where hydrodynamic theories such as Toner-Tu remain valid.

## Features

- Physics-constrained architecture with hard mass conservation and positivity
- Correct two-sided spectral convolution for rfft2
- Built-in ABP simulator with FFT Gaussian coarse-graining
- Baselines: MLP, Unconstrained FNO, Toner-Tu closure
- Stability analysis: Lyapunov exponent and spectral energy tracking
- AMP training on GPU
- Rotation augmentation for 90-degree equivariance
- Normalization modes for consistent training and evaluation
- Expanded metrics and automatic artifact export

## Installation

### Requirements

```bash
pip install torch numpy matplotlib
```

### Optional (for GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

```bash
python main.py
```

This runs the full pipeline:
1. Verification suite
2. Data generation
3. Model training
4. Trajectory evaluation (ID and OOD)
5. Stability analysis
6. Plots and metrics export
7. Ablation study

## Outputs

By default, artifacts are written to `results/`:
- `loss_curves.png`
- `rho_snapshot.png`
- `polmag_snapshot.png`
- `energy_spectrum.png`
- `structure_factor.png`
- `rollout_diagnostics.png`
- `metrics.json`
- `history.csv`

Control output behavior with `TrainConfig`:
- `results_dir`
- `save_plots`
- `save_metrics`
- `plot_rollout_steps`

## Architecture

Input: `[rho, Px, Py]` with shape `(B, 3, H, W)`

```
Input --> Lift (1x1 Conv + GELU)
  -> FNO backbone (L layers)
  -> Branch A: Flux J = (Jx, Jy)  -> drho/dt = -div(J)
  -> Branch B: Polar tendency G  -> dP/dt = G
```

### Physics Constraints

| Constraint | Implementation | Guarantee |
|---|---|---|
| Mass conservation | drho/dt = -div(J) (spectral) | Integral of rho is constant |
| Positivity | rho_{t+dt} = Softplus(rho_t + dt * drho/dt) | rho >= 0 |
| Translation equivariance | Fourier representation | Inherent |
| Rotational equivariance | 90-degree augmentation | Enforced |
| Entropy regularization | mean(rho * log rho) | Discourages clustering |

## Evaluation Metrics

`evaluate_trajectory` returns:
- `l2_rho`, `l2_P` (RMSE)
- `mae_rho`, `mae_P`
- `rel_l2_rho`, `rel_l2_P`
- `corr_rho`, `corr_P` (final-step correlation)
- `mass_error` (time-averaged relative mass error)
- `spectral_err` (energy spectrum RMSE)
- `structure_err` (structure factor RMSE)
- `pol_mag_rmse` (final-step polarization magnitude RMSE)
- `stable`

## Normalization

Two modes are supported in `ABPDataset` and evaluation:
- `physics` (default): rho scaled by mean density, P standardized
- `standard`: per-channel mean and std

## Configuration

```python
@dataclass
class TrainConfig:
    # Architecture
    width: int = 64
    n_layers: int = 4
    k_max: int = 16

    # Training
    n_epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Loss weights
    lambda_c: float = 1e-3
    lambda_e: float = 1e-4
    lambda_r: float = 0.1
    gamma_rollout: float = 0.9

    # Rollout stability
    rollout_steps: int = 4
    rollout_start_epoch: int = 10

    # Grid
    H: int = 64
    W: int = 64
    dt_field: float = 0.05

    # Augmentation
    augment: bool = True

    # Normalization and outputs
    normalize_mode: str = "physics"
    results_dir: str = "results"
    plot_rollout_steps: int = 30
    save_plots: bool = True
    save_metrics: bool = True
```

## ABP Simulation Parameters

```python
@dataclass
class ABPParams:
    N: int = 1024
    L: float = 32.0
    v0: float = 1.0
    D_t: float = 0.1
    D_r: float = 1.0
    dt: float = 0.005
    grid_H: int = 64
    grid_W: int = 64
    kernel_sigma: float = 1.0
```

## Baseline Models

| Model | Description | Constraints |
|---|---|---|
| MLPClosure | Pixel-wise MLP (1x1 Convs) | None |
| UnconstrainedFNO | Standard FNO | None |
| TonerTuClosure | Fitted Toner-Tu PDE | Near-equilibrium |
| PhysicsConstrainedFNO | Proposed method | Mass + Positivity |

### Toner-Tu Equations

```
partial_t rho = -v0 * div(P) + D_rho * laplacian(rho)
partial_t P   = -lambda (P dot grad) P - (alpha + beta |P|^2) P
               - (v0/2) * grad(rho) + D_P * laplacian(P)
```
