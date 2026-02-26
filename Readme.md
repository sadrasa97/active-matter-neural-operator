# Physics-Constrained Fourier Neural Operator for Active Matter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
A physics-constrained Fourier Neural Operator (FNO) for learning macroscopic closure models of Active Brownian Particle (ABP) systems beyond the Toner-Tu validity regime.

## Overview

This project implements a **data-driven approach** to discover macroscopic evolution equations for active matter systems directly from microscopic simulations, without relying on phenomenological closure approximations.

### Key Innovation

> A physics-constrained neural operator can learn a **stable and generalizable macroscopic closure model** for systems of self-propelled particles, beyond the regime where hydrodynamic theories such as Toner-Tu equations remain valid.

## Features

- **Physics-Constrained Architecture**: Hard constraints for mass conservation and positivity
- **Spectral Neural Operators**: Fourier Neural Operator with correct two-sided mode selection
- **Active Brownian Particle Simulator**: Built-in ABP simulation with coarse-graining
- **Baseline Comparisons**: MLP, Unconstrained FNO, and fitted Toner-Tu PDE
- **Stability Analysis**: Lyapunov exponent estimation and spectral energy tracking
- **Automatic Mixed Precision**: GPU-accelerated training with AMP
- **Rotational Equivariance**: Data augmentation for rotation symmetry

## Installation

### Requirements

```bash
pip install torch numpy
```

### Optional (for enhanced performance)

```bash
# CUDA support (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Run Full Pipeline

```bash
python main.py
```

This executes:
1. **Verification Suite** — Validates physics constraints and gradient flow
2. **Data Generation** — Simulates ABP trajectories across parameter regimes
3. **Model Training** — Trains physics-constrained FNO with rollout loss
4. **Trajectory Evaluation** — In-distribution and OOD generalization tests
5. **Stability Analysis** — Lyapunov exponent and spectral energy evolution
6. **Ablation Study** — Compares against baseline models
- `rollout_diagnostics.png`
- `metrics.json`
- `history.csv`

Control output behavior with `TrainConfig`:
- `results_dir`
- `save_plots`
- `save_metrics`
- `plot_rollout_steps`

## Architecture

### Physics-Constrained FNO

```
Input: [ρ, Px, Py]  (B, 3, H, W)
  │
  ├─→ Lift: 3 → width (Conv2d + GELU)
  │
  ├─→ FNO Backbone (L layers)
  │     ├─ SpectralConv2d (Fourier mixing)
  │     └─ Pointwise Conv2d (channel mixing)
  │
  ├─→ Branch A: Flux J_θ = (Jx, Jy)
  │     └─→ ∂_t ρ = -∇·J_θ  [mass-conserving]
  │
  └─→ Branch B: Polar tendency G_θ = (∂_t Px, ∂_t Py)
```

### Physics Constraints

| Constraint | Implementation | Guarantee |
|------------|----------------|-----------|
| **Mass Conservation** | ∂_t ρ = -∇·J (spectral divergence) | ∫ ∂_t ρ dx = 0 exactly |
| **Positivity** | ρ_{t+Δt} = Softplus(ρ_t + Δt·∂_t ρ) | ρ ≥ 0 everywhere |
| **Translation Equivariance** | Spectral representation | Inherent in Fourier space |
| **Rotational Equivariance** | 90° rotation augmentation | Enforced via data augmentation |
| **Entropy Regularization** | L_ent = λ_e · mean(ρ log ρ) | Discourages unphysical clustering |

### Spectral Convolution

Correct two-sided mode selection for real FFT:

```
û = rfft2(u)  →  shape (B, C, H, W//2+1)

Retained modes:
  - Positive ky:  rows 0 : k_max
  - Negative ky:  rows H-k_max : H
  - Positive kx:  cols 0 : k_max

Complex weights: R_pos, R_neg ∈ ℂ^(C_in × C_out × k_max × k_max)
```

### Mathematical Formulation

### Microscopic Dynamics (ABP)

Overdamped 2D Active Brownian Particles:

```
ẋ_i = v₀ e(θ_i) + √(2D_t) η_i
θ̇_i = √(2D_r) ξ_i

e(θ_i) = (cos θ_i, sin θ_i)
```

### Coarse-Grained Fields

```
ρ(x,t) = Σ_i W(x - x_i(t))          [density]
P(x,t) = Σ_i e(θ_i) W(x - x_i(t))   [polarization]
```

### Target Operator

Learn macroscopic closure without moment truncation:

```
∂_t ρ = F_θ(ρ, P)   enforced as: ∂_t ρ := -∇·J_θ(ρ, P)
∂_t P = G_θ(ρ, P)
```

### Training Objective

```
L = L_data + λ_c L_continuity + λ_e L_entropy + λ_r L_rollout

L_data     = ||ρ_pred - ρ_true||² + ||P_pred - P_true||²
L_entropy  = mean(ρ log ρ)
L_rollout  = (1/T) Σ_{t=1}^T γ^t · MSE(ρ^t_pred, ρ^t_true)
```

## Configuration

### Training Parameters (`TrainConfig`)

```python
@dataclass
class TrainConfig:
    # Architecture
    width: int = 64       # Feature width
    n_layers: int = 4     # FNO layers
    k_max: int = 16       # Fourier modes

    # Training
    n_epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Loss weights
    lambda_e: float = 1e-4    # Entropy regularization
    lambda_r: float = 0.1     # Rollout loss
    gamma_rollout: float = 0.9

    # Rollout stability
    rollout_steps: int = 4
    rollout_start_epoch: int = 10
```

### ABP Simulation Parameters

```python
@dataclass
class ABPParams:
    N: int = 1024         # Number of particles
    L: float = 32.0       # Box size (periodic)
    v0: float = 1.0       # Self-propulsion speed
    D_t: float = 0.1      # Translational diffusion
    D_r: float = 1.0      # Rotational diffusion
    dt: float = 0.005     # Integration timestep
    grid_H: int = 64      # Coarse-grid resolution
    grid_W: int = 64
    kernel_sigma: float = 1.0  # Gaussian smoothing
```

## Baseline Models

The code includes three baseline models for comparison:

| Model | Description | Constraints |
|-------|-------------|-------------|
| **MLPClosure** | Pixel-wise MLP (1×1 Convs) | None |
| **UnconstrainedFNO** | Standard FNO | None |
| **TonerTuClosure** | Fitted Toner-Tu PDE | Near-equilibrium theory |
| **PhysicsConstrainedFNO** | Proposed method | Mass + Positivity |

### Toner-Tu Equations

```
∂_t ρ = -v₀ ∇·P + D_ρ ∇²ρ
∂_t P = -λ(P·∇)P - (α + β|P|²)P - (v₀/2)∇ρ + D_P ∇²P
```

Parameters (α, β, λ, D_ρ, D_P) are fitted from data.

## Evaluation Metrics

### Trajectory Evaluation

- **L2 Error**: RMSE on density and polarization trajectories
- **R2 Score**: Coefficient of determination over full trajectories
- **Mass Error**: |∫ρ_pred - ∫ρ_true| / ∫ρ_true (mean + max over time)
- **Negativity Fraction**: Fraction of predicted ρ values below 0
- **Stability**: Binary (no NaN/Inf during rollout)
- **Spectral Error**: MSE of isotropic energy spectrum E(k)
- **Structure Factor Error**: MSE of static structure factor S(k)

### Stability Analysis

- **Lyapunov Exponent**: λ₁ = mean(log(||δz_{t+1}|| / ||δz_t||) / Δt)
  - λ₁ < 0 → stable (perturbations decay)
  - λ₁ > 0 → chaotic (perturbations grow)

- **Spectral Energy Evolution**: E(k, t) over rollout time

## Performance Notes

### Computational Complexity

```
SpectralConv2d: O(B · C² · k_max² + B · C · H·W · log(H·W))
FNO Forward:    O(L · B · C² · H·W · log(H·W))
```

### Memory Footprint (float32)

```
Activations: B × n_layers × width × H × W × 4 bytes
Weights:     n_layers × width² × k_max² × 8 bytes (complex64)

Example (B=8, L=4, width=64, H=W=64, k_max=16):
  Activations: ~536 MB
  Weights:     ~33 MB
```

### Optimization Tips

1. `torch.compile(model)` — 2-3× speedup (PyTorch 2.0+)
2. Mixed precision (`amp.autocast()`) — ~2× memory reduction
3. Increase `k_max` to 16-24 for better spectral resolution
4. Use `cudnn.benchmark = True` for fixed input sizes

## Project Structure

```
FNO/
├── main.py                     # Complete implementation
├── README.md                   # This file
├── abp_fno_checkpoint.pt       # Saved checkpoint (after training)
└── .gitignore
```

## Output Example

```
2026-02-25 10:30:15 [INFO] Device: cuda | AMP: True
2026-02-25 10:30:16 [INFO] ==================================================
2026-02-25 10:30:16 [INFO] Physics-Constrained FNO for ABP Macroscopic Closure
2026-02-25 10:30:16 [INFO] ==================================================

[Phase 1] Verification suite...
[✓] SpectralConv2d: shape OK, grad_norm=1.234e-03
[✓] Mass conservation: max|∫∂_tρ dx| = 2.34e-06 < 1.00e-04
[✓] Positivity: min ρ_new = 1.23e-04 ≥ 0
...

[Phase 4] Training physics-constrained FNO...
  Model parameters: 1,234,567
  Ep    1 | tr=2.34e-02 | val=2.45e-02 | lr=3.00e-04 | 12.3s
  Ep    5 | tr=1.23e-02 | val=1.34e-02 | lr=2.85e-04 | 11.8s
  ...
  Ep   30 | tr=4.56e-03 | val=5.12e-03 | lr=1.20e-05 | 11.5s

[Phase 7] Ablation study...
╔══════════════════════════════════════════════════════════════════╗
│ ABLATION TABLE — In-Distribution (Val)                           │
╠══════════════════════════════════════════════════════════════════╣
│ Model                l2_rho     l2_P     mass_error  stable     │
│ physics_fno          1.23e-03   2.34e-03 1.23e-05    1.0        │
│ unconstrained_fno    2.34e-03   3.45e-03 4.56e-02    1.0        │
│ mlp                  5.67e-03   6.78e-03 1.23e-01    0.0        │
│ toner_tu             8.90e-03   9.01e-03 2.34e-04    1.0        │
╚══════════════════════════════════════════════════════════════════╝
```

## Theoretical Background

### Why Beyond Toner-Tu?

Toner-Tu equations are derived under assumptions:
- Weak noise (small fluctuations)
- Near-ordered state (small |P|)
- Gradient expansion truncation at low order

These break down in:
- **High activity regimes** (large v₀)
- **Motility-Induced Phase Separation (MIPS)**
- **Strong noise conditions**

### Neural Operator Advantage

Learning the closure operator F_θ, G_θ directly from data:
- No moment truncation bias
- Captures non-local spatial correlations (via Fourier modes)
- Generalizes to unseen parameter regimes
- Maintains physical structure (mass, positivity)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sadrasa97_abp_fno_2026,
  author = {Sadra Saadati},
  title = {Physics-Constrained Neural Operator for Active Matter},
  year = {2026},
  url = {https://github.com/sadrasa97/active-matter-neural-operator}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **GitHub**: [@sadrasa97](https://github.com/sadrasa97)
- **Repository**: [active-matter-neural-operator](https://github.com/sadrasa97/active-matter-neural-operator)

---

*This project implements research-grade code for physics-constrained machine learning in active matter systems.*
