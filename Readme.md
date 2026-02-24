---

# 3. Methodology

## 3.1 Problem Statement

We investigate whether a physics-constrained neural operator can learn a **stable and generalizable macroscopic closure model** for systems of self-propelled particles modeled as Active Brownian Particle (ABP), beyond the regime where hydrodynamic theories such as Toner–Tu equations remain valid.

Specifically, we aim to learn the macroscopic evolution operator:

[
(\rho, \mathbf{P}) \mapsto \partial_t (\rho, \mathbf{P})
]

where:

* (\rho(x,t)): coarse-grained density field
* (\mathbf{P}(x,t)): polarization field

without relying on moment-closure approximations.

---

# 3.2 Microscopic System: Active Brownian Particles

We consider (N) overdamped ABPs in 2D:

[
\dot{\mathbf{x}}_i = v_0 \mathbf{e}(\theta_i) + \sqrt{2D_t},\boldsymbol{\eta}_i(t)
]
[
\dot{\theta}_i = \sqrt{2D_r},\xi_i(t)
]

where:

* (v_0): self-propulsion speed
* (D_t): translational diffusion
* (D_r): rotational diffusion
* (\mathbf{e}(\theta_i) = (\cos\theta_i, \sin\theta_i))

We simulate the system under periodic boundary conditions across parameter regimes including:

* Low and high density
* Strong noise
* Motility-induced phase separation (MIPS) regime

These regimes intentionally extend beyond Toner–Tu validity.

---

# 3.3 Coarse-Graining Procedure

We construct Eulerian fields on a uniform grid:

### Density

[
\rho(x,t) = \sum_i W(x - x_i(t))
]

### Polarization

[
\mathbf{P}(x,t) = \sum_i \mathbf{e}(\theta_i) W(x - x_i(t))
]

where (W) is a Gaussian kernel with bandwidth comparable to interaction length scale.

Fields are normalized to ensure:

[
\int \rho(x,t),dx = N
]

This yields time-series data:

[
{(\rho^k, \mathbf{P}^k)}_{k=1}^{T}
]

for multiple parameter sets.

---

# 3.4 Target Macroscopic Closure Formulation

Instead of assuming Toner–Tu structure, we learn:

[
\partial_t \rho = \mathcal{F}*\theta(\rho, \mathbf{P})
]
[
\partial_t \mathbf{P} = \mathcal{G}*\theta(\rho, \mathbf{P})
]

where (\mathcal{F}*\theta) and (\mathcal{G}*\theta) are nonlinear operators parameterized by a neural operator.

This avoids explicit moment truncation.

---

# 3.5 Neural Operator Architecture

We adopt a physics-constrained version of the Fourier Neural Operator (FNO).

### 3.5.1 Spectral Operator Layer

For input field (u(x)):

[
\hat{u}_k = \mathcal{F}[u](k)
]

Truncated Fourier modes are transformed via learned kernel:

[
\hat{v}_k = R_k \hat{u}_k
]

where (R_k) is a complex weight tensor.

Inverse transform yields spatial output.

Stacking L layers gives:

[
u_{l+1} = \sigma( W u_l + \mathcal{K}_\theta(u_l) )
]

---

# 3.6 Physics Constraints

To avoid black-box behavior, we enforce physical structure explicitly.

---

## 3.6.1 Mass Conservation

Continuity requires:

[
\partial_t \rho + \nabla \cdot \mathbf{J} = 0
]

We enforce this by:

1. Learning flux field (\mathbf{J}_\theta)
2. Defining:

[
\partial_t \rho := -\nabla \cdot \mathbf{J}_\theta
]

This guarantees:

[
\frac{d}{dt}\int \rho dx = 0
]

exactly (up to discretization).

---

## 3.6.2 Rotational and Translational Equivariance

We enforce symmetry by:

* Training with random domain rotations
* Using spectral representation (naturally translation equivariant)
* Weight tying for isotropic kernels

---

## 3.6.3 Positivity Constraint

Density positivity is enforced via:

[
\rho_{t+\Delta t} = \text{Softplus}(\rho_t + \Delta t,\partial_t \rho)
]

---

## 3.6.4 Energy-like Regularization

We introduce entropy-inspired regularization:

[
\mathcal{L}_{entropy} = \lambda_e \int \rho \log \rho , dx
]

to discourage unphysical clustering artifacts.

---

# 3.7 Training Objective

For discrete timestep prediction:

[
\mathcal{L} =
\mathcal{L}_{data}

* \lambda_c \mathcal{L}_{continuity}
* \lambda_e \mathcal{L}_{entropy}
  ]

where:

### Data Loss

[
\mathcal{L}_{data} =
|\rho^{pred} - \rho^{true}|_2^2
+
|\mathbf{P}^{pred} - \mathbf{P}^{true}|_2^2
]

### Stability Loss

Long-horizon rollout consistency over T steps.

---

# 3.8 Generalization Regime

Training:

* Parameter subset: (v_0 \in [v_{min}, v_{mid}])

Testing:

* Unseen propulsion speeds
* Unseen noise strengths
* Higher density regimes

Goal:
Demonstrate operator-level generalization beyond Toner–Tu approximation.

---

# 3.9 Baseline Models

We compare against:

1. Toner–Tu PDE fitted to data
2. MLP-based closure model
3. Unconstrained FNO

Evaluation metrics:

* L2 trajectory error
* Long-time stability
* Phase transition prediction accuracy
* Energy spectrum preservation

---

# 3.10 Stability Analysis

We evaluate:

1. Linear perturbation growth rates
2. Spectral energy distribution
3. Absence of numerical blow-up in long rollout

Additionally, we examine whether learned operator preserves emergent band structures.

---

# 3.11 Key Hypothesis

We test:

> A physics-constrained neural operator can learn a stable, generalizable macroscopic closure beyond the weak-noise and near-equilibrium regime where Toner–Tu theory is valid.

---

# 3.12 Expected Contributions

1. Data-driven closure beyond moment expansion.
2. Physics-constrained neural operator for active matter.
3. Stability and long-time generalization analysis.
4. Insight into emergent hydrodynamic structure.

---

# 3.13 Implementation Details

* Spatial resolution: 64×64 or 128×128
* Fourier modes retained: 16–24
* Depth: 4–6 operator layers
* Optimizer: AdamW
* Spectral normalization for stability
* Mixed precision training

---

# 3.14 Reproducibility

* Fixed random seeds
* Open-source simulation code
* Dataset generation scripts included
* Hyperparameter grid provided

---

# Final Positioning

This methodology is not:

* Pure ML benchmarking
* Pure active matter simulation

It directly addresses the **closure problem** in non-equilibrium active systems using structured operator learning.

---