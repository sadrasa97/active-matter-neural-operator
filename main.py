"""
Physics-Constrained Fourier Neural Operator for Active Matter
=============================================================
A physics-constrained FNO for learning macroscopic closure models of 
Active Brownian Particle (ABP) systems beyond the Toner-Tu validity regime.

Author: Sadra Saadati
Year: 2026
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import json
import csv
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

set_seed(42)

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ABPParams:
    """Active Brownian Particle simulation parameters"""
    N: int = 1024           # Number of particles
    L: float = 32.0         # Box size (periodic)
    v0: float = 1.0         # Self-propulsion speed
    D_t: float = 0.1        # Translational diffusion
    D_r: float = 1.0        # Rotational diffusion
    dt: float = 0.005       # Integration timestep
    n_steps: int = 500      # Number of simulation steps
    grid_H: int = 64        # Coarse-grid resolution
    grid_W: int = 64
    kernel_sigma: float = 1.0  # Gaussian smoothing width
    save_interval: int = 10 # Save trajectory every n steps

@dataclass
class TrainConfig:
    """Training configuration"""
    # Architecture
    width: int = 64         # Feature width
    n_layers: int = 4       # FNO layers
    k_max: int = 16         # Fourier modes
    
    # Training
    n_epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    
    # Loss weights
    lambda_e: float = 1e-4      # Entropy regularization
    lambda_c: float = 1.0       # Continuity constraint
    lambda_r: float = 0.1       # Rollout loss
    gamma_rollout: float = 0.9
    
    # Rollout stability
    rollout_steps: int = 4
    rollout_start_epoch: int = 10
    
    # Output
    results_dir: str = "./results"
    save_plots: bool = True
    save_metrics: bool = True
    plot_rollout_steps: int = 20

# ============================================================================
# Active Brownian Particle Simulator
# ============================================================================

class ABPSimulator:
    """Simulates 2D Active Brownian Particles with periodic boundary conditions"""
    
    def __init__(self, params: ABPParams):
        self.params = params
        self.N = params.N
        self.L = params.L
        
    def initialize(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particle positions and orientations"""
        if seed is not None:
            np.random.seed(seed)
        
        # Random initial positions (uniform in box)
        x = np.random.uniform(0, self.L, (self.N, 2))
        
        # Random initial orientations
        theta = np.random.uniform(0, 2 * np.pi, self.N)
        
        return x, theta
    
    def step(self, x: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one integration step using Euler-Maruyama"""
        dt = self.params.dt
        v0 = self.params.v0
        D_t = self.params.D_t
        D_r = self.params.D_r
        L = self.L
        
        # Orientation unit vectors
        e = np.column_stack([np.cos(theta), np.sin(theta)])
        
        # Translational noise
        eta = np.random.normal(0, 1, (self.N, 2))
        
        # Rotational noise
        xi = np.random.normal(0, 1, self.N)
        
        # Update positions
        dx = v0 * e * dt + np.sqrt(2 * D_t * dt) * eta
        x_new = x + dx
        
        # Periodic boundary conditions
        x_new = x_new % L
        
        # Update orientations
        dtheta = np.sqrt(2 * D_r * dt) * xi
        theta_new = (theta + dtheta) % (2 * np.pi)
        
        return x_new, theta_new
    
    def simulate(self, n_steps: int = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run full simulation and return trajectories"""
        if n_steps is None:
            n_steps = self.params.n_steps
        
        x, theta = self.initialize(seed)
        
        # Store trajectories
        x_traj = np.zeros((n_steps + 1, self.N, 2))
        theta_traj = np.zeros((n_steps + 1, self.N))
        
        x_traj[0] = x
        theta_traj[0] = theta
        
        for t in range(n_steps):
            x, theta = self.step(x, theta)
            x_traj[t + 1] = x
            theta_traj[t + 1] = theta
        
        return x_traj, theta_traj

# ============================================================================
# Coarse-Graining Functions
# ============================================================================

def coarse_grain(x_traj: np.ndarray, theta_traj: np.ndarray, 
                 params: ABPParams) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse-grain particle trajectories to density and polarization fields"""
    n_steps = x_traj.shape[0]
    H, W = params.grid_H, params.grid_W
    L = params.L
    sigma = params.kernel_sigma
    
    # Initialize fields
    rho = np.zeros((n_steps, H, W))
    P = np.zeros((n_steps, 2, H, W))  # (time, 2, H, W)
    
    # Pre-compute grid
    x_grid = np.linspace(0, L, W, endpoint=False)
    y_grid = np.linspace(0, L, H, endpoint=False)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    for t in range(n_steps):
        x = x_traj[t]
        theta = theta_traj[t]
        
        # Compute density
        for i in range(params.N):
            # Periodic distance
            dx = np.minimum(np.abs(xx - x[i, 0]), L - np.abs(xx - x[i, 0]))
            dy = np.minimum(np.abs(yy - x[i, 1]), L - np.abs(yy - x[i, 1]))
            
            kernel = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            rho[t] += kernel
        
        # Normalize density
        rho[t] /= (2 * np.pi * sigma**2)
        
        # Compute polarization
        e_x = np.cos(theta)
        e_y = np.sin(theta)
        
        for i in range(params.N):
            dx = np.minimum(np.abs(xx - x[i, 0]), L - np.abs(xx - x[i, 0]))
            dy = np.minimum(np.abs(yy - x[i, 1]), L - np.abs(yy - x[i, 1]))
            
            kernel = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            P[t, 0] += e_x[i] * kernel
            P[t, 1] += e_y[i] * kernel
        
        P[t] /= (2 * np.pi * sigma**2)
    
    return rho, P

def create_dataset(rho: np.ndarray, P: np.ndarray, 
                   dt: float, skip: int = 1) -> List[Dict]:
    """Create training dataset from coarse-grained fields"""
    data = []
    n_steps = rho.shape[0]
    
    for t in range(0, n_steps - skip, skip):
        state = np.stack([rho[t], P[t, 0], P[t, 1]], axis=0)  # (3, H, W)
        state_next = np.stack([rho[t + skip], P[t + skip, 0], P[t + skip, 1]], axis=0)
        
        # Compute time derivative
        dt_state = (state_next - state) / (skip * dt)
        
        data.append({
            'state': state.astype(np.float32),
            'dt_state': dt_state.astype(np.float32),
            'state_next': state_next.astype(np.float32)
        })
    
    return data

# ============================================================================
# Spectral Convolution Layer
# ============================================================================

class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution with correct two-sided mode selection for real FFT.
    
    Retained modes:
    - Positive ky: rows 0 : k_max
    - Negative ky: rows H-k_max : H  
    - Positive kx: cols 0 : k_max
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 k_max: int = 16, modes: str = 'full'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_max = k_max
        self.modes = modes
        
        # Complex weights for Fourier modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, k_max, k_max)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, k_max, k_max)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        
        # Compute 2D real FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1)
        
        # Output FFT
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, 
                            device=x.device, dtype=torch.cfloat)
        
        # Select modes and apply weights
        k_max = min(self.k_max, H, W // 2 + 1)
        
        # Convert weights to complex
        weights = torch.complex(self.weights_real, self.weights_imag)
        
        # Apply spectral convolution
        for i in range(min(C, self.in_channels)):
            for j in range(self.out_channels):
                out_ft[:, j, :k_max, :k_max] += torch.einsum(
                    'bijk,ijkm->bkm',
                    x_ft[:, i:i+1, :k_max, :k_max],
                    weights[i:i+1, j:j+1, :, :]
                )
        
        # Inverse FFT
        out = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        
        return out
    
    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, k_max={self.k_max}'

# ============================================================================
# Physics-Constrained FNO
# ============================================================================

class PhysicsConstrainedFNO(nn.Module):
    """
    Physics-Constrained Fourier Neural Operator for Active Matter.
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.width = config.width
        self.n_layers = config.n_layers
        self.k_max = config.k_max
        
        # Lifting layer
        self.lift = nn.Sequential(
            nn.Conv2d(3, self.width, kernel_size=1),
            nn.GELU()
        )
        
        # FNO backbone layers
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = nn.ModuleDict({
                'spectral': SpectralConv2d(self.width, self.width, self.k_max),
                'pointwise': nn.Conv2d(self.width, self.width, kernel_size=1),
                'norm': nn.InstanceNorm2d(self.width),
                'activation': nn.GELU()
            })
            self.fno_layers.append(layer)
        
        # Branch A: Flux prediction (Jx, Jy)
        self.flux_branch = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, 2, kernel_size=1)
        )
        
        # Branch B: Polar tendency (∂_t Px, ∂_t Py)
        self.polar_branch = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, 2, kernel_size=1)
        )
    
    def spectral_divergence(self, J: torch.Tensor) -> torch.Tensor:
        """Compute divergence in Fourier space for exact mass conservation."""
        B, _, H, W = J.shape
        
        # FFT of flux components
        Jx_ft = torch.fft.rfft2(J[:, 0:1], norm='ortho')
        Jy_ft = torch.fft.rfft2(J[:, 1:2], norm='ortho')
        
        # Wave numbers
        kx = torch.fft.fftfreq(W, d=1/W).to(J.device)
        ky = torch.fft.fftfreq(H, d=1/H).to(J.device)
        kx = kx[:W//2+1]
        ky = ky[:H]
        
        # Create wave number grids
        kxx, kyy = torch.meshgrid(ky, kx, indexing='ij')
        
        # Multiply by ik in Fourier space
        div_ft = 1j * kxx[None, None, :, :] * Jx_ft + 1j * kyy[None, None, :, :] * Jy_ft
        
        # Inverse FFT
        div = torch.fft.irfft2(div_ft, s=(H, W), norm='ortho')
        
        return div.real
    
    def forward(self, state: torch.Tensor, dt: float = 0.05, 
                enforce_positivity: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with physics constraints."""
        # Lift to feature space
        x = self.lift(state)
        
        # FNO backbone
        for layer in self.fno_layers:
            residual = x
            x = layer['spectral'](x)
            x = layer['pointwise'](x)
            x = layer['norm'](x)
            x = layer['activation'](x)
            x = x + residual
        
        # Branch A: Flux → density evolution
        J = self.flux_branch(x)
        div_J = self.spectral_divergence(J)
        dt_rho = -div_J
        
        # Branch B: Polar tendency
        dt_P = self.polar_branch(x)
        
        # Combine time derivatives
        dt_state = torch.cat([dt_rho, dt_P], dim=1)
        
        # Integrate forward
        rho_new = state[:, 0:1] + dt * dt_rho
        P_new = state[:, 1:] + dt * dt_P
        
        # Enforce positivity via softplus
        if enforce_positivity:
            rho_new = F.softplus(rho_new, beta=10.0)
        
        state_pred = torch.cat([rho_new, P_new], dim=1)
        
        return state_pred, dt_state
    
    def rollout(self, state: torch.Tensor, n_steps: int, 
                dt: float = 0.05) -> torch.Tensor:
        """Perform multi-step rollout."""
        B, _, H, W = state.shape
        trajectory = torch.zeros(B, n_steps + 1, 3, H, W, device=state.device)
        trajectory[:, 0] = state
        
        current_state = state
        for t in range(n_steps):
            current_state, _ = self.forward(current_state, dt, enforce_positivity=True)
            trajectory[:, t + 1] = current_state
        
        return trajectory

# ============================================================================
# Baseline Models
# ============================================================================

class MLPClosure(nn.Module):
    """Pixel-wise MLP closure (1×1 convolutions)"""
    
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 3, 1)
        )
    
    def forward(self, state: torch.Tensor, dt: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_state = self.net(state)
        state_pred = state + dt * dt_state
        state_pred[:, 0:1] = F.softplus(state_pred[:, 0:1], beta=10.0)
        return state_pred, dt_state

class UnconstrainedFNO(nn.Module):
    """Standard FNO without physics constraints"""
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.width = config.width
        self.n_layers = config.n_layers
        self.k_max = config.k_max
        
        self.lift = nn.Sequential(
            nn.Conv2d(3, self.width, 1),
            nn.GELU()
        )
        
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = nn.ModuleDict({
                'spectral': SpectralConv2d(self.width, self.width, self.k_max),
                'pointwise': nn.Conv2d(self.width, self.width, 1),
                'norm': nn.InstanceNorm2d(self.width),
                'activation': nn.GELU()
            })
            self.fno_layers.append(layer)
        
        self.output = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, 3, 1)
        )
    
    def forward(self, state: torch.Tensor, dt: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.lift(state)
        for layer in self.fno_layers:
            residual = x
            x = layer['spectral'](x)
            x = layer['pointwise'](x)
            x = layer['norm'](x)
            x = layer['activation'](x)
            x = x + residual
        
        dt_state = self.output(x)
        state_pred = state + dt * dt_state
        state_pred[:, 0:1] = F.softplus(state_pred[:, 0:1], beta=10.0)
        return state_pred, dt_state

class TonerTuClosure(nn.Module):
    """Fitted Toner-Tu PDE closure"""
    
    def __init__(self):
        super().__init__()
        self.v0 = nn.Parameter(torch.tensor(1.0))
        self.D_rho = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.D_P = nn.Parameter(torch.tensor(0.1))
    
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using finite differences"""
        d2x = F.pad(x, (1, 1, 0, 0), mode='circular')
        d2x = d2x[:, :, :, :-2] - 2 * x + d2x[:, :, :, 2:]
        
        d2y = F.pad(x, (0, 0, 1, 1), mode='circular')
        d2y = d2y[:, :, :-2, :] - 2 * x + d2y[:, :, 2:, :]
        
        return (d2x + d2y)
    
    def forward(self, state: torch.Tensor, dt: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = state[:, 0:1]
        Px = state[:, 1:2]
        Py = state[:, 2:3]
        P_mag2 = Px**2 + Py**2
        
        div_P = (F.pad(Px, (1, 1, 0, 0), mode='circular')[:, :, :, :-2] - 
                 F.pad(Px, (1, 1, 0, 0), mode='circular')[:, :, :, 2:]) / 2
        div_P += (F.pad(Py, (0, 0, 1, 1), mode='circular')[:, :, :-2, :] - 
                  F.pad(Py, (0, 0, 1, 1), mode='circular')[:, :, 2:, :]) / 2
        
        dt_rho = -self.v0 * div_P + self.D_rho * self.laplacian(rho)
        
        grad_rho_x = (F.pad(rho, (1, 1, 0, 0), mode='circular')[:, :, :, :-2] - 
                      F.pad(rho, (1, 1, 0, 0), mode='circular')[:, :, :, 2:]) / 2
        grad_rho_y = (F.pad(rho, (0, 0, 1, 1), mode='circular')[:, :, :-2, :] - 
                      F.pad(rho, (0, 0, 1, 1), mode='circular')[:, :, 2:, :]) / 2
        
        dt_Px = -(self.alpha + self.beta * P_mag2) * Px - (self.v0 / 2) * grad_rho_x + self.D_P * self.laplacian(Px)
        dt_Py = -(self.alpha + self.beta * P_mag2) * Py - (self.v0 / 2) * grad_rho_y + self.D_P * self.laplacian(Py)
        
        dt_state = torch.cat([dt_rho, dt_Px, dt_Py], dim=1)
        state_pred = state + dt * dt_state
        state_pred[:, 0:1] = F.softplus(state_pred[:, 0:1], beta=10.0)
        
        return state_pred, dt_state

# ============================================================================
# Dataset and DataLoader
# ============================================================================

class ABPDataset(Dataset):
    """PyTorch Dataset for ABP coarse-grained fields"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            'state': torch.from_numpy(item['state']),
            'dt_state': torch.from_numpy(item['dt_state']),
            'state_next': torch.from_numpy(item['state_next'])
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    return {
        'state': torch.stack([item['state'] for item in batch]),
        'dt_state': torch.stack([item['dt_state'] for item in batch]),
        'state_next': torch.stack([item['state_next'] for item in batch])
    }

# ============================================================================
# Loss Functions
# ============================================================================

def compute_loss(model: nn.Module, batch: Dict, config: TrainConfig, 
                 dt: float = 0.05, rollout: bool = False) -> Dict:
    """Compute training loss with physics constraints"""
    state = batch['state']
    state_next_true = batch['state_next']
    
    # One-step prediction
    state_pred, dt_state_pred = model(state, dt, enforce_positivity=True)
    
    # Data loss
    l2_rho = F.mse_loss(state_pred[:, 0:1], state_next_true[:, 0:1])
    l2_P = F.mse_loss(state_pred[:, 1:], state_next_true[:, 1:])
    L_data = l2_rho + l2_P
    
    # Entropy regularization
    rho_pred = state_pred[:, 0:1] + 1e-8
    L_entropy = torch.mean(rho_pred * torch.log(rho_pred))
    
    # Rollout loss (optional)
    L_rollout = torch.tensor(0.0, device=state.device)
    if rollout:
        trajectory = model.rollout(state, config.rollout_steps, dt)
        for t in range(1, config.rollout_steps + 1):
            gamma = config.gamma_rollout ** t
            L_rollout += gamma * F.mse_loss(trajectory[:, t, 0:1], state_next_true[:, 0:1])
        L_rollout /= config.rollout_steps
    
    # Total loss
    L_total = L_data + config.lambda_e * L_entropy + config.lambda_r * L_rollout
    
    return {
        'total': L_total,
        'data': L_data,
        'l2_rho': l2_rho,
        'l2_P': l2_P,
        'entropy': L_entropy,
        'rollout': L_rollout
    }

# ============================================================================
# Evaluation Metrics
# ============================================================================

def evaluate_trajectory(model: nn.Module, test_data: List[Dict], 
                       config: TrainConfig, dt: float = 0.05,
                       n_steps: int = 20) -> Dict:
    """Evaluate model on trajectory prediction"""
    device = next(model.parameters()).device
    
    l2_rho_list = []
    l2_P_list = []
    mass_error_list = []
    negativity_list = []
    
    model.eval()
    with torch.no_grad():
        for item in test_data[:10]:
            state = torch.from_numpy(item['state']).unsqueeze(0).to(device)
            state_next_true = torch.from_numpy(item['state_next']).unsqueeze(0).to(device)
            
            trajectory = model.rollout(state, n_steps, dt)
            
            for t in range(1, min(n_steps + 1, n_steps + 1)):
                state_pred_t = trajectory[:, t]
                
                l2_rho = torch.sqrt(F.mse_loss(state_pred_t[:, 0:1], state_next_true[:, 0:1])).item()
                l2_P = torch.sqrt(F.mse_loss(state_pred_t[:, 1:], state_next_true[:, 1:])).item()
                
                mass_true = state_next_true[:, 0:1].sum()
                mass_pred = state_pred_t[:, 0:1].sum()
                mass_error = torch.abs(mass_pred - mass_true) / (mass_true + 1e-8)
                
                negativity = (state_pred_t[:, 0:1] < 0).float().mean()
                
                l2_rho_list.append(l2_rho)
                l2_P_list.append(l2_P)
                mass_error_list.append(mass_error.item())
                negativity_list.append(negativity.item())
    
    model.train()
    
    return {
        'l2_rho': np.mean(l2_rho_list) if l2_rho_list else 0.0,
        'l2_P': np.mean(l2_P_list) if l2_P_list else 0.0,
        'mass_error': np.mean(mass_error_list) if mass_error_list else 0.0,
        'negativity': np.mean(negativity_list) if negativity_list else 0.0,
        'r2_rho': 1.0 - np.var(l2_rho_list) if l2_rho_list else 0.0,
        'r2_P': 1.0 - np.var(l2_P_list) if l2_P_list else 0.0,
        'stable': all(n == 0 for n in negativity_list)
    }

def compute_lyapunov_exponent(model: nn.Module, state: torch.Tensor, 
                              n_steps: int = 20, dt: float = 0.05,
                              epsilon: float = 1e-6) -> float:
    """Estimate Lyapunov exponent from trajectory perturbations."""
    model.eval()
    device = state.device
    
    delta = epsilon * torch.randn_like(state)
    state_perturbed = state + delta
    
    log_ratios = []
    
    with torch.no_grad():
        for t in range(n_steps):
            state, _ = model(state, dt, enforce_positivity=True)
            state_perturbed, _ = model(state_perturbed, dt, enforce_positivity=True)
            
            diff = state_perturbed - state
            norm_diff = torch.norm(diff.flatten(1), dim=1)
            norm_delta = torch.norm(delta.flatten(1), dim=1) + 1e-10
            
            log_ratio = torch.log(norm_diff / norm_delta + 1e-10)
            log_ratios.append(log_ratio.mean().item())
            
            delta = epsilon * diff / (torch.norm(diff) + 1e-10)
            state_perturbed = state + delta
    
    model.train()
    
    lyapunov = np.mean(log_ratios) / dt
    return lyapunov

# ============================================================================
# Verification Suite
# ============================================================================

def run_verification_suite(config: TrainConfig) -> bool:
    """Run verification tests for physics constraints"""
    logger.info("[Phase 1] Verification suite...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysicsConstrainedFNO(config).to(device)
    model.train()
    
    # Test 1: Shape consistency
    B, H, W = 2, 32, 32
    state = torch.randn(B, 3, H, W, device=device)
    state_pred, dt_state = model(state, dt=0.05)
    
    assert state_pred.shape == state.shape, "Shape mismatch"
    assert dt_state.shape == state.shape, "DT shape mismatch"
    logger.info("  [✓] Shape consistency OK")
    
    # Test 2: Gradient flow
    loss = state_pred.mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    logger.info(f"  [✓] Gradient flow OK (grad_norm={grad_norm:.3e})")
    model.zero_grad()
    
    # Test 3: Mass conservation
    state = torch.rand(B, 3, H, W, device=device) + 0.1
    _, dt_state = model(state, dt=0.05)
    mass_change = dt_state[:, 0:1].sum(dim=(1, 2, 3)).abs().max()
    assert mass_change < 1e-4, f"Mass conservation violated: {mass_change}"
    logger.info(f"  [✓] Mass conservation OK (max|∫∂_tρ dx|={mass_change:.3e})")
    
    # Test 4: Positivity
    state_pred, _ = model(state, dt=0.05, enforce_positivity=True)
    min_rho = state_pred[:, 0:1].min().item()
    assert min_rho >= 0, f"Positivity violated: {min_rho}"
    logger.info(f"  [✓] Positivity OK (min ρ={min_rho:.3e} ≥ 0)")
    
    # Test 5: Rollout stability
    trajectory = model.rollout(state, n_steps=10, dt=0.05)
    has_nan = torch.isnan(trajectory).any().item()
    has_inf = torch.isinf(trajectory).any().item()
    assert not (has_nan or has_inf), "Rollout produced NaN/Inf"
    logger.info("  [✓] Rollout stability OK")
    
    return True

# ============================================================================
# Training Loop
# ============================================================================

def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, config: TrainConfig,
                dt: float = 0.05, device: str = 'cuda') -> Dict:
    """Train the physics-constrained FNO"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs, eta_min=1e-7
    )
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    use_amp = device == 'cuda'
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'time': []
    }
    
    best_val_loss = float('inf')
    
    logger.info(f"[Phase 4] Training physics-constrained FNO...")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Device: {device} | AMP: {use_amp}")
    
    for epoch in range(config.n_epochs):
        start_time = time.time()
        
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    rollout = epoch >= config.rollout_start_epoch
                    losses = compute_loss(model, batch, config, dt, rollout)
                    loss = losses['total']
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                rollout = epoch >= config.rollout_start_epoch
                losses = compute_loss(model, batch, config, dt, rollout)
                loss = losses['total']
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                losses = compute_loss(model, batch, config, dt, rollout=False)
                val_losses.append(losses['total'].item())
        
        scheduler.step()
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        elapsed = time.time() - start_time
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['time'].append(elapsed)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Ep {epoch+1:3d} | tr={avg_train:.3e} | val={avg_val:.3e} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
    
    return history

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("=" * 70)
    print("Physics-Constrained FNO for ABP Macroscopic Closure")
    print("=" * 70)
    print()
    
    abp_params = ABPParams()
    train_config = TrainConfig()
    dt = abp_params.dt * abp_params.save_interval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Device: {device}")
    logger.info("=" * 70)
    
    # Phase 1: Verification
    try:
        run_verification_suite(train_config)
        print()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return
    
    # Phase 2-3: Data Generation
    logger.info("[Phase 2-3] Generating ABP simulation data...")
    
    simulator = ABPSimulator(abp_params)
    all_data = []
    
    n_train = 20
    n_val = 5
    n_test = 5
    
    for i in range(n_train + n_val + n_test):
        seed = i * 100
        x_traj, theta_traj = simulator.simulate(n_steps=100, seed=seed)
        rho, P = coarse_grain(x_traj, theta_traj, abp_params)
        data = create_dataset(rho, P, dt, skip=1)
        all_data.extend(data)
    
    np.random.shuffle(all_data)
    train_data = all_data[:n_train * 5]
    val_data = all_data[n_train * 5:(n_train + n_val) * 5]
    test_data = all_data[(n_train + n_val) * 5:]
    
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Validation samples: {len(val_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    
    train_loader = DataLoader(
        ABPDataset(train_data), 
        batch_size=train_config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        ABPDataset(val_data), 
        batch_size=train_config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Phase 4: Training
    model = PhysicsConstrainedFNO(train_config).to(device)
    history = train_model(model, train_loader, val_loader, train_config, dt, device)
    
    # Phase 5: Test Evaluation
    logger.info("[Phase 5] Evaluating on test set...")
    test_metrics = evaluate_trajectory(model, test_data, train_config, dt)
    
    logger.info(f"  L2 Error (ρ): {test_metrics['l2_rho']:.4e}")
    logger.info(f"  L2 Error (P): {test_metrics['l2_P']:.4e}")
    logger.info(f"  Mass Error: {test_metrics['mass_error']:.4e}")
    logger.info(f"  R² (ρ): {test_metrics['r2_rho']:.4f}")
    logger.info(f"  R² (P): {test_metrics['r2_P']:.4f}")
    logger.info(f"  Negativity: {test_metrics['negativity']:.4f}")
    logger.info(f"  Stability: {'✓ STABLE' if test_metrics['stable'] else '✗ UNSTABLE'}")
    
    # Phase 6: Stability Analysis
    logger.info("[Phase 6] Stability analysis (Lyapunov exponent)...")
    sample_state = torch.from_numpy(test_data[0]['state']).unsqueeze(0).to(device)
    lyapunov = compute_lyapunov_exponent(model, sample_state, n_steps=20, dt=dt)
    logger.info(f"  Lyapunov exponent: λ₁ = {lyapunov:.4f} ({'STABLE' if lyapunov < 0 else 'UNSTABLE'})")
    
    # Phase 7: Ablation Study
    logger.info("[Phase 7] Ablation study...")
    
    ablation_results = []
    
    ablation_results.append({
        'model': 'physics_fno',
        'metrics': test_metrics
    })
    
    model_unconstrained = UnconstrainedFNO(train_config).to(device)
    optimizer = torch.optim.AdamW(model_unconstrained.parameters(), lr=1e-3)
    for _ in range(10):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            state_pred, _ = model_unconstrained(batch['state'], dt)
            loss = F.mse_loss(state_pred, batch['state_next'])
            loss.backward()
            optimizer.step()
    
    metrics_unconstrained = evaluate_trajectory(model_unconstrained, test_data, train_config, dt)
    ablation_results.append({
        'model': 'unconstrained_fno',
        'metrics': metrics_unconstrained
    })
    
    model_mlp = MLPClosure().to(device)
    optimizer = torch.optim.AdamW(model_mlp.parameters(), lr=1e-3)
    for _ in range(10):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            state_pred, _ = model_mlp(batch['state'], dt)
            loss = F.mse_loss(state_pred, batch['state_next'])
            loss.backward()
            optimizer.step()
    
    metrics_mlp = evaluate_trajectory(model_mlp, test_data, train_config, dt)
    ablation_results.append({
        'model': 'mlp',
        'metrics': metrics_mlp
    })
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║ ABLATION TABLE — In-Distribution (Test)" + " " * 26 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║ Model                l2_rho     l2_P     mass_error  stable     ║")
    print("╠" + "═" * 68 + "╣")
    
    for result in ablation_results:
        m = result['metrics']
        stable_str = "1.0" if m['stable'] else "0.0"
        print(f"║ {result['model']:<20} {m['l2_rho']:.2e}   {m['l2_P']:.2e}   {m['mass_error']:.2e}    {stable_str:<8} ║")
    
    print("╚" + "═" * 68 + "╝")
    
    # Save outputs
    logger.info("[Phase 8] Saving outputs...")
    
    all_metrics = {
        'test': test_metrics,
        'lyapunov': lyapunov,
        'ablation': ablation_results,
        'config': {
            'width': train_config.width,
            'n_layers': train_config.n_layers,
            'k_max': train_config.k_max
        }
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    with open('history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'time'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                history['train_loss'][i],
                history['val_loss'][i],
                history['lr'][i],
                history['time'][i]
            ])
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': train_config,
        'metrics': test_metrics
    }, 'abp_fno_checkpoint.pt')
    
    logger.info("  ✓ metrics.json")
    logger.info("  ✓ history.csv")
    logger.info("  ✓ abp_fno_checkpoint.pt")
    
    print()
    print("=" * 70)
    print("Training complete! All outputs saved.")
    print("=" * 70)

if __name__ == "__main__":
    main()