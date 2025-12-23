"""
Physics-Consistency Loss for Map-Conditioned Localization.

Enforces consistency between predicted UE positions and observed radio measurements
by comparing measurements to precomputed radio map features at predicted locations.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .differentiable_lookup import differentiable_lookup


@dataclass
class PhysicsLossConfig:
    """Configuration for physics-consistency loss."""
    
    # Feature weights (importance of each feature in loss)
    feature_weights: Dict[str, float] = None
    
    # Map extent in meters (x_min, y_min, x_max, y_max)
    map_extent: Tuple[float, float, float, float] = (0.0, 0.0, 512.0, 512.0)
    
    # Loss function ('mse' or 'huber')
    loss_type: str = 'mse'
    
    # Huber delta (only used if loss_type='huber')
    huber_delta: float = 1.0
    
    # Whether to normalize features before computing loss
    normalize_features: bool = True
    
    # Padding mode for out-of-bounds positions
    padding_mode: str = 'border'
    
    def __post_init__(self):
        if self.feature_weights is None:
            # Default weights from IMPLEMENTATION_GUIDE.md
            self.feature_weights = {
                'path_gain': 1.0,   # High weight, most reliable
                'toa': 0.5,         # Medium, affected by NLOS bias
                'aoa': 0.3,         # Lower, high measurement noise
                'snr': 0.8,         # High, good signal quality indicator
                'sinr': 0.8,        # High, good signal quality indicator
                'throughput': 0.2,  # Lower, depends on scheduler/load
                'bler': 0.2,        # Lower, depends on channel conditions
            }


class PhysicsLoss(nn.Module):
    """
    Physics-consistency loss using precomputed radio maps.
    
    Compares observed radio features (from measurements) with simulated features
    (from precomputed radio maps) at the predicted UE position. Uses differentiable
    bilinear interpolation to sample map features at arbitrary positions.
    
    Loss enforces that:
        predicted_position → radio_map[predicted_position] ≈ observed_features
    
    This acts as a physics-based regularizer that keeps predictions consistent
    with electromagnetic propagation physics encoded in the radio maps.
    """
    
    def __init__(self, config: PhysicsLossConfig):
        super().__init__()
        self.config = config
        
        # Convert feature weights to tensor for efficient computation
        # Assume feature order: [path_gain, toa, aoa, snr, sinr, throughput, bler]
        feature_names = ['path_gain', 'toa', 'aoa', 'snr', 'sinr', 'throughput', 'bler']
        weights = [config.feature_weights.get(name, 1.0) for name in feature_names]
        self.register_buffer('feature_weights', torch.tensor(weights))
        
    def forward(
        self,
        predicted_xy: torch.Tensor,
        observed_features: torch.Tensor,
        radio_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics-consistency loss.
        
        Args:
            predicted_xy: (batch, 2) predicted UE positions in meters [x, y]
            observed_features: (batch, C) observed radio features from measurements
                Features: [path_gain, toa, aoa, snr, sinr, throughput, bler]
            radio_maps: (batch, C, H, W) precomputed radio map features
                Same feature channels as observed_features
                
        Returns:
            loss: scalar physics-consistency loss
            
        Example:
            >>> physics_loss = PhysicsLoss(PhysicsLossConfig())
            >>> predicted_xy = torch.tensor([[100.0, 200.0], [150.0, 250.0]])
            >>> observed = torch.randn(2, 7)  # 7 features
            >>> radio_maps = torch.randn(2, 7, 512, 512)
            >>> loss = physics_loss(predicted_xy, observed, radio_maps)
        """
        # Sample radio map features at predicted positions
        simulated_features = differentiable_lookup(
            predicted_xy=predicted_xy,
            radio_maps=radio_maps,
            map_extent=self.config.map_extent,
            padding_mode=self.config.padding_mode,
        )
        
        # Normalize features if configured
        if self.config.normalize_features:
            # Normalize each feature to zero mean, unit variance
            obs_mean = observed_features.mean(dim=0, keepdim=True)
            obs_std = observed_features.std(dim=0, keepdim=True) + 1e-6
            observed_norm = (observed_features - obs_mean) / obs_std
            simulated_norm = (simulated_features - obs_mean) / obs_std
        else:
            observed_norm = observed_features
            simulated_norm = simulated_features
        
        # Compute per-feature residuals
        residuals = observed_norm - simulated_norm  # (batch, C)
        
        # Apply loss function
        if self.config.loss_type == 'mse':
            feature_losses = residuals ** 2
        elif self.config.loss_type == 'huber':
            feature_losses = torch.nn.functional.huber_loss(
                simulated_norm,
                observed_norm,
                reduction='none',
                delta=self.config.huber_delta,
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}")
        
        # Apply feature weights: (batch, C) * (C,) -> (batch, C)
        weighted_losses = feature_losses * self.feature_weights.unsqueeze(0)
        
        # Average over features and batch
        loss = weighted_losses.sum(dim=1).mean()
        
        return loss
    
    def compute_per_feature_loss(
        self,
        predicted_xy: torch.Tensor,
        observed_features: torch.Tensor,
        radio_maps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss per feature for analysis.
        
        Returns:
            Dict mapping feature name to scalar loss
        """
        # Sample features
        simulated_features = differentiable_lookup(
            predicted_xy=predicted_xy,
            radio_maps=radio_maps,
            map_extent=self.config.map_extent,
            padding_mode=self.config.padding_mode,
        )
        
        # Normalize if configured
        if self.config.normalize_features:
            obs_mean = observed_features.mean(dim=0, keepdim=True)
            obs_std = observed_features.std(dim=0, keepdim=True) + 1e-6
            observed_norm = (observed_features - obs_mean) / obs_std
            simulated_norm = (simulated_features - obs_mean) / obs_std
        else:
            observed_norm = observed_features
            simulated_norm = simulated_features
        
        # Compute per-feature losses
        residuals = observed_norm - simulated_norm
        feature_losses = residuals ** 2  # (batch, C)
        
        # Average over batch
        mean_losses = feature_losses.mean(dim=0)  # (C,)
        
        # Map to feature names
        feature_names = ['path_gain', 'toa', 'aoa', 'snr', 'sinr', 'throughput', 'bler']
        loss_dict = {name: mean_losses[i] for i, name in enumerate(feature_names)}
        
        return loss_dict


def compute_physics_loss(
    predicted_xy: torch.Tensor,
    observed_features: torch.Tensor,
    radio_maps: torch.Tensor,
    feature_weights: Optional[torch.Tensor] = None,
    map_extent: Tuple[float, float, float, float] = (0.0, 0.0, 512.0, 512.0),
) -> torch.Tensor:
    """
    Functional API for physics loss (no module instantiation needed).
    
    Args:
        predicted_xy: (batch, 2) predicted positions
        observed_features: (batch, C) observed features
        radio_maps: (batch, C, H, W) precomputed maps
        feature_weights: (C,) optional feature weights (default: all 1.0)
        map_extent: (x_min, y_min, x_max, y_max) in meters
        
    Returns:
        loss: scalar
    """
    # Sample features at predicted positions
    simulated_features = differentiable_lookup(
        predicted_xy=predicted_xy,
        radio_maps=radio_maps,
        map_extent=map_extent,
    )
    
    # Compute residuals
    residuals = (observed_features - simulated_features) ** 2  # (batch, C)
    
    # Apply weights if provided
    if feature_weights is not None:
        residuals = residuals * feature_weights.unsqueeze(0)
    
    # Average over features and batch
    loss = residuals.sum(dim=1).mean()
    
    return loss
