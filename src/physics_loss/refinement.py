"""
Inference-Time Position Refinement using Physics Loss.

Provides gradient-based optimization to refine network predictions using
physics-consistency loss. Useful for high-stakes predictions or when
network confidence is low.
"""

import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .physics_loss import compute_physics_loss


@dataclass
class RefineConfig:
    """Configuration for position refinement."""
    
    # Number of gradient descent steps
    num_steps: int = 5
    
    # Learning rate (in meters per step)
    learning_rate: float = 0.5
    
    # Map extent for coordinate normalization
    map_extent: Tuple[float, float, float, float] = (0.0, 0.0, 512.0, 512.0)
    
    # Feature weights (if None, use uniform weights)
    feature_weights: Optional[torch.Tensor] = None
    
    # Whether to clip refined position to map extent
    clip_to_extent: bool = True
    
    # Minimum confidence threshold to trigger refinement
    # (if None, always refine)
    min_confidence_threshold: Optional[float] = None


def refine_position(
    initial_xy: torch.Tensor,
    observed_features: torch.Tensor,
    radio_maps: torch.Tensor,
    config: RefineConfig,
    confidence: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Refine predicted positions using gradient-based optimization.
    
    Uses physics loss to iteratively adjust positions to better match observed
    measurements with precomputed radio map features.
    
    Args:
        initial_xy: (batch, 2) initial position estimates from network
        observed_features: (batch, C) observed radio features
        radio_maps: (batch, C, H, W) precomputed radio maps
        config: Refinement configuration
        confidence: (batch,) optional confidence scores (0-1)
            If provided, only refine samples below threshold
            
    Returns:
        refined_xy: (batch, 2) refined positions
        info: Dictionary with refinement statistics:
            - 'loss_initial': initial physics loss
            - 'loss_final': final physics loss after refinement
            - 'distance_moved': L2 distance between initial and refined
            - 'num_refined': number of samples that were refined
            
    Example:
        >>> config = RefineConfig(num_steps=5, learning_rate=0.5)
        >>> initial_xy = torch.tensor([[100.0, 200.0], [150.0, 250.0]])
        >>> observed = torch.randn(2, 7)
        >>> radio_maps = torch.randn(2, 7, 512, 512)
        >>> refined_xy, info = refine_position(initial_xy, observed, radio_maps, config)
        >>> print(f"Moved {info['distance_moved'].mean():.2f} meters")
    """
    batch_size = initial_xy.shape[0]
    device = initial_xy.device
    
    # Determine which samples to refine
    if confidence is not None and config.min_confidence_threshold is not None:
        # Only refine low-confidence predictions
        refine_mask = confidence < config.min_confidence_threshold
        num_refined = refine_mask.sum().item()
    else:
        # Refine all samples
        refine_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_refined = batch_size
    
    # Initialize refined positions (copy initial)
    refined_xy = initial_xy.clone()
    
    # Compute initial loss for all samples
    with torch.no_grad():
        loss_initial = compute_physics_loss(
            initial_xy,
            observed_features,
            radio_maps,
            feature_weights=config.feature_weights,
            map_extent=config.map_extent,
        )
    
    if num_refined == 0:
        # No samples to refine
        info = {
            'loss_initial': loss_initial,
            'loss_final': loss_initial,
            'distance_moved': torch.zeros(batch_size, device=device),
            'num_refined': 0,
        }
        return refined_xy, info
    
    # Extract samples to refine
    xy_to_refine = initial_xy[refine_mask].detach().clone().requires_grad_(True)
    obs_to_refine = observed_features[refine_mask]
    maps_to_refine = radio_maps[refine_mask]
    
    # Optimizer (Adam for better convergence)
    optimizer = torch.optim.Adam([xy_to_refine], lr=config.learning_rate)
    
    # Gradient descent
    for step in range(config.num_steps):
        optimizer.zero_grad()
        
        # Compute physics loss
        loss = compute_physics_loss(
            xy_to_refine,
            obs_to_refine,
            maps_to_refine,
            feature_weights=config.feature_weights,
            map_extent=config.map_extent,
        )
        
        # Backprop and update
        loss.backward()
        optimizer.step()
        
        # Clip to map extent if configured
        if config.clip_to_extent:
            x_min, y_min, x_max, y_max = config.map_extent
            with torch.no_grad():
                xy_to_refine[:, 0].clamp_(x_min, x_max)
                xy_to_refine[:, 1].clamp_(y_min, y_max)
    
    # Update refined positions
    refined_xy[refine_mask] = xy_to_refine.detach()
    
    # Compute final loss
    with torch.no_grad():
        loss_final = compute_physics_loss(
            refined_xy,
            observed_features,
            radio_maps,
            feature_weights=config.feature_weights,
            map_extent=config.map_extent,
        )
    
    # Compute distance moved
    distance_moved = torch.zeros(batch_size, device=device)
    distance_moved[refine_mask] = torch.norm(
        refined_xy[refine_mask] - initial_xy[refine_mask],
        dim=-1,
    )
    
    info = {
        'loss_initial': loss_initial,
        'loss_final': loss_final,
        'distance_moved': distance_moved,
        'num_refined': num_refined,
    }
    
    return refined_xy, info


def batch_refine_positions(
    initial_xy: torch.Tensor,
    observed_features: torch.Tensor,
    radio_maps: torch.Tensor,
    config: RefineConfig,
    top_k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Refine multiple candidate positions and select the best one.
    
    Useful when network produces multiple candidates (e.g., from coarse head).
    Refines each candidate and selects the one with lowest physics loss.
    
    Args:
        initial_xy: (batch, K, 2) K candidate positions per sample
        observed_features: (batch, C) observed features
        radio_maps: (batch, C, H, W) precomputed maps
        config: Refinement configuration
        top_k: Number of candidates (K)
        
    Returns:
        best_xy: (batch, 2) best refined position per sample
        best_loss: (batch,) physics loss of best position
    """
    batch_size, K, _ = initial_xy.shape
    device = initial_xy.device
    
    # Expand observed features and maps for all candidates
    obs_expanded = observed_features.unsqueeze(1).expand(-1, K, -1)  # (batch, K, C)
    maps_expanded = radio_maps.unsqueeze(1).expand(-1, K, -1, -1, -1)  # (batch, K, C, H, W)
    
    # Reshape for refinement: (batch*K, ...)
    xy_flat = initial_xy.view(-1, 2)
    obs_flat = obs_expanded.reshape(-1, obs_expanded.shape[-1])
    maps_flat = maps_expanded.reshape(-1, *maps_expanded.shape[2:])
    
    # Refine all candidates
    refined_flat, _ = refine_position(
        xy_flat,
        obs_flat,
        maps_flat,
        config,
        confidence=None,  # Refine all candidates
    )
    
    # Reshape back: (batch, K, 2)
    refined_xy = refined_flat.view(batch_size, K, 2)
    
    # Compute physics loss for each refined candidate
    losses = torch.zeros(batch_size, K, device=device)
    for k in range(K):
        losses[:, k] = compute_physics_loss(
            refined_xy[:, k],
            observed_features,
            radio_maps,
            feature_weights=config.feature_weights,
            map_extent=config.map_extent,
        )
    
    # Select best candidate (lowest loss)
    best_indices = losses.argmin(dim=1)  # (batch,)
    best_xy = refined_xy[torch.arange(batch_size), best_indices]  # (batch, 2)
    best_loss = losses[torch.arange(batch_size), best_indices]  # (batch,)
    
    return best_xy, best_loss
