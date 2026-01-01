"""
Differentiable Lookup for Physics Loss.

Provides differentiable bilinear interpolation for sampling precomputed radio map
features at predicted UE positions. Uses PyTorch's F.grid_sample for full gradient support.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def normalize_coords(
    xy_meters: torch.Tensor,
    map_extent: Tuple[float, float, float, float],
) -> torch.Tensor:
    """
    Normalize metric coordinates to [-1, 1] for grid_sample.
    
    Args:
        xy_meters: (batch, 2) coordinates in meters [x, y]
        map_extent: (x_min, y_min, x_max, y_max) in meters
        
    Returns:
        xy_norm: (batch, 2) normalized to [-1, 1] where:
            - (-1, -1) corresponds to (x_min, y_min)
            - (1, 1) corresponds to (x_max, y_max)
    
    Note:
        grid_sample expects (x, y) order where x is horizontal (width) 
        and y is vertical (height).
    """
    x_min, y_min, x_max, y_max = map_extent
    
    # Extract x, y
    x = xy_meters[..., 0]
    y = xy_meters[..., 1]
    
    # Normalize to [-1, 1]
    # X: West(-1) -> East(1)
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    
    # Y: South(1) -> North(-1)
    # Standard grid_sample: -1 is Top, 1 is Bottom
    # Map Coords: Min Y is South, Max Y is North
    # We want North (Max Y) -> Top (-1)
    # We want South (Min Y) -> Bottom (1)
    # y_norm = 1 - 2 * (y - min) / (max - min)
    y_norm = 1 - 2 * (y - y_min) / (y_max - y_min)
    
    # Stack back
    xy_norm = torch.stack([x_norm, y_norm], dim=-1)
    
    return xy_norm


def differentiable_lookup(
    predicted_xy: torch.Tensor,
    radio_maps: torch.Tensor,
    map_extent: Tuple[float, float, float, float],
    padding_mode: str = 'border',
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Differentiable bilinear interpolation of radio map features.
    
    This function samples precomputed radio map features at predicted UE positions
    using bilinear interpolation. The operation is fully differentiable w.r.t.
    predicted_xy, enabling gradient-based optimization.
    
    Args:
        predicted_xy: (batch, 2) predicted positions in meters [x, y]
        radio_maps: (batch, C, H, W) precomputed feature grids where:
            - C: number of features (e.g., 7: path_gain, toa, aoa, snr, sinr, throughput, bler)
            - H, W: spatial resolution (e.g., 512x512 for 1m/pixel)
        map_extent: (x_min, y_min, x_max, y_max) in meters
        padding_mode: How to handle out-of-bounds coordinates
            - 'zeros': use 0 for out-of-bounds
            - 'border': clamp to edge values (default, more stable)
            - 'reflection': reflect at boundary
        align_corners: If True, corner pixels are aligned with corners of extent
        
    Returns:
        sampled_features: (batch, C) interpolated feature values at predicted positions
        
    Example:
        >>> predicted_xy = torch.tensor([[100.0, 200.0], [150.0, 250.0]])  # 2 positions
        >>> radio_maps = torch.randn(2, 7, 512, 512)  # 2 scenes, 7 features
        >>> map_extent = (0.0, 0.0, 512.0, 512.0)  # 512m x 512m
        >>> features = differentiable_lookup(predicted_xy, radio_maps, map_extent)
        >>> features.shape  # (2, 7)
        
    Note:
        - Gradients flow through predicted_xy for physics-based optimization
        - radio_maps should be precomputed and frozen (no gradients needed)
        - For training, use with physics_loss to enforce consistency
        - For inference, use with refine_position for gradient-based refinement
    """
    batch_size = predicted_xy.shape[0]
    num_features = radio_maps.shape[1]
    
    # Normalize coordinates to [-1, 1]
    xy_norm = normalize_coords(predicted_xy, map_extent)
    
    # Reshape for grid_sample: (batch, 1, 1, 2)
    # grid_sample expects (N, H_out, W_out, 2) for sampling grid
    # We want to sample at a single point, so H_out=1, W_out=1
    xy_grid = xy_norm.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, 2)
    
    # Bilinear interpolation (fully differentiable)
    # Input: (batch, C, H, W)
    # Grid: (batch, 1, 1, 2)
    # Output: (batch, C, 1, 1)
    sampled = F.grid_sample(
        radio_maps,
        xy_grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    
    # Remove spatial dimensions: (batch, C, 1, 1) -> (batch, C)
    sampled_features = sampled.squeeze(-1).squeeze(-1)
    
    return sampled_features


def batch_differentiable_lookup(
    predicted_xy: torch.Tensor,
    radio_maps: torch.Tensor,
    map_extent: Tuple[float, float, float, float],
    padding_mode: str = 'border',
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Batch version of differentiable_lookup for multiple positions per sample.
    
    Useful when sampling multiple candidate positions (e.g., top-K from coarse head).
    
    Args:
        predicted_xy: (batch, K, 2) K predicted positions per sample
        radio_maps: (batch, C, H, W) precomputed feature grids
        map_extent: (x_min, y_min, x_max, y_max) in meters
        padding_mode: How to handle out-of-bounds coordinates
        align_corners: If True, corner pixels are aligned
        
    Returns:
        sampled_features: (batch, K, C) features for each position
    """
    batch_size, K, _ = predicted_xy.shape
    num_features = radio_maps.shape[1]
    
    # Normalize all positions: (batch, K, 2)
    xy_norm = normalize_coords(predicted_xy.view(-1, 2), map_extent)
    xy_norm = xy_norm.view(batch_size, K, 2)
    
    # Reshape for grid_sample: (batch, K, 1, 2)
    xy_grid = xy_norm.unsqueeze(2)  # (batch, K, 1, 2)
    
    # Sample: (batch, C, K, 1)
    sampled = F.grid_sample(
        radio_maps,
        xy_grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    
    # Reshape: (batch, C, K, 1) -> (batch, K, C)
    sampled_features = sampled.squeeze(-1).permute(0, 2, 1)
    
    return sampled_features
