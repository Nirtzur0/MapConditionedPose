"""
Physics Loss Module for Differentiable Physics Regularization (M4).

This module provides physics-consistency loss using precomputed Sionna radio maps
with differentiable bilinear interpolation. It enables the model to learn from both
supervised labels and physics-based constraints.

Components:
    - RadioMapGenerator: Generates precomputed Sionna radio maps
    - differentiable_lookup: Bilinear interpolation using F.grid_sample
    - PhysicsLoss: Multi-feature weighted MSE physics-consistency loss
    - refine_position: Gradient-based position refinement at inference time
"""

from .differentiable_lookup import differentiable_lookup, normalize_coords
from .physics_loss import PhysicsLoss, compute_physics_loss, PhysicsLossConfig
from .refinement import refine_position, RefineConfig

__all__ = [
    'differentiable_lookup',
    'normalize_coords',
    'PhysicsLoss',
    'PhysicsLossConfig',
    'compute_physics_loss',
    'refine_position',
    'RefineConfig',
]
