"""
Tests for M4: Physics Loss Module

Tests cover:
- Differentiable lookup with F.grid_sample
- Physics-consistency loss computation
- Radio map generation (mocked Sionna)
- Gradient flow verification
- Position refinement
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.physics_loss import (
    differentiable_lookup,
    normalize_coords,
    PhysicsLoss,
    PhysicsLossConfig,
    compute_physics_loss,
    refine_position,
    RefineConfig,
)


class TestDifferentiableLookup:
    """Test differentiable bilinear interpolation."""
    
    def test_normalize_coords(self):
        """Test coordinate normalization to [-1, 1]."""
        xy_meters = torch.tensor([
            [0.0, 0.0],      # (x_min, y_min) -> (-1, -1)
            [512.0, 512.0],  # (x_max, y_max) -> (1, 1)
            [256.0, 256.0],  # center -> (0, 0)
        ])
        map_extent = (0.0, 0.0, 512.0, 512.0)
        
        xy_norm = normalize_coords(xy_meters, map_extent)
        
        # Note: y-axis is inverted for grid_sample normalization.
        expected = torch.tensor([
            [-1.0, 1.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ])
        
        assert xy_norm.shape == (3, 2)
        assert torch.allclose(xy_norm, expected, atol=1e-5)
    
    def test_differentiable_lookup_single(self):
        """Test lookup at single point."""
        # Create simple radio map (batch=1, features=1, 4x4 spatial)
        radio_map = torch.tensor([[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]]])  # (1, 1, 4, 4)
        
        # Sample at center (should be bilinear interp of center 4 values)
        predicted_xy = torch.tensor([[2.0, 2.0]])  # Center of 4x4 grid (0-4 range)
        map_extent = (0.0, 0.0, 4.0, 4.0)
        
        sampled = differentiable_lookup(predicted_xy, radio_map, map_extent)
        
        # Center should interpolate 6, 7, 10, 11 -> ~8.5
        assert sampled.shape == (1, 1)
        assert sampled[0, 0] > 6.0 and sampled[0, 0] < 11.0
    
    def test_differentiable_lookup_batch(self):
        """Test batched lookup."""
        batch_size = 4
        num_features = 7
        H, W = 128, 128
        
        radio_maps = torch.randn(batch_size, num_features, H, W)
        predicted_xy = torch.rand(batch_size, 2) * 128.0  # Random positions
        map_extent = (0.0, 0.0, 128.0, 128.0)
        
        sampled = differentiable_lookup(predicted_xy, radio_maps, map_extent)
        
        assert sampled.shape == (batch_size, num_features)
        assert not torch.isnan(sampled).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through lookup."""
        radio_maps = torch.randn(2, 3, 32, 32)
        predicted_xy = torch.tensor([[16.0, 16.0], [8.0, 24.0]], requires_grad=True)
        map_extent = (0.0, 0.0, 32.0, 32.0)
        
        sampled = differentiable_lookup(predicted_xy, radio_maps, map_extent)
        
        # Compute loss
        loss = sampled.sum()
        loss.backward()
        
        # Check gradients exist
        assert predicted_xy.grad is not None
        assert predicted_xy.grad.shape == (2, 2)
        assert not torch.isnan(predicted_xy.grad).any()
    
    def test_out_of_bounds_border(self):
        """Test padding mode 'border' for out-of-bounds coordinates."""
        radio_maps = torch.ones(1, 1, 10, 10) * 5.0  # Uniform map
        predicted_xy = torch.tensor([[-10.0, -10.0]])  # Way out of bounds
        map_extent = (0.0, 0.0, 10.0, 10.0)
        
        sampled = differentiable_lookup(
            predicted_xy, radio_maps, map_extent, padding_mode='border'
        )
        
        # Should clamp to edge value (5.0)
        assert torch.allclose(sampled, torch.tensor([[5.0]]), atol=1e-4)


class TestPhysicsLoss:
    """Test physics-consistency loss."""
    
    def test_physics_loss_init(self):
        """Test PhysicsLoss initialization."""
        channel_names = ('path_gain', 'toa', 'aoa', 'snr', 'sinr', 'throughput', 'bler')
        weights = {'path_gain': 1.0, 'toa': 0.5}
        config = PhysicsLossConfig(channel_names=channel_names, feature_weights=weights)
        loss_fn = PhysicsLoss(config)
        
        assert loss_fn.feature_weights.shape == (7,)
        assert loss_fn.feature_weights[0] == 1.0  # path_gain
        assert loss_fn.feature_weights[1] == 0.5  # toa
    
    def test_physics_loss_forward(self):
        """Test physics loss computation."""
        channel_names = tuple(f'feat_{i}' for i in range(7))
        config = PhysicsLossConfig(normalize_features=False, channel_names=channel_names)
        loss_fn = PhysicsLoss(config)
        
        batch_size = 8
        num_features = 7
        
        predicted_xy = torch.rand(batch_size, 2) * 128.0
        observed_features = torch.randn(batch_size, num_features)
        radio_maps = torch.randn(batch_size, num_features, 128, 128)
        
        loss = loss_fn(predicted_xy, observed_features, radio_maps)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0.0  # Non-negative (MSE)
        assert not torch.isnan(loss)
    
    def test_physics_loss_perfect_match(self):
        """Test that loss is zero when prediction matches observation."""
        channel_names = tuple(f'feat_{i}' for i in range(7))
        config = PhysicsLossConfig(normalize_features=False, channel_names=channel_names)
        loss_fn = PhysicsLoss(config)
        
        # Create uniform radio map
        radio_maps = torch.ones(2, 7, 32, 32) * 10.0
        
        # Observed features match map
        observed_features = torch.ones(2, 7) * 10.0
        
        # Any position should give same features (uniform map)
        predicted_xy = torch.tensor([[16.0, 16.0], [8.0, 24.0]])
        
        loss = loss_fn(predicted_xy, observed_features, radio_maps)
        
        # Loss should be very small (nearly zero)
        assert loss < 1e-4
    
    def test_gradient_flow_through_loss(self):
        """Test gradient flow through physics loss."""
        channel_names = tuple(f'feat_{i}' for i in range(7))
        config = PhysicsLossConfig(normalize_features=False, channel_names=channel_names)  # Disable normalization for single sample
        loss_fn = PhysicsLoss(config)
        
        predicted_xy = torch.tensor([[25.0, 50.0]], requires_grad=True)
        observed_features = torch.randn(1, 7)
        radio_maps = torch.randn(1, 7, 128, 128)
        
        loss = loss_fn(predicted_xy, observed_features, radio_maps)
        loss.backward()
        
        assert predicted_xy.grad is not None
        assert not torch.isnan(predicted_xy.grad).any()
    
    def test_per_feature_loss(self):
        """Test per-feature loss computation."""
        channel_names = ('path_gain', 'toa', 'aoa', 'snr', 'sinr', 'throughput', 'bler')
        config = PhysicsLossConfig(normalize_features=False, channel_names=channel_names)
        loss_fn = PhysicsLoss(config)
        
        predicted_xy = torch.rand(4, 2) * 128.0
        observed_features = torch.randn(4, 7)
        radio_maps = torch.randn(4, 7, 128, 128)
        
        loss_dict = loss_fn.compute_per_feature_loss(
            predicted_xy, observed_features, radio_maps
        )
        
        assert len(loss_dict) == 7
        assert 'path_gain' in loss_dict
        assert 'toa' in loss_dict
        assert all(v >= 0.0 for v in loss_dict.values())
    
    def test_functional_api(self):
        """Test functional compute_physics_loss."""
        predicted_xy = torch.rand(4, 2) * 128.0
        observed_features = torch.randn(4, 7)
        radio_maps = torch.randn(4, 7, 128, 128)
        
        loss = compute_physics_loss(
            predicted_xy, observed_features, radio_maps
        )
        
        assert loss.ndim == 0
        assert loss >= 0.0


class TestPositionRefinement:
    """Test gradient-based position refinement."""
    
    def test_refine_position_basic(self):
        """Test basic position refinement."""
        config = RefineConfig(num_steps=5, learning_rate=0.5)
        
        # Create a peaked radio map (higher values in center)
        x = torch.linspace(0, 32, 32)
        y = torch.linspace(0, 32, 32)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        center_x, center_y = 16.0, 16.0
        radio_map = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / 50.0)
        radio_maps = radio_map.unsqueeze(0).unsqueeze(0).expand(2, 7, -1, -1)  # (2, 7, 32, 32)
        
        # Observed features are peak value
        observed_features = torch.ones(2, 7)
        
        # Initial guess is off-center
        initial_xy = torch.tensor([[12.0, 12.0], [20.0, 20.0]])
        
        refined_xy, info = refine_position(
            initial_xy, observed_features, radio_maps, config
        )
        
        assert refined_xy.shape == (2, 2)
        assert info['num_refined'] == 2
        assert info['loss_final'] <= info['loss_initial']  # Should improve
    
    def test_refine_with_confidence_threshold(self):
        """Test selective refinement based on confidence."""
        config = RefineConfig(
            num_steps=5,
            learning_rate=0.5,
            min_confidence_threshold=0.5,
        )
        
        radio_maps = torch.randn(4, 7, 32, 32)
        observed_features = torch.randn(4, 7)
        initial_xy = torch.rand(4, 2) * 32.0
        confidence = torch.tensor([0.9, 0.3, 0.7, 0.2])  # Only indices 1,3 should refine
        
        refined_xy, info = refine_position(
            initial_xy, observed_features, radio_maps, config, confidence
        )
        
        assert info['num_refined'] == 2  # Only low-confidence samples
        assert refined_xy.shape == (4, 2)
        
        # High-confidence samples should not move
        assert torch.allclose(refined_xy[0], initial_xy[0])
        assert torch.allclose(refined_xy[2], initial_xy[2])
    
    def test_gradient_descent_converges(self):
        """Test that refinement reduces physics loss."""
        config = RefineConfig(num_steps=10, learning_rate=1.0)
        
        # Create simple linear gradient map
        radio_maps = torch.linspace(0, 1, 64).view(1, 1, 8, 8).expand(1, 7, -1, -1)
        
        # Target: high values (near 1.0)
        observed_features = torch.ones(1, 7) * 0.9
        
        # Initial: low values (near 0.0)
        initial_xy = torch.tensor([[1.0, 1.0]])
        
        refined_xy, info = refine_position(
            initial_xy, observed_features, radio_maps, config
        )
        
        # Should move towards high-value region
        assert info['loss_final'] < info['loss_initial']
        assert info['distance_moved'][0] > 0
    
    def test_clip_to_extent(self):
        """Test that refinement respects map extent."""
        config = RefineConfig(
            num_steps=20,
            learning_rate=10.0,  # Large LR to potentially go out of bounds
            clip_to_extent=True,
            map_extent=(0.0, 0.0, 32.0, 32.0),
        )
        
        radio_maps = torch.randn(1, 7, 32, 32)
        observed_features = torch.randn(1, 7)
        initial_xy = torch.tensor([[16.0, 16.0]])
        
        refined_xy, info = refine_position(
            initial_xy, observed_features, radio_maps, config
        )
        
        # Check bounds
        assert (refined_xy >= 0.0).all()
        assert (refined_xy <= 32.0).all()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_pipeline(self):
        """Test full pipeline: lookup -> loss -> refinement."""
        batch_size = 4
        num_features = 7
        
        # Generate synthetic radio maps
        radio_maps = torch.randn(batch_size, num_features, 128, 128)
        
        # Generate ground truth positions
        true_xy = torch.rand(batch_size, 2) * 128.0
        
        # Sample observed features at true positions
        observed_features = differentiable_lookup(
            true_xy, radio_maps, (0.0, 0.0, 128.0, 128.0)
        )
        
        # Start with noisy predictions
        noise = torch.randn(batch_size, 2) * 10.0
        initial_xy = true_xy + noise
        initial_xy = torch.clamp(initial_xy, 0.0, 128.0)
        
        # Compute initial loss
        channel_names = tuple(f'feat_{i}' for i in range(7))
        config = PhysicsLossConfig(
            map_extent=(0.0, 0.0, 128.0, 128.0),
            normalize_features=False,  # Disable for stability in test
            channel_names=channel_names
        )
        loss_fn = PhysicsLoss(config)
        initial_loss = loss_fn(initial_xy, observed_features, radio_maps)
        
        # Refine positions
        refine_config = RefineConfig(
            num_steps=10,
            learning_rate=1.0,
            map_extent=(0.0, 0.0, 128.0, 128.0),
        )
        refined_xy, info = refine_position(
            initial_xy, observed_features, radio_maps, refine_config
        )
        
        # Verify refinement info
        assert info['num_refined'] == batch_size
        assert torch.all(info['distance_moved'] >= 0.0)
        
        # Check that positions were updated (at least some moved)
        assert info['distance_moved'].sum() > 0.0
        
        # Check that refined positions are within bounds
        assert torch.all(refined_xy >= 0.0)
        assert torch.all(refined_xy <= 128.0)
    
    def test_training_step_simulation(self):
        """Simulate training step with physics loss."""
        # Model prediction (simulated) - use parameter to ensure it's a leaf
        batch_size = 8
        predicted_xy_param = torch.nn.Parameter(torch.rand(batch_size, 2) * 128.0)
        
        # Ground truth
        true_xy = torch.rand(batch_size, 2) * 128.0
        
        # Radio maps and observed features
        radio_maps = torch.randn(batch_size, 7, 128, 128)
        observed_features = torch.randn(batch_size, 7)
        
        # Compute supervised loss (simple L2)
        supervised_loss = torch.nn.functional.mse_loss(predicted_xy_param, true_xy)
        
        # Compute physics loss
        physics_loss = compute_physics_loss(
            predicted_xy_param, observed_features, radio_maps
        )
        
        # Combined loss
        lambda_phys = 0.1
        total_loss = supervised_loss + lambda_phys * physics_loss
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        assert predicted_xy_param.grad is not None
        assert not torch.isnan(predicted_xy_param.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
