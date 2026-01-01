"""
Tests to diagnose position-related bugs:
1. Large validation median error
2. UE positions appearing the same in Comet visualizations
"""

import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import zarr

sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.radio_dataset import RadioLocalizationDataset
from src.models.ue_localization_model import UELocalizationModel
from src.training import UELocalizationLightning


class TestPositionDataLoading:
    """Test that positions are being loaded and normalized correctly."""
    
    def test_position_diversity_in_dataset(self, tmp_path):
        """Test that dataset has diverse UE positions, not all the same."""
        # Use the specific known good dataset
        zarr_path = Path("data/processed/sionna_dataset/dataset_20260101_184259.zarr")
        if not zarr_path.exists():
            pytest.skip("Test dataset not available")
        
        # Load dataset
        dataset = RadioLocalizationDataset(
            zarr_path=str(zarr_path),
            split='train',
            normalize=True
        )
        
        # Check we have samples
        assert len(dataset) > 0, "Dataset is empty"
        
        # Get multiple samples (sample widely to get diverse positions)
        num_samples = min(100, len(dataset))
        sample_indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
        positions = []
        
        for idx in sample_indices:
            sample = dataset[idx]
            pos = sample['position']
            positions.append(pos.numpy())
        
        positions = np.array(positions)
        
        # Check positions are diverse (not all the same)
        pos_std = positions.std(axis=0)
        print(f"\nPosition statistics:")
        print(f"  Mean: {positions.mean(axis=0)}")
        print(f"  Std:  {pos_std}")
        print(f"  Min:  {positions.min(axis=0)}")
        print(f"  Max:  {positions.max(axis=0)}")
        print(f"  First 5 positions:\n{positions[:5]}")
        
        # Positions should have non-zero standard deviation
        assert pos_std[0] > 0.01, f"X positions have no diversity: std={pos_std[0]}"
        assert pos_std[1] > 0.01, f"Y positions have no diversity: std={pos_std[1]}"
        
        # Check positions are in normalized range [0, 1]
        assert (positions >= 0).all(), "Some positions are negative"
        assert (positions <= 1).all(), "Some positions exceed 1.0"
    
    def test_position_to_cell_grid_conversion(self):
        """Test that position to cell_grid conversion is correct."""
        grid_size = 32
        sample_extent = 512.0
        cell_size = sample_extent / grid_size
        
        # Test corner cases
        test_cases = [
            # (x, y) in meters -> expected (grid_x, grid_y)
            (0, 0, 0, 31),  # Bottom-left -> top-left in image coords
            (512, 512, 31, 0),  # Top-right -> bottom-right in image coords (clamped)
            (256, 256, 16, 16),  # Center
            (8, 8, 0, 31),  # Near bottom-left (y clamped to max)
            (504, 504, 31, 0),  # Near top-right (y clamped to min)
        ]
        
        for x, y, expected_gx, expected_gy in test_cases:
            norm_x = x / sample_extent
            norm_y = y / sample_extent
            
            grid_x = int(norm_x * grid_size)
            grid_y = int((1.0 - norm_y) * grid_size)
            
            # Ensure within bounds
            grid_x = max(0, min(grid_x, grid_size - 1))
            grid_y = max(0, min(grid_y, grid_size - 1))
            
            print(f"Position ({x:.0f}, {y:.0f}) -> norm({norm_x:.3f}, {norm_y:.3f}) -> grid({grid_x}, {grid_y})")
            
            assert grid_x == expected_gx, f"X grid mismatch for ({x}, {y}): got {grid_x}, expected {expected_gx}"
            assert grid_y == expected_gy, f"Y grid mismatch for ({x}, {y}): got {grid_x}, expected {expected_gy}"
    
    def test_cell_grid_to_position_conversion(self):
        """Test that cell_grid to position conversion matches the inverse."""
        grid_size = 32
        cell_size = 16.0  # 512 / 32
        
        # Test some cell indices
        test_indices = [0, 31, 32, 1023, 512]  # Various positions in grid
        
        for cell_idx in test_indices:
            # Convert to grid coordinates
            row = cell_idx // grid_size
            col = cell_idx % grid_size
            
            # Convert to position (cell center) using the same logic as FineHead
            y = (grid_size - 1 - row + 0.5) * cell_size
            x = (col + 0.5) * cell_size
            
            print(f"Cell {cell_idx} (row={row}, col={col}) -> ({x:.1f}, {y:.1f})")
            
            # Verify bounds
            assert 0 <= x <= 512, f"X out of bounds: {x}"
            assert 0 <= y <= 512, f"Y out of bounds: {y}"
            
            # Verify the conversion is invertible
            norm_x = x / 512.0
            norm_y = y / 512.0
            
            # Convert back to grid
            back_gx = int(norm_x * grid_size)
            back_gy = int((1.0 - norm_y) * grid_size)
            
            # Clamp
            back_gx = max(0, min(back_gx, grid_size - 1))
            back_gy = max(0, min(back_gy, grid_size - 1))
            
            back_idx = back_gy * grid_size + back_gx
            
            # Should be close (within 1 cell due to rounding)
            assert abs(back_idx - cell_idx) <= 1, f"Conversion not invertible: {cell_idx} -> ({x}, {y}) -> {back_idx}"


class TestModelPredictionDiversity:
    """Test that model produces diverse predictions, not all the same."""
    
    def test_model_output_diversity(self):
        """Test that model outputs diverse predictions for different inputs."""
        # Create a small test model
        config = {
            'model': {
                'radio_encoder': {
                    'num_cells': 4,
                    'num_beams': 8,
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'max_seq_len': 100,
                    'rt_features_dim': 8,
                    'phy_features_dim': 8,
                    'mac_features_dim': 4,
                },
                'map_encoder': {
                    'img_size': 64,
                    'patch_size': 16,
                    'in_channels': 10,
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'use_e2_equivariant': True,
                    'radio_map_channels': 5,
                    'osm_map_channels': 5,
                },
                'fusion': {
                    'd_radio': 64,
                    'd_map': 64,
                    'd_fusion': 64,
                    'nhead': 4,
                    'dropout': 0.1,
                },
                'coarse_head': {
                    'd_input': 64,
                    'grid_size': 32,
                    'dropout': 0.1,
                },
                'fine_head': {
                    'd_input': 64,
                    'd_hidden': 64,
                    'top_k': 3,
                    'sigma_min': 0.01,
                    'dropout': 0.1,
                },
                'grid_size': 32,
                'scene_extent': 512,
            },
            'dataset': {
                'scene_extent': 512,
            }
        }
        
        model = UELocalizationModel(config)
        model.eval()
        
        batch_size = 8
        seq_len = 10
        
        # Create diverse inputs
        measurements = {
            'cell_ids': torch.randint(0, 4, (batch_size, seq_len)),
            'beam_ids': torch.randint(0, 8, (batch_size, seq_len)),
            'rt_features': torch.randn(batch_size, seq_len, 8),
            'phy_features': torch.randn(batch_size, seq_len, 8),
            'mac_features': torch.randn(batch_size, seq_len, 4),
            'timestamps': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float(),
            'mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
        
        # Create different maps for each sample
        radio_maps = torch.randn(batch_size, 5, 64, 64)
        osm_maps = torch.randn(batch_size, 5, 64, 64)
        
        with torch.no_grad():
            outputs = model(measurements, radio_maps, osm_maps)
        
        predictions = outputs['predicted_position']
        
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Predictions:\n{predictions}")
        
        # Check predictions are diverse (lower threshold since model trained on buggy data)
        pred_std = predictions.std(dim=0)
        print(f"Prediction std: {pred_std}")
        
        # We expect at least some diversity in predictions for different inputs
        # Note: Lower threshold since model may have been trained on buggy normalized data
        assert pred_std[0] > 0.01, f"X predictions have no diversity: std={pred_std[0]}"
        assert pred_std[1] > 0.01, f"Y predictions have no diversity: std={pred_std[1]}"
        
        # Check predictions are in reasonable range (normalized [0, 1] or meters)
        print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")


class TestVisualizationBug:
    """Test to reproduce the visualization bug where positions appear the same."""
    
    def test_validation_step_position_logging(self):
        """Test that validation step logs correct positions."""
        # This would require a full training setup
        # For now, we'll test the position normalization logic
        
        # Simulate what happens in validation_step
        batch_size = 4
        
        # Normalized positions [0, 1]
        true_pos = torch.tensor([
            [0.1, 0.2],
            [0.5, 0.5],
            [0.8, 0.9],
            [0.3, 0.7],
        ])
        
        pred_pos = torch.tensor([
            [0.15, 0.25],
            [0.45, 0.55],
            [0.75, 0.85],
            [0.35, 0.65],
        ])
        
        extent = torch.tensor([512.0] * batch_size)
        
        # Compute errors in meters
        errors = torch.norm(pred_pos - true_pos, dim=-1) * extent
        
        print(f"\nTrue positions (normalized): \n{true_pos}")
        print(f"Pred positions (normalized): \n{pred_pos}")
        print(f"Errors (meters): {errors}")
        
        # Check errors are reasonable
        assert (errors > 0).all(), "Some errors are zero (predictions match exactly)"
        assert (errors < extent).all(), "Some errors exceed scene extent"
        
        # Check visualization coordinates
        h, w = 256, 256
        for i in range(batch_size):
            true_px = true_pos[i, 0].item() * (w - 1)
            true_py = (1.0 - true_pos[i, 1].item()) * (h - 1)  # Flip Y
            
            pred_px = pred_pos[i, 0].item() * (w - 1)
            pred_py = (1.0 - pred_pos[i, 1].item()) * (h - 1)  # Flip Y
            
            print(f"Sample {i}: true=({true_px:.0f}, {true_py:.0f}), pred=({pred_px:.0f}, {pred_py:.0f})")
            
            # Positions should be different
            assert abs(true_px - pred_px) > 1 or abs(true_py - pred_py) > 1, \
                f"Sample {i}: positions are too close in pixel space"


class TestErrorComputation:
    """Test that error computation is correct."""
    
    def test_error_computation_logic(self):
        """Test that errors are computed correctly."""
        # Normalized positions
        pred_pos = torch.tensor([[0.5, 0.5]])
        true_pos = torch.tensor([[0.6, 0.6]])
        extent = torch.tensor([512.0])
        
        # Compute error
        error_norm = torch.norm(pred_pos - true_pos, dim=-1)
        error_meters = error_norm * extent
        
        # Expected: sqrt((0.1)^2 + (0.1)^2) * 512 = sqrt(0.02) * 512 ≈ 72.4 meters
        expected = np.sqrt(0.02) * 512.0
        
        print(f"\nError (normalized): {error_norm.item():.4f}")
        print(f"Error (meters): {error_meters.item():.2f}")
        print(f"Expected: {expected:.2f}")
        
        assert abs(error_meters.item() - expected) < 0.1, "Error computation is incorrect"
    
    def test_large_error_scenarios(self):
        """Test scenarios that might cause large errors."""
        extent = torch.tensor([512.0])
        
        # Scenario 1: Prediction at one corner, GT at opposite corner
        pred_pos = torch.tensor([[0.0, 0.0]])
        true_pos = torch.tensor([[1.0, 1.0]])
        error = torch.norm(pred_pos - true_pos, dim=-1) * extent
        
        # Maximum possible error: sqrt(2) * 512 ≈ 724 meters
        max_error = np.sqrt(2) * 512.0
        print(f"\nMax error (diagonal): {error.item():.2f} meters")
        assert abs(error.item() - max_error) < 1.0
        
        # Scenario 2: Position is always the same (the bug!)
        pred_pos = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        true_pos = torch.tensor([[0.1, 0.1], [0.5, 0.9], [0.9, 0.5]])
        errors = torch.norm(pred_pos - true_pos, dim=-1) * extent.expand(3)
        
        print(f"Errors when pred is always (0.5, 0.5): {errors}")
        
        # If model always predicts center, errors should vary based on GT
        assert errors.std() > 10, "Errors should vary if GT positions are diverse"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
