"""
Tests for training loop logic, specifically augmentation and label consistency.
"""

import pytest
import torch
import yaml
from pathlib import Path
from src.training import UELocalizationLightning
from src.datasets.augmentations import RadioAugmentation

@pytest.fixture
def training_config(tmp_path):
    """Create a minimal training config."""
    config = {
        'dataset': {
            'scene_extent': 512.0,
        },
        'training': {
            'batch_size': 4,
            'loss': {
                'coarse_weight': 1.0,
                'fine_weight': 1.0,
                'use_physics_loss': False,
            },
            'augmentation': {
                'random_flip': True,
                'random_rotation': True,
                'scale_range': [0.8, 1.2],
            }
        },
        'model': {
            'radio_encoder': {
                'num_cells': 2, 'num_beams': 4, 'd_model': 64, 'nhead': 4, 
                'num_layers': 1, 'dropout': 0.1, 'max_seq_len': 10,
                'rt_features_dim': 16, 'phy_features_dim': 8, 'mac_features_dim': 6
            },
            'map_encoder': {
                'img_size': 256, 'patch_size': 16, 'in_channels': 10, 
                'd_model': 64, 'nhead': 4, 'num_layers': 1, 'dropout': 0.1,
                'radio_map_channels': 5, 'osm_map_channels': 5
            },
            'fusion': {'d_fusion': 64, 'nhead': 4, 'dropout': 0.1},
            'coarse_head': {'grid_size': 32, 'dropout': 0.1},
            'fine_head': {'top_k': 5, 'd_hidden': 64, 'dropout': 0.1}
        }
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path)

def test_training_step_augmentation_consistency(training_config):
    """Test that training_step correctly updates labels after augmentation."""
    model = UELocalizationLightning(training_config)
    model.model.eval() # Prevent weight updates from creating NaNs for now
    
    # Create a dummy batch
    B = 2
    grid_size = model.model.grid_size
    
    batch = {
        'measurements': {
            'rt_features': torch.randn(B, 10, 16),
            'phy_features': torch.randn(B, 10, 8),
            'mac_features': torch.randn(B, 10, 6),
            'cell_ids': torch.zeros(B, 10, dtype=torch.long),
            'beam_ids': torch.zeros(B, 10, dtype=torch.long),
            'timestamps': torch.arange(10).float().unsqueeze(0).expand(B, -1),
            'mask': torch.ones(B, 10, dtype=torch.bool),
        },
        'radio_map': torch.randn(B, 5, 256, 256),
        'osm_map': torch.randn(B, 5, 256, 256),
        'position': torch.tensor([[0.3, 0.3], [0.7, 0.7]]), # Normalized [0, 1]
        'cell_grid': torch.tensor([0, 0]), # Initial dummy values
    }
    
    # Manually set initial correct grid
    pos = batch['position']
    gx = (pos[:, 0] * grid_size).long()
    gy = (pos[:, 1] * grid_size).long()
    batch['cell_grid'] = gy * grid_size + gx
    
    # Mocking self.forward and self.model.compute_loss to inspect calls
    original_compute_loss = model.model.compute_loss
    
    captured_targets = []
    def mocked_compute_loss(outputs, targets, loss_weights):
        captured_targets.append(targets)
        return original_compute_loss(outputs, targets, loss_weights)
    
    model.model.compute_loss = mocked_compute_loss
    
    # Run training step
    # Important: Set manual seed for reproducibility of augmentation
    torch.manual_seed(42)
    model.training_step(batch, 0)
    
    # Check consistency
    targets = captured_targets[0]
    aug_pos = targets['position']
    aug_grid = targets['cell_grid']
    
    # Re-calculate correct grid from augmented position
    expected_gx = (aug_pos[:, 0] * grid_size).long()
    expected_gy = (aug_pos[:, 1] * grid_size).long()
    expected_grid = expected_gy * grid_size + expected_gx
    
    assert torch.equal(aug_grid, expected_grid), "cell_grid was not updated correctly after augmentation"
    assert aug_pos.min() >= 0.0 and aug_pos.max() <= 1.0, "Augmented position went out of bounds"
    
def test_no_nan_with_augmentations(training_config):
    """Verify that forward + loss doesn't produce NaNs with various augmentations."""
    model = UELocalizationLightning(training_config)
    
    B = 4
    grid_size = model.model.grid_size
    
    for i in range(10): # Run multiple times with different random seeds
        torch.manual_seed(i)
        
        batch = {
            'measurements': {
                'rt_features': torch.randn(B, 10, 16),
                'phy_features': torch.randn(B, 10, 8),
                'mac_features': torch.randn(B, 10, 6),
                'cell_ids': torch.zeros(B, 10, dtype=torch.long),
                'beam_ids': torch.zeros(B, 10, dtype=torch.long),
                'timestamps': torch.arange(10).float().unsqueeze(0).expand(B, -1),
                'mask': torch.ones(B, 10, dtype=torch.bool),
            },
            'radio_map': torch.randn(B, 5, 256, 256),
            'osm_map': torch.randn(B, 5, 256, 256),
            'position': torch.rand(B, 2),
            'cell_grid': torch.zeros(B, dtype=torch.long),
        }
        
        # Grid initialization
        gx = (batch['position'][:, 0] * grid_size).long()
        gy = (batch['position'][:, 1] * grid_size).long()
        batch['cell_grid'] = gy * grid_size + gx
        
        loss = model.training_step(batch, 0)
        
        assert not torch.isnan(loss), f"NaN loss encountered at iteration {i}"
        assert not torch.isinf(loss), f"Inf loss encountered at iteration {i}"
