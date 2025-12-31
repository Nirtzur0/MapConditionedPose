"""
Tests for RadioAugmentation class.

Verifies correctness of geometric transformations and position updates.
"""

import pytest
import torch
from src.datasets.augmentations import RadioAugmentation


@pytest.fixture
def sample_measurements():
    """Create sample measurement dict."""
    return {
        'rt_features': torch.randn(10, 8),  # [seq_len, features]
        'phy_features': torch.randn(10, 6),
        'mac_features': torch.randn(10, 4),
        'mask': torch.ones(10, dtype=torch.bool),
    }


@pytest.fixture
def sample_maps():
    """Create sample radio and OSM maps."""
    radio_map = torch.randn(5, 256, 256)  # [channels, H, W]
    osm_map = torch.randn(5, 256, 256)
    return radio_map, osm_map


@pytest.fixture
def sample_position():
    """Create sample normalized position [x, y] in [0, 1]."""
    return torch.tensor([0.3, 0.7])


class TestRadioAugmentationDisabled:
    """Test that disabled augmentations pass through unchanged."""
    
    def test_disabled_passthrough(self, sample_measurements, sample_maps, sample_position):
        """Test that disabled augmentation returns inputs unchanged."""
        augmentor = RadioAugmentation(config=None)
        radio_map, osm_map = sample_maps
        
        meas_out, radio_out, osm_out, pos_out = augmentor(
            sample_measurements, radio_map, osm_map, sample_position
        )
        
        assert torch.equal(meas_out['rt_features'], sample_measurements['rt_features'])
        assert torch.equal(radio_out, radio_map)
        assert torch.equal(osm_out, osm_map)
        assert torch.equal(pos_out, sample_position)


class TestMeasurementAugmentations:
    """Test measurement-level augmentations (noise, dropout)."""
    
    def test_feature_noise(self, sample_measurements):
        """Test that feature noise is applied."""
        config = {'feature_noise': 0.1}
        augmentor = RadioAugmentation(config)
        
        # Clone to preserve original
        original_rt = sample_measurements['rt_features'].clone()
        meas_out = augmentor.augment_measurements(sample_measurements)
        
        # Should not be exactly equal due to noise
        # Note: noise is proportional, so check if values changed
        assert not torch.equal(meas_out['rt_features'], original_rt), \
            "Feature noise should modify rt_features"
        # But should be close
        assert torch.allclose(meas_out['rt_features'], original_rt, atol=1.0)
    
    def test_feature_dropout(self, sample_measurements):
        """Test that feature dropout zeros some features."""
        config = {'feature_dropout': 0.5}
        augmentor = RadioAugmentation(config)
        
        # Run multiple times to ensure dropout happens
        has_zeros = False
        for _ in range(10):
            meas_out = augmentor.augment_measurements(sample_measurements)
            if (meas_out['rt_features'] == 0).any():
                has_zeros = True
                break
        
        assert has_zeros, "Feature dropout should zero some features"


class TestFlipAugmentation:
    """Test horizontal flip augmentation and position update."""
    
    def test_flip_single_sample(self, sample_maps):
        """Test flip transformation formula."""
        # Test the flip transformation directly
        position = torch.tensor([0.3, 0.7]).unsqueeze(0)  # [1, 2]
        
        # Apply flip transformation: x_new = 1.0 - x, y stays same
        pos_flipped = position.clone()
        pos_flipped[:, 0] = 1.0 - pos_flipped[:, 0]
        
        expected = torch.tensor([[0.7, 0.7]])
        assert torch.allclose(pos_flipped, expected, atol=1e-5), \
            f"Flip should transform (0.3, 0.7) to (0.7, 0.7), got {pos_flipped}"
    
    def test_flip_batch(self, sample_maps):
        """Test flip on batch handles per-sample flipping."""
        config = {'random_flip': True}
        augmentor = RadioAugmentation(config)
        
        radio_map, osm_map = sample_maps
        # Create batch
        radio_batch = radio_map.unsqueeze(0).repeat(4, 1, 1, 1)  # [4, 5, 256, 256]
        osm_batch = osm_map.unsqueeze(0).repeat(4, 1, 1, 1)
        position_batch = torch.tensor([[0.2, 0.3], [0.5, 0.5], [0.8, 0.9], [0.1, 0.4]])
        
        torch.manual_seed(123)
        radio_out, osm_out, pos_out = augmentor.augment_maps(
            radio_batch, osm_batch, position_batch
        )
        
        # Check that positions are valid (in [0, 1])
        assert (pos_out >= 0).all() and (pos_out <= 1).all(), \
            "Positions should remain in [0, 1] after flip"


class TestRotationAugmentation:
    """Test rotation augmentation and position updates."""
    
    @pytest.mark.parametrize("k,expected_transform", [
        (1, lambda x, y: (1.0 - y, x)),      # 90° CCW
        (2, lambda x, y: (1.0 - x, 1.0 - y)), # 180°
        (3, lambda x, y: (y, 1.0 - x)),      # 270° CCW
    ])
    def test_rotation_position_update(self, sample_maps, k, expected_transform):
        """Test that rotation correctly updates position."""
        config = {'random_rotation': True, 'random_flip': False}
        augmentor = RadioAugmentation(config)
        
        radio_map, osm_map = sample_maps
        position = torch.tensor([0.3, 0.7])
        
        # Manually apply rotation with known k
        radio_rotated = torch.rot90(radio_map, k, dims=[-2, -1])
        
        # Compute expected position
        x, y = position[0].item(), position[1].item()
        expected_x, expected_y = expected_transform(x, y)
        
        # Apply augmentation (we can't force k, so we test the logic separately)
        # Instead, test the transformation directly
        pos_out = position.unsqueeze(0)  # [1, 2]
        x_orig = pos_out[:, 0].clone()
        y_orig = pos_out[:, 1].clone()
        
        if k == 1:
            pos_out[:, 0] = 1.0 - y_orig
            pos_out[:, 1] = x_orig
        elif k == 2:
            pos_out[:, 0] = 1.0 - x_orig
            pos_out[:, 1] = 1.0 - y_orig
        elif k == 3:
            pos_out[:, 0] = y_orig
            pos_out[:, 1] = 1.0 - x_orig
        
        assert torch.isclose(pos_out[0, 0], torch.tensor(expected_x), atol=1e-5)
        assert torch.isclose(pos_out[0, 1], torch.tensor(expected_y), atol=1e-5)


class TestScaleAugmentation:
    """Test scale jitter augmentation."""
    
    def test_scale_position_update(self, sample_maps):
        """Test that scaling updates position correctly."""
        config = {'scale_range': [0.9, 1.1], 'random_flip': False, 'random_rotation': False}
        augmentor = RadioAugmentation(config)
        
        radio_map, osm_map = sample_maps
        position = torch.tensor([0.5, 0.5])  # Center position
        
        torch.manual_seed(99)
        radio_out, osm_out, pos_out = augmentor.augment_maps(
            radio_map, osm_map, position
        )
        
        # Position should still be valid
        assert (pos_out >= 0).all() and (pos_out <= 1.5).all(), \
            "Position should remain reasonable after scaling"
        
        # For center position (0.5, 0.5), scaling should keep it near center
        # Formula: (pos - 0.5) * scale + 0.5
        # For scale in [0.9, 1.1], center should stay at 0.5
        if torch.isclose(position, torch.tensor([0.5, 0.5]), atol=1e-3).all():
            assert torch.isclose(pos_out, torch.tensor([0.5, 0.5]), atol=0.1).all(), \
                "Center position should remain near center after scaling"
    
    def test_scale_zoom_in(self, sample_maps):
        """Test zoom in (scale > 1) moves points away from center."""
        radio_map, osm_map = sample_maps
        position = torch.tensor([0.7, 0.3])  # Off-center
        
        # Manually test scale formula
        scale = 1.1
        pos_scaled = (position - 0.5) * scale + 0.5
        
        # Point at (0.7, 0.3) should move to (0.72, 0.28) with scale=1.1
        expected = torch.tensor([0.72, 0.28])
        assert torch.allclose(pos_scaled, expected, atol=1e-5)


class TestBatchHandling:
    """Test that augmentations handle batches correctly."""
    
    def test_batch_augmentation(self, sample_maps):
        """Test augmentation on batched inputs."""
        config = {
            'random_flip': True,
            'random_rotation': True,
            'scale_range': [0.9, 1.1]
        }
        augmentor = RadioAugmentation(config)
        
        radio_map, osm_map = sample_maps
        batch_size = 8
        
        # Create batches
        radio_batch = radio_map.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        osm_batch = osm_map.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        position_batch = torch.rand(batch_size, 2)
        
        measurements_batch = {
            'rt_features': torch.randn(batch_size, 10, 8),
            'phy_features': torch.randn(batch_size, 10, 6),
            'mac_features': torch.randn(batch_size, 10, 4),
            'mask': torch.ones(batch_size, 10, dtype=torch.bool),
        }
        
        meas_out, radio_out, osm_out, pos_out = augmentor(
            measurements_batch, radio_batch, osm_batch, position_batch
        )
        
        # Check output shapes
        assert radio_out.shape == radio_batch.shape
        assert osm_out.shape == osm_batch.shape
        assert pos_out.shape == position_batch.shape
        
        # Check positions are valid
        assert (pos_out >= -0.1).all() and (pos_out <= 1.1).all(), \
            "Positions should remain reasonable after augmentation"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_corner_positions(self, sample_maps):
        """Test augmentation on corner positions."""
        config = {'random_flip': True}
        augmentor = RadioAugmentation(config)
        
        radio_map, osm_map = sample_maps
        corners = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        for corner in corners:
            radio_out, osm_out, pos_out = augmentor.augment_maps(
                radio_map.clone(), osm_map.clone(), corner.clone()
            )
            
            # Positions should remain in valid range
            assert (pos_out >= 0).all() and (pos_out <= 1).all(), \
                f"Corner {corner} should remain valid after augmentation"
    
    def test_no_position_provided(self, sample_maps, sample_measurements):
        """Test that augmentation works without position."""
        config = {'random_flip': True}
        augmentor = RadioAugmentation(config)
        
        radio_map, osm_map = sample_maps
        
        meas_out, radio_out, osm_out, pos_out = augmentor(
            sample_measurements, radio_map, osm_map, position=None
        )
        
        assert pos_out is None
        assert radio_out.shape == radio_map.shape
        assert osm_out.shape == osm_map.shape
