#!/usr/bin/env python3
"""
Quick verification that E2 equivariant encoder is properly integrated.
Uses minimal memory for testing in resource-constrained environments.
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_basic_import():
    """Test 1: Basic import"""
    print("Test 1: Importing E2EquivariantMapEncoder...", end=" ")
    try:
        from src.models import E2EquivariantMapEncoder, MapEncoder
        assert MapEncoder is E2EquivariantMapEncoder
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_instantiation():
    """Test 2: Model instantiation"""
    print("Test 2: Instantiating small model...", end=" ")
    try:
        from src.models import E2EquivariantMapEncoder
        model = E2EquivariantMapEncoder(
            img_size=32,
            in_channels=10,
            d_model=16,
            num_heads=2,
            num_layers=1,
            num_group_elements=4,
        )
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_forward_pass():
    """Test 3: Forward pass"""
    print("Test 3: Running forward pass...", end=" ")
    try:
        from src.models import E2EquivariantMapEncoder
        model = E2EquivariantMapEncoder(
            img_size=32,
            in_channels=10,
            d_model=16,
            num_heads=2,
            num_layers=1,
            num_group_elements=4,
        )
        model.eval()
        
        with torch.no_grad():
            radio_map = torch.randn(1, 5, 32, 32)
            osm_map = torch.randn(1, 5, 32, 32)
            spatial_tokens, cls_token = model(radio_map, osm_map)
        
        assert spatial_tokens.shape[0] == 1
        assert cls_token.shape == (1, 16)
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_rotation_invariance():
    """Test 4: Rotation invariance"""
    print("Test 4: Testing rotation invariance...", end=" ")
    try:
        from src.models import E2EquivariantMapEncoder
        model = E2EquivariantMapEncoder(
            img_size=32,
            in_channels=10,
            d_model=16,
            num_heads=2,
            num_layers=1,
            num_group_elements=4,
            dropout=0.0,
        )
        model.eval()
        
        torch.manual_seed(42)
        radio_map = torch.randn(1, 5, 32, 32)
        osm_map = torch.randn(1, 5, 32, 32)
        
        with torch.no_grad():
            _, cls_original = model(radio_map, osm_map)
            
            # Rotate 90 degrees
            radio_rot = torch.rot90(radio_map, k=1, dims=[2, 3])
            osm_rot = torch.rot90(osm_map, k=1, dims=[2, 3])
            _, cls_rotated = model(radio_rot, osm_rot)
        
        diff = torch.abs(cls_original - cls_rotated).mean().item()
        
        if diff < 0.1:
            print(f"✓ PASS (diff={diff:.4f})")
            return True
        else:
            print(f"✗ FAIL (diff={diff:.4f}, expected <0.1)")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_model_integration():
    """Test 5: Full model integration"""
    print("Test 5: Testing UELocalizationModel integration...", end=" ")
    try:
        import yaml
        from src.models import UELocalizationModel
        
        # Minimal config
        config = {
            'model': {
                'radio_encoder': {
                    'num_cells': 6,
                    'num_beams': 64,
                    'd_model': 16,
                    'nhead': 2,
                    'num_layers': 1,
                    'dropout': 0.1,
                    'max_seq_len': 5,
                    'rt_features_dim': 10,
                    'phy_features_dim': 8,
                    'mac_features_dim': 6,
                },
                'map_encoder': {
                    'img_size': 32,
                    'patch_size': 8,
                    'in_channels': 10,
                    'd_model': 16,
                    'nhead': 2,
                    'num_layers': 1,
                    'num_group_elements': 4,
                    'dropout': 0.1,
                    'radio_map_channels': 5,
                    'osm_map_channels': 5,
                },
                'fusion': {
                    'd_fusion': 16,
                    'nhead': 2,
                    'dropout': 0.1,
                },
                'coarse_head': {
                    'grid_size': 8,
                    'dropout': 0.1,
                },
                'fine_head': {
                    'top_k': 3,
                    'd_hidden': 16,
                    'dropout': 0.1,
                },
            },
            'dataset': {
                'name': 'test',
                'grid_size': 8,
            }
        }
        
        model = UELocalizationModel(config)
        assert model.map_encoder.__class__.__name__ == 'E2EquivariantMapEncoder'
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("E2 Equivariant Map Encoder - Quick Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_import,
        test_instantiation,
        test_forward_pass,
        test_rotation_invariance,
        test_model_integration,
    ]
    
    results = [test() for test in tests]
    
    print()
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
