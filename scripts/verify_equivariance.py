import torch
import sys
import os
import math

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.map_encoder import E2EquivariantMapEncoder

def test_rotation_invariance():
    """Test that E2 encoder produces invariant outputs under rotation."""
    print("\n=== Testing E2 Equivariant Encoder: Rotation Invariance ===")
    
    # Initialize E2 equivariant model
    model = E2EquivariantMapEncoder(
        img_size=128,  # Smaller for faster testing
        in_channels=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_group_elements=4,  # p4 group (4 rotations)
        dropout=0.0,  # No dropout for deterministic testing
    )
    model.eval()
    
    # Create random input
    torch.manual_seed(42)
    B = 2
    radio_map = torch.randn(B, 5, 128, 128)
    osm_map = torch.randn(B, 5, 128, 128)
    
    print(f"Input shape: {radio_map.shape}")
    
    # Test different rotations
    rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°
    outputs = []
    
    with torch.no_grad():
        for k in rotations:
            # Rotate inputs
            if k > 0:
                radio_rot = torch.rot90(radio_map, k=k, dims=[-2, -1])
                osm_rot = torch.rot90(osm_map, k=k, dims=[-2, -1])
            else:
                radio_rot = radio_map
                osm_rot = osm_map
            
            # Forward pass
            _, cls_token = model(radio_rot, osm_rot)
            outputs.append(cls_token)
            print(f"Rotation {k*90}°: CLS token norm = {cls_token.norm(dim=1).mean().item():.4f}")
    
    # Compute pairwise differences
    print("\n=== Rotation Invariance Test (CLS tokens should be similar) ===")
    max_diff = 0.0
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            diff = torch.abs(outputs[i] - outputs[j]).mean().item()
            max_diff = max(max_diff, diff)
            print(f"Difference between {rotations[i]*90}° and {rotations[j]*90}°: {diff:.6f}")
    
    if max_diff < 0.1:
        print(f"\n✓ [PASS] Model IS rotation invariant (max diff = {max_diff:.6f})")
        return True
    else:
        print(f"\n✗ [FAIL] Model is NOT fully rotation invariant (max diff = {max_diff:.6f})")
        return False


def test_reflection_invariance():
    """Test that E2 encoder with p4m produces invariant outputs under reflection."""
    print("\n=== Testing E2 Equivariant Encoder: Reflection Invariance ===")
    
    # Initialize E2 equivariant model with reflections
    model = E2EquivariantMapEncoder(
        img_size=128,
        in_channels=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_group_elements=8,  # p4m group (4 rotations x 2 reflections)
        dropout=0.0,
    )
    model.eval()
    
    # Create random input
    torch.manual_seed(42)
    B = 2
    radio_map = torch.randn(B, 5, 128, 128)
    osm_map = torch.randn(B, 5, 128, 128)
    
    with torch.no_grad():
        # Original
        _, cls_original = model(radio_map, osm_map)
        
        # Horizontal flip
        radio_flip_h = torch.flip(radio_map, dims=[3])
        osm_flip_h = torch.flip(osm_map, dims=[3])
        _, cls_flip_h = model(radio_flip_h, osm_flip_h)
        
        # Vertical flip
        radio_flip_v = torch.flip(radio_map, dims=[2])
        osm_flip_v = torch.flip(osm_map, dims=[2])
        _, cls_flip_v = model(radio_flip_v, osm_flip_v)
    
    diff_h = torch.abs(cls_original - cls_flip_h).mean().item()
    diff_v = torch.abs(cls_original - cls_flip_v).mean().item()
    
    print(f"Difference after horizontal flip: {diff_h:.6f}")
    print(f"Difference after vertical flip: {diff_v:.6f}")
    
    max_diff = max(diff_h, diff_v)
    if max_diff < 0.1:
        print(f"\n✓ [PASS] Model IS reflection invariant (max diff = {max_diff:.6f})")
        return True
    else:
        print(f"\n✗ [FAIL] Model is NOT fully reflection invariant (max diff = {max_diff:.6f})")
        return False


def test_spatial_equivariance():
    """Test that spatial features transform appropriately under rotation."""
    print("\n=== Testing E2 Equivariant Encoder: Spatial Feature Equivariance ===")
    
    model = E2EquivariantMapEncoder(
        img_size=64,  # Smaller for visualization
        in_channels=10,
        d_model=32,
        num_heads=4,
        num_layers=1,
        num_group_elements=4,
        dropout=0.0,
    )
    model.eval()
    
    # Create input with a clear spatial pattern
    torch.manual_seed(42)
    B = 1
    radio_map = torch.randn(B, 5, 64, 64)
    osm_map = torch.randn(B, 5, 64, 64)
    
    with torch.no_grad():
        # Original spatial grid
        grid_original = model.get_spatial_grid(radio_map, osm_map)
        
        # Rotated input
        radio_rot = torch.rot90(radio_map, k=1, dims=[-2, -1])
        osm_rot = torch.rot90(osm_map, k=1, dims=[-2, -1])
        grid_rotated_input = model.get_spatial_grid(radio_rot, osm_rot)
        
        # Rotate original grid for comparison
        grid_original_rotated = torch.rot90(grid_original, k=1, dims=[-2, -1])
    
    # Compute error
    mse = torch.mean((grid_original_rotated - grid_rotated_input) ** 2).item()
    max_diff = torch.max(torch.abs(grid_original_rotated - grid_rotated_input)).item()
    
    print(f"Spatial grid shape: {grid_original.shape}")
    print(f"MSE between Rot(Enc(x)) and Enc(Rot(x)): {mse:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    
    if mse < 0.01:
        print(f"\n✓ [PASS] Spatial features ARE equivariant (MSE = {mse:.6f})")
        return True
    else:
        print(f"\n✗ [FAIL] Spatial features are NOT fully equivariant (MSE = {mse:.6f})")
        print("Note: Some difference is expected due to interpolation and numerical precision")
        return False


def main():
    print("=" * 70)
    print("E(2) Equivariant Vision Transformer Verification")
    print("=" * 70)
    
    results = []
    
    # Test 1: Rotation invariance (CLS token)
    results.append(("Rotation Invariance", test_rotation_invariance()))
    
    # Test 2: Reflection invariance (CLS token)
    results.append(("Reflection Invariance", test_reflection_invariance()))
    
    # Test 3: Spatial equivariance
    results.append(("Spatial Equivariance", test_spatial_equivariance()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - E2 Equivariance Verified!")
    else:
        print("✗ SOME TESTS FAILED - Review results above")
    print("=" * 70)


if __name__ == "__main__":
    main()
