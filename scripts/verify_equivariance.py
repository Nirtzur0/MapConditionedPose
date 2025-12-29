import torch
import sys
import os
import math

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.map_encoder import MapEncoder

def test_equivariance():
    print("Testing MapEncoder Rotation Equivariance (with escnn)...")
    
    # Initialize model with escnn-based EquivariantPatchEmbedding
    model = MapEncoder(
        img_size=256,
        patch_size=16,
        in_channels=10,
        d_model=64,  # Small model for speed (must be divisible by 4 for C4 repr)
        nhead=4,
        num_layers=2
    )
    model.eval()
    
    # Create random input
    B = 1
    radio_map = torch.randn(B, 5, 256, 256)
    osm_map = torch.randn(B, 5, 256, 256)
    
    # 1. Forward pass with original
    with torch.no_grad():
        out_original = model.get_spatial_grid(radio_map, osm_map)
        
    # 2. Rotate inputs by 90 degrees
    radio_rot = torch.rot90(radio_map, k=1, dims=[-2, -1])
    osm_rot = torch.rot90(osm_map, k=1, dims=[-2, -1])
    
    # 3. Forward pass with rotated
    with torch.no_grad():
        out_rotated_input = model.get_spatial_grid(radio_rot, osm_rot)

    # 4. Rotate original output to compare
    out_original_rotated = torch.rot90(out_original, k=1, dims=[-2, -1])
    
    # Check dimensions
    print(f"Output shape: {out_original.shape}")
    
    # Compute error
    mse = torch.mean((out_original_rotated - out_rotated_input) ** 2)
    max_diff = torch.max(torch.abs(out_original_rotated - out_rotated_input))
    
    print(f"MSE between Rot(Enc(x)) and Enc(Rot(x)): {mse.item():.6f}")
    print(f"Max difference: {max_diff.item():.6f}")
    
    # escnn with C4 should give perfect equivariance up to numerical precision
    if mse < 1e-5:
        print("\n[PASS] Model IS rotation equivariant (escnn C4).")
    elif mse < 0.1:
        print("\n[PARTIAL] Model shows improved rotation stability (MSE < 0.1).")
    else:
        print(f"\n[FAIL] Model is NOT rotation equivariant (MSE = {mse.item():.4f}).")


if __name__ == "__main__":
    test_equivariance()
