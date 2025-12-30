"""
Example: Using Variable-Length Data Support in Zarr Writer

This example demonstrates how to use the new max dimensions feature
to properly handle scenes with varying numbers of cells and paths.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_generation.zarr_writer import ZarrDatasetWriter
from src.data_generation.zarr_utils import compute_max_dimensions, log_dimension_statistics


def create_sample_scene_data(num_samples: int, num_cells: int, num_paths: int) -> dict:
    """Create sample scene data with specified dimensions."""
    return {
        'positions': np.random.uniform(-500, 500, (num_samples, 3)),
        'timestamps': np.arange(num_samples) * 0.2,
        'rt/path_gains': np.random.randn(num_samples, num_cells, num_paths) + 
                         1j * np.random.randn(num_samples, num_cells, num_paths),
        'rt/path_delays': np.sort(np.random.uniform(0, 1e-6, (num_samples, num_cells, num_paths)), axis=-1),
        'rt/rms_delay_spread': np.random.uniform(10e-9, 200e-9, (num_samples, num_cells)),
        'phy_fapi/rsrp': np.random.uniform(-100, -60, (num_samples, num_cells)),
        'phy_fapi/rsrq': np.random.uniform(-15, -5, (num_samples, num_cells)),
        'phy_fapi/sinr': np.random.uniform(-5, 20, (num_samples, num_cells)),
        'phy_fapi/cqi': np.random.randint(0, 16, (num_samples, num_cells)),
        'phy_fapi/ri': np.random.randint(1, 5, (num_samples, num_cells)),
        'phy_fapi/pmi': np.random.randint(0, 16, (num_samples, num_cells)),
        'mac_rrc/serving_cell_id': np.random.randint(0, num_cells, num_samples),
        'mac_rrc/timing_advance': np.random.randint(0, 1000, num_samples),
    }


def main():
    print("=" * 80)
    print("Variable-Length Data Support Example")
    print("=" * 80)
    
    # Create scenes with varying dimensions
    print("\n1. Creating scenes with varying dimensions...")
    scenes = [
        ('scene_001', create_sample_scene_data(num_samples=100, num_cells=1, num_paths=50)),
        ('scene_002', create_sample_scene_data(num_samples=150, num_cells=2, num_paths=150)),
        ('scene_003', create_sample_scene_data(num_samples=120, num_cells=1, num_paths=100)),
    ]
    
    print(f"   Created {len(scenes)} scenes:")
    for scene_id, data in scenes:
        path_gains_shape = data['rt/path_gains'].shape
        print(f"   - {scene_id}: {path_gains_shape[0]} samples, "
              f"{path_gains_shape[1]} cells, {path_gains_shape[2]} paths")
    
    # Compute max dimensions
    print("\n2. Computing maximum dimensions across all scenes...")
    scene_data_list = [data for _, data in scenes]
    max_dims = compute_max_dimensions(scene_data_list)
    
    print("   Max dimensions:")
    for key, shape in sorted(max_dims.items()):
        if 'rt/' in key:  # Only show RT arrays for brevity
            print(f"   - {key}: {shape}")
    
    # Log statistics
    print("\n3. Dimension statistics:")
    log_dimension_statistics(scene_data_list)
    
    # Create Zarr writer with max dimensions
    print("\n4. Creating Zarr dataset with max dimensions...")
    output_dir = Path("test_variable_length_output")
    output_dir.mkdir(exist_ok=True)
    
    writer = ZarrDatasetWriter(output_dir=output_dir, chunk_size=50)
    
    # Set max dimensions BEFORE appending data
    writer.set_max_dimensions(max_dims)
    
    # Append all scenes
    print("\n5. Appending scenes to dataset...")
    for scene_id, scene_data in scenes:
        writer.append(scene_data, scene_id=scene_id)
        print(f"   ✓ Appended {scene_id}")
    
    # Finalize
    print("\n6. Finalizing dataset...")
    store_path = writer.finalize()
    print(f"   ✓ Dataset saved to: {store_path}")
    
    # Verify the dataset
    print("\n7. Verifying dataset structure...")
    import zarr
    store = zarr.open(str(store_path), mode='r')
    
    print(f"   Total samples: {store.attrs.get('num_samples', 'unknown')}")
    print(f"   Total scenes: {store.attrs.get('num_scenes', 'unknown')}")
    
    # Check array shapes
    if 'rt' in store and 'path_gains' in store['rt']:
        path_gains_shape = store['rt/path_gains'].shape
        print(f"   path_gains shape: {path_gains_shape}")
        print(f"   Expected: ({sum(d['rt/path_gains'].shape[0] for _, d in scenes)}, 2, 150)")
    
    # Check metadata
    if 'metadata' in store:
        if 'actual_num_cells' in store['metadata']:
            actual_cells = store['metadata/actual_num_cells'][:]
            print(f"   actual_num_cells: min={actual_cells.min()}, max={actual_cells.max()}")
        
        if 'actual_num_paths' in store['metadata']:
            actual_paths = store['metadata/actual_num_paths'][:]
            print(f"   actual_num_paths: min={actual_paths.min()}, max={actual_paths.max()}")
    
    print("\n" + "=" * 80)
    print("✓ Example completed successfully!")
    print("=" * 80)
    print(f"\nDataset location: {store_path}")
    print("\nKey takeaways:")
    print("  1. Compute max dimensions BEFORE creating the writer")
    print("  2. Call set_max_dimensions() before appending any data")
    print("  3. Actual dimensions are tracked per sample in metadata")
    print("  4. No data loss - all paths are preserved!")


if __name__ == "__main__":
    main()
