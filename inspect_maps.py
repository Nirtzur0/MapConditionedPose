
import zarr
import numpy as np
import sys
from pathlib import Path

def inspect_dataset(zarr_path):
    print(f"Inspecting: {zarr_path}")
    try:
        store = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"Failed to open Zarr: {e}")
        return

    # Check Radio Maps
    if 'radio_maps' in store:
        rm = store['radio_maps']
        print(f"Radio Maps Shape: {rm.shape}")
        if rm.shape[0] > 0:
            sample = rm[0]
            print(f"Sample 0 Radio Map Stats:")
            print(f"  Min: {np.min(sample)}")
            print(f"  Max: {np.max(sample)}")
            print(f"  Mean: {np.mean(sample)}")
            print(f"  Non-zero count: {np.count_nonzero(sample)}")
            print(f"  Unique values (first 10): {np.unique(sample)[:10]}")
    else:
        print("No 'radio_maps' group found.")

    # Check OSM Maps
    if 'osm_maps' in store:
        om = store['osm_maps']
        print(f"OSM Maps Shape: {om.shape}")
        if om.shape[0] > 0:
            sample = om[0]
            print(f"Sample 0 OSM Map Stats:")
            print(f"  Min: {np.min(sample)}")
            print(f"  Max: {np.max(sample)}")
            print(f"  Mean: {np.mean(sample)}")
            print(f"  Non-zero count: {np.count_nonzero(sample)}")
            print(f"  Unique values (first 10): {np.unique(sample)[:10]}")
    else:
        print("No 'osm_maps' group found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to Zarr dataset")
    args = parser.parse_args()

    if args.dataset:
        inspect_dataset(args.dataset)
    else:
        # Find the latest dataset
        base_dir = Path("data/processed/quick_test_dataset")
        if not base_dir.exists():
             print(f"{base_dir} does not exist.")
             sys.exit(1)
             
        zarr_files = sorted(base_dir.glob("dataset_*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if zarr_files:
            inspect_dataset(str(zarr_files[0]))
        else:
            print("No datasets found.")
