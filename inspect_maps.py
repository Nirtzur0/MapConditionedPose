
import zarr
import numpy as np
import sys
from pathlib import Path

def inspect_dataset(zarr_path):
    print(f"Inspecting: {zarr_path}")
    store = zarr.open(str(zarr_path), mode='r')
    
    if 'radio_maps' not in store:
        print("ERROR: 'radio_maps' array not found in Zarr!")
        print("Keys:", list(store.keys()))
        return

    radio_maps = store['radio_maps']
    print(f"Radio Maps Shape: {radio_maps.shape}")
    
    if radio_maps.shape[0] == 0:
        print("Radio Maps array is empty.")
        return

    # Check stats for each map and channel
    for i in range(radio_maps.shape[0]):
        rmap = radio_maps[i]
        print(f"\n--- Scene {i} ---")
        for c in range(rmap.shape[0]):
            ch_data = rmap[c]
            # Filter out the "void" values (-200 or similar low values) to see signal stats
            mask = ch_data > -180
            
            min_val = np.min(ch_data)
            max_val = np.max(ch_data)
            mean_val = np.mean(ch_data)
            
            valid_count = np.sum(mask)
            total_count = ch_data.size
            valid_percent = (valid_count / total_count) * 100
            
            print(f"  Channel {c}: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, Valid > -180dB: {valid_percent:.2f}%")
            
            if valid_percent < 0.01 and c == 0: # Check Path Gain (usually channel 0)
                print("  WARNING: Almost no valid signal in Path Gain channel!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", help="Path to .zarr dataset")
    args = parser.parse_args()
    
    inspect_dataset(args.zarr_path)
