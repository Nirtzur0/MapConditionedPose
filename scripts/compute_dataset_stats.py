
import sys
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import numcodecs

def compute_stats(zarr_path: Path):
    print(f"Opening dataset: {zarr_path}")
    store = zarr.open(str(zarr_path), mode='r+')
    
    normalization = {}
    
    # Groups to process
    groups = ['rt', 'phy_fapi', 'mac_rrc']
    
    for group_name in groups:
        if group_name not in store:
            continue
            
        print(f"Processing group: {group_name}")
        group = store[group_name]
        normalization[group_name] = {}
        
        for key in group.keys():
            # Skip if group (recursive) or protected
            if isinstance(group[key], zarr.Group):
                continue
                
            print(f"  Computing stats for {key}...")
            # We load full array to compute precision stats
            # Warning: High RAM usage for huge datasets. 
            # Ideally chunked computation, but for now 4GB dataset is fine.
            data = group[key][:]
            
            # Check dtype
            if not np.issubdtype(data.dtype, np.number):
                continue
                
            # Handle complex
            if np.iscomplexobj(data):
                data = np.abs(data)
                
            # Handle NaN/Inf
            valid_mask = np.isfinite(data)
            if not np.any(valid_mask):
                mean, std = 0.0, 1.0
            else:
                valid_data = data[valid_mask]
                mean = float(np.mean(valid_data))
                std = float(np.std(valid_data))
                
            # Avoid zero std
            if std < 1e-9:
                std = 1.0
                
            normalization[group_name][key] = {
                'mean': float(mean),
                'std': float(std)
            }
            print(f"    Mean: {mean:.4f}, Std: {std:.4f}")

    # Write to metadata
    if 'metadata' not in store:
        store.create_group('metadata')
    
    # Store in metadata group as separate arrays or JSON?
    # Dataset reader expects: store['metadata']['normalization'][key]['mean']
    # Let's save as JSON in attributes first for inspection, 
    # but the reader code I saw uses store['metadata']['normalization'][key]... which implies subgroups?
    # Let's check reader:
    # return { key: { 'mean': store...['mean'][:], ... } }
    # So it expects actual Zarr arrays.
    
    meta_grp = store['metadata']
    if 'normalization' not in meta_grp:
        norm_grp = meta_grp.create_group('normalization')
    else:
        norm_grp = meta_grp['normalization']
        
    for grp_key, features in normalization.items():
        for feat_key, stats in features.items():
            full_key = f"{grp_key}/{feat_key}"
            
            if full_key in norm_grp:
                del norm_grp[full_key]
            
            feat_grp = norm_grp.create_group(full_key)
            
            # Zarr v2/v3 compatibility: explicit shape required
            feat_grp.create_dataset('mean', shape=(1,), chunks=(1,), dtype=np.float32, data=np.array([stats['mean']], dtype=np.float32))
            feat_grp.create_dataset('std', shape=(1,), chunks=(1,), dtype=np.float32, data=np.array([stats['std']], dtype=np.float32))
            
    print("Normalization stats written to metadata.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .zarr dataset")
    args = parser.parse_args()
    
    compute_stats(Path(args.path))
