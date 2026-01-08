#!/usr/bin/env python3
"""Inspect zarr structure."""
import zarr
import sys

zarr_path = sys.argv[1] if len(sys.argv) > 1 else 'data/synthetic/dataset_20260101_094443.zarr'
store = zarr.open(zarr_path, mode='r')

def show_tree(g, indent=0, max_depth=3):
    if indent >= max_depth:
        return
    try:
        keys = list(g.keys())
        for k in keys[:30]:  # Limit output
            print('  '*indent + k)
            try:
                if hasattr(g[k], 'keys'):
                    show_tree(g[k], indent+1, max_depth)
            except:
                pass
    except:
        pass

print(f"Inspecting: {zarr_path}")
print("="*60)
show_tree(store)
