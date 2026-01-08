#!/usr/bin/env python3
"""Inspect LMDB database structure."""
import lmdb
import pickle
import sys

lmdb_path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/experiment/data/dataset_20260107_211306_train.lmdb'

env = lmdb.open(lmdb_path, readonly=True, lock=False)

with env.begin() as txn:
    # Check metadata
    metadata_bytes = txn.get(b'__metadata__')
    if metadata_bytes:
        metadata = pickle.loads(metadata_bytes)
        print("Metadata keys:", list(metadata.keys()))
        print("Metadata content:")
        for k, v in metadata.items():
            if isinstance(v, dict):
                print(f"  {k}: {list(v.keys()) if v else 'empty dict'}")
            else:
                print(f"  {k}: {v}")
    else:
        print("No metadata found!")
    
    # Count samples
    sample_count = 0
    cursor = txn.cursor()
    for key, value in cursor:
        if key.startswith(b'sample_'):
            sample_count += 1
    
    print(f"\nTotal samples: {sample_count}")
    
    # Show first sample
    if sample_count > 0:
        first_sample_bytes = txn.get(b'sample_000000')
        if first_sample_bytes:
            first_sample = pickle.loads(first_sample_bytes)
            print(f"\nFirst sample keys: {list(first_sample.keys())}")

env.close()
