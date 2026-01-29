#!/usr/bin/env python3
"""
Debug script to run the notebook logic directly
"""

import sys
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd

# Setup
VERBOSE = True
project_root = Path('/home/ubuntu/projects/MapConditionedPose')
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Clear src modules
to_remove = [key for key in sys.modules.keys() if key.startswith('src.')]
for key in to_remove:
    del sys.modules[key]

print("=== Notebook Debug Script ===")

# Imports
from src.training import UELocalizationLightning
from src.notebook_plot_helpers import (
    resolve_checkpoint_and_config,
    resolve_lmdb_paths,
    visualize_feature_histograms,
)

# Direct dataset setup
USE_DIRECT_DATASET = True
if USE_DIRECT_DATASET:
    DATASET_PATH = '/home/ubuntu/projects/MapConditionedPose/outputs/experiment_2tx_2vars_1000ue/data/dataset_20260128_144040_test.lmdb'
    BASE_CONFIG_PATH = '/home/ubuntu/projects/MapConditionedPose/outputs/experiment_2tx_2vars_1000ue/config.yaml'
    print(f"Using direct dataset: {DATASET_PATH}")
    print(f"Config: {BASE_CONFIG_PATH}")
    
    # Load config
    with open(BASE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Config loaded: {list(config.keys())}")
    
    # Extract dataset parameters from config
    dataset_config = config.get('dataset', {})
    map_resolution = dataset_config.get('map_resolution', 1.0)
    scene_extent = dataset_config.get('scene_extent', 512)
    normalize = dataset_config.get('normalize', True)
    handle_missing = dataset_config.get('handle_missing', 'mask')
    
    print(f"Dataset params: resolution={map_resolution}, extent={scene_extent}, normalize={normalize}")
    
    # Create dataset directly
    from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset
    dataset = LMDBRadioLocalizationDataset(
        lmdb_path=DATASET_PATH,
        split='test',
        map_resolution=map_resolution,
        scene_extent=scene_extent,
        normalize=normalize,
        handle_missing=handle_missing,
    )
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"DataLoader created")
    
    # Get a batch
    TARGET_BATCH_IDX = 1
    batch_iter = iter(val_loader)
    for i in range(TARGET_BATCH_IDX + 1):
        try:
            batch = next(batch_iter)
        except StopIteration:
            print(f"Reached end of dataset at batch {i}")
            break
    
    print(f"Loaded batch {TARGET_BATCH_IDX}")
    print(f"Batch keys: {batch.keys()}")
    if 'measurements' in batch:
        print(f"Measurements keys: {batch['measurements'].keys()}")
        for key in ['rt_features', 'phy_features', 'mac_features']:
            if key in batch['measurements']:
                shape = batch['measurements'][key].shape
                print(f"  {key}: {shape}")
    
    # Test histogram visualization
    print("\n=== Testing Histogram Visualization ===")
    try:
        visualize_feature_histograms(batch, max_dims=12, use_mask=True)
        print("Histogram visualization completed successfully!")
    except Exception as e:
        print(f"Error in histogram visualization: {e}")
        import traceback
        traceback.print_exc()

else:
    print("Using checkpoint resolution...")

print("\n=== Debug Complete ===")
