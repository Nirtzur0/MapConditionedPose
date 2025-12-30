"""
Helper utilities for variable-length data handling in Zarr datasets.

This module provides utilities to compute maximum dimensions across scenes
for proper variable-length array storage.
"""

import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def compute_max_dimensions(scene_data_list: List[Dict[str, np.ndarray]]) -> Dict[str, tuple]:
    """
    Compute maximum dimensions across all scenes for variable-length arrays.
    
    Args:
        scene_data_list: List of scene data dictionaries, each containing arrays
                        with keys like 'rt/path_gains', 'rt/path_delays', etc.
    
    Returns:
        Dictionary mapping array keys to maximum shape tuples (excluding batch dim)
        e.g., {'rt/path_gains': (2, 150), 'rt/path_delays': (2, 150)}
    
    Example:
        >>> scene1 = {'rt/path_gains': np.zeros((100, 1, 74))}
        >>> scene2 = {'rt/path_gains': np.zeros((200, 2, 150))}
        >>> max_dims = compute_max_dimensions([scene1, scene2])
        >>> max_dims['rt/path_gains']
        (2, 150)
    """
    max_dims = {}
    
    # Collect all unique keys
    all_keys = set()
    for scene_data in scene_data_list:
        all_keys.update(scene_data.keys())
    
    # For each key, find maximum dimensions
    for key in all_keys:
        shapes = []
        for scene_data in scene_data_list:
            if key in scene_data:
                value = scene_data[key]
                if isinstance(value, np.ndarray) and value.size > 0:
                    # Skip batch dimension (first dimension)
                    shapes.append(value.shape[1:])
        
        if shapes:
            # Compute element-wise maximum across all shapes
            max_shape = shapes[0]
            for shape in shapes[1:]:
                # Ensure shapes have same rank
                if len(shape) == len(max_shape):
                    max_shape = tuple(max(a, b) for a, b in zip(max_shape, shape))
                else:
                    logger.warning(f"Inconsistent shape ranks for {key}: {shapes}")
                    # Take the shape with more dimensions
                    if len(shape) > len(max_shape):
                        max_shape = shape
            
            max_dims[key] = max_shape
            logger.debug(f"Max dimensions for {key}: {max_shape}")
    
    return max_dims


def extract_rt_dimensions(scene_data: Dict[str, np.ndarray]) -> Dict[str, int]:
    """
    Extract actual RT dimensions (num_cells, num_paths) from scene data.
    
    Args:
        scene_data: Scene data dictionary
    
    Returns:
        Dictionary with 'num_cells' and 'num_paths' keys
    """
    dims = {'num_cells': 0, 'num_paths': 0}
    
    # Try to extract from path_gains first
    if 'rt/path_gains' in scene_data:
        shape = scene_data['rt/path_gains'].shape
        if len(shape) >= 2:
            dims['num_cells'] = shape[1]
        if len(shape) >= 3:
            dims['num_paths'] = shape[2]
        elif len(shape) == 2:
            dims['num_paths'] = shape[1]
    
    # Fallback to path_delays
    elif 'rt/path_delays' in scene_data:
        shape = scene_data['rt/path_delays'].shape
        if len(shape) >= 2:
            dims['num_cells'] = shape[1]
            if len(shape) >= 3:
                dims['num_paths'] = shape[2]
    
    return dims


def log_dimension_statistics(scene_data_list: List[Dict[str, np.ndarray]]):
    """
    Log statistics about dimensions across scenes for debugging.
    
    Args:
        scene_data_list: List of scene data dictionaries
    """
    if not scene_data_list:
        return
    
    # Collect dimension statistics
    num_cells_list = []
    num_paths_list = []
    
    for scene_data in scene_data_list:
        dims = extract_rt_dimensions(scene_data)
        if dims['num_cells'] > 0:
            num_cells_list.append(dims['num_cells'])
        if dims['num_paths'] > 0:
            num_paths_list.append(dims['num_paths'])
    
    if num_cells_list:
        logger.info(f"Number of cells across scenes: min={min(num_cells_list)}, "
                   f"max={max(num_cells_list)}, mean={np.mean(num_cells_list):.1f}")
    
    if num_paths_list:
        logger.info(f"Number of paths across scenes: min={min(num_paths_list)}, "
                   f"max={max(num_paths_list)}, mean={np.mean(num_paths_list):.1f}")


if __name__ == "__main__":
    # Test the utilities
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test data with varying dimensions
    scene1 = {
        'rt/path_gains': np.zeros((100, 1, 50)),
        'rt/path_delays': np.zeros((100, 1, 50)),
    }
    
    scene2 = {
        'rt/path_gains': np.zeros((200, 2, 150)),
        'rt/path_delays': np.zeros((200, 2, 150)),
    }
    
    scene3 = {
        'rt/path_gains': np.zeros((150, 1, 100)),
        'rt/path_delays': np.zeros((150, 1, 100)),
    }
    
    scenes = [scene1, scene2, scene3]
    
    # Compute max dimensions
    max_dims = compute_max_dimensions(scenes)
    print("\nMax Dimensions:")
    for key, shape in max_dims.items():
        print(f"  {key}: {shape}")
    
    # Log statistics
    print("\nDimension Statistics:")
    log_dimension_statistics(scenes)
    
    # Test dimension extraction
    print("\nDimension Extraction:")
    for i, scene in enumerate(scenes, 1):
        dims = extract_rt_dimensions(scene)
        print(f"  Scene {i}: {dims}")
