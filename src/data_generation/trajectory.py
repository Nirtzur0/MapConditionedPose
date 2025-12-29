"""
UE trajectory sampling logic for data generation.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def sample_ue_trajectories(
    scene_metadata: Dict,
    num_ue_per_tile: int,
    ue_height_range: Tuple[float, float],
    ue_velocity_range: Tuple[float, float],
    num_reports_per_ue: int,
    report_interval_ms: float,
    offset: Tuple[float, float] = (0.0, 0.0)
) -> List[np.ndarray]:
    """
    Samples UE positions and trajectories within scene bounds.
    
    Args:
        scene_metadata: Scene metadata including bounding box.
        num_ue_per_tile: Number of UEs to sample.
        ue_height_range: Min/max UE height in meters.
        ue_velocity_range: Min/max UE velocity in m/s.
        num_reports_per_ue: Number of measurement reports per UE trajectory.
        report_interval_ms: Time between reports in milliseconds.
        offset: (x, y) offset to subtract from sampled positions.
        
    Returns:
        List of [num_reports, 3] position arrays for UE trajectories.
    """
    # Get scene bounds
    bbox = scene_metadata.get('bbox', {})
    x_min = bbox.get('x_min', -500)
    x_max = bbox.get('x_max', 500)
    y_min = bbox.get('y_min', -500)
    y_max = bbox.get('y_max', 500)
    
    trajectories = []
    # Stratified Sampling (Grid-Jittered)
    # Guarantees uniform coverage by dividing the area into a grid
    num_ues = num_ue_per_tile
    
    # Calculate aspect ratio to distribute grid cells
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Determine grid dimensions (cols * rows ≈ num_ues)
    # cols / rows ≈ aspect_ratio  =>  cols ≈ rows * aspect_ratio
    # rows * (rows * aspect_ratio) ≈ num_ues  =>  rows^2 ≈ num_ues / aspect_ratio
    num_rows = int(np.sqrt(num_ues / aspect_ratio))
    num_cols = int(num_ues / num_rows) if num_rows > 0 else 1
    
    # Adjust to match exact count or close to it
    actual_ues = num_rows * num_cols
    
    # Grid cell sizes
    cell_w = width / num_cols
    cell_h = height / num_rows
    
    logger.info(f"Using Stratified Sampling: {num_cols}x{num_rows} grid ({actual_ues} UEs). bbox: {width:.0f}x{height:.0f}m")
    
    trajectories = []
    
    # Generate grid cells
    cells = []
    for r in range(num_rows):
        for c in range(num_cols):
            cells.append((r, c))
            
    # If we need more UEs to match exact request, add random cells
    # (Though usually acceptable to be slightly under/over, let's just fill the grid primarily)
    
    for i in range(actual_ues):
        r, c = cells[i]
        
        # Cell bounds
        cell_x_min = x_min + c * cell_w
        cell_y_min = y_min + r * cell_h
        
        # Sample ONE point uniformly within this cell (Jittered)
        x0 = np.random.uniform(cell_x_min, cell_x_min + cell_w)
        y0 = np.random.uniform(cell_y_min, cell_y_min + cell_h)
        z0 = np.random.uniform(*ue_height_range)
        
        speed = np.random.uniform(*ue_velocity_range)
        direction = np.random.uniform(0, 2*np.pi)
        vx = speed * np.cos(direction)
        vy = speed * np.sin(direction)
        
        # Generate trajectory
        trajectory = []
        for t in range(num_reports_per_ue):
            dt = t * report_interval_ms / 1000.0
            x = x0 + vx * dt
            y = y0 + vy * dt
            z = z0
            
            # Clip to bounds
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)
            
            # Apply offset to convert to local coordinates
            x_local = x - offset[0]
            y_local = y - offset[1]
            
            trajectory.append([x_local, y_local, z])
        
        trajectories.append(np.array(trajectory))
        
    # Add any remaining UEs randomly if grid count was less than requested
    remaining = num_ues - actual_ues
    if remaining > 0:
        logger.debug(f"Adding {remaining} extra random UEs to match requested count.")
        for _ in range(remaining):
            x0 = np.random.uniform(x_min, x_max)
            y0 = np.random.uniform(y_min, y_max)
            z0 = np.random.uniform(*ue_height_range)
            
            speed = np.random.uniform(*ue_velocity_range)
            direction = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(direction)
            vy = speed * np.sin(direction)
            
            trajectory = []
            for t in range(num_reports_per_ue):
                dt = t * report_interval_ms / 1000.0
                x = x0 + vx * dt
                y = y0 + vy * dt
                z = z0
                
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                
                x_local = x - offset[0]
                y_local = y - offset[1]
                trajectory.append([x_local, y_local, z])
            trajectories.append(np.array(trajectory))

    return trajectories
