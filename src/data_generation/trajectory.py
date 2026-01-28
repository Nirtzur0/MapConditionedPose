"""
UE trajectory sampling logic for data generation.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

def sample_ue_trajectories(
    scene_metadata: Dict,
    num_ue_per_tile: int,
    ue_height_range: Tuple[float, float],
    ue_velocity_range: Tuple[float, float],
    num_reports_per_ue: int,
    report_interval_ms: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    building_height_map: Optional[np.ndarray] = None,
    max_attempts_per_ue: int = 25,
    enforce_unique_positions: bool = False,
    min_ue_separation_m: float = 1.0,
    sampling_margin_m: float = 0.0
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

    # Optionally shrink sampling bounds to avoid edge cases.
    sample_x_min, sample_x_max = x_min, x_max
    sample_y_min, sample_y_max = y_min, y_max
    if sampling_margin_m and sampling_margin_m > 0:
        candidate_x_min = x_min + sampling_margin_m
        candidate_x_max = x_max - sampling_margin_m
        candidate_y_min = y_min + sampling_margin_m
        candidate_y_max = y_max - sampling_margin_m
        if candidate_x_min < candidate_x_max and candidate_y_min < candidate_y_max:
            sample_x_min, sample_x_max = candidate_x_min, candidate_x_max
            sample_y_min, sample_y_max = candidate_y_min, candidate_y_max
            logger.info(
                "Applying UE sampling margin %.1fm: bbox %.1fx%.1fm -> %.1fx%.1fm",
                sampling_margin_m,
                (x_max - x_min),
                (y_max - y_min),
                (sample_x_max - sample_x_min),
                (sample_y_max - sample_y_min),
            )
        else:
            logger.warning(
                "Sampling margin %.1fm too large for bbox %.1fx%.1fm; using full bounds.",
                sampling_margin_m,
                (x_max - x_min),
                (y_max - y_min),
            )
    
    def _is_in_building(x_pos: float, y_pos: float) -> bool:
        if building_height_map is None:
            return False
        bbox = scene_metadata.get('bbox', {})
        if not all(k in bbox for k in ('x_min', 'y_min', 'x_max', 'y_max')):
            return False
        x_min_local = bbox['x_min']
        x_max_local = bbox['x_max']
        y_min_local = bbox['y_min']
        y_max_local = bbox['y_max']
        width = building_height_map.shape[1]
        height = building_height_map.shape[0]
        px = int(np.floor(x_pos - x_min_local))
        py = int(np.floor(y_max_local - y_pos))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        return building_height_map[py, px] > 0

    trajectories = []
    # Stratified Sampling (Grid-Jittered)
    # Guarantees uniform coverage by dividing the area into a grid
    num_ues = num_ue_per_tile
    
    # Calculate aspect ratio to distribute grid cells
    width = sample_x_max - sample_x_min
    height = sample_y_max - sample_y_min
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
    
    used_positions: List[Tuple[float, float]] = []
    used_position_keys = set()

    def _is_unique_positions(candidate_xy: List[Tuple[float, float]]) -> bool:
        if not enforce_unique_positions:
            return True
        if min_ue_separation_m <= 0:
            for cx, cy in candidate_xy:
                key = (round(cx, 3), round(cy, 3))
                if key in used_position_keys:
                    return False
            return True
        if not used_positions:
            return True
        used_arr = np.asarray(used_positions, dtype=np.float32)
        for cx, cy in candidate_xy:
            dists = np.hypot(used_arr[:, 0] - cx, used_arr[:, 1] - cy)
            if np.any(dists < min_ue_separation_m):
                return False
        return True

    def _mark_used(candidate_xy: List[Tuple[float, float]]) -> None:
        if not enforce_unique_positions:
            return
        for cx, cy in candidate_xy:
            used_positions.append((cx, cy))
            if min_ue_separation_m <= 0:
                used_position_keys.add((round(cx, 3), round(cy, 3)))

    def _build_stationary_trajectory(x0: float, y0: float, z0: float) -> np.ndarray:
        """Fallback trajectory: fixed position across all reports."""
        traj = np.zeros((num_reports_per_ue, 3), dtype=np.float32)
        traj[:, 0] = x0 - offset[0]
        traj[:, 1] = y0 - offset[1]
        traj[:, 2] = z0
        return traj

    def _fallback_trajectory() -> np.ndarray:
        """Fallback that guarantees valid in-bounds coordinates (avoid zeros/negatives)."""
        for _ in range(max_attempts_per_ue):
            x0 = np.random.uniform(sample_x_min, sample_x_max)
            y0 = np.random.uniform(sample_y_min, sample_y_max)
            if _is_in_building(x0, y0):
                continue
            if not _is_unique_positions([(x0, y0)]):
                continue
            z0 = np.random.uniform(*ue_height_range)
            _mark_used([(x0, y0)])
            return _build_stationary_trajectory(x0, y0, z0)
        # If building map is too dense, fall back to any in-bounds point.
        x0 = np.random.uniform(sample_x_min, sample_x_max)
        y0 = np.random.uniform(sample_y_min, sample_y_max)
        z0 = np.random.uniform(*ue_height_range)
        if enforce_unique_positions and not _is_unique_positions([(x0, y0)]):
            logger.warning("Unique UE placement could not be satisfied in fallback; using closest available point.")
        else:
            _mark_used([(x0, y0)])
        return _build_stationary_trajectory(x0, y0, z0)

    for i in range(actual_ues):
        r, c = cells[i]
        
        # Cell bounds
        cell_x_min = sample_x_min + c * cell_w
        cell_y_min = sample_y_min + r * cell_h
        
        attempt = 0
        trajectory = None
        while attempt < max_attempts_per_ue:
            attempt += 1
            # Sample ONE point uniformly within this cell (Jittered)
            x0 = np.random.uniform(cell_x_min, cell_x_min + cell_w)
            y0 = np.random.uniform(cell_y_min, cell_y_min + cell_h)
            z0 = np.random.uniform(*ue_height_range)

            speed = np.random.uniform(*ue_velocity_range)
            direction = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(direction)
            vy = speed * np.sin(direction)

            # Generate trajectory
            candidate = []
            candidate_global = []
            invalid = False
            for t in range(num_reports_per_ue):
                dt = t * report_interval_ms / 1000.0
                x = x0 + vx * dt
                y = y0 + vy * dt
                z = z0

                # Clip to bounds
                x = np.clip(x, sample_x_min, sample_x_max)
                y = np.clip(y, sample_y_min, sample_y_max)

                if _is_in_building(x, y):
                    invalid = True
                    break

                # Apply offset to convert to local coordinates
                x_local = x - offset[0]
                y_local = y - offset[1]

                candidate.append([x_local, y_local, z])
                candidate_global.append((x, y))

            if not invalid and len(candidate) == num_reports_per_ue:
                if not _is_unique_positions(candidate_global):
                    continue
                trajectory = np.array(candidate)
                _mark_used(candidate_global)
                break

        if trajectory is None:
            logger.warning(
                "Failed to sample building-free trajectory after %d attempts; using fallback in-bounds trajectory.",
                max_attempts_per_ue,
            )
            trajectory = _fallback_trajectory()

        trajectories.append(trajectory)
        
    # Add any remaining UEs randomly if grid count was less than requested
    remaining = num_ues - actual_ues
    if remaining > 0:
        logger.debug(f"Adding {remaining} extra random UEs to match requested count.")
        for _ in range(remaining):
            attempt = 0
            trajectory = None
            while attempt < max_attempts_per_ue:
                attempt += 1
                x0 = np.random.uniform(sample_x_min, sample_x_max)
                y0 = np.random.uniform(sample_y_min, sample_y_max)
                z0 = np.random.uniform(*ue_height_range)

                speed = np.random.uniform(*ue_velocity_range)
                direction = np.random.uniform(0, 2*np.pi)
                vx = speed * np.cos(direction)
                vy = speed * np.sin(direction)

                candidate = []
                candidate_global = []
                invalid = False
                for t in range(num_reports_per_ue):
                    dt = t * report_interval_ms / 1000.0
                    x = x0 + vx * dt
                    y = y0 + vy * dt
                    z = z0

                    x = np.clip(x, sample_x_min, sample_x_max)
                    y = np.clip(y, sample_y_min, sample_y_max)

                    if _is_in_building(x, y):
                        invalid = True
                        break

                    x_local = x - offset[0]
                    y_local = y - offset[1]
                    candidate.append([x_local, y_local, z])
                    candidate_global.append((x, y))

                if not invalid and len(candidate) == num_reports_per_ue:
                    if not _is_unique_positions(candidate_global):
                        continue
                    trajectory = np.array(candidate)
                    _mark_used(candidate_global)
                    break

            if trajectory is None:
                logger.warning(
                    "Failed to sample building-free random trajectory after %d attempts; using fallback in-bounds trajectory.",
                    max_attempts_per_ue,
                )
                trajectory = _fallback_trajectory()
            trajectories.append(trajectory)

    return trajectories
