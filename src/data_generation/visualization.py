"""
Visualization utilities for data generation (3D rendering and map plotting).
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def render_scene_3d(scene: Any, scene_id: str, metadata: Dict, output_dir: Path):
    """
    Render 3D visualizations of the scene using Sionna's PBR renderer.
    Generates Top-Down and Isometric views.
    """
    if scene is None:
        logger.warning(f"Scene is None, skipping 3D render for {scene_id}")
        return

    try:
        from sionna.rt import Camera
        
        logger.info(f"Rendering 3D visualizations for {scene_id}...")
        
        # Create output directory
        viz_dir = output_dir / "3d_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        safe_scene_id = scene_id.replace("/", "_").replace("\\", "_")
        
        # Get scene bounds
        bbox = metadata.get('bbox', {})
        x_min = bbox.get('x_min', -500)
        x_max = bbox.get('x_max', 500)
        y_min = bbox.get('y_min', -500)
        y_max = bbox.get('y_max', 500)
        
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)
        
        # --- View 1: Top-Down ---
        cam_z = max_dim * 1.5 
        
        logger.debug(f"Creating Top-Down Camera at ({cx}, {cy}, {cam_z})")
        
        # Try to set clipping planes if possible after creation or via different args
        # For now, stick to basic args to ensure creation succeeds
        cam_top = Camera(
            name=f"cam_top_{safe_scene_id}", # Try adding name
            position=[cx, cy, cam_z],
            look_at=[cx, cy, 0],
        )
        # Try setting standard mitsuba params if exposed
        # if hasattr(cam_top, 'near_clip'): cam_top.near_clip = 1.0
        # if hasattr(cam_top, 'far_clip'): cam_top.far_clip = 100000.0
        
        # Render
        out_path_top = viz_dir / f"{safe_scene_id}_top_down.png"
        logger.info(f"Rendering to {out_path_top}")
        scene.render_to_file(
            camera=cam_top,
            filename=str(out_path_top),
            resolution=(1024, 768)
        )
        
        # --- View 2: Isometric ---
        iso_dist = max_dim * 0.8
        
        cam_iso = Camera(
            name=f"cam_iso_{safe_scene_id}",
            position=[cx - iso_dist, cy - iso_dist, max_dim * 0.6],
            look_at=[cx, cy, 0],
        )
        
        out_path_iso = viz_dir / f"{safe_scene_id}_isometric.png"
        logger.info(f"Rendering to {out_path_iso}")
        scene.render_to_file(
            camera=cam_iso,
            filename=str(out_path_iso),
            resolution=(1024, 768)
        )
        
        logger.info(f"Saved 3D renders -> {viz_dir}")

    except Exception as e:
        import traceback
        logger.error(f"3D Rendering failed for {scene_id}: {e}")
        logger.error(traceback.format_exc())


def normalize_map(data: np.ndarray, fixed_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Helper: Percentile-based normalization with fallback."""
    data = np.asarray(data, dtype=np.float32)
    # Replace Inf with NaN for percentile calc
    valid_data = np.where(np.isinf(data), np.nan, data)
    
    # Filter out "background" values (e.g. -200 or 0 for linear) if they dominate
    # For Path Gain (dB), background is usually <= -150
    signal_mask = valid_data > -150
    
    if not np.any(signal_mask):
        # No signal at all
        return np.zeros_like(data)
    
    if fixed_range:
        vmin, vmax = fixed_range
    else:
        # Calculate percentiles only on "signal" pixels to avoid background skewing
        if np.sum(signal_mask) > 10: # Sufficient samples
            vmin = np.nanpercentile(valid_data[signal_mask], 2)
            vmax = np.nanpercentile(valid_data[signal_mask], 98)
        else:
            vmin = np.nanmin(valid_data)
            vmax = np.nanmax(valid_data)
    
    # Safety check
    if vmax - vmin < 1e-6:
        vmax = vmin + 10.0 # Arbitrary range to avoid div/0
    
    # Normalize
    # Use entire array (including background)
    normalized = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    return normalized


def save_map_visualizations(scene_id: str, radio_map: np.ndarray, osm_map: np.ndarray, output_dir: Path):
    """
    Save radio and OSM map visualizations as reference images.
    
    Args:
        scene_id: Scene identifier for naming
        radio_map: [C, H, W] radio map array
        osm_map: [C, H, W] OSM map array
        output_dir: Directory to save visualizations
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create output directory inside the dataset output folder
        viz_dir = output_dir / "map_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean scene_id for filename (replace / with _)
        safe_scene_id = scene_id.replace("/", "_").replace("\\", "_")
        
        # Create figure with radio and OSM maps
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Radio Map - Path Gain (first channel)
        # Use fixed range for Path Gain to ensure consistent visualization
        # -120 dB (weak) to -50 dB (strong) covers typical cellular range
        if radio_map.shape[0] > 0:
            radio_img = normalize_map(radio_map[0], fixed_range=(-120, -50))
            im0 = axes[0].imshow(radio_img, cmap='inferno', origin='lower')
            axes[0].set_title(f"Radio Map (Path Gain)\n{scene_id}")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], shrink=0.8, label="Normalized Path Gain (-120 to -50 dB)")
        
        # OSM Map - RGB visualization (R:Height, G:Footprint, B:Road+Terrain)
        # Assuming channels: Height, Material, Footprint, Road, Terrain
        if osm_map.ndim == 3:
            h, w = osm_map.shape[1], osm_map.shape[2]
            osm_height = normalize_map(osm_map[0]) if osm_map.shape[0] > 0 else np.zeros((h, w))
            osm_footprint = normalize_map(osm_map[2]) if osm_map.shape[0] > 2 else np.zeros((h, w))
            osm_road = osm_map[3] if osm_map.shape[0] > 3 else np.zeros((h, w))
            osm_terrain = osm_map[4] if osm_map.shape[0] > 4 else np.zeros((h, w))
            osm_blue = normalize_map(np.maximum(osm_road, osm_terrain * 0.3))
            osm_rgb = np.stack([osm_height, osm_footprint, osm_blue], axis=-1)
            
            axes[1].imshow(osm_rgb, origin='lower')
            axes[1].set_title(f"OSM Map (R:Height G:Build B:Road/Terrain)\n{scene_id}")
            axes[1].axis("off")
        
        plt.tight_layout()
        
        # Save combined figure
        output_path = viz_dir / f"{safe_scene_id}_maps.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved map visualization: {output_path}")
        
        # Also save individual channel visualizations for detailed inspection
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
        
        # Radio map channels
        channel_names = ['Path Gain', 'SNR', 'SINR', 'Throughput', 'BLER']
        for i in range(min(3, radio_map.shape[0])):
            img = normalize_map(radio_map[i])
            # Skip Ch2 (AoA) in individual plots if it's not provided
            axes2[0, i].imshow(img, cmap='viridis', origin='lower')
            title = channel_names[i] if i < len(channel_names) else f'Ch{i}'
            axes2[0, i].set_title(f"Radio Ch{i}: {title}")
            axes2[0, i].axis("off")
        
        # OSM map channels
        osm_names = ['Height', 'Material', 'Footprint', 'Road', 'Terrain']
        # Map indices to plot 0, 1, 2 in the loop to Height, Footprint, Terrain
        # Indices in osm_map: 0:Height, 1:Material, 2:Footprint, 3:Road, 4:Terrain
        plot_indices = [0, 2, 4] # Select most interesting ones
        
        for i in range(3):
            ch_idx = plot_indices[i]
            if ch_idx < osm_map.shape[0]:
                img = normalize_map(osm_map[ch_idx])
                axes2[1, i].imshow(img, cmap='gray', origin='lower')
                axes2[1, i].set_title(f"OSM Ch{ch_idx}: {osm_names[ch_idx]}")
            axes2[1, i].axis("off")
        
        plt.tight_layout()
        
        detailed_path = viz_dir / f"{safe_scene_id}_maps_detailed.png"
        fig2.savefig(detailed_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)
        
        logger.info(f"Saved detailed map visualization: {detailed_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save map visualization for {scene_id}: {e}")
