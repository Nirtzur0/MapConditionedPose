"""
Visualization utilities for data generation (3D rendering and map plotting).
Uses Sionna's native rendering capabilities for proper radio map visualization.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import sys
import os

if TYPE_CHECKING:
    from sionna.rt import Scene, RadioMap

logger = logging.getLogger(__name__)


def render_scene_3d(
    scene: Any, 
    scene_id: str, 
    metadata: Dict, 
    output_dir: Path, 
    radio_map: Optional[Any] = None  # Should be sionna.rt.RadioMap (PlanarRadioMap)
):
    """
    Render 3D visualizations of the scene using Sionna's native renderer.
    Generates Top-Down and Isometric views with optional radio map overlay.
    
    Args:
        scene: Sionna RT Scene object
        scene_id: Identifier for the scene
        metadata: Scene metadata dict
        output_dir: Output directory path
        radio_map: Optional Sionna RadioMap object (from RadioMapSolver). 
                   If provided, renders coverage overlay using Sionna's native support.
    """
    if scene is None:
        logger.warning(f"Scene is None, skipping 3D render for {scene_id}")
        return

    try:
        from sionna.rt import Camera
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        logger.info(f"Rendering 3D visualizations for {scene_id}...")
        
        # Create output directory
        viz_dir = output_dir / "3d_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        safe_scene_id = scene_id.replace("/", "_").replace("\\", "_")
        
        # Get scene bounds from Mitsuba scene bbox
        try:
            bbox = scene.mi_scene.bbox()
            x_min, y_min, z_min = bbox.min.x, bbox.min.y, bbox.min.z
            x_max, y_max, z_max = bbox.max.x, bbox.max.y, bbox.max.z
            logger.info(f"Scene BBox: Min=({x_min:.1f}, {y_min:.1f}, {z_min:.1f}), Max=({x_max:.1f}, {y_max:.1f}, {z_max:.1f})")
        except Exception as e:
            logger.warning(f"Could not get scene bbox: {e}, using metadata")
            bbox_meta = metadata.get('bbox', {})
            x_min = bbox_meta.get('x_min', -250)
            x_max = bbox_meta.get('x_max', 250)
            y_min = bbox_meta.get('y_min', -250)
            y_max = bbox_meta.get('y_max', 250)
            z_min = 0
            z_max = 100
        
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)
        ground_z = z_min
        
        # --- View 1: Top-Down ---
        dist_z = max_dim * 1.5 
        cam_height = ground_z + dist_z
        
        cam_top = Camera(
            position=[float(cx + 0.1), float(cy + 0.1), float(cam_height)],
            look_at=[float(cx), float(cy), float(ground_z)], 
        )
        
        # Render with optional radio map overlay (Sionna native)
        # Only attempt if radio_map is a Sionna RadioMap object, not a numpy array
        if radio_map is not None and not isinstance(radio_map, np.ndarray):
            logger.info("Rendering Top-Down with radio map coverage (Sionna native)...")
            try:
                fig = scene.render(
                    camera=cam_top,
                    radio_map=radio_map,
                    rm_metric='path_gain',
                    rm_db_scale=True,
                    rm_vmin=-120,
                    rm_vmax=-50,
                    rm_show_color_bar=True,
                    show_devices=True,
                    resolution=(1024, 768)
                )
                out_path_coverage = viz_dir / f"{safe_scene_id}_top_down_coverage.png"
                fig.savefig(out_path_coverage, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved coverage map: {out_path_coverage}")
            except Exception as e:
                logger.warning(f"Sionna native coverage render failed: {e}")
        
        # Plain render (no radio map)
        out_path_top = viz_dir / f"{safe_scene_id}_top_down.png"
        logger.info(f"Rendering Top-Down plain to {out_path_top}")
        scene.render_to_file(
            camera=cam_top,
            filename=str(out_path_top),
            resolution=(1024, 768)
        )
        
        # --- View 2: Isometric ---
        iso_dist = max_dim * 0.8
        cam_iso_z = ground_z + (max_dim * 0.6)
        
        cam_iso = Camera(
            position=[float(cx - iso_dist), float(cy - iso_dist), float(cam_iso_z)],
            look_at=[float(cx), float(cy), float(ground_z)], 
        )
        
        out_path_iso = viz_dir / f"{safe_scene_id}_isometric.png"
        logger.info(f"Rendering Isometric to {out_path_iso}")
        scene.render_to_file(
            camera=cam_iso,
            filename=str(out_path_iso),
            resolution=(1024, 768)
        )
        
        # Isometric with coverage (only if radio_map is a Sionna object)
        if radio_map is not None and not isinstance(radio_map, np.ndarray):
            try:
                logger.info("Rendering Isometric with coverage...")
                fig_iso = scene.render(
                    camera=cam_iso,
                    radio_map=radio_map,
                    rm_metric='path_gain',
                    rm_db_scale=True,
                    rm_vmin=-120,
                    rm_vmax=-50,
                    rm_show_color_bar=True,
                    show_devices=True,
                    resolution=(1024, 768)
                )
                out_path_iso_cov = viz_dir / f"{safe_scene_id}_isometric_coverage.png"
                fig_iso.savefig(out_path_iso_cov, dpi=150, bbox_inches='tight')
                plt.close(fig_iso)
                logger.info(f"Saved isometric coverage: {out_path_iso_cov}")
            except Exception as e:
                logger.warning(f"Isometric coverage render failed: {e}")
        
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
            axes2[0, i].imshow(img, cmap='viridis', origin='lower')
            title = channel_names[i] if i < len(channel_names) else f'Ch{i}'
            axes2[0, i].set_title(f"Radio Ch{i}: {title}")
            axes2[0, i].axis("off")
        
        # OSM map channels
        osm_names = ['Height', 'Material', 'Footprint', 'Road', 'Terrain']
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


if __name__ == "__main__":
    """
    Test entry point for debugging visualizations using Sionna native rendering.
    Usage: python src/data_generation/visualization.py [scene_id]
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python src/data_generation/visualization.py <scene_id>")
        print("Example: python src/data_generation/visualization.py austin_texas/scene_-97.765_30.27")
        # Try to find a default scene
        base_dir = Path("data/scenes")
        if base_dir.exists():
            scenes = list(base_dir.rglob("scene.xml"))
            if scenes:
                default_scene = scenes[0].parent
                scene_id = str(default_scene.relative_to(base_dir))
                print(f"No scene specified. Using first found: {scene_id}")
            else:
                sys.exit(1)
        else:
            print("No data/scenes directory found.")
            sys.exit(1)
    else:
        scene_id = sys.argv[1]

    try:
        import sionna.rt as rt
        
        project_root = Path(__file__).parent.parent.parent
        scene_dir = project_root / "data/scenes"
        output_dir = project_root / "data/processed/test_viz"
        
        print(f"Loading scene: {scene_id}")
        scene_path = scene_dir / scene_id / "scene.xml"
        
        if not scene_path.exists():
            print(f"Scene file not found: {scene_path}")
            sys.exit(1)
            
        scene = rt.load_scene(str(scene_path))
        scene.frequency = 3.5e9
        print("Scene loaded.")
        
        # Setup antenna arrays for radio map generation
        scene.tx_array = rt.PlanarArray(num_rows=8, num_cols=8, 
                                         vertical_spacing=0.5, horizontal_spacing=0.5,
                                         pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, 
                                         pattern="iso", polarization="V")
        
        # Get scene bounds
        bbox = scene.mi_scene.bbox()
        cx = (bbox.min.x + bbox.max.x) / 2
        cy = (bbox.min.y + bbox.max.y) / 2
        width = min(bbox.max.x - bbox.min.x, 500.0)
        height = min(bbox.max.y - bbox.min.y, 500.0)
        ground_z = bbox.min.z
        
        # Add transmitter
        tx = rt.Transmitter(name="TX_1", position=[float(cx), float(cy), float(ground_z + 30)])
        scene.add(tx)
        print(f"Added transmitter at center: ({cx:.1f}, {cy:.1f}, {ground_z + 30:.1f})")
        
        # Generate radio map using Sionna's RadioMapSolver
        print("Generating radio map...")
        solver = rt.RadioMapSolver()
        radio_map = solver(
            scene, 
            center=[float(cx), float(cy), float(ground_z + 1.5)],
            size=[float(width), float(height)],
            cell_size=[5.0, 5.0],
            orientation=[0.0, 0.0, 0.0],
            max_depth=5,
            diffraction=True
        )
        print(f"Radio map generated: {radio_map.path_gain.shape}")
        
        # Metadata mockup
        metadata = {
            'bbox': {
                'x_min': bbox.min.x, 'x_max': bbox.max.x,
                'y_min': bbox.min.y, 'y_max': bbox.max.y
            }
        }
        
        # Render with coverage
        render_scene_3d(scene, scene_id, metadata, output_dir, radio_map=radio_map)
        print("Done.")
        
    except ImportError:
        print("Sionna not available. Cannot test rendering.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
