#!/usr/bin/env python3
"""
Generate Precomputed Radio Maps for Physics Loss.

This script generates radio maps for all scenes in the dataset using Sionna RT.
Radio maps are saved in Zarr format for efficient loading during training.

Usage:
    python scripts/generate_radio_maps.py --scenes-dir data/scenes --output-dir data/radio_maps
    python scripts/generate_radio_maps.py --config configs/radio_map.yaml --parallel 4
"""

import argparse
import logging
from pathlib import Path
import yaml
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    import sionna as sn
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False

from src.physics_loss import RadioMapGenerator, RadioMapConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_scene(scene_path: Path) -> 'sn.rt.Scene':
    """Load Sionna scene from XML file."""
    if not SIONNA_AVAILABLE:
        raise ImportError("Sionna is required. Install with: pip install sionna")
    
    logger.info(f"Loading scene: {scene_path}")
    scene = sn.rt.load_scene(str(scene_path))
    return scene


def extract_cell_sites(scene: 'sn.rt.Scene') -> List[dict]:
    """
    Extract cell site configurations from scene.
    
    Returns list of dicts with keys: position, orientation, tx_power, antenna
    """
    # This is a placeholder - actual implementation depends on scene format
    # In practice, cell sites might be stored in scene metadata or separate file
    
    # For now, return dummy cell sites
    # TODO: Parse from scene.xml or accompanying metadata file
    cell_sites = [
        {
            'position': (256.0, 256.0, 30.0),  # Center of 512x512 map, 30m height
            'orientation': (0.0, 0.0),  # (azimuth, elevation) in degrees
            'tx_power': 43.0,  # dBm
            'antenna': 'default',
        }
    ]
    
    logger.info(f"Found {len(cell_sites)} cell sites")
    return cell_sites


def _save_radio_map_plots(
    radio_map: 'np.ndarray',
    feature_names: List[str],
    output_dir: Path,
    scene_id: str,
):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available; skipping radio map plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    num_features = radio_map.shape[0]
    cols = min(3, num_features)
    rows = int(np.ceil(num_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx >= num_features:
            ax.axis('off')
            continue
        data = radio_map[idx]
        ax.imshow(data, cmap='inferno')
        title = feature_names[idx] if idx < len(feature_names) else f"ch_{idx}"
        ax.set_title(title)
        ax.axis('off')

    fig.tight_layout()
    out_path = output_dir / f"{scene_id}_radio_maps.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_radio_map_for_scene(
    scene_path: Path,
    output_dir: Path,
    config: RadioMapConfig,
    save_plots: bool = False,
    plots_dir: Optional[Path] = None,
) -> Path:
    """
    Generate radio map for a single scene.
    
    Args:
        scene_path: Path to scene XML file
        output_dir: Output directory for radio maps
        config: Radio map configuration
        
    Returns:
        Path to generated Zarr file
    """
    scene_id = scene_path.stem
    
    try:
        # Load scene
        scene = load_scene(scene_path)
        
        # Extract cell sites
        cell_sites = extract_cell_sites(scene)
        
        # Create generator
        generator = RadioMapGenerator(config)
        
        # Generate radio map
        logger.info(f"Generating radio map for {scene_id}...")
        radio_map = generator.generate_for_scene(scene, cell_sites, show_progress=True)
        
        # Save optional debug plots
        if save_plots:
            plot_dir = plots_dir or (output_dir / "plots")
            _save_radio_map_plots(radio_map, config.features, plot_dir, scene_id)

        # Save to Zarr
        metadata = {
            'scene_path': str(scene_path),
            'num_cell_sites': len(cell_sites),
        }
        output_path = generator.save_to_zarr(radio_map, scene_id, metadata)
        
        logger.info(f"Saved radio map to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate radio map for {scene_id}: {e}")
        raise


def generate_radio_maps_parallel(
    scene_paths: List[Path],
    output_dir: Path,
    config: RadioMapConfig,
    num_workers: int = 1,
    save_plots: bool = False,
    plots_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Generate radio maps for multiple scenes in parallel.
    
    Args:
        scene_paths: List of scene XML paths
        output_dir: Output directory
        config: Radio map configuration
        num_workers: Number of parallel workers
        
    Returns:
        List of generated Zarr paths
    """
    output_paths = []
    
    if num_workers == 1:
        # Sequential processing
        for scene_path in tqdm(scene_paths, desc="Generating radio maps"):
            output_path = generate_radio_map_for_scene(
                scene_path,
                output_dir,
                config,
                save_plots=save_plots,
                plots_dir=plots_dir,
            )
            output_paths.append(output_path)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    generate_radio_map_for_scene,
                    scene_path,
                    output_dir,
                    config,
                    save_plots,
                    plots_dir,
                ): scene_path
                for scene_path in scene_paths
            }
            
            with tqdm(total=len(scene_paths), desc="Generating radio maps") as pbar:
                for future in as_completed(futures):
                    scene_path = futures[future]
                    try:
                        output_path = future.result()
                        output_paths.append(output_path)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed for {scene_path}: {e}")
                        pbar.update(1)
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Generate precomputed radio maps")
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file (optional)'
    )
    parser.add_argument(
        '--scenes-dir',
        type=str,
        required=True,
        help='Directory containing scene XML files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/radio_maps',
        help='Output directory for radio maps (default: data/radio_maps)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='Map resolution in meters per pixel (default: 1.0)'
    )
    parser.add_argument(
        '--map-size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='Map size in pixels [width height] (default: 512 512)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.xml',
        help='File pattern for scene files (default: *.xml)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save debug plots of radio maps'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        help='Directory for radio map plots (default: <output_dir>/plots)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = RadioMapConfig(**config_dict)
    else:
        # Use CLI arguments
        config = RadioMapConfig(
            resolution=args.resolution,
            map_size=tuple(args.map_size),
            output_dir=Path(args.output_dir),
        )
    
    # Find scene files
    scenes_dir = Path(args.scenes_dir)
    if not scenes_dir.exists():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    
    scene_paths = sorted(scenes_dir.glob(args.pattern))
    if not scene_paths:
        raise ValueError(f"No scene files found matching {args.pattern} in {scenes_dir}")
    
    logger.info(f"Found {len(scene_paths)} scenes")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    
    # Check Sionna availability
    if not SIONNA_AVAILABLE:
        logger.error("Sionna is not available. Please install with: pip install sionna")
        return 1
    
    # Generate radio maps
    logger.info(f"Generating radio maps with {args.parallel} workers...")
    output_paths = generate_radio_maps_parallel(
        scene_paths,
        output_dir,
        config,
        num_workers=args.parallel,
        save_plots=args.save_plots,
        plots_dir=plots_dir,
    )
    
    logger.info(f"Successfully generated {len(output_paths)} radio maps")
    logger.info(f"Radio maps saved to: {output_dir}")
    
    # Print summary
    total_size = sum(p.stat().st_size for p in output_paths if p.exists())
    logger.info(f"Total size: {total_size / 1024**2:.1f} MB")
    
    return 0


if __name__ == '__main__':
    exit(main())
