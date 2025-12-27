#!/usr/bin/env python3
"""
Scene Generation Example Script (M1)
Demonstrates deep Geo2SigMap integration for UE localization

Usage:
    # Single scene
    python scripts/scene_generation/generate_scenes.py --area "Boulder, CO" --output data/scenes
    
    # Multiple tiles
    python scripts/scene_generation/generate_scenes.py --bbox -105.30 40.00 -105.20 40.05 \
                                       --tiles --output data/scenes
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scene_generation import (
    SceneGenerator,
    MaterialRandomizer,
    SitePlacer,
    TileGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D scenes with deep Geo2SigMap integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Scene definition
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--area",
        type=str,
        help="Area name (e.g., 'Boulder, CO') - will query small bbox",
    )
    scene_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box in WGS84 coordinates",
    )
    scene_group.add_argument(
        "--polygon",
        type=Path,
        help="Path to JSON file with polygon coordinates",
    )
    
    # Processing mode
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="Generate multiple tiles (use with --bbox)",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=500,
        help="Tile size in meters (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=50,
        help="Tile overlap in meters (default: 50)",
    )
    
    # Site configuration
    parser.add_argument(
        "--num-tx",
        type=int,
        default=3,
        help="Number of transmitter sites per scene (default: 3)",
    )
    parser.add_argument(
        "--num-rx",
        type=int,
        default=10,
        help="Number of receiver sites per scene (default: 10)",
    )
    parser.add_argument(
        "--site-strategy",
        choices=["grid", "random", "isd"],
        default="random",
        help="Site placement strategy (default: random)",
    )
    parser.add_argument(
        "--fixed-sites",
        action="store_true",
        help="Disable random number of sites (use --num-tx exactly)",
    )
    parser.add_argument(
        "--isd",
        type=float,
        help="Inter-site distance for 'isd' strategy (meters)",
    )
    
    # Materials
    parser.add_argument(
        "--randomize-materials",
        action="store_true",
        help="Enable material domain randomization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    
    # Terrain / LiDAR
    parser.add_argument(
        "--use-lidar",
        action="store_true",
        help="Enable LiDAR point cloud usage for terrain/building heights",
    )
    parser.add_argument(
        "--use-dem",
        action="store_true",
        help="Enable DEM terrain usage",
    )
    parser.add_argument(
        "--hag-tiff",
        type=Path,
        help="Path to pre-calculated HAG GeoTIFF",
    )
    
    # Paths
    parser.add_argument(
        "--geo2sigmap-path",
        type=Path,
        default="/home/ubuntu/projects/geo2sigmap/package/src",
        help="Path to geo2sigmap package src",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./data/scenes",
        help="Output directory for scenes (default: ./data/scenes)",
    )
    
    return parser.parse_args()


def load_polygon(polygon_path: Path):
    """Load polygon from JSON file."""
    with open(polygon_path) as f:
        data = json.load(f)
    
    if "coordinates" in data:
        return data["coordinates"]
    elif "polygon" in data:
        return data["polygon"]
    else:
        return data


def get_simple_bbox(area_name: str):
    """Get a small bbox for testing (hardcoded locations)."""
    # Hardcoded test locations (approx 1km x 1km box)
    locations = {
        "Boulder, CO": (-105.275, 40.015, -105.265, 40.025),
        "Boulder": (-105.275, 40.015, -105.265, 40.025),
        "Durham, NC": (-78.940, 36.000, -78.930, 36.010),
        "Durham": (-78.940, 36.000, -78.930, 36.010),
        "Austin, TX": (-97.745, 30.265, -97.735, 30.275),
        "Austin": (-97.745, 30.265, -97.735, 30.275),
        "Chicago, IL": (-87.635, 41.875, -87.625, 41.885),
        "Chicago": (-87.635, 41.875, -87.625, 41.885),
    }
    
    for key, bbox in locations.items():
        if key.lower() in area_name.lower():
            logger.info(f"Using small test bbox for {key}: {bbox}")
            return bbox
    
    raise ValueError(
        f"Unknown area: {area_name}\n"
        f"Available: {list(locations.keys())}\n"
        f"Or use --bbox to specify coordinates"
    )


def bbox_to_polygon(bbox):
    """Convert bbox to polygon (counter-clockwise)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),  # Close polygon
    ]


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("M1 Scene Generation - Deep Geo2SigMap Integration")
    logger.info("=" * 60)
    
    # Determine polygon
    if args.area:
        bbox = get_simple_bbox(args.area)
        polygon_wgs84 = bbox_to_polygon(bbox)
        scene_id = args.area.replace(" ", "_").replace(",", "").lower()
    elif args.bbox:
        polygon_wgs84 = bbox_to_polygon(args.bbox)
        scene_id = f"scene_{args.bbox[0]}_{args.bbox[1]}"
    else:
        polygon_wgs84 = load_polygon(args.polygon)
        scene_id = args.polygon.stem
    
    logger.info(f"Scene ID: {scene_id}")
    logger.info(f"Polygon: {polygon_wgs84[:3]}... ({len(polygon_wgs84)} vertices)")
    
    # Initialize components
    logger.info("\nInitializing scene generator...")
    material_randomizer = MaterialRandomizer(
        enable_randomization=args.randomize_materials,
        seed=args.seed,
    )
    site_placer = SitePlacer(
        strategy=args.site_strategy,
        seed=args.seed,
    )

    # Randomize number of sites if not fixed
    num_tx = args.num_tx
    if not args.fixed_sites:
        # Default to 1-3 if num_tx is at its default (3) or unspecified
        upper_limit = min(args.num_tx if args.num_tx > 0 else 3, 3)
        num_tx = random.randint(1, upper_limit)
        logger.info(f"Randomized number of transmitters: {num_tx} (1-{upper_limit})")
    
    # Tile mode or single scene?
    if args.tiles:
        logger.info(f"Tile mode: {args.tile_size}m tiles with {args.overlap}m overlap")
        
        tile_gen = TileGenerator(
            material_randomizer=material_randomizer,
            site_placer=site_placer,
        )
        
        metadata_list = tile_gen.generate_tiles(
            bbox_wgs84=args.bbox,
            tile_size_meters=args.tile_size,
            overlap_meters=args.overlap,
            num_tx_per_tile=args.num_tx,
            num_rx_per_tile=args.num_rx,
        )
        
        logger.info(f"\n✓ Generated {len(metadata_list)} tiles")
        
        # Save aggregated metadata
        output_path = args.output / "tiles_metadata.json"
        tile_gen.aggregate_metadata(metadata_list, output_path)
        
    else:
        pass # Avoid logging "Single scene mode" which confuses users running multi-city jobs
        logger.info(f"Generating single scene for area: {scene_id} (Full Bounding Box)")
        
        scene_gen = SceneGenerator(
            geo2sigmap_path=str(args.geo2sigmap_path),
            material_randomizer=material_randomizer,
            site_placer=site_placer,
            output_dir=args.output,
        )
        
        # Create site config
        site_config = {
            'num_tx': num_tx,
            'num_rx': args.num_rx,
            'strategy': args.site_strategy,
        }
        if args.isd:
            site_config['isd'] = args.isd
        
        metadata = scene_gen.generate(
            polygon_points=polygon_wgs84,
            scene_id=scene_id,
            site_config=site_config,
            terrain_config={
                'use_lidar': args.use_lidar,
                'use_dem': args.use_dem,
                'hag_tiff_path': str(args.hag_tiff) if args.hag_tiff else None,
            },
        )
        
        logger.info(f"\n✓ Generated scene: {metadata['scene_id']}")
        logger.info(f"  Buildings: {metadata.get('num_buildings', 'N/A')}")
        logger.info(f"  TX sites: {len([s for s in metadata.get('sites', []) if s.get('site_type')=='tx'])}")
        logger.info(f"  RX sites: {len([s for s in metadata.get('sites', []) if s.get('site_type')=='rx'])}")
        logger.info(f"  Materials: {metadata.get('materials', {})}")
        logger.info(f"  Output: {metadata.get('scene_dir', args.output)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Scene generation complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Verify scene.xml in Mitsuba")
    logger.info("  2. Run M2 data generation with Sionna RT")
    logger.info("  3. Check metadata.json for site configurations")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        sys.exit(1)
