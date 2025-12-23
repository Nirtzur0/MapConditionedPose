#!/usr/bin/env python3
"""
Scene Generation Example Script (M1)
Demonstrates deep Geo2SigMap integration for UE localization

Usage:
    # Single scene
    python scripts/generate_scenes.py --area "Boulder, CO" --output data/scenes
    
    # Multiple tiles
    python scripts/generate_scenes.py --bbox -105.30 40.00 -105.20 40.05 \
                                       --tiles --output data/scenes
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
        default="grid",
        help="Site placement strategy (default: grid)",
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
    # Hardcoded test locations
    locations = {
        "Boulder, CO": (-105.30, 40.00, -105.25, 40.03),
        "Boulder": (-105.30, 40.00, -105.25, 40.03),
        "Durham, NC": (-78.95, 35.98, (-78.90, 36.01),
        "Durham": (-78.95, 35.98, -78.90, 36.01),
    }
    
    for key, bbox in locations.items():
        if key.lower() in area_name.lower():
            logger.info(f"Using bbox for {key}: {bbox}")
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
        logger.info("Single scene mode")
        
        scene_gen = SceneGenerator(
            geo2sigmap_path=str(args.geo2sigmap_path),
            material_randomizer=material_randomizer,
            site_placer=site_placer,
        )
        
        metadata = scene_gen.generate(
            polygon_points=polygon_wgs84,
            scene_id=scene_id,
            name=f"Scene {scene_id}",
            folder=str(args.output),
            num_tx_sites=args.num_tx,
            num_rx_sites=args.num_rx,
        )
        
        logger.info(f"\n✓ Generated scene: {metadata['scene_id']}")
        logger.info(f"  Buildings: {metadata['num_buildings']}")
        logger.info(f"  TX sites: {len([s for s in metadata['sites'] if s['site_type']=='tx'])}")
        logger.info(f"  RX sites: {len([s for s in metadata['sites'] if s['site_type']=='rx'])}")
        logger.info(f"  Materials: {metadata['materials']}")
        logger.info(f"  Output: {args.output / scene_id}")
    
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Location specification
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument(
        "--city",
        type=str,
        help="City name (e.g., 'Boulder, Colorado')",
    )
    location_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box coordinates",
    )
    location_group.add_argument(
        "--point",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        help="Center point coordinates",
    )
    
    # Scene configuration
    parser.add_argument(
        "--tiles",
        type=int,
        default=5,
        help="Number of tiles to generate",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=512.0,
        help="Tile size in meters",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between tiles in meters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/scenes"),
        help="Output directory for scenes",
    )
    
    # Material configuration
    parser.add_argument(
        "--randomize-materials",
        action="store_true",
        default=True,
        help="Enable material domain randomization",
    )
    parser.add_argument(
        "--no-randomize-materials",
        action="store_false",
        dest="randomize_materials",
        help="Disable material randomization",
    )
    
    # Terrain options
    parser.add_argument(
        "--lidar-terrain",
        action="store_true",
        help="Use USGS LiDAR data for terrain (requires data)",
    )
    parser.add_argument(
        "--dem-terrain",
        action="store_true",
        help="Use USGS 1m DEM data for terrain (requires data)",
    )
    
    # OSM server
    parser.add_argument(
        "--osm-server",
        type=str,
        default="https://overpass-api.de/api/interpreter",
        help="OSM Overpass API server",
    )
    
    args = parser.parse_args()
    
    # Check geo2sigmap availability
    if not GEO2SIGMAP_AVAILABLE:
        logger.error(
            "geo2sigmap package not found. Install with:\n"
            "  cd ../geo2sigmap/package\n"
            "  pip install -e ."
        )
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Scene Generation (M1) - Using Geo2SigMap Pipeline")
    logger.info("=" * 60)
    
    # Initialize batch generator
    batch_gen = BatchSceneGenerator(
        output_dir=args.output,
        randomize_materials=args.randomize_materials,
        osm_server=args.osm_server,
        enable_lidar_terrain=args.lidar_terrain,
        enable_dem_terrain=args.dem_terrain,
    )
    
    # Generate scenes based on input type
    if args.city:
        logger.info(f"\nGenerating {args.tiles} tiles for {args.city}")
        tiles = batch_gen.generate_city_tiles(
            city_name=args.city,
            num_tiles=args.tiles,
            tile_size_m=args.tile_size,
        )
    
    elif args.bbox:
        logger.info(f"\nGenerating {args.tiles} tiles from bounding box")
        min_lon, min_lat, max_lon, max_lat = args.bbox
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        tiles = batch_gen.generate_grid_tiles(
            center_point=(center_lon, center_lat),
            tile_size_m=args.tile_size,
            num_tiles=args.tiles,
            overlap_m=args.overlap,
            base_name="custom_tile",
        )
    
    elif args.point:
        logger.info(f"\nGenerating {args.tiles} tiles around point {args.point}")
        tiles = batch_gen.generate_grid_tiles(
            center_point=tuple(args.point),
            tile_size_m=args.tile_size,
            num_tiles=args.tiles,
            overlap_m=args.overlap,
            base_name="point_tile",
        )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Successfully generated {len(tiles)} tiles")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    logger.info("\nGenerated tiles:")
    for tile in tiles:
        logger.info(f"  [{tile['scene_id']}]")
        logger.info(f"    Location: {tile['bbox']}")
        logger.info(f"    Materials: {tile['materials']}")
        logger.info(f"    Output: {tile['output_dir']}")
    
    logger.info(f"\nMetadata saved to: {args.output / 'scenes_metadata.json'}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify scenes (check .xml and .ply files in output directory)")
    logger.info("  2. Proceed to M2: python scripts/generate_dataset.py")


if __name__ == "__main__":
    main()
