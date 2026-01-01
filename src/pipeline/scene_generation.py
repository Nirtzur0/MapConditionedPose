"""
Scene Generation Pipeline Module

Handles the generation of 3D scenes with transmitter sites.
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import Optional
from easydict import EasyDict
import yaml

from src.utils.logging_utils import print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def generate_scenes(args, project_root: Path, scene_dir: Path, log_section_func, run_command_func):
    """
    Generate 3D scenes with transmitter sites using GIS data.
    """
    if args.skip_scenes:
        print_info("Skipping scene generation")
        return

    log_section_func("STEP 1: Generate Scenes")

    cmd = [sys.executable, "scripts/scene_generation/generate_scenes.py"]
    
    # Check for custom Overpass API URL in environment
    import os
    overpass_url = os.environ.get("OVERPASS_URL")
    if overpass_url:
        print_info(f"Using custom Overpass API: [dim]{overpass_url}[/dim]")
        cmd.extend(["--osm-server", overpass_url])

    if args.scene_config:
        print_info(f"Using scene config: [bold]{args.scene_config.name}[/bold]")
        with open(args.scene_config, 'r') as f:
            config = EasyDict(yaml.safe_load(f)).scene_generation
        cities = getattr(config, "cities", None)
        if cities:
            for city_cfg in cities:
                city = EasyDict(city_cfg)
                city_cmd = [sys.executable, "scripts/scene_generation/generate_scenes.py"]

                if city.get('bounding_box'):
                    city_cmd.extend(["--bbox"] + [str(c) for c in city.bounding_box])
                    scene_slug = city.get('slug') or city.get('name', 'custom_bbox')
                    scene_slug = scene_slug.replace(", ", "_").replace(" ", "_").lower()
                elif city.get('name'):
                    city_cmd.extend(["--area", city.name])
                    scene_slug = city.name.replace(", ", "_").replace(" ", "_").lower()
                else:
                    raise RuntimeError("Each city entry needs name or bounding_box")

                scene_dir_city = project_root / "data" / "scenes" / scene_slug
                city_cmd.extend(["--output", str(scene_dir_city)])

                # Add custom Overpass API if present
                if overpass_url:
                    city_cmd.extend(["--osm-server", overpass_url])

                tiles_cfg = city.get('tiles') or config.city.tiles
                if tiles_cfg and tiles_cfg.num_tiles > 0:
                    city_cmd.append("--tiles")
                    city_cmd.extend(["--tile-size", str(tiles_cfg.tile_size_m)])
                    city_cmd.extend(["--overlap", str(tiles_cfg.overlap_m)])
                    # Note: TileGenerator will be called by scripts/scene_generation/generate_scenes.py which now gets --osm-server

                sites_cfg = city.get('sites') or config.sites
                if sites_cfg:
                    city_cmd.extend(["--num-tx", str(sites_cfg.num_sites_per_tile)])
                    city_cmd.extend(["--site-strategy", sites_cfg.placement_strategy])

                # Terrain Configuration
                osm_cfg = config.get('osm')
                if osm_cfg and osm_cfg.get('terrain'):
                    terrain = osm_cfg.terrain
                    if terrain.get('use_lidar', False) or terrain.get('source', '').lower() == 'lidar':
                         city_cmd.append("--use-lidar")
                    if terrain.get('use_dem', False) or terrain.get('source', '').lower() == 'dem':
                         city_cmd.append("--use-dem")

                if args.clean and scene_dir_city.exists():
                    print_info(f"Cleaning: [dim]{scene_slug}[/dim]")
                    shutil.rmtree(scene_dir_city)

                run_command_func(city_cmd, f"Scene Generation ({scene_slug})")
            return

        if config.city.bounding_box:
            cmd.extend(["--bbox"] + [str(c) for c in config.city.bounding_box])
        elif config.city.name:
            cmd.extend(["--area", config.city.name])

        scene_dir = project_root / "data" / "scenes" / config.city.name.replace(", ", "_").replace(" ", "_").lower()
        cmd.extend(["--output", str(scene_dir)])

        tiles_cfg = config.city.tiles
        if tiles_cfg and tiles_cfg.num_tiles > 0:
            cmd.append("--tiles")
            cmd.extend(["--tile-size", str(tiles_cfg.tile_size_m)])
            cmd.extend(["--overlap", str(tiles_cfg.overlap_m)])

        if config.sites:
            cmd.extend(["--num-tx", str(config.sites.num_sites_per_tile)])
            cmd.extend(["--site-strategy", config.sites.placement_strategy])

        # Terrain Configuration
        osm_cfg = config.get('osm')
        if osm_cfg and osm_cfg.get('terrain'):
            terrain = osm_cfg.terrain
            if terrain.get('use_lidar', False) or terrain.get('source', '').lower() == 'lidar':
                    cmd.append("--use-lidar")
            if terrain.get('use_dem', False) or terrain.get('source', '').lower() == 'dem':
                    cmd.append("--use-dem")

    else:
        bbox_str = f"{args.bbox[0]:.3f}, {args.bbox[1]:.3f}, {args.bbox[2]:.3f}, {args.bbox[3]:.3f}"
        print_info(f"Bounding box: [dim]{bbox_str}[/dim]")
        cmd.extend([
            "--bbox",
            str(args.bbox[0]),  # west
            str(args.bbox[1]),  # south
            str(args.bbox[2]),  # east
            str(args.bbox[3]),  # north
            "--output", str(scene_dir),
            "--num-tx", str(args.num_tx),
            "--site-strategy", args.site_strategy,
        ])
        if args.tiles:
            cmd.append("--tiles")

    # Clean existing scenes if requested
    if args.clean and scene_dir.exists():
        print_info(f"Cleaning: [dim]{scene_dir}[/dim]")
        shutil.rmtree(scene_dir)

    run_command_func(cmd, "Scene Generation")

    # Verify scenes were created
    if not scene_dir.exists():
        raise RuntimeError(f"Scene directory not created: {scene_dir}")

    scene_count = len(list(scene_dir.glob("scene_*")))
    print_success(f"Created {scene_count} scene(s)")