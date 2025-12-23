"""
Tile Generation for Large-Scale Scene Creation
Handles spatial tiling, batch processing, and coordinate transformations
"""

import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
from pyproj import Transformer
import yaml

from .core import SceneGenerator
from .materials import MaterialRandomizer
from .sites import SitePlacer

logger = logging.getLogger(__name__)


class TileGenerator:
    """
    Generate multiple scene tiles for large geographic areas.
    
    Features:
    - Grid-based tiling with configurable overlap
    - WGS84 (lat/lon) to UTM coordinate transformation
    - Batch scene generation with consistent materials
    - Metadata aggregation for downstream M2 processing
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        material_randomizer: Optional[MaterialRandomizer] = None,
        site_placer: Optional[SitePlacer] = None,
    ):
        """
        Initialize tile generator.
        
        Args:
            config_path: Path to scene_generation.yaml config
            material_randomizer: Material randomizer instance
            site_placer: Site placer instance
        """
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        self.material_randomizer = material_randomizer or MaterialRandomizer()
        self.site_placer = site_placer or SitePlacer()
        
        # Lazy-load scene generator (only when needed)
        self._scene_generator = None
        
        logger.info("TileGenerator initialized")
    
    @property
    def scene_generator(self):
        """Lazy-load scene generator."""
        if self._scene_generator is None:
            self._scene_generator = SceneGenerator(
                geo2sigmap_path=self.config['geo2sigmap']['package_path'],
                material_randomizer=self.material_randomizer,
                site_placer=self.site_placer,
            )
        return self._scene_generator
    
    def _default_config(self) -> Dict:
        """Default configuration if no config file provided."""
        return {
            'geo2sigmap': {
                'package_path': '/home/ubuntu/projects/geo2sigmap/package/src',
            },
            'tiling': {
                'tile_size_meters': 500,
                'overlap_meters': 50,
            },
            'osm': {
                'building_levels': 5,
            },
            'output': {
                'base_dir': './data/scenes',
            },
        }
    
    def generate_tiles(
        self,
        bbox_wgs84: Tuple[float, float, float, float],  # (lon_min, lat_min, lon_max, lat_max)
        tile_size_meters: Optional[float] = None,
        overlap_meters: Optional[float] = None,
        num_tx_per_tile: int = 3,
        num_rx_per_tile: int = 10,
    ) -> List[Dict]:
        """
        Generate scene tiles covering a bounding box.
        
        Args:
            bbox_wgs84: WGS84 bounding box (lon_min, lat_min, lon_max, lat_max)
            tile_size_meters: Tile size in meters (overrides config)
            overlap_meters: Tile overlap in meters (overrides config)
            num_tx_per_tile: Transmitters per tile
            num_rx_per_tile: Receivers per tile
            
        Returns:
            List of metadata dictionaries for all tiles
        """
        tile_size_meters = tile_size_meters or self.config['tiling']['tile_size_meters']
        overlap_meters = overlap_meters or self.config['tiling']['overlap_meters']
        
        # Convert WGS84 bbox to UTM
        tiles_utm = self._create_tile_grid(bbox_wgs84, tile_size_meters, overlap_meters)
        
        logger.info(f"Generating {len(tiles_utm)} tiles ({tile_size_meters}m x {tile_size_meters}m)")
        
        all_metadata = []
        for tile_idx, tile_utm in enumerate(tiles_utm):
            logger.info(f"Processing tile {tile_idx+1}/{len(tiles_utm)}")
            
            # Generate scene for this tile
            metadata = self._generate_tile(
                tile_idx=tile_idx,
                tile_utm=tile_utm,
                num_tx=num_tx_per_tile,
                num_rx=num_rx_per_tile,
            )
            
            all_metadata.append(metadata)
        
        logger.info(f"Generated {len(all_metadata)} tiles successfully")
        return all_metadata
    
    def _create_tile_grid(
        self,
        bbox_wgs84: Tuple[float, float, float, float],
        tile_size_meters: float,
        overlap_meters: float,
    ) -> List[Dict]:
        """
        Create tile grid in UTM coordinates.
        
        Args:
            bbox_wgs84: WGS84 bbox (lon_min, lat_min, lon_max, lat_max)
            tile_size_meters: Tile size
            overlap_meters: Tile overlap
            
        Returns:
            List of tile dictionaries with UTM bounds and WGS84 polygon
        """
        lon_min, lat_min, lon_max, lat_max = bbox_wgs84
        
        # Determine UTM zone from center point
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        hemisphere = 'north' if center_lat >= 0 else 'south'
        
        # Create WGS84 -> UTM transformer
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        
        # Convert bbox corners to UTM
        xmin_utm, ymin_utm = transformer.transform(lon_min, lat_min)
        xmax_utm, ymax_utm = transformer.transform(lon_max, lat_max)
        
        # Generate tile grid with overlap
        stride = tile_size_meters - overlap_meters
        
        tiles = []
        x = xmin_utm
        tile_x = 0
        while x < xmax_utm:
            y = ymin_utm
            tile_y = 0
            while y < ymax_utm:
                # Tile bounds in UTM
                tile_xmin = x
                tile_ymin = y
                tile_xmax = min(x + tile_size_meters, xmax_utm)
                tile_ymax = min(y + tile_size_meters, ymax_utm)
                
                # Convert back to WGS84 for OSM query
                tile_polygon_wgs84 = self._utm_to_wgs84_polygon(
                    (tile_xmin, tile_ymin, tile_xmax, tile_ymax),
                    transformer,
                )
                
                tiles.append({
                    'tile_id': f"tile_{tile_x}_{tile_y}",
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'bounds_utm': (tile_xmin, tile_ymin, tile_xmax, tile_ymax),
                    'polygon_wgs84': tile_polygon_wgs84,
                    'utm_zone': utm_zone,
                    'hemisphere': hemisphere,
                })
                
                y += stride
                tile_y += 1
            
            x += stride
            tile_x += 1
        
        logger.info(f"Created {len(tiles)} tiles ({tile_x} x {tile_y} grid)")
        return tiles
    
    def _utm_to_wgs84_polygon(
        self,
        bounds_utm: Tuple[float, float, float, float],
        transformer: Transformer,
    ) -> List[Tuple[float, float]]:
        """
        Convert UTM bounds to WGS84 polygon (for OSM query).
        
        Args:
            bounds_utm: (xmin, ymin, xmax, ymax) in UTM
            transformer: pyproj Transformer (WGS84 -> UTM)
            
        Returns:
            List of (lon, lat) polygon vertices
        """
        xmin, ymin, xmax, ymax = bounds_utm
        
        # Create polygon vertices (counter-clockwise)
        utm_vertices = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),  # Close polygon
        ]
        
        # Transform to WGS84 (invert transformer)
        inv_transformer = Transformer.from_crs(
            transformer.target_crs,
            transformer.source_crs,
            always_xy=True,
        )
        
        wgs84_vertices = []
        for x_utm, y_utm in utm_vertices:
            lon, lat = inv_transformer.transform(x_utm, y_utm)
            wgs84_vertices.append((lon, lat))
        
        return wgs84_vertices
    
    def _generate_tile(
        self,
        tile_idx: int,
        tile_utm: Dict,
        num_tx: int,
        num_rx: int,
    ) -> Dict:
        """
        Generate scene for a single tile.
        
        Args:
            tile_idx: Tile index
            tile_utm: Tile dictionary with bounds and polygon
            num_tx: Number of transmitters
            num_rx: Number of receivers
            
        Returns:
            Scene metadata dictionary
        """
        scene_id = f"scene_{tile_utm['tile_id']}"
        
        # Generate scene using SceneGenerator
        metadata = self.scene_generator.generate(
            polygon_points=tile_utm['polygon_wgs84'],
            scene_id=scene_id,
            name=scene_id,
            folder=self.config['output']['base_dir'],
            building_levels=self.config['osm']['building_levels'],
            num_tx_sites=num_tx,
            num_rx_sites=num_rx,
        )
        
        # Add tile-specific metadata
        metadata['tile'] = {
            'tile_id': tile_utm['tile_id'],
            'tile_x': tile_utm['tile_x'],
            'tile_y': tile_utm['tile_y'],
            'bounds_utm': tile_utm['bounds_utm'],
            'utm_zone': tile_utm['utm_zone'],
            'hemisphere': tile_utm['hemisphere'],
        }
        
        return metadata
    
    def aggregate_metadata(
        self,
        tile_metadata_list: List[Dict],
        output_path: Path,
    ) -> None:
        """
        Aggregate metadata from all tiles into single file.
        
        Args:
            tile_metadata_list: List of metadata dictionaries
            output_path: Output YAML file path
        """
        aggregated = {
            'num_tiles': len(tile_metadata_list),
            'tiles': tile_metadata_list,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(aggregated, f, default_flow_style=False)
        
        logger.info(f"Aggregated metadata saved to {output_path}")


if __name__ == "__main__":
    # Test tile generation
    logging.basicConfig(level=logging.INFO)
    
    # Boulder, CO region (small test area)
    bbox_wgs84 = (-105.30, 40.00, -105.20, 40.05)  # ~10km x 5km
    
    tile_gen = TileGenerator()
    
    # Generate 2x2 grid of tiles
    metadata_list = tile_gen.generate_tiles(
        bbox_wgs84=bbox_wgs84,
        tile_size_meters=500,
        overlap_meters=50,
        num_tx_per_tile=1,
        num_rx_per_tile=5,
    )
    
    print(f"\nGenerated {len(metadata_list)} tiles")
    for meta in metadata_list[:3]:
        print(f"  {meta['scene_id']}: {meta['num_buildings']} buildings, {len(meta['sites'])} sites")
