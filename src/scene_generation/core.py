"""
Core Scene Generator
Extended from Scene Builder with UE localization requirements
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import importlib.util
from pyproj import Transformer

logger = logging.getLogger(__name__)


def _import_scene_builder():
    """
    Import scene_builder Scene class.
    
    Returns:
        Tuple of (Scene class, ITU_MATERIALS)
    """
    # Import from migrated scene_builder module
    try:
        # Try relative import first (when running as package)
        from ..scene_builder import Scene, ITU_MATERIALS
    except ImportError:
        # Fallback to absolute import (when running tests)
        import sys
        from pathlib import Path
        # Add src to path if not already there
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from scene_builder import Scene, ITU_MATERIALS
    return Scene, ITU_MATERIALS


# Will be set during SceneGenerator initialization
SceneBuilderScene = None
ITU_MATERIALS = None



class SceneGenerator:
    """
    Scene generator with material randomization and site placement.
    Extends Scene Builder functionality for UE localization training.
    """
    
    def __init__(
        self,
        material_randomizer: Optional['MaterialRandomizer'] = None,
        site_placer: Optional['SitePlacer'] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize scene generator.
        
        Args:
            material_randomizer: MaterialRandomizer instance (creates default if None)
            site_placer: SitePlacer instance (creates default if None)
            output_dir: Base output directory for scenes (defaults to ./data/scenes)
        """
        # Import scene_builder components
        global SceneBuilderScene, ITU_MATERIALS
        if SceneBuilderScene is None:
            logger.info("Loading Scene Builder")
            SceneBuilderScene, ITU_MATERIALS = _import_scene_builder()
        
        # Import after scene_builder to avoid circular dependency
        from .materials import MaterialRandomizer as _MaterialRandomizer
        from .sites import SitePlacer as _SitePlacer
        
        # Use scene_builder Scene directly
        self.scene = SceneBuilderScene()
        
        # Initialize components
        self.material_randomizer = material_randomizer or _MaterialRandomizer()
        self.site_placer = site_placer or _SitePlacer()
        self.output_dir = output_dir or Path("./data/scenes")
        
        logger.info("SceneGenerator initialized with Scene Builder integration")
    
    def generate(
        self,
        polygon_points: List[Tuple[float, float]],
        scene_id: str,
        materials: Optional[Dict[str, str]] = None,
        site_config: Optional[Dict] = None,
        terrain_config: Optional[Dict] = None,
        osm_server_addr: Optional[str] = None,
    ) -> Dict:
        """
        Generate a complete scene with buildings, terrain, and sites.
        
        Args:
            polygon_points: List of (lon, lat) defining scene boundary
            scene_id: Unique scene identifier
            materials: Dict with 'ground', 'rooftop', 'wall' material IDs
            site_config: Site placement configuration
            terrain_config: Terrain generation configuration
            osm_server_addr: Custom Overpass API endpoint
            
        Returns:
            Scene metadata dictionary
        """
        scene_dir = self.output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating scene: {scene_id}")

        # Compute projected bounds once for placement/metadata
        bounds = self._compute_bounds(polygon_points)
        
        # Sample materials if not provided
        if materials is None:
            materials = self.material_randomizer.sample()
        
        # Terrain configuration
        terrain_cfg = terrain_config or {}
        use_lidar = terrain_cfg.get('use_lidar', False)
        use_dem = terrain_cfg.get('use_dem', False)
        hag_tiff_path = terrain_cfg.get('hag_tiff_path', None)

        if use_lidar and importlib.util.find_spec("pdal") is None:
            logger.warning("PDAL not available; disabling LiDAR terrain generation for this scene.")
            use_lidar = False
            use_dem = False
        
        # Generate base scene using scene_builder
        self.scene(
            points=polygon_points,
            data_dir=str(scene_dir),
            hag_tiff_path=hag_tiff_path,
            osm_server_addr=osm_server_addr or "https://overpass-api.de/api/interpreter",
            lidar_calibration=use_lidar,
            generate_building_map=True,
            ground_material_type=materials['ground'],
            rooftop_material_type=materials['rooftop'],
            wall_material_type=materials['wall'],
            lidar_terrain=use_lidar,
            dem_terrain=use_dem,
            gen_lidar_terrain_only=False,
        )

        # Add site placement
        if site_config:
            sites = self._place_sites(scene_dir, site_config, bounds)
        else:
            sites = []
        
        # Create comprehensive metadata
        metadata = self._create_metadata(
            scene_id=scene_id,
            scene_dir=scene_dir,
            polygon_points=polygon_points,
            materials=materials,
            sites=sites,
            bounds=bounds,
            terrain_config=terrain_cfg,
        )
        
        # Save metadata
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Scene {scene_id} generated at {scene_dir}")
        return metadata
    
    def _place_sites(
        self,
        scene_dir: Path,
        site_config: Dict,
        bounds: Dict,
    ) -> List[Dict]:
        """
        Place base stations in the scene.
        
        Args:
            scene_dir: Scene output directory
            site_config: Configuration for site placement
            
        Returns:
            List of site dictionaries with positions and configurations
        """
        # Load scene XML to ensure scene exists
        xml_path = scene_dir / "scene.xml"
        if not xml_path.exists():
            logger.warning(f"Scene XML not found at {xml_path}, skipping site placement")
            return []

        # Use precomputed projected bounds (meters)
        scene_bounds = (
            bounds.get('x_min', -500),
            bounds.get('y_min', -500),
            bounds.get('x_max', 500),
            bounds.get('y_max', 500)
        )
        
        # Use site placer to determine positions
        sites = self.site_placer.place(
            bounds=scene_bounds,
            num_tx=site_config.get('num_tx', 3),
            num_rx=site_config.get('num_rx', 0),
            height_tx=site_config.get('height_m', 25.0),
            isd_meters=site_config.get('isd'),
        )
        
        # Convert Site objects to dicts
        site_dicts = []
        for site in sites:
            site_dicts.append({
                'site_id': site.site_id,
                'site_type': site.site_type,
                'position': site.position,
                'height': site.position[2],
                'antenna': {
                    'pattern': site.antenna.pattern,
                    'orientation': site.antenna.orientation,
                    'polarization': site.antenna.polarization,
                },
                'cell_id': site.cell_id,
                'sector_id': site.sector_id,
                'power_dbm': site.power_dbm,
            })
        
        # Update scene XML with site/antenna information
        self._add_sites_to_xml(xml_path, site_dicts)
        
        return site_dicts
    
    def _add_sites_to_xml(self, xml_path: Path, sites: List[Dict]) -> None:
        """
        Add site/transmitter definitions to Mitsuba XML.
        
        Args:
            xml_path: Path to scene.xml
            sites: List of site dictionaries
        """
        # Sionna manages transmitters/receivers programmatically, not in XML
        # We'll save site metadata to JSON instead
        logger.debug(f"Site information will be stored in metadata.json, not scene.xml")
        logger.debug(f"Sionna RT will add {len(sites)} transmitters programmatically")
    
    def _create_metadata(
        self,
        scene_id: str,
        scene_dir: Path,
        polygon_points: List[Tuple[float, float]],
        materials: Dict[str, str],
        sites: List[Dict],
        bounds: Dict,
        terrain_config: Dict,
    ) -> Dict:
        """Create comprehensive scene metadata."""
        # Calculate bounding box
        lons = [p[0] for p in polygon_points]
        lats = [p[1] for p in polygon_points]
        bbox_wgs84 = {
            'lon_min': min(lons),
            'lat_min': min(lats),
            'lon_max': max(lons),
            'lat_max': max(lats),
        }
        
        metadata = {
            'scene_id': scene_id,
            'scene_dir': str(scene_dir),
            # Keep both WGS84 bbox and projected meter bounds
            'bbox_wgs84': bbox_wgs84,
            'bbox': {
                'x_min': bounds.get('x_min'),
                'x_max': bounds.get('x_max'),
                'y_min': bounds.get('y_min'),
                'y_max': bounds.get('y_max'),
            },
            'polygon': polygon_points,
            'materials': materials,
            'material_properties': {
                'ground': self.material_randomizer.get_material_properties(materials['ground']),
                'rooftop': self.material_randomizer.get_material_properties(materials['rooftop']),
                'wall': self.material_randomizer.get_material_properties(materials['wall']),
            },
            'sites': sites,
            'num_sites': len(sites),
            'bounds': bounds,
            'terrain': {
                'type': 'lidar' if terrain_config.get('use_lidar') 
                       else 'dem' if terrain_config.get('use_dem') 
                       else 'flat',
                'config': terrain_config,
            },
            'files': {
                'xml': str(scene_dir / 'scene.xml'),
                'building_map': str(scene_dir / 'building_map.png'),
                'metadata': str(scene_dir / 'metadata.json'),
            }
        }

        return metadata

    def _compute_bounds(self, polygon_points: List[Tuple[float, float]]) -> Dict:
        """Compute projected bounds (meters) from lon/lat polygon."""
        if not polygon_points:
            return {}

        # Use UTM zone derived from first point
        lon0, lat0 = polygon_points[0]
        transformer = Transformer.from_crs(
            "EPSG:4326",
            _get_utm_epsg_code(lon0, lat0),
            always_xy=True,
        )

        xs, ys = [], []
        for lon, lat in polygon_points:
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)

        return {
            'x_min': float(min(xs)),
            'x_max': float(max(xs)),
            'y_min': float(min(ys)),
            'y_max': float(max(ys)),
            'utm_epsg': transformer.target_crs.to_string(),
        }


def _get_utm_epsg_code(lon: float, lat: float):
    """Resolve UTM CRS using local copy of scene_builder utils (avoids external dependency)."""
    try:
        from scene_builder.utils import get_utm_epsg_code_from_gps
        return get_utm_epsg_code_from_gps(lon, lat)
    except Exception:
        # Fallback: derive UTM zone manually
        zone = int((lon + 180) / 6) + 1
        hemisphere = 32600 if lat >= 0 else 32700
        return hemisphere + zone


if __name__ == "__main__":
    # Test scene generation
    logging.basicConfig(level=logging.INFO)
    
    # Define a small test area
    test_polygon = [
        (-105.2720, 40.0160),  # Boulder, CO
        (-105.2690, 40.0160),
        (-105.2690, 40.0140),
        (-105.2720, 40.0140),
        (-105.2720, 40.0160),
    ]
    
    generator = SceneGenerator(output_dir=Path("data/raw/scenes"))
    
    metadata = generator.generate(
        polygon_points=test_polygon,
        scene_id="test_boulder",
        site_config={
            'strategy': 'grid',
            'num_sites': 4,
            'height_m': 25.0,
        }
    )
    
    logger.info(f"Generated scene: {metadata['scene_id']}")
    logger.info(f"Location: {metadata['bbox']}")
    logger.info(f"Materials: {metadata['materials']}")
    logger.info(f"Sites: {len(metadata['sites'])}")
