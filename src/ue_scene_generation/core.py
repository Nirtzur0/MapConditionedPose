"""
Core Scene Generator
Extended from Geo2SigMap Scene class with UE localization requirements
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import importlib.util

logger = logging.getLogger(__name__)


def _import_geo2sigmap(geo2sigmap_path: str):
    """
    Import geo2sigmap Scene class.
    
    Args:
        geo2sigmap_path: Path to geo2sigmap/package/src (unused, kept for API compat)
        
    Returns:
        Tuple of (Scene class, ITU_MATERIALS)
    """
    # Import from installed geo2sigmap package
    from scene_generation.core import Scene, ITU_MATERIALS
    return Scene, ITU_MATERIALS


# Will be set during SceneGenerator initialization
Geo2SigMapScene = None
ITU_MATERIALS = None



class SceneGenerator:
    """
    Scene generator with material randomization and site placement.
    Extends Geo2SigMap functionality for UE localization training.
    """
    
    def __init__(
        self,
        geo2sigmap_path: str,
        material_randomizer: Optional['MaterialRandomizer'] = None,
        site_placer: Optional['SitePlacer'] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize scene generator.
        
        Args:
            geo2sigmap_path: Path to geo2sigmap/package/src
            material_randomizer: MaterialRandomizer instance (creates default if None)
            site_placer: SitePlacer instance (creates default if None)
            output_dir: Base output directory for scenes (defaults to ./data/scenes)
        """
        # Import geo2sigmap components
        global Geo2SigMapScene, ITU_MATERIALS
        if Geo2SigMapScene is None:
            logger.info(f"Loading Geo2SigMap from {geo2sigmap_path}")
            Geo2SigMapScene, ITU_MATERIALS = _import_geo2sigmap(geo2sigmap_path)
        
        # Import after geo2sigmap to avoid circular dependency
        from .materials import MaterialRandomizer as _MaterialRandomizer
        from .sites import SitePlacer as _SitePlacer
        
        # Use geo2sigmap Scene directly
        self.scene = Geo2SigMapScene()
        
        # Initialize components
        self.material_randomizer = material_randomizer or _MaterialRandomizer()
        self.site_placer = site_placer or _SitePlacer()
        self.output_dir = output_dir or Path("./data/scenes")
        
        logger.info("SceneGenerator initialized with deep Geo2SigMap integration")
    
    def generate(
        self,
        polygon_points: List[Tuple[float, float]],
        scene_id: str,
        materials: Optional[Dict[str, str]] = None,
        site_config: Optional[Dict] = None,
        terrain_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a complete scene with buildings, terrain, and sites.
        
        Args:
            polygon_points: List of (lon, lat) defining scene boundary
            scene_id: Unique scene identifier
            materials: Dict with 'ground', 'rooftop', 'wall' material IDs
            site_config: Site placement configuration
            terrain_config: Terrain generation configuration
            
        Returns:
            Scene metadata dictionary
        """
        scene_dir = self.output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating scene: {scene_id}")
        
        # Sample materials if not provided
        if materials is None:
            materials = self.material_randomizer.sample()
        
        # Terrain configuration
        terrain_cfg = terrain_config or {}
        use_lidar = terrain_cfg.get('use_lidar', False)
        use_dem = terrain_cfg.get('use_dem', False)
        hag_tiff_path = terrain_cfg.get('hag_tiff_path', None)
        
        # Generate base scene using geo2sigmap
        self.scene(
            points=polygon_points,
            data_dir=str(scene_dir),
            hag_tiff_path=hag_tiff_path,
            osm_server_addr="https://overpass-api.de/api/interpreter",
            lidar_calibration=False,
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
            sites = self._place_sites(scene_dir, site_config)
        else:
            sites = []
        
        # Create comprehensive metadata
        metadata = self._create_metadata(
            scene_id=scene_id,
            scene_dir=scene_dir,
            polygon_points=polygon_points,
            materials=materials,
            sites=sites,
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
    ) -> List[Dict]:
        """
        Place base stations in the scene.
        
        Args:
            scene_dir: Scene output directory
            site_config: Configuration for site placement
            
        Returns:
            List of site dictionaries with positions and configurations
        """
        # Load scene XML to get dimensions/bounds
        xml_path = scene_dir / "scene.xml"
        if not xml_path.exists():
            logger.warning(f"Scene XML not found at {xml_path}, skipping site placement")
            return []
        
        # Parse metadata for bounds
        metadata_path = scene_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                meta = json.load(f)
            bounds = meta.get('bounds', {})
            # Convert to (xmin, ymin, xmax, ymax)
            scene_bounds = (
                bounds.get('x_min', -500),
                bounds.get('y_min', -500),
                bounds.get('x_max', 500),
                bounds.get('y_max', 500)
            )
        else:
            # Default bounds
            scene_bounds = (-500, -500, 500, 500)
        
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
        terrain_config: Dict,
    ) -> Dict:
        """Create comprehensive scene metadata."""
        # Calculate bounding box
        lons = [p[0] for p in polygon_points]
        lats = [p[1] for p in polygon_points]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        
        metadata = {
            'scene_id': scene_id,
            'scene_dir': str(scene_dir),
            'bbox': bbox,
            'polygon': polygon_points,
            'materials': materials,
            'material_properties': {
                'ground': self.material_randomizer.get_material_properties(materials['ground']),
                'rooftop': self.material_randomizer.get_material_properties(materials['rooftop']),
                'wall': self.material_randomizer.get_material_properties(materials['wall']),
            },
            'sites': sites,
            'num_sites': len(sites),
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
    
    print(f"Generated scene: {metadata['scene_id']}")
    print(f"Location: {metadata['bbox']}")
    print(f"Materials: {metadata['materials']}")
    print(f"Sites: {len(metadata['sites'])}")
