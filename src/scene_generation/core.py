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
    Import geo2sigmap Scene class avoiding circular imports.
    
    Args:
        geo2sigmap_path: Path to geo2sigmap/package/src
        
    Returns:
        Tuple of (Scene class, ITU_MATERIALS)
    """
    # Load core module directly by file path
    core_module_path = Path(geo2sigmap_path) / "scene_generation" / "core.py"
    itu_module_path = Path(geo2sigmap_path) / "scene_generation" / "itu_materials.py"
    
    if not core_module_path.exists():
        raise ImportError(f"Geo2SigMap not found at {geo2sigmap_path}")
    
    # Load core module
    spec_core = importlib.util.spec_from_file_location("geo2sigmap_core", core_module_path)
    geo2sigmap_core = importlib.util.module_from_spec(spec_core)
    spec_core.loader.exec_module(geo2sigmap_core)
    
    # Load ITU materials module
    spec_itu = importlib.util.spec_from_file_location("geo2sigmap_itu", itu_module_path)
    geo2sigmap_itu = importlib.util.module_from_spec(spec_itu)
    spec_itu.loader.exec_module(geo2sigmap_itu)
    
    return geo2sigmap_core.Scene, geo2sigmap_itu.ITU_MATERIALS


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
    ):
        """
        Initialize scene generator.
        
        Args:
            geo2sigmap_path: Path to geo2sigmap/package/src
            material_randomizer: MaterialRandomizer instance (creates default if None)
            site_placer: SitePlacer instance (creates default if None)
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
        # Load scene XML to get dimensions
        xml_path = scene_dir / "scene.xml"
        if not xml_path.exists():
            logger.warning(f"Scene XML not found at {xml_path}, skipping site placement")
            return []
        
        # Use site placer to determine positions
        sites = self.site_placer.place(
            scene_dir=scene_dir,
            strategy=site_config.get('strategy', 'grid'),
            num_sites=site_config.get('num_sites', 4),
            height_m=site_config.get('height_m', 25.0),
            antenna_config=site_config.get('antenna', {}),
        )
        
        # Update scene XML with site/antenna information
        self._add_sites_to_xml(xml_path, sites)
        
        return sites
    
    def _add_sites_to_xml(self, xml_path: Path, sites: List[Dict]) -> None:
        """
        Add site/transmitter definitions to Mitsuba XML.
        
        Args:
            xml_path: Path to scene.xml
            sites: List of site dictionaries
        """
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Add sites as sensors (for Sionna RT)
        for site in sites:
            sensor = ET.SubElement(root, 'sensor', type='perspective')
            sensor.set('id', site['id'])
            
            # Position
            transform = ET.SubElement(sensor, 'transform', name='to_world')
            lookat = ET.SubElement(transform, 'lookat')
            lookat.set('origin', f"{site['position'][0]}, {site['position'][1]}, {site['position'][2]}")
            lookat.set('target', f"{site['position'][0]}, {site['position'][1]}, 0")
            lookat.set('up', "0, 0, 1")
            
            # Store antenna config as metadata
            metadata = ET.SubElement(sensor, 'metadata')
            for key, value in site.get('antenna', {}).items():
                meta = ET.SubElement(metadata, 'string', name=key)
                meta.set('value', str(value))
        
        # Write updated XML
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        logger.debug(f"Added {len(sites)} sites to {xml_path}")
    
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
