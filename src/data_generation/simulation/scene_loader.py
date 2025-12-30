"""
Scene loading and management for Sionna RT simulations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try importing Sionna
try:
    import sionna
    from sionna.rt import load_scene
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available; SceneLoader will operate in limited mode.")


class SceneLoader:
    """
    Handles loading and managing Sionna RT scenes.
    
    Responsibilities:
    - Load Mitsuba XML scenes into Sionna
    - Load scene metadata from JSON files
    - Configure scene-level settings (frequency, synthetic arrays)
    """
    
    def __init__(self, scene_dir: Path, carrier_frequency_hz: float):
        """
        Initialize the scene loader.
        
        Args:
            scene_dir: Base directory containing scene subdirectories
            carrier_frequency_hz: Carrier frequency for RT simulations
        """
        self.scene_dir = Path(scene_dir)
        self.carrier_frequency_hz = carrier_frequency_hz
        
    def load_scene(self, scene_path: Path) -> Optional[Any]:
        """
        Load a Mitsuba XML scene into Sionna RT.
        
        Args:
            scene_path: Path to scene.xml file
            
        Returns:
            Loaded Sionna scene object, or None if Sionna unavailable
        """
        if not SIONNA_AVAILABLE:
            logger.warning("Sionna unavailable; cannot load scene.")
            return None
            
        logger.info(f"Loading Sionna scene from {scene_path}...")
        scene = load_scene(str(scene_path))
        
        # Apply scene-level settings
        scene.frequency = self.carrier_frequency_hz
        scene.synthetic_array = True
        
        # Clear any existing transmitters
        if hasattr(scene, 'transmitters'):
            tx_names = list(scene.transmitters.keys())
            if tx_names:
                for name in tx_names:
                    scene.remove(name)
        
        logger.info(f"Scene loaded: frequency={self.carrier_frequency_hz/1e9:.2f} GHz.")
        return scene
    
    def load_metadata(self, scene_id: str) -> Dict:
        """
        Load scene metadata from JSON file.
        
        Args:
            scene_id: Scene identifier (subdirectory name)
            
        Returns:
            Dictionary containing scene metadata
        """
        metadata_path = self.scene_dir / scene_id / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Scene metadata not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {scene_id} ({len(metadata.get('sites', []))} sites).")
        return metadata
    
    def get_scene_bounds(self, scene: Any) -> Optional[Dict]:
        """
        Extract bounding box information from a scene.
        
        Args:
            scene: Sionna scene object
            
        Returns:
            Dictionary with bbox info, or None if unavailable
        """
        if scene is None or not hasattr(scene, 'mi_scene'):
            return None
            
        try:
            bbox = scene.mi_scene.bbox()
            return {
                'x_min': float(bbox.min.x),
                'y_min': float(bbox.min.y),
                'z_min': float(bbox.min.z),
                'x_max': float(bbox.max.x),
                'y_max': float(bbox.max.y),
                'z_max': float(bbox.max.z),
            }
        except Exception as e:
            logger.warning(f"Could not extract scene bounds: {e}")
            return None
