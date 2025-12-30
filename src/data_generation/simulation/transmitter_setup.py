"""
Transmitter and receiver setup for Sionna RT simulations.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Try importing Sionna
try:
    import sionna
    from sionna.rt import Transmitter, Receiver, PlanarArray
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available; TransmitterSetup will operate in limited mode.")


class TransmitterSetup:
    """
    Handles transmitter and receiver configuration for Sionna RT.
    
    Responsibilities:
    - Create and configure transmitter antenna arrays
    - Setup cell site transmitters with proper positioning
    - Setup UE receivers
    - Handle ground clamping for realistic positioning
    """
    
    def __init__(self, carrier_frequency_hz: float):
        """
        Initialize transmitter setup.
        
        Args:
            carrier_frequency_hz: Carrier frequency for antenna array selection
        """
        self.carrier_frequency_hz = carrier_frequency_hz
        
    def setup_transmitters(
        self,
        scene: Any,
        site_positions: np.ndarray,
        site_metadata: Dict
    ) -> List[Any]:
        """
        Setup cell site transmitters with antenna arrays.
        
        Args:
            scene: Sionna scene object
            site_positions: Array of transmitter positions [N, 3]
            site_metadata: Metadata containing orientation info
            
        Returns:
            List of created transmitter objects
        """
        if not SIONNA_AVAILABLE or scene is None:
            return []
        
        transmitters = []
        
        # Create antenna array based on frequency (shared across transmitters)
        array = self._create_antenna_array()
        scene.tx_array = array
        
        for site_idx, pos in enumerate(site_positions):
            # Get site-specific metadata
            azimuth, downtilt = self._get_site_orientation(site_metadata, site_idx)
            
            tx = Transmitter(
                name=f"BS_{site_idx}",
                position=pos if isinstance(pos, list) else pos.tolist(),
                orientation=[azimuth, downtilt, 0.0]
            )
            
            try:
                # Clamp transmitter to ground level if needed
                self._clamp_to_ground(scene, pos, site_idx, tx)
                
                scene.add(tx)
                transmitters.append(tx)
                logger.debug(f"Added transmitter BS_{site_idx} at {pos}")
            except Exception as e:
                logger.warning(f"Failed to add transmitter BS_{site_idx}: {e}")
        
        return transmitters
    
    def setup_receiver(self, scene: Any, ue_position: np.ndarray, name: str) -> Optional[str]:
        """
        Setup UE receiver at given position.
        
        Args:
            scene: Sionna scene object
            ue_position: Receiver position [x, y, z]
            name: Receiver name
            
        Returns:
            Receiver name if successful, None otherwise
        """
        if not SIONNA_AVAILABLE or scene is None:
            return None
        
        ue_array = PlanarArray(
            num_rows=1, num_cols=2,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )
        scene.rx_array = ue_array
        
        rx = Receiver(
            name=name,
            position=ue_position.tolist(),
            orientation=[0.0, 0.0, 0.0]
        )
        
        scene.add(rx)
        return name
    
    def _create_antenna_array(self) -> Any:
        """
        Create antenna array based on carrier frequency.
        
        Returns:
            PlanarArray object configured for the frequency band
        """
        if self.carrier_frequency_hz < 10e9:
            # Sub-6 GHz: 8x8 array, vertical polarization
            return PlanarArray(
                num_rows=8, num_cols=8,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V"
            )
        else:
            # mmWave: 16x16 array, vertical polarization
            return PlanarArray(
                num_rows=16, num_cols=16,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V"
            )
    
    def _get_site_orientation(self, site_metadata: Dict, site_idx: int) -> tuple:
        """
        Extract azimuth and downtilt from site metadata.
        
        Args:
            site_metadata: Metadata dictionary
            site_idx: Site index
            
        Returns:
            Tuple of (azimuth, downtilt) in degrees
        """
        azimuth = 0.0
        downtilt = 10.0
        
        if isinstance(site_metadata, list):
            if site_idx < len(site_metadata):
                meta = site_metadata[site_idx]
                azimuth = meta.get('orientation', [0, 0, 0])[0] if 'orientation' in meta else 0.0
                downtilt = meta.get('orientation', [0, 10, 0])[1] if 'orientation' in meta else 10.0
        elif isinstance(site_metadata, dict):
            azimuth = site_metadata.get(f'site_{site_idx}_azimuth', 0.0)
            downtilt = site_metadata.get(f'site_{site_idx}_downtilt', 10.0)
        
        return azimuth, downtilt
    
    def _clamp_to_ground(self, scene: Any, pos: np.ndarray, site_idx: int, tx: Any) -> None:
        """
        Clamp transmitter position to ground level with minimum height.
        
        Args:
            scene: Sionna scene object
            pos: Position array [x, y, z]
            site_idx: Site index for logging
            tx: Transmitter object to update
        """
        if not SIONNA_AVAILABLE:
            return
        
        try:
            import drjit as dr
            import mitsuba as mi
            
            z_start = 30000.0
            
            # Create ray from above
            o = mi.Point3f(pos[0], pos[1], z_start)
            d = mi.Vector3f(0.0, 0.0, -1.0)
            ray = mi.Ray3f(o, d)
            
            # Ray cast to find ground
            si = scene.mi_scene.ray_intersect(ray)
            
            if si.is_valid():
                # Ground found
                dist_t = si.t.numpy()[0]
                ground_z = z_start - dist_t
                
                min_tx_h = 25.0
                if pos[2] < ground_z + min_tx_h:
                    logger.debug(f"Adjusting Site {site_idx} height: {pos[2]:.1f} -> {ground_z+min_tx_h:.1f}")
                    pos[2] = ground_z + min_tx_h
                    tx.position = pos if isinstance(pos, list) else pos.tolist()
                    
        except Exception as clamp_e:
            if site_idx == 0:  # Log once
                pass  # Debug mainly
            
            # Fallback to AABB Z-min
            try:
                if hasattr(scene, 'aabb'):
                    z_min = float(scene.aabb[0, 2])
                    if pos[2] < z_min:
                        if z_min > 500 and pos[2] < 100:
                            pos[2] += z_min
                            tx.position = pos if isinstance(pos, list) else pos.tolist()
            except:
                pass
