"""
Radio Map Generator for Physics Loss.

Generates precomputed Sionna radio maps with comprehensive RT/PHY/SYS features
for use in physics-consistency loss. Maps are saved in Zarr format for efficient loading.
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import logging

try:
    import sionna as sn
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logging.warning("Sionna not available. RadioMapGenerator will not work.")


@dataclass
class RadioMapConfig:
    """Configuration for radio map generation."""
    
    # Spatial resolution
    resolution: float = 1.0  # meters per pixel
    map_size: Tuple[int, int] = (512, 512)  # (width, height) in pixels
    map_extent: Tuple[float, float, float, float] = (0.0, 0.0, 512.0, 512.0)  # (x_min, y_min, x_max, y_max)
    
    # Height for radio map computation
    ue_height: float = 1.5  # meters above ground
    
    # Features to compute
    features: List[str] = field(default_factory=lambda: [
        'path_gain',   # dB
        'toa',         # ns
        'aoa',         # degrees
        'snr',         # dB
        'sinr',        # dB
        'throughput',  # Mbps
        'bler',        # 0-1
    ])
    
    # Ray tracing parameters
    max_depth: int = 5  # Maximum number of reflections
    num_samples: int = 10000000  # Number of rays per cell site
    
    # PHY layer parameters
    carrier_frequency: float = 3.5e9  # Hz
    bandwidth: float = 100e6  # Hz
    tx_power: float = 43.0  # dBm
    noise_figure: float = 9.0  # dB
    
    # Output format
    output_dir: Path = Path("data/radio_maps")
    compression: str = "zstd"  # Zarr compression
    compression_level: int = 3
    
    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class RadioMapGenerator:
    """
    Generates precomputed Sionna radio maps for physics loss.
    
    For each scene, computes a multi-channel radio map covering the full extent
    with comprehensive RT/PHY/SYS features. Maps are stored in Zarr format for
    efficient loading during training.
    
    Storage:
        Per scene: ~50-100 MB (7 features × 512×512 × float32)
        Total for 10 scenes: ~1 GB
    
    Example:
        >>> config = RadioMapConfig()
        >>> generator = RadioMapGenerator(config)
        >>> radio_map = generator.generate_for_scene(scene, cell_sites)
        >>> generator.save_to_zarr(radio_map, "scene_001")
    """
    
    def __init__(self, config: RadioMapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not SIONNA_AVAILABLE:
            raise ImportError(
                "Sionna is required for RadioMapGenerator. "
                "Install with: pip install sionna"
            )
    
    def generate_for_scene(
        self,
        scene: 'sn.rt.Scene',
        cell_sites: List[Dict],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate radio map for a scene with given cell sites.
        
        Args:
            scene: Sionna RT scene
            cell_sites: List of cell site configs with keys:
                - 'position': (x, y, z) in meters
                - 'orientation': (azimuth, elevation) in degrees
                - 'tx_power': dBm
                - 'antenna': antenna pattern name
            show_progress: Whether to show progress bar
            
        Returns:
            radio_map: (C, H, W) array where C is number of features
        """
        self.logger.info(f"Generating radio map for scene with {len(cell_sites)} cell sites")
        
        # Configure grid parameters
        width, height = self.config.map_size
        x_min, y_min, x_max, y_max = self.config.map_extent
        
        # Calculate center and size for RadioMapSolver
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        size_x = x_max - x_min
        size_y = y_max - y_min
        
        # Initialize feature maps
        num_features = len(self.config.features)
        radio_map = np.zeros((num_features, height, width), dtype=np.float32)
        
        # Set default values for specific features
        for i, feature in enumerate(self.config.features):
            if feature in ['path_gain', 'snr', 'sinr']:
                radio_map[i] = -np.inf
            elif feature == 'toa':
                radio_map[i] = np.inf

        # Validate and setup scene transmitters
        # If scene has no transmitters, we MUST try to add them from cell_sites
        if len(scene.transmitters) == 0:
            self.logger.info(f"Scene has no transmitters. Adding {len(cell_sites)} from config.")
            if len(cell_sites) == 0:
                self.logger.warning("No cell sites provided and no transmitters in scene. Map will be empty.")
            else:
                # Need to configure tx_array first
                if not hasattr(scene, 'tx_array') or scene.tx_array is None:
                    # Default array (similar to MultiLayerDataGenerator logic)
                    if self.config.carrier_frequency < 10e9:
                         scene.tx_array = sn.rt.PlanarArray(
                            num_rows=8, num_cols=8,
                            vertical_spacing=0.5, horizontal_spacing=0.5,
                            pattern="iso", polarization="V"
                        )
                    else:
                        scene.tx_array = sn.rt.PlanarArray(
                            num_rows=16, num_cols=16,
                            vertical_spacing=0.5, horizontal_spacing=0.5,
                            pattern="iso", polarization="V"
                        )
                
                # Add transmitters with height enforcement
                for i, site in enumerate(cell_sites):
                     # Parse site dict
                     pos = site.get('position')
                     if pos is None: continue
                     
                     # Enforce minimum height for transmitters (e.g. 25m) to ensure coverage
                     # unless explicitly marked as 'small_cell' or similar (not available here)
                     min_tx_h = 25.0
                     if pos[2] < min_tx_h:
                         self.logger.warning(f"  Site {i} height {pos[2]:.1f}m too low. Bumping to {min_tx_h}m.")
                         pos[2] = min_tx_h
                     
                     # Orientation
                     orient = site.get('orientation', [0, 0, 0])
                     
                     tx = sn.rt.Transmitter(
                         name=f"Gen_TX_{i}",
                         position=pos,
                         orientation=orient
                     )
                     scene.add(tx)
        
        # Ensure tx_array is present (RadioMapSolver requirement)
        if not hasattr(scene, 'tx_array') or scene.tx_array is None:
             self.logger.warning("Scene missing tx_array. Setting default.")
             scene.tx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        try:
            # Create solver and compute radio map
            solver = sn.rt.RadioMapSolver()
            
            # center uses z=1.5 default if not specified, but we should match config
            # RadioMapSolver expects center as Point3f or array
            center = [center_x, center_y, self.config.ue_height]
            size = [size_x, size_y]
            orientation = [0.0, 0.0, 0.0]
            
            # Run solver
            self.logger.info(f"Running RadioMapSolver (center={center}, size={size})...")
            # cell_size needs to be array-like [dx, dy]
            cell_size = [size_x/width, size_y/height]
            
            # Enable advanced physics for better urban coverage
            rm = solver(
                scene, 
                center=center, 
                size=size, 
                cell_size=cell_size, 
                orientation=orientation,
                diffraction=True,  # Enable diffraction
                # scattering=True, # Not supported in RadioMapSolver __call__ apparently
                # num_samples=self.config.num_samples # Not supported in RadioMapSolver __call__
            )
            
            # Debug available attributes
            self.logger.info(f"Solver output attributes: {[a for a in dir(rm) if not a.startswith('_')]}")
            
            # Extract features
            # rm.path_gain is [num_tx, height, width]
            # We take max over transmitters for coverage map
            
            if 'path_gain' in self.config.features:
                idx = self.config.features.index('path_gain')
                # Path Gain
                try:
                    pg_linear = rm.path_gain.numpy() 
                except:
                    pg_linear = rm.path_gain 
                    
                pg_db = 10 * np.log10(np.maximum(pg_linear, 1e-15))
                # Take max over TXs
                if pg_db.shape[0] > 0:
                    radio_map[idx] = np.max(pg_db, axis=0)
            
            # ToA / AoA / Delay Spread
            # Check if available in rm
            if 'toa' in self.config.features and hasattr(rm, 'time_of_arrival'):
                 # If implemented in future Sionna versions
                 pass
            
            if 'snr' in self.config.features:
                idx = self.config.features.index('snr')
                # RSS in dBm (if available) or compute from Path Gain
                try:
                    rss_linear = rm.rss.numpy()
                except:
                    rss_linear = rm.rss
                    
                rss_dbm = 10 * np.log10(np.maximum(rss_linear, 1e-18)) + 30
                
                # Thermal noise power
                k_boltzmann = 1.380649e-23
                T = 290
                noise_power_dbm = 10 * np.log10(k_boltzmann * T * self.config.bandwidth) + 30 + self.config.noise_figure
                
                snr_db = rss_dbm - noise_power_dbm
                if snr_db.shape[0] > 0:
                    radio_map[idx] = np.max(snr_db, axis=0)

            if 'sinr' in self.config.features:
                idx = self.config.features.index('sinr')
                try:
                    sinr_linear = rm.sinr.numpy()
                except:
                    sinr_linear = rm.sinr
                    
                sinr_db = 10 * np.log10(np.maximum(sinr_linear, 1e-12))
                if sinr_db.shape[0] > 0:
                    radio_map[idx] = np.max(sinr_db, axis=0)

            # SYS features (throughput, BLER) - derived from SNR
            if 'throughput' in self.config.features and 'snr' in self.config.features:
                idx = self.config.features.index('throughput')
                snr_val = radio_map[self.config.features.index('snr')]
                # Shannon capacity
                snr_linear = 10**(snr_val/10)
                capacity_mbps = (self.config.bandwidth * np.log2(1 + snr_linear)) / 1e6
                radio_map[idx] = capacity_mbps
            
            if 'bler' in self.config.features and 'snr' in self.config.features:
                idx = self.config.features.index('bler')
                snr_val = radio_map[self.config.features.index('snr')]
                bler = np.exp(-snr_val / 10)
                bler = np.clip(bler, 0.0, 1.0)
                radio_map[idx] = bler
                
            self.logger.warning("RadioMapSolver does not provide ToA/AoA. These features will remain default (Inf/0).")
            
        except Exception as e:
            self.logger.error(f"Error in RadioMapSolver: {e}")
            # If solver fails, return empty map (already initialized)
            pass

        # Post-process to remove Infs/NaNs
        # Replace -inf in dB features with min value (e.g. -200 dB)
        radio_map[np.isneginf(radio_map)] = -200.0
        
        # Replace inf (e.g. in ToA) with 0 or max value
        # For ToA, if no signal, 0 is safer for NN than a huge number which implies huge delay
        radio_map[np.isposinf(radio_map)] = 0.0
        
        # Replace NaNs with 0
        radio_map[np.isnan(radio_map)] = 0.0

        self.logger.info(f"Radio map generation complete. Shape: {radio_map.shape}")
        self.logger.info(f"Final Map Stats: Min={np.min(radio_map):.2f}, Max={np.max(radio_map):.2f}")
        return radio_map
    
    def save_to_zarr(
        self,
        radio_map: np.ndarray,
        scene_id: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save radio map to Zarr format.
        
        Args:
            radio_map: (C, H, W) feature map
            scene_id: Unique scene identifier
            metadata: Optional metadata to store
            
        Returns:
            Path to saved Zarr file
        """
        output_path = self.config.output_dir / f"{scene_id}.zarr"
        
        # Create Zarr store
        store = zarr.open(str(output_path), mode='w')
        
        # Save radio map with compression
        store.create_dataset(
            'radio_map',
            data=radio_map,
            chunks=(1, 256, 256),
            compression=self.config.compression,
            compression_opts={'level': self.config.compression_level},
        )
        
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'scene_id': scene_id,
            'resolution': self.config.resolution,
            'map_size': self.config.map_size,
            'map_extent': self.config.map_extent,
            'features': self.config.features,
            'ue_height': self.config.ue_height,
        })
        store.attrs.update(metadata)
        
        self.logger.info(f"Saved radio map to {output_path}")
        return output_path
    
    def load_from_zarr(self, scene_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Load radio map from Zarr format.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            radio_map: (C, H, W) feature map
            metadata: Dictionary of metadata
        """
        zarr_path = self.config.output_dir / f"{scene_id}.zarr"
        
        if not zarr_path.exists():
            raise FileNotFoundError(f"Radio map not found: {zarr_path}")
        
        store = zarr.open(str(zarr_path), mode='r')
        radio_map = store['radio_map'][:]
        metadata = dict(store.attrs)
        
        return radio_map, metadata
