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
    
    # Features to compute (Aligned with 3GPP/Dataset features)
    features: List[str] = field(default_factory=lambda: [
        'rsrp',        # dBm
        'rsrq',        # dB
        'sinr',        # dB
        'cqi',         # 0-15
        'throughput',  # Mbps
        'path_gain',   # dB
        'snr',         # dB
        'rms_ds',      # seconds (log scale?)
        'mean_delay',  # seconds
        'k_factor',    # dB
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
    """
    
    def __init__(self, config: RadioMapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not SIONNA_AVAILABLE:
            raise ImportError(
                "Sionna is required for RadioMapGenerator. "
                "Install with: pip install sionna"
            )
    
    def _get_ground_level(self, scene: 'sn.rt.Scene', xy: List[float]) -> float:
        """Determines ground level at given (x,y) by casting a ray from above."""
        try:
            # Use Mitsuba directly if available for reliable ray casting
            import mitsuba as mi
            
            z_start = 30000.0
            o = mi.Point3f(float(xy[0]), float(xy[1]), z_start)
            d = mi.Vector3f(0.0, 0.0, -1.0)
            ray = mi.Ray3f(o, d)
            
            si = scene.mi_scene.ray_intersect(ray)
            if si.is_valid():
                t = si.t.numpy()[0]
                return z_start - t
            return 0.0
        except Exception:
            try:
                if hasattr(scene, 'aabb'):
                    return float(scene.aabb[0, 2])
            except:
                pass
            return 0.0

    def generate_for_scene(
        self,
        scene: 'sn.rt.Scene',
        cell_sites: List[Dict],
        show_progress: bool = True,
        return_sionna_object: bool = False,
    ) -> np.ndarray:
        """Generate radio map for a scene."""
        self.logger.info(f"Generating radio map for scene with {len(cell_sites)} cell sites")
        
        # Configure grid parameters
        width, height = self.config.map_size
        x_min, y_min, x_max, y_max = self.config.map_extent
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        size_x = x_max - x_min
        size_y = y_max - y_min
        
        # Initialize feature maps
        num_features = len(self.config.features)
        radio_map = np.zeros((num_features, height, width), dtype=np.float32)
        
        # Initialize with reasonable defaults (floor values)
        for i, feature in enumerate(self.config.features):
            if feature in ['rsrp', 'rsrq', 'sinr', 'path_gain', 'snr']:
                radio_map[i] = -200.0 # dB floor
            elif feature == 'cqi':
                radio_map[i] = 0.0
            elif feature == 'throughput':
                radio_map[i] = 0.0
            elif feature in ['rms_ds', 'mean_delay']:
                radio_map[i] = 0.0 # Seconds
            elif feature == 'k_factor':
                radio_map[i] = -100.0 # dB (approximating no LOS?)

        # Validate and setup scene transmitters
        if len(scene.transmitters) == 0:
            self.logger.info(f"Adding {len(cell_sites)} transmitters from config.")
            if not hasattr(scene, 'tx_array') or scene.tx_array is None:
                scene.tx_array = sn.rt.PlanarArray(num_rows=8, num_cols=8, pattern="iso", polarization="V")
            
            for i, site in enumerate(cell_sites):
                 pos = list(site.get('position'))
                 if pos is None: continue
                 
                 ground_z = self._get_ground_level(scene, pos[:2])
                 min_tx_h = 25.0
                 if pos[2] < ground_z + min_tx_h:
                     pos[2] = ground_z + min_tx_h
                 
                 orient = site.get('orientation', [0, 0, 0])
                 tx = sn.rt.Transmitter(f"Gen_TX_{i}", pos, orient)
                 scene.add(tx)
        
        if not hasattr(scene, 'tx_array') or scene.tx_array is None:
             scene.tx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        try:
            solver = sn.rt.RadioMapSolver()
            center_ground_z = self._get_ground_level(scene, [center_x, center_y])
            ue_z = center_ground_z + self.config.ue_height
            
            center = [center_x, center_y, ue_z]
            size = [size_x, size_y]
            cell_size = [size_x/width, size_y/height]
            
            self.logger.info(f"Running RadioMapSolver...")
            rm = solver(
                scene, center=center, size=size, cell_size=cell_size,
                orientation=[0.,0.,0.], diffraction=True,
                max_depth=self.config.max_depth
            )
            sionna_rm = rm
            
            # --- Feature Extraction [Sionna Native -> 3GPP] ---
            
            # 1. Path Gain -> RSRP
            # RSRP = PathGain + TxPower
            pg_linear = None
            if hasattr(rm, 'path_gain'):
                try: pg_linear = rm.path_gain.numpy()
                except: pg_linear = rm.path_gain
            
            if pg_linear is not None:
                pg_db = 10 * np.log10(np.maximum(pg_linear, 1e-18))
                max_pg_db = np.max(pg_db, axis=0) # Max over TXs
                
                if 'rsrp' in self.config.features:
                    idx = self.config.features.index('rsrp')
                    radio_map[idx] = max_pg_db + self.config.tx_power
                
                if 'path_gain' in self.config.features:
                    idx = self.config.features.index('path_gain')
                    radio_map[idx] = max_pg_db

                if 'snr' in self.config.features:
                    idx = self.config.features.index('snr')
                    # Calculate thermal noise power in dBm
                    # K_b * T * B * F
                    # Thermal noise density: -174 dBm/Hz
                    noise_power_dbm = -174 + 10 * np.log10(self.config.bandwidth) + self.config.noise_figure
                    # SNR = RSRP - NoisePower
                    snr_val = (max_pg_db + self.config.tx_power) - noise_power_dbm
                    radio_map[idx] = snr_val
            
            # 2. SINR -> SINR, RSRQ, CQI, Throughput
            sinr_db_map = None
            if hasattr(rm, 'sinr'):
                try: sinr_lin = rm.sinr.numpy()
                except: sinr_lin = rm.sinr
                
                sinr_db = 10 * np.log10(np.maximum(sinr_lin, 1e-12))
                sinr_db_map = np.max(sinr_db, axis=0)
                
                if 'sinr' in self.config.features:
                    idx = self.config.features.index('sinr')
                    radio_map[idx] = sinr_db_map
            
            # Derived Metrics from SINR
            if sinr_db_map is not None:
                # RSRQ
                if 'rsrq' in self.config.features:
                    idx = self.config.features.index('rsrq')
                    # RSRQ approx: N / (1 + 1/SINR) -> in dB
                    sinr_lin_val = 10**(sinr_db_map/10.0)
                    rsrq_lin = sinr_lin_val / (sinr_lin_val + 1.0)
                    radio_map[idx] = 10 * np.log10(rsrq_lin + 1e-10) # ~ -3 to -20 dB range
                
                # CQI (Linear mapping approximation)
                if 'cqi' in self.config.features:
                    idx = self.config.features.index('cqi')
                    # Map -5dB to 25dB -> 0 to 15
                    cqi = (sinr_db_map + 5.0) / 2.0
                    radio_map[idx] = np.clip(cqi, 0, 15)
                
                # Throughput (Shannon)
                if 'throughput' in self.config.features:
                    idx = self.config.features.index('throughput')
                    snr_lin_val = 10**(sinr_db_map/10.0)
                    bw_mhz = self.config.bandwidth / 1e6
                    capacity = bw_mhz * np.log2(1 + snr_lin_val)
                    radio_map[idx] = capacity

        except Exception as e:
            self.logger.error(f"Error in RadioMapSolver: {e}")
            self.logger.exception(e)
            sionna_rm = None

        if return_sionna_object:
            return radio_map, sionna_rm
        return radio_map
    
    def save_to_zarr(self, radio_map: np.ndarray, scene_id: str, metadata: Optional[Dict] = None) -> Path:
        """Save radio map to Zarr format."""
        output_path = self.config.output_dir / f"{scene_id}.zarr"
        store = zarr.open(str(output_path), mode='w')
        store.create_dataset('radio_map', data=radio_map, chunks=(1, 256, 256),
                           compression=self.config.compression, compression_opts={'level': self.config.compression_level})
        
        if metadata is None: metadata = {}
        metadata.update({
            'scene_id': scene_id,
            'resolution': self.config.resolution,
            'map_size': self.config.map_size,
            'map_extent': self.config.map_extent,
            'features': self.config.features,
            'ue_height': self.config.ue_height
        })
        store.attrs.update(metadata)
        return output_path
    
    def load_from_zarr(self, scene_id: str) -> Tuple[np.ndarray, Dict]:
        """Load radio map from Zarr format."""
        zarr_path = self.config.output_dir / f"{scene_id}.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"Radio map not found: {zarr_path}")
        store = zarr.open(str(zarr_path), mode='r')
        return store['radio_map'][:], dict(store.attrs)
