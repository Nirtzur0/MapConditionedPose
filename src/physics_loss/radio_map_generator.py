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
        
        # Create radio map solver
        solver = sn.rt.RadioMapSolver(scene)
        
        # Configure grid
        width, height = self.config.map_size
        x_min, y_min, x_max, y_max = self.config.map_extent
        
        # Create sampling grid
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        z = np.full_like(x, self.config.ue_height)
        
        # Initialize feature maps
        num_features = len(self.config.features)
        radio_map = np.zeros((num_features, height, width), dtype=np.float32)
        
        # Process each cell site
        for site_idx, site in enumerate(cell_sites):
            self.logger.info(f"Processing cell site {site_idx + 1}/{len(cell_sites)}")
            
            # Set transmitter position and orientation
            tx_pos = site['position']
            tx_orient = site.get('orientation', (0.0, 0.0))
            
            # Compute paths using ray tracing
            paths = solver.compute_paths(
                tx=tx_pos,
                rx_positions=(x, y, z),
                max_depth=self.config.max_depth,
                num_samples=self.config.num_samples,
            )
            
            # Extract RT features
            if 'path_gain' in self.config.features:
                idx = self.config.features.index('path_gain')
                # Path gain in dB
                path_gain = 10 * np.log10(np.abs(paths.a) ** 2 + 1e-12)
                radio_map[idx] = np.maximum(radio_map[idx], path_gain.reshape(height, width))
            
            if 'toa' in self.config.features:
                idx = self.config.features.index('toa')
                # Time of arrival in ns
                toa = paths.tau * 1e9
                # Use minimum ToA (first arrival)
                toa_map = np.where(paths.tau > 0, toa, np.inf).min(axis=-1)
                radio_map[idx] = np.where(
                    radio_map[idx] == 0,
                    toa_map.reshape(height, width),
                    np.minimum(radio_map[idx], toa_map.reshape(height, width))
                )
            
            if 'aoa' in self.config.features:
                idx = self.config.features.index('aoa')
                # Angle of arrival (azimuth)
                aoa = np.arctan2(paths.theta[..., 1], paths.theta[..., 0]) * 180 / np.pi
                # Use strongest path's AoA
                strongest_idx = np.argmax(np.abs(paths.a), axis=-1)
                aoa_map = np.take_along_axis(aoa, strongest_idx[..., None], axis=-1).squeeze(-1)
                radio_map[idx] = aoa_map.reshape(height, width)
            
            # Compute PHY features (SNR, SINR)
            if 'snr' in self.config.features or 'sinr' in self.config.features:
                # Received signal power
                rx_power_dbm = self.config.tx_power + path_gain
                
                # Thermal noise power
                k_boltzmann = 1.380649e-23  # J/K
                T = 290  # K (room temperature)
                noise_power_dbm = 10 * np.log10(
                    k_boltzmann * T * self.config.bandwidth
                ) + 30 + self.config.noise_figure
                
                if 'snr' in self.config.features:
                    idx = self.config.features.index('snr')
                    snr_db = rx_power_dbm - noise_power_dbm
                    radio_map[idx] = np.maximum(radio_map[idx], snr_db.reshape(height, width))
                
                if 'sinr' in self.config.features:
                    idx = self.config.features.index('sinr')
                    # For now, SINR = SNR (no interference model)
                    # In multi-cell scenario, would add interference from other cells
                    sinr_db = rx_power_dbm - noise_power_dbm
                    radio_map[idx] = np.maximum(radio_map[idx], sinr_db.reshape(height, width))
            
            # SYS features (throughput, BLER) - simplified models
            if 'throughput' in self.config.features:
                idx = self.config.features.index('throughput')
                # Shannon capacity approximation
                snr_linear = 10 ** (radio_map[self.config.features.index('snr')] / 10)
                capacity_bps = self.config.bandwidth * np.log2(1 + snr_linear)
                capacity_mbps = capacity_bps / 1e6
                radio_map[idx] = np.maximum(radio_map[idx], capacity_mbps)
            
            if 'bler' in self.config.features:
                idx = self.config.features.index('bler')
                # Simplified BLER model based on SNR
                snr_db = radio_map[self.config.features.index('snr')]
                # BLER decreases with SNR (simple exponential model)
                bler = np.exp(-snr_db / 10)
                bler = np.clip(bler, 0.0, 1.0)
                radio_map[idx] = bler
        
        self.logger.info(f"Radio map generation complete. Shape: {radio_map.shape}")
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
