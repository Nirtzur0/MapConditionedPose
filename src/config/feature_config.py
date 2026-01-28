"""
Feature Configuration for Cellular Positioning

This module provides a centralized configuration for all features used in the
positioning system. It defines:
- Which features are available from each layer (RT, PHY, MAC)
- Feature dimensions and shapes
- Feature normalization parameters
- Which features are enabled for training

The configuration ensures consistency between:
- Data generation (Sionna feature extraction)
- Dataset loading (LMDB -> PyTorch)
- Model architecture (input dimensions)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import yaml
from pathlib import Path


class FeatureSource(Enum):
    """Source of feature extraction."""
    SIONNA_PATHS = "sionna_paths"       # Direct from paths object (paths.a, paths.tau, etc.)
    SIONNA_CFR = "sionna_cfr"           # Derived from channel frequency response
    DERIVED = "derived"                  # Computed from other features
    SIMULATED = "simulated"              # Simulated/approximated values


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str                           # Feature name (e.g., "rsrp", "path_gains")
    storage_key: str                    # Key in storage (e.g., "phy_fapi/rsrp")
    shape: Tuple[str, ...]              # Shape description (e.g., ("batch", "cells"), ("batch", "cells", "paths"))
    dtype: str = "float32"              # Data type
    source: FeatureSource = FeatureSource.DERIVED
    default_value: float = 0.0          # Default if missing
    normalize: bool = True              # Whether to normalize
    norm_mean: Optional[float] = None   # Mean for normalization
    norm_std: Optional[float] = None    # Std for normalization
    enabled: bool = True                # Whether to include in model input
    description: str = ""               # Human-readable description
    
    def flat_dim(self, max_cells: int = 8, max_paths: int = 64, num_subcarriers: int = 64) -> int:
        """Calculate flattened dimension for this feature."""
        dim = 1
        for s in self.shape:
            if s == "batch":
                continue  # Batch dimension not counted
            elif s == "cells":
                dim *= max_cells
            elif s == "paths":
                dim *= max_paths
            elif s == "subcarriers":
                dim *= num_subcarriers
            elif s == "antennas":
                dim *= 4  # Default 2x2 MIMO
            elif isinstance(s, int):
                dim *= s
        return dim


@dataclass
class LayerFeatureConfig:
    """Configuration for a feature layer (RT, PHY, or MAC)."""
    name: str
    features: List[FeatureSpec] = field(default_factory=list)
    
    def total_dim(self, max_cells: int = 8, max_paths: int = 64, num_subcarriers: int = 64) -> int:
        """Total input dimension for this layer (enabled features only)."""
        return sum(
            f.flat_dim(max_cells, max_paths, num_subcarriers)
            for f in self.features if f.enabled
        )
    
    def enabled_features(self) -> List[FeatureSpec]:
        """Get list of enabled features."""
        return [f for f in self.features if f.enabled]
    
    def get_storage_keys(self) -> List[str]:
        """Get storage keys for all enabled features."""
        return [f.storage_key for f in self.features if f.enabled]


@dataclass
class FeatureConfig:
    """
    Complete feature configuration for the positioning system.
    
    This is the single source of truth for feature dimensions across:
    - Data generation
    - Dataset loading
    - Model architecture
    """
    # Layer configurations
    rt_layer: LayerFeatureConfig = field(default_factory=lambda: LayerFeatureConfig("rt"))
    phy_layer: LayerFeatureConfig = field(default_factory=lambda: LayerFeatureConfig("phy_fapi"))
    mac_layer: LayerFeatureConfig = field(default_factory=lambda: LayerFeatureConfig("mac_rrc"))
    
    # Dimension limits
    max_cells: int = 8
    max_paths: int = 64
    max_beams: int = 64
    num_subcarriers: int = 64  # Downsampled CFR resolution
    
    # CFR configuration (Channel Frequency Response)
    cfr_enabled: bool = True
    cfr_num_subcarriers: int = 64  # Downsampled from full bandwidth
    cfr_magnitude_only: bool = True  # Use |H| instead of complex H
    
    # PMI configuration
    pmi_codebook_size: int = 16  # Type I single-panel codebook
    pmi_compute_method: str = "svd"  # "svd", "codebook", or "random"
    
    @property
    def rt_features_dim(self) -> int:
        """Total RT features dimension."""
        return self.rt_layer.total_dim(self.max_cells, self.max_paths, self.num_subcarriers)
    
    @property
    def phy_features_dim(self) -> int:
        """Total PHY features dimension."""
        base = self.phy_layer.total_dim(self.max_cells, self.max_paths, self.num_subcarriers)
        if self.cfr_enabled:
            # Add CFR dimension: [cells, subcarriers] if magnitude only, else x2 for complex
            cfr_dim = self.max_cells * self.cfr_num_subcarriers
            if not self.cfr_magnitude_only:
                cfr_dim *= 2  # Real + Imag
            base += cfr_dim
        return base
    
    @property
    def mac_features_dim(self) -> int:
        """Total MAC features dimension."""
        return self.mac_layer.total_dim(self.max_cells, self.max_paths, self.num_subcarriers)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration dict for model initialization."""
        return {
            'rt_features_dim': self.rt_features_dim,
            'phy_features_dim': self.phy_features_dim,
            'mac_features_dim': self.mac_features_dim,
            'max_cells': self.max_cells,
            'max_paths': self.max_paths,
            'max_beams': self.max_beams,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'rt_layer': {
                'name': self.rt_layer.name,
                'features': [
                    {
                        'name': f.name,
                        'storage_key': f.storage_key,
                        'shape': f.shape,
                        'enabled': f.enabled,
                        'default_value': f.default_value,
                    }
                    for f in self.rt_layer.features
                ]
            },
            'phy_layer': {
                'name': self.phy_layer.name,
                'features': [
                    {
                        'name': f.name,
                        'storage_key': f.storage_key,
                        'shape': f.shape,
                        'enabled': f.enabled,
                        'default_value': f.default_value,
                    }
                    for f in self.phy_layer.features
                ]
            },
            'mac_layer': {
                'name': self.mac_layer.name,
                'features': [
                    {
                        'name': f.name,
                        'storage_key': f.storage_key,
                        'shape': f.shape,
                        'enabled': f.enabled,
                        'default_value': f.default_value,
                    }
                    for f in self.mac_layer.features
                ]
            },
            'max_cells': self.max_cells,
            'max_paths': self.max_paths,
            'max_beams': self.max_beams,
            'num_subcarriers': self.num_subcarriers,
            'cfr_enabled': self.cfr_enabled,
            'cfr_num_subcarriers': self.cfr_num_subcarriers,
            'cfr_magnitude_only': self.cfr_magnitude_only,
            'pmi_codebook_size': self.pmi_codebook_size,
            'pmi_compute_method': self.pmi_compute_method,
        }
    
    @classmethod
    def from_yaml(cls, path: str) -> 'FeatureConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureConfig':
        """Create from dictionary."""
        config = cls()
        
        # Update scalar fields
        for key in ['max_cells', 'max_paths', 'max_beams', 'num_subcarriers',
                    'cfr_enabled', 'cfr_num_subcarriers', 'cfr_magnitude_only',
                    'pmi_codebook_size', 'pmi_compute_method']:
            if key in data:
                setattr(config, key, data[key])
        
        # Parse layer configs if present
        if 'rt_layer' in data:
            config.rt_layer = cls._parse_layer(data['rt_layer'], 'rt')
        if 'phy_layer' in data:
            config.phy_layer = cls._parse_layer(data['phy_layer'], 'phy_fapi')
        if 'mac_layer' in data:
            config.mac_layer = cls._parse_layer(data['mac_layer'], 'mac_rrc')
        
        return config
    
    @staticmethod
    def _parse_layer(data: Dict, default_name: str) -> LayerFeatureConfig:
        """Parse a layer configuration from dict."""
        name = data.get('name', default_name)
        features = []
        for f_data in data.get('features', []):
            features.append(FeatureSpec(
                name=f_data['name'],
                storage_key=f_data['storage_key'],
                shape=tuple(f_data.get('shape', ('batch',))),
                enabled=f_data.get('enabled', True),
                default_value=f_data.get('default_value', 0.0),
                description=f_data.get('description', ''),
            ))
        return LayerFeatureConfig(name=name, features=features)


def get_default_feature_config() -> FeatureConfig:
    """
    Create the default feature configuration.
    
    This defines all features extracted from Sionna and used in the model.
    """
    config = FeatureConfig()
    
    # --- RT Layer Features ---
    # These come directly from Sionna's ray tracing paths object
    config.rt_layer = LayerFeatureConfig(
        name="rt",
        features=[
            # Per-cell aggregate features (most useful for positioning)
            FeatureSpec(
                name="toa",
                storage_key="rt/toa",
                shape=("batch", "cells"),
                source=FeatureSource.SIONNA_PATHS,
                description="Time of Arrival - first path delay per cell (seconds)",
            ),
            FeatureSpec(
                name="rms_delay_spread",
                storage_key="rt/rms_delay_spread",
                shape=("batch", "cells"),
                source=FeatureSource.DERIVED,
                description="RMS Delay Spread - multipath channel dispersion (seconds)",
            ),
            FeatureSpec(
                name="rms_angular_spread",
                storage_key="rt/rms_angular_spread",
                shape=("batch", "cells"),
                source=FeatureSource.DERIVED,
                description="RMS Angular Spread - angular scattering (radians)",
            ),
            FeatureSpec(
                name="k_factor",
                storage_key="rt/k_factor",
                shape=("batch", "cells"),
                source=FeatureSource.DERIVED,
                description="Rician K-factor - LoS vs NLoS indicator (dB)",
            ),
            FeatureSpec(
                name="is_nlos",
                storage_key="rt/is_nlos",
                shape=("batch", "cells"),
                dtype="bool",
                source=FeatureSource.DERIVED,
                description="NLoS flag - True if K-factor < 0 dB",
            ),
            FeatureSpec(
                name="num_paths",
                storage_key="rt/num_paths",
                shape=("batch", "cells"),
                dtype="int32",
                source=FeatureSource.SIONNA_PATHS,
                description="Number of valid paths per cell",
            ),
            # Per-path features (variable length, padded to max_paths)
            FeatureSpec(
                name="path_gains",
                storage_key="rt/path_gains",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Complex path coefficients magnitude (linear)",
                enabled=False,  # Disable by default - high dimensionality
            ),
            FeatureSpec(
                name="path_delays",
                storage_key="rt/path_delays",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Per-path propagation delays (seconds)",
                enabled=False,  # Disable by default - high dimensionality
            ),
            FeatureSpec(
                name="path_aoa_azimuth",
                storage_key="rt/path_aoa_azimuth",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Angle of Arrival azimuth per path (radians)",
                enabled=False,
            ),
            FeatureSpec(
                name="path_aoa_elevation",
                storage_key="rt/path_aoa_elevation",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Angle of Arrival elevation per path (radians)",
                enabled=False,
            ),
            FeatureSpec(
                name="path_aod_azimuth",
                storage_key="rt/path_aod_azimuth",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Angle of Departure azimuth per path (radians)",
                enabled=False,
            ),
            FeatureSpec(
                name="path_aod_elevation",
                storage_key="rt/path_aod_elevation",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Angle of Departure elevation per path (radians)",
                enabled=False,
            ),
            FeatureSpec(
                name="path_doppler",
                storage_key="rt/path_doppler",
                shape=("batch", "cells", "paths"),
                source=FeatureSource.SIONNA_PATHS,
                description="Doppler shift per path (Hz)",
                enabled=False,
            ),
        ]
    )
    
    # --- PHY/FAPI Layer Features ---
    # These are derived from the Channel Frequency Response (CFR)
    config.phy_layer = LayerFeatureConfig(
        name="phy_fapi",
        features=[
            FeatureSpec(
                name="rsrp",
                storage_key="phy_fapi/rsrp",
                shape=("batch", "cells"),
                source=FeatureSource.SIONNA_CFR,
                default_value=-120.0,
                description="Reference Signal Received Power (dBm)",
            ),
            FeatureSpec(
                name="rsrq",
                storage_key="phy_fapi/rsrq",
                shape=("batch", "cells"),
                source=FeatureSource.DERIVED,
                default_value=-20.0,
                description="Reference Signal Received Quality (dB)",
            ),
            FeatureSpec(
                name="sinr",
                storage_key="phy_fapi/sinr",
                shape=("batch", "cells"),
                source=FeatureSource.DERIVED,
                default_value=-10.0,
                description="Signal-to-Interference-plus-Noise Ratio (dB)",
            ),
            FeatureSpec(
                name="cqi",
                storage_key="phy_fapi/cqi",
                shape=("batch", "cells"),
                dtype="int32",
                source=FeatureSource.DERIVED,
                default_value=7.0,
                description="Channel Quality Indicator (0-15)",
            ),
            FeatureSpec(
                name="ri",
                storage_key="phy_fapi/ri",
                shape=("batch", "cells"),
                dtype="int32",
                source=FeatureSource.SIONNA_CFR,
                default_value=1.0,
                description="Rank Indicator (1-8)",
            ),
            FeatureSpec(
                name="pmi",
                storage_key="phy_fapi/pmi",
                shape=("batch", "cells"),
                dtype="int32",
                source=FeatureSource.SIONNA_CFR,
                default_value=0.0,
                description="Precoding Matrix Indicator - optimal precoder index",
            ),
            FeatureSpec(
                name="capacity_mbps",
                storage_key="phy_fapi/capacity_mbps",
                shape=("batch", "cells"),
                source=FeatureSource.SIONNA_CFR,
                default_value=0.0,
                description="Shannon capacity estimate (Mbps)",
            ),
            FeatureSpec(
                name="condition_number",
                storage_key="phy_fapi/condition_number",
                shape=("batch", "cells"),
                source=FeatureSource.SIONNA_CFR,
                default_value=1.0,
                description="Channel matrix condition number",
            ),
            # Channel Frequency Response (CFR) - the raw channel estimate
            FeatureSpec(
                name="cfr_magnitude",
                storage_key="phy_fapi/cfr_magnitude",
                shape=("batch", "cells", "subcarriers"),
                source=FeatureSource.SIONNA_CFR,
                default_value=0.0,
                description="Channel Frequency Response magnitude |H(f)|",
                enabled=True,  # This is the new key feature!
            ),
            FeatureSpec(
                name="cfr_phase",
                storage_key="phy_fapi/cfr_phase",
                shape=("batch", "cells", "subcarriers"),
                source=FeatureSource.SIONNA_CFR,
                default_value=0.0,
                description="Channel Frequency Response phase angle(H(f))",
                enabled=False,  # Optional - phase can be noisy
            ),
        ]
    )
    
    # --- MAC/RRC Layer Features ---
    config.mac_layer = LayerFeatureConfig(
        name="mac_rrc",
        features=[
            FeatureSpec(
                name="serving_cell_id",
                storage_key="mac_rrc/serving_cell_id",
                shape=("batch",),
                dtype="int32",
                source=FeatureSource.DERIVED,
                default_value=0.0,
                description="Serving cell Physical Cell ID",
            ),
            FeatureSpec(
                name="timing_advance",
                storage_key="mac_rrc/timing_advance",
                shape=("batch", "cells"),
                dtype="int32",
                source=FeatureSource.DERIVED,
                default_value=0.0,
                description="Timing Advance command (TA index)",
            ),
            # The following are simulated/approximated
            FeatureSpec(
                name="neighbor_cell_ids",
                storage_key="mac_rrc/neighbor_cell_ids",
                shape=("batch", 8),  # Up to 8 neighbors
                dtype="int32",
                source=FeatureSource.DERIVED,
                default_value=0.0,
                description="Neighbor cell IDs sorted by RSRP",
                enabled=False,
            ),
        ]
    )
    
    return config


# Singleton instance for convenience
_default_config: Optional[FeatureConfig] = None


def get_feature_config() -> FeatureConfig:
    """Get the singleton feature configuration."""
    global _default_config
    if _default_config is None:
        _default_config = get_default_feature_config()
    return _default_config


def set_feature_config(config: FeatureConfig) -> None:
    """Set the singleton feature configuration."""
    global _default_config
    _default_config = config
