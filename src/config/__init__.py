"""
Configuration module for Cellular Positioning Research.

Provides centralized configuration for:
- Feature definitions (RT, PHY, MAC layers)
- Model hyperparameters
- Training settings
"""

from .feature_config import (
    FeatureConfig,
    FeatureSpec,
    FeatureSource,
    LayerFeatureConfig,
    get_feature_config,
    set_feature_config,
    get_default_feature_config,
)
from .feature_schema import (
    RTFeatureIndex,
    PHYFeatureIndex,
    MACFeatureIndex,
    RT_FEATURE_DIM,
    PHY_FEATURE_DIM,
    MAC_FEATURE_DIM,
    RADIO_MAP_CHANNELS,
    PHYSICS_OBSERVED_FEATURES,
    RT_ZARR_KEYS,
    PHY_ZARR_KEYS,
    MAC_ZARR_KEYS,
)

__all__ = [
    'FeatureConfig',
    'FeatureSpec',
    'FeatureSource',
    'LayerFeatureConfig',
    'get_feature_config',
    'set_feature_config',
    'get_default_feature_config',
    'RTFeatureIndex',
    'PHYFeatureIndex',
    'MACFeatureIndex',
    'RT_FEATURE_DIM',
    'PHY_FEATURE_DIM',
    'MAC_FEATURE_DIM',
    'RADIO_MAP_CHANNELS',
    'PHYSICS_OBSERVED_FEATURES',
    'RT_ZARR_KEYS',
    'PHY_ZARR_KEYS',
    'MAC_ZARR_KEYS',
]
