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

__all__ = [
    'FeatureConfig',
    'FeatureSpec',
    'FeatureSource',
    'LayerFeatureConfig',
    'get_feature_config',
    'set_feature_config',
    'get_default_feature_config',
]
