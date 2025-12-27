"""
Multi-Layer Data Generation (M2)
Sionna RT + PHY/SYS feature extraction for UE localization training
"""

from .features import (
    RTFeatureExtractor,
    PHYFAPIFeatureExtractor,
    MACRRCFeatureExtractor,
)
from .multi_layer_generator import MultiLayerDataGenerator
from .radio_map_generator import RadioMapGenerator, RadioMapConfig
from .measurement_utils import (
    compute_rsrp,
    compute_rsrq,
    compute_cqi,
    compute_rank_indicator,
    compute_timing_advance,
    add_measurement_dropout,
)

# Zarr writer is optional
try:
    from .zarr_writer import ZarrDatasetWriter
    __all__ = [
        "RTFeatureExtractor",
        "PHYFAPIFeatureExtractor",
        "MACRRCFeatureExtractor",
        "MultiLayerDataGenerator",
        "RadioMapGenerator",
        "RadioMapConfig",
        "ZarrDatasetWriter",
        # Measurement utils
        "compute_rsrp",
        "compute_rsrq",
        "compute_cqi",
        "compute_rank_indicator",
        "compute_timing_advance",
        "add_measurement_dropout",
    ]
except ImportError:
    __all__ = [
        "RTFeatureExtractor",
        "PHYFAPIFeatureExtractor",
        "MACRRCFeatureExtractor",
        "MultiLayerDataGenerator",
        # Measurement utils
        "compute_rsrp",
        "compute_rsrq",
        "compute_cqi",
        "compute_rank_indicator",
        "compute_timing_advance",
        "add_measurement_dropout",
    ]
