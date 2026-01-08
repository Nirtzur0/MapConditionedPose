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

# LMDB writer (preferred)
try:
    from .lmdb_writer import LMDBDatasetWriter
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

# Zarr writer (DEPRECATED - kept for backward compatibility only)
ZARR_AVAILABLE = False
# Uncomment to enable legacy Zarr support:
# try:
#     from .zarr_writer import ZarrDatasetWriter
#     ZARR_AVAILABLE = True
# except ImportError:
#     ZARR_AVAILABLE = False

# Build __all__ list
_base_exports = [
    "RTFeatureExtractor",
    "PHYFAPIFeatureExtractor",
    "MACRRCFeatureExtractor",
    "MultiLayerDataGenerator",
    "RadioMapGenerator",
    "RadioMapConfig",
    # Measurement utils
    "compute_rsrp",
    "compute_rsrq",
    "compute_cqi",
    "compute_rank_indicator",
    "compute_timing_advance",
    "add_measurement_dropout",
]

if LMDB_AVAILABLE:
    _base_exports.append("LMDBDatasetWriter")
if ZARR_AVAILABLE:
    _base_exports.append("ZarrDatasetWriter")  # Deprecated

__all__ = _base_exports
