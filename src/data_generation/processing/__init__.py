"""
Processing subpackage for data aggregation and realism.

This package contains classes for:
- Data stacking and padding
- Measurement realism (dropout, quantization)
"""

from .data_stacker import DataStacker
from .measurement_processor import MeasurementProcessor

__all__ = [
    'DataStacker',
    'MeasurementProcessor',
]
