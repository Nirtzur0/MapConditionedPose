"""
PyTorch Dataset for Loading M2 LMDB Features

Loads precomputed multi-layer features from M2 data generation:
- RT layer: path gains, ToA, AoA, AoD, Doppler, RMS-DS
- PHY/FAPI layer: RSRP, RSRQ, SINR, CQI, RI, PMI, beam measurements
- MAC/RRC layer: TA, cell IDs, throughput, BLER

No TensorFlow/Sionna needed - all features are precomputed NumPy arrays.

This module now uses LMDB for efficient multiprocessing support.
For backward compatibility, the class name remains RadioLocalizationDataset
but internally uses LMDBRadioDataset.
"""

# Import from LMDB dataset implementation
from .lmdb_dataset import LMDBRadioDataset as RadioLocalizationDataset
from .lmdb_dataset import collate_fn

# For backward compatibility, export all necessary components
__all__ = ['RadioLocalizationDataset', 'collate_fn']
