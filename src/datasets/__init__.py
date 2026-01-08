"""
M3: PyTorch Datasets for Training

Datasets for loading precomputed M2 features from storage.
Supports both LMDB (preferred, multiprocessing-friendly) and Zarr (legacy).
"""

from .radio_dataset import RadioLocalizationDataset

# LMDB dataset (preferred)
try:
    from .lmdb_dataset import LMDBRadioLocalizationDataset
    __all__ = ['RadioLocalizationDataset', 'LMDBRadioLocalizationDataset']
except ImportError:
    __all__ = ['RadioLocalizationDataset']
