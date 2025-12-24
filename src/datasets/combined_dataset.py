"""
Combined PyTorch Dataset for Loading from Multiple M2 Zarr Feature Sets
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict

from .radio_dataset import RadioLocalizationDataset

class CombinedRadioLocalizationDataset(Dataset):
    """
    A dataset that concatenates multiple RadioLocalizationDataset instances.

    Args:
        zarr_paths (List[str]): A list of paths to Zarr datasets.
        split (str): The dataset split to load ('train', 'val', or 'test').
        **kwargs: Additional arguments to pass to each RadioLocalizationDataset.
    """
    def __init__(self, zarr_paths: List[str], split: str, **kwargs):
        self.datasets = [RadioLocalizationDataset(path, split=split, **kwargs) for path in zarr_paths]
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.concat_dataset[idx]
