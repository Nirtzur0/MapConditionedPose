"""
PyTorch Dataset for Loading M2 Zarr Features

Loads precomputed multi-layer features from M2 data generation:
- RT layer: path gains, ToA, AoA, AoD, Doppler, RMS-DS
- PHY/FAPI layer: RSRP, RSRQ, SINR, CQI, RI, PMI, beam measurements
- MAC/RRC layer: TA, cell IDs, throughput, BLER

No TensorFlow/Sionna needed - all features are precomputed NumPy arrays.
"""

import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

from .augmentations import RadioAugmentation

logger = logging.getLogger(__name__)


class RadioLocalizationDataset(Dataset):
    """PyTorch Dataset for UE localization with multi-layer radio measurements.
    
    Loads precomputed features from M2 Zarr storage. All features are
    already computed by Sionna during data generation, stored as NumPy
    arrays, and loaded directly into PyTorch tensors.
    
    Args:
        zarr_path: Path to Zarr dataset (e.g., 'data/synthetic/dataset_city01.zarr')
        split: 'train', 'val', or 'test'
        map_resolution: Meters per pixel for maps (default: 1.0)
        scene_extent: Scene size in meters (default: 512)
        normalize: Whether to normalize features (default: True)
        handle_missing: How to handle missing values: 'mask', 'zero', 'mean'
    """
    
    def __init__(
        self,
        zarr_path: str,
        split: str = 'train',
        map_resolution: float = 1.0,
        scene_extent: int = 512,
        normalize: bool = True,
        handle_missing: str = 'mask',
        augmentation: Optional[Dict] = None,
    ):
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.map_resolution = map_resolution
        self.scene_extent = self._resolve_scene_extent(scene_extent)
        self.normalize = normalize
        self.handle_missing = handle_missing
        self._dataset_origin = None
        
        # Augmentation (only applied during training)
        self.augmentor = RadioAugmentation(augmentation if split == 'train' else None)
        
        # Open Zarr store (read-only)
        self.store = zarr.open(str(self.zarr_path), mode='r')
        
        # Get split indices
        self.indices = self._get_split_indices()
        
        # Load normalization stats if available
        self.norm_stats = self._load_normalization_stats()
        self._dataset_origin = self._infer_dataset_origin()
        
        if self.augmentor.enabled:
            logger.info(f"Augmentation enabled for training: {list(self.augmentor.config.keys())}")
        
        logger.info(f"Loaded {len(self)} samples for {split} split from {zarr_path}")
    
    def _get_split_indices(self) -> np.ndarray:
        """Get indices for train/val/test split."""
        # Check if pre-split indices exist in metadata
        if 'metadata' in self.store and f'{self.split}_indices' in self.store['metadata']:
            return self.store['metadata'][f'{self.split}_indices'][:]
        
        # Otherwise, create split based on total samples
        total_samples = len(self.store['positions/ue_x'])
        indices = np.arange(total_samples)
        
        # Default split: 70% train, 15% val, 15% test
        np.random.seed(42)  # Reproducible splits
        np.random.shuffle(indices)
        
        if self.split == 'all':
            return indices
            
        # Standard splits
        if self.split == 'train_80':
            n_split = int(0.8 * total_samples)
            return indices[:n_split]
        elif self.split == 'val_20':
            n_split = int(0.8 * total_samples)
            return indices[n_split:]
            
        # Default legacy split: 70% train, 15% val, 15% test
        n_train = int(0.7 * total_samples)
        n_val = int(0.15 * total_samples)
        
        if self.split == 'train':
            return indices[:n_train]
        elif self.split == 'val':
            return indices[n_train:n_train + n_val]
        else:  # test
            return indices[n_train + n_val:]

    @staticmethod
    def _resolve_scene_extent(scene_extent: Optional[float]) -> float:
        if isinstance(scene_extent, (list, tuple)):
            if len(scene_extent) == 4:
                return float(abs(scene_extent[2] - scene_extent[0]))
            return float(scene_extent[-1])
        return float(scene_extent) if scene_extent is not None else 512.0
    
    def _load_normalization_stats(self) -> Optional[Dict]:
        """Load pre-computed normalization statistics."""
        if not self.normalize:
            return None
        
        if 'metadata' in self.store and 'normalization' in self.store['metadata']:
            return {
                key: {
                    'mean': self.store['metadata']['normalization'][key]['mean'][:],
                    'std': self.store['metadata']['normalization'][key]['std'][:],
                }
                for key in self.store['metadata']['normalization'].keys()
            }
        return None

    def _infer_dataset_origin(self) -> Optional[Tuple[float, float]]:
        if 'positions' not in self.store:
            return None
        try:
            ue_x = np.asarray(self.store['positions/ue_x'])
            ue_y = np.asarray(self.store['positions/ue_y'])
            if ue_x.size == 0 or ue_y.size == 0:
                return None
            return float(np.nanmin(ue_x)), float(np.nanmin(ue_y))
        except Exception:
            return None
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.
        
        Returns a dictionary containing:
            - measurements: Temporal sequence data (RT, PHY, MAC features, IDs, timestamps, mask)
            - radio_map: Radio signal map [5, H, W]
            - osm_map: OSM building/geometry map [5, H, W]
            - position: Ground truth (x, y) in meters
            - cell_grid: Ground truth coarse grid cell index
        """
        # Get actual index in Zarr store
        zarr_idx = int(self.indices[idx])
        
        # Load ground truth position from individual arrays
        ue_x = float(self.store['positions/ue_x'][zarr_idx])
        ue_y = float(self.store['positions/ue_y'][zarr_idx])
        
        # Load scene bounding box; fall back to dataset origin if missing/invalid.
        scene_bbox = None
        if 'metadata' in self.store and 'scene_bbox' in self.store['metadata']:
            scene_bbox = self.store['metadata/scene_bbox'][zarr_idx]
        if scene_bbox is not None:
            scene_bbox = np.asarray(scene_bbox, dtype=np.float32)
            bbox_valid = np.isfinite(scene_bbox).all() and np.any(np.abs(scene_bbox) > 1e-6)
        else:
            bbox_valid = False

        if bbox_valid:
            x_min, y_min = scene_bbox[0], scene_bbox[1]
        elif self._dataset_origin is not None:
            x_min, y_min = self._dataset_origin
        else:
            x_min, y_min = 0.0, 0.0

        # Convert to local coordinates
        local_x = ue_x - x_min
        local_y = ue_y - y_min
        
        position = torch.tensor([local_x, local_y], dtype=torch.float32)
        
        
        # Convert to grid cell for coarse supervision (grid_size: e.g., 32x32 for scene)
        grid_size = 32
        cell_size = self.scene_extent / grid_size
        # Clip positions to valid range [0, scene_extent]
        clamped_x = max(0, min(position[0].item(), self.scene_extent - 1))
        clamped_y = max(0, min(position[1].item(), self.scene_extent - 1))
        grid_x = int(clamped_x / cell_size)
        grid_y = int(clamped_y / cell_size)
        # Ensure grid indices are within bounds
        grid_x = max(0, min(grid_x, grid_size - 1))
        grid_y = max(0, min(grid_y, grid_size - 1))
        cell_grid = torch.tensor(
            grid_y * grid_size + grid_x,
            dtype=torch.long
        )
        
        # Load measurement sequence
        measurements = self._load_measurements(zarr_idx)
        
        # Load maps and resize to model's expected dimensions (256x256)
        radio_map = self._load_radio_map(zarr_idx)
        osm_map = self._load_osm_map(zarr_idx)
        
        # Resize maps if needed (model expects 256x256)
        target_size = 256
        if radio_map.shape[1] != target_size or radio_map.shape[2] != target_size:
            radio_map = torch.nn.functional.interpolate(
                radio_map.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        if osm_map.shape[1] != target_size or osm_map.shape[2] != target_size:
            osm_map = torch.nn.functional.interpolate(
                osm_map.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Apply augmentations (training only)
        measurements, radio_map, osm_map = self.augmentor(measurements, radio_map, osm_map)
        
        return {
            'measurements': measurements,
            'radio_map': radio_map,
            'osm_map': osm_map,
            'position': position[:2],  # Only x, y (drop z if present)
            'cell_grid': cell_grid,
        }
    
    
    def _load_measurements(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process temporal measurement sequences for a given index."""
        zarr_idx = idx
        
        # Extract and average RT layer features
        rt_data = {}
        if 'rt' in self.store:
            rt_group = self.store['rt']
            rt_features = []
            
            # Aggregate available RT features
            if 'path_gains' in rt_group: rt_features.append(np.mean(np.abs(rt_group['path_gains'][zarr_idx])))
            if 'path_delays' in rt_group: rt_features.append(np.mean(rt_group['path_delays'][zarr_idx]))
            if 'path_aoa_azimuth' in rt_group: rt_features.append(np.mean(rt_group['path_aoa_azimuth'][zarr_idx]))
            if 'path_aoa_elevation' in rt_group: rt_features.append(np.mean(rt_group['path_aoa_elevation'][zarr_idx]))
            if 'path_aod_azimuth' in rt_group: rt_features.append(np.mean(rt_group['path_aod_azimuth'][zarr_idx]))
            if 'path_aod_elevation' in rt_group: rt_features.append(np.mean(rt_group['path_aod_elevation'][zarr_idx]))
            if 'path_doppler' in rt_group: rt_features.append(np.mean(rt_group['path_doppler'][zarr_idx]))
            if 'rms_delay_spread' in rt_group: rt_features.append(rt_group['rms_delay_spread'][zarr_idx])
            if 'k_factor' in rt_group: rt_features.append(rt_group['k_factor'][zarr_idx])
            if 'num_paths' in rt_group: rt_features.append(float(rt_group['num_paths'][zarr_idx]))
            
            if rt_features:
                rt_data['features'] = torch.tensor(
                    np.array(rt_features, dtype=np.float32),
                    dtype=torch.float32
                ).unsqueeze(0)
        
        # Extract and average PHY/FAPI layer features
        phy_data = {}
        if 'phy_fapi' in self.store:
            phy_group = self.store['phy_fapi']
            phy_features = []
            
            # Aggregate available PHY features
            if 'rsrp' in phy_group: phy_features.append(np.mean(phy_group['rsrp'][zarr_idx]) if phy_group['rsrp'][zarr_idx].size > 0 else -120.0)
            if 'rsrq' in phy_group: phy_features.append(np.mean(phy_group['rsrq'][zarr_idx]) if phy_group['rsrq'][zarr_idx].size > 0 else -20.0)
            if 'sinr' in phy_group: phy_features.append(np.mean(phy_group['sinr'][zarr_idx]) if phy_group['sinr'][zarr_idx].size > 0 else -10.0)
            if 'cqi' in phy_group: phy_features.append(np.mean(phy_group['cqi'][zarr_idx]) if phy_group['cqi'][zarr_idx].size > 0 else 7.0)
            if 'ri' in phy_group: phy_features.append(np.mean(phy_group['ri'][zarr_idx]) if phy_group['ri'][zarr_idx].size > 0 else 1.0)
            if 'pmi' in phy_group: phy_features.append(np.mean(phy_group['pmi'][zarr_idx]) if phy_group['pmi'][zarr_idx].size > 0 else 0.0)
            
            if phy_features:
                phy_data['features'] = torch.tensor(
                    np.array(phy_features, dtype=np.float32),
                    dtype=torch.float32
                )
        
        # Extract and average MAC/RRC layer features
        mac_data = {}
        if 'mac_rrc' in self.store:
            mac_group = self.store['mac_rrc']
            mac_features = []

            def _scalar(value, default=0.0):
                arr = np.array(value)
                if arr.size == 0:
                    return float(default)
                return float(np.mean(arr))
            
            if 'serving_cell_id' in mac_group: mac_features.append(_scalar(mac_group['serving_cell_id'][zarr_idx], default=0.0))
            if 'timing_advance' in mac_group: mac_features.append(_scalar(mac_group['timing_advance'][zarr_idx], default=0.0))
            if 'phr' in mac_group: mac_features.append(_scalar(mac_group['phr'][zarr_idx], default=0.0))
            if 'throughput' in mac_group: mac_features.append(_scalar(mac_group['throughput'][zarr_idx], default=0.0))
            if 'bler' in mac_group: mac_features.append(_scalar(mac_group['bler'][zarr_idx], default=0.0))
            
            if mac_features:
                mac_data['features'] = torch.tensor(
                    np.array(mac_features, dtype=np.float32),
                    dtype=torch.float32
                )
        
        # Initialize cell/beam IDs and timestamps
        cell_ids = torch.zeros(1, dtype=torch.long)
        beam_ids = torch.zeros(1, dtype=torch.long)
        timestamps = torch.zeros(1, dtype=torch.float32)
        
        # Load from metadata if available
        if 'metadata' in self.store:
            if 'cell_ids' in self.store['metadata']:
                cell_ids = torch.tensor(self.store['metadata']['cell_ids'][idx], dtype=torch.long)
            if 'beam_ids' in self.store['metadata']:
                beam_ids = torch.tensor(self.store['metadata']['beam_ids'][idx], dtype=torch.long)
            if 'timestamps' in self.store['metadata']:
                timestamps = torch.tensor(self.store['metadata']['timestamps'][idx], dtype=torch.float32)
        
        # Create mask (True for valid, False for padding/missing)
        seq_len = max(rt_data.get('features', torch.empty(0, 0)).shape[0], 1)
        mask = torch.ones(seq_len, dtype=torch.bool)
        
        # Handle missing values: mask out NaNs in sequence data
        if self.handle_missing == 'mask':
            if 'features' in rt_data: mask &= ~torch.isnan(rt_data['features']).any(dim=-1)
            if 'features' in phy_data: phy_data['features'] = torch.nan_to_num(phy_data['features'], nan=0.0)
            if 'features' in mac_data: mac_data['features'] = torch.nan_to_num(mac_data['features'], nan=0.0)
        elif self.handle_missing == 'zero':
            # Replace NaNs with zero
            if 'features' in rt_data: rt_data['features'] = torch.nan_to_num(rt_data['features'], nan=0.0)
            if 'features' in phy_data: phy_data['features'] = torch.nan_to_num(phy_data['features'], nan=0.0)
            if 'features' in mac_data: mac_data['features'] = torch.nan_to_num(mac_data['features'], nan=0.0)
        
        # Normalize features if requested
        if self.normalize and self.norm_stats:
            rt_data = self._normalize_features(rt_data, 'rt')
            phy_data = self._normalize_features(phy_data, 'phy')
            mac_data = self._normalize_features(mac_data, 'mac')
        
        # Pad/truncate features to model's expected dimensions
        rt_features = rt_data.get('features', torch.zeros(seq_len, 10))
        if rt_features.shape[-1] < 10: rt_features = torch.cat([rt_features, torch.zeros(rt_features.shape[0], 10 - rt_features.shape[-1])], dim=-1)
        elif rt_features.shape[-1] > 10: rt_features = rt_features[..., :10]
            
        phy_features = phy_data.get('features', torch.zeros(8))
        if len(phy_features) < 8: phy_features = torch.cat([phy_features, torch.zeros(8 - len(phy_features))], dim=0)
        elif len(phy_features) > 8: phy_features = phy_features[:8]
            
        mac_features = mac_data.get('features', torch.zeros(6))
        if len(mac_features) < 6: mac_features = torch.cat([mac_features, torch.zeros(6 - len(mac_features))], dim=0)
        elif len(mac_features) > 6: mac_features = mac_features[:6]
        
        return {
            'rt_features': rt_features,
            'phy_features': phy_features,
            'mac_features': mac_features,
            'cell_ids': cell_ids if len(cell_ids) > 0 else torch.zeros(seq_len, dtype=torch.long),
            'beam_ids': beam_ids if len(beam_ids) > 0 else torch.zeros(seq_len, dtype=torch.long),
            'timestamps': timestamps if len(timestamps) > 0 else torch.arange(seq_len, dtype=torch.float32),
            'mask': mask,
        }
    
    def _load_radio_map(self, idx: int) -> torch.Tensor:
        """Load and process a precomputed Sionna radio map."""
        # Indirection: Use scene_index to find the correct map
        scene_idx = 0
        if 'metadata' in self.store and 'scene_indices' in self.store['metadata']:
            scene_idx = int(self.store['metadata']['scene_indices'][idx])
        
        if 'radio_maps' not in self.store:
            # Return dummy map if not available, padded to 5 channels
            H = W = int(self.scene_extent / self.map_resolution)
            return torch.zeros(5, H, W, dtype=torch.float32)
        
        # Load radio map for the specific scene
        # Handle shape [NumScenes, C, H, W] or [NumScenes, H, W, C]
        radio_map = torch.tensor(self.store['radio_maps'][scene_idx], dtype=torch.float32)
        
        # Ensure channel-first format [C, H, W]
        if radio_map.dim() == 3:
             # Check if channel is last [H, W, C]
             if radio_map.shape[-1] < radio_map.shape[0]: 
                 radio_map = radio_map.permute(2, 0, 1)
        
        return radio_map[:5]
    
    def _load_osm_map(self, idx: int) -> torch.Tensor:
        """Load and process an OSM building/geometry map."""
        scene_idx = 0
        if 'metadata' in self.store and 'scene_indices' in self.store['metadata']:
            scene_idx = int(self.store['metadata']['scene_indices'][idx])

        if 'osm_maps' not in self.store:
            # Return dummy map if not available, padded to 5 channels
            H = W = int(self.scene_extent / self.map_resolution)
            return torch.zeros(5, H, W, dtype=torch.float32)
        
        # Load OSM map for the specific scene
        osm_map = torch.tensor(self.store['osm_maps'][scene_idx], dtype=torch.float32)
        
        # Ensure channel-first format [C, H, W]
        if osm_map.dim() == 3:
             if osm_map.shape[-1] < osm_map.shape[0]:
                 osm_map = osm_map.permute(2, 0, 1)
        
        # Pad to 5 channels if needed
        if osm_map.shape[0] < 5:
            padding = torch.zeros(5 - osm_map.shape[0], osm_map.shape[1], osm_map.shape[2])
            osm_map = torch.cat([osm_map, padding], dim=0)
        
        return osm_map[:5]
    
    def _normalize_features(
        self,
        data: Dict,
        layer: str
    ) -> Dict:
        """Normalize features using pre-computed stats."""
        if 'features' not in data or self.norm_stats is None:
            return data
        
        if layer in self.norm_stats:
            mean = torch.tensor(self.norm_stats[layer]['mean'], dtype=torch.float32)
            std = torch.tensor(self.norm_stats[layer]['std'], dtype=torch.float32)
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            
            data['features'] = (data['features'] - mean) / std
        
        return data


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences.
    
    Pads measurement sequences to the maximum length in the current batch.
    """
    # Find max sequence length in batch
    max_seq_len = max(
        sample['measurements']['rt_features'].shape[0]
        for sample in batch
    )
    
    # Initialize padded batch structure
    padded_batch = {
        'measurements': {
            'rt_features': [],
            'phy_features': [],
            'mac_features': [],
            'cell_ids': [],
            'beam_ids': [],
            'timestamps': [],
            'mask': [],
        },
        'radio_map': [],
        'osm_map': [],
        'position': [],
        'cell_grid': [],
    }
    
    for sample in batch:
        meas = sample['measurements']
        seq_len = meas['rt_features'].shape[0]
        
        # Pad features to max_seq_len
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            
            padded_batch['measurements']['rt_features'].append(
                torch.cat([
                    meas['rt_features'],
                    torch.zeros(pad_len, meas['rt_features'].shape[1])
                ])
            )
            padded_batch['measurements']['phy_features'].append(
                torch.cat([
                    meas['phy_features'].unsqueeze(0).expand(seq_len, -1),
                    torch.zeros(pad_len, meas['phy_features'].shape[0])
                ])
            )
            padded_batch['measurements']['mac_features'].append(
                torch.cat([
                    meas['mac_features'].unsqueeze(0).expand(seq_len, -1),
                    torch.zeros(pad_len, meas['mac_features'].shape[0])
                ])
            )
            
            # Pad IDs and timestamps
            padded_batch['measurements']['cell_ids'].append(
                torch.cat([meas['cell_ids'], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_batch['measurements']['beam_ids'].append(
                torch.cat([meas['beam_ids'], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_batch['measurements']['timestamps'].append(
                torch.cat([meas['timestamps'], torch.zeros(pad_len)])
            )
            
            # Pad mask (False for padded values)
            padded_batch['measurements']['mask'].append(
                torch.cat([meas['mask'], torch.zeros(pad_len, dtype=torch.bool)])
            )
        else: # No padding needed
            padded_batch['measurements']['rt_features'].append(meas['rt_features'])
            padded_batch['measurements']['phy_features'].append(
                meas['phy_features'].unsqueeze(0).expand(seq_len, -1)
            )
            padded_batch['measurements']['mac_features'].append(
                meas['mac_features'].unsqueeze(0).expand(seq_len, -1)
            )
            padded_batch['measurements']['cell_ids'].append(meas['cell_ids'])
            padded_batch['measurements']['beam_ids'].append(meas['beam_ids'])
            padded_batch['measurements']['timestamps'].append(meas['timestamps'])
            padded_batch['measurements']['mask'].append(meas['mask'])
        
        # Stack maps and targets
        padded_batch['radio_map'].append(sample['radio_map'])
        padded_batch['osm_map'].append(sample['osm_map'])
        padded_batch['position'].append(sample['position'])
        padded_batch['cell_grid'].append(sample['cell_grid'])
    
    # Stack all tensors
    return {
        'measurements': {
            k: torch.stack(v) for k, v in padded_batch['measurements'].items()
        },
        'radio_map': torch.stack(padded_batch['radio_map']),
        'osm_map': torch.stack(padded_batch['osm_map']),
        'position': torch.stack(padded_batch['position']),
        'cell_grid': torch.stack(padded_batch['cell_grid']),
    }
