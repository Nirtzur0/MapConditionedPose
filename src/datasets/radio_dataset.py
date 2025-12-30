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
    
    # --- Feature Schemas ---
    RT_SCHEMA = [
        ('path_gains', 'path_gains'),
        ('path_delays', 'path_delays'),
        ('path_aoa_azimuth', 'path_aoa_azimuth'),
        ('path_aoa_elevation', 'path_aoa_elevation'),
        ('path_aod_azimuth', 'path_aod_azimuth'),
        ('path_aod_elevation', 'path_aod_elevation'),
        ('path_doppler', 'path_doppler'),
        ('rms_delay_spread', 'rms_delay_spread'),
        ('k_factor', 'k_factor'),
        ('num_paths', 'num_paths'),
        ('toa', 'toa'),
        ('is_nlos', 'is_nlos'),
        ('rms_angular_spread', 'rms_angular_spread'),
    ]
    
    PHY_SCHEMA = [
        ('rsrp', 'rsrp', -120.0),
        ('rsrq', 'rsrq', -20.0),
        ('sinr', 'sinr', -10.0),
        ('cqi', 'cqi', 7.0),
        ('ri', 'ri', 1.0),
        ('pmi', 'pmi', 0.0),
        ('capacity_mbps', 'capacity_mbps', 0.0),
        ('condition_number', 'condition_number', 1.0),
    ]
    
    MAC_SCHEMA = [
        ('serving_cell_id', 'serving_cell_id', 0.0),
        ('timing_advance', 'timing_advance', 0.0),
        ('phr', 'phr', 0.0),
        ('throughput', 'throughput', 0.0),
        ('bler', 'bler', 0.0),
    ]

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
        if not zarr_path or str(zarr_path) == '.':
            raise ValueError(f"Invalid zarr_path provided to RadioLocalizationDataset: '{zarr_path}'")
        
        logger.debug(f"Opening Zarr store at: {self.zarr_path}")
        try:
            self.store = zarr.open(str(self.zarr_path), mode='r')
        except Exception as e:
            logger.error(f"Failed to open Zarr store at {self.zarr_path}: {e}")
            raise
        
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
        
        # --- SCENE-BASED SPLIT (Prevents Leakage) ---
        if 'metadata' in self.store and 'scene_ids' in self.store['metadata']:
            all_scene_ids = self.store['metadata/scene_ids'][:]
            unique_scenes = np.unique(all_scene_ids)
            n_scenes = len(unique_scenes)
            
            # If multi-scene, split by scene ID
            if n_scenes > 1:
                # Deterministic shuffle of SCENES
                rng = np.random.RandomState(42)
                shuffled_scenes = np.sort(unique_scenes) # Sort first for stability
                rng.shuffle(shuffled_scenes)
                
                n_train = int(0.7 * n_scenes)
                n_val = int(0.15 * n_scenes)
                
                train_scenes = set(shuffled_scenes[:n_train])
                val_scenes = set(shuffled_scenes[n_train:n_train+n_val])
                test_scenes = set(shuffled_scenes[n_train+n_val:])
                
                # Assign target set
                if self.split in ['train', 'train_80']:
                    target_scenes = train_scenes
                    # For train_80, we just take 80% logic, basically train + half val? 
                    # Simpler to stick to standard 70/15/15 for now or adapt if specific split requested
                    if self.split == 'train_80': # Legacy adaption
                         n_tr_80 = int(0.8 * n_scenes)
                         target_scenes = set(shuffled_scenes[:n_tr_80])

                elif self.split in ['val', 'val_20']:
                    target_scenes = val_scenes
                    if self.split == 'val_20':
                         n_tr_80 = int(0.8 * n_scenes)
                         target_scenes = set(shuffled_scenes[n_tr_80:])

                elif self.split == 'test':
                    target_scenes = test_scenes
                else:
                    return np.arange(len(all_scene_ids)) # 'all'

                # Find indices belonging to target scenes
                # Verify performance: np.isin is okay for millions of items?
                # scene_ids is object array of strings. 
                # Faster: convert unique list to dictionary map?
                # Or just use np.isin
                mask = np.isin(all_scene_ids, list(target_scenes))
                return np.where(mask)[0]
        
        # --- FALLBACK: RANDOM SPLIT (Single scene or no metadata) ---
        total_samples = self.store['positions/ue_x'].shape[0]
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
        
        if 'metadata' not in self.store or 'normalization' not in self.store['metadata']:
            return None
            
        norm_grp = self.store['metadata']['normalization']
        stats = {}
        
        # Helper to extract scalar mean/std for a key
        def get_stat(layer_grp, key):
            # Check nested key "layer/key" first (Zarr v2 style if flattened?)
            # But the structure is likely: norm_grp[layer][key]['mean']
            try:
                if layer_grp in norm_grp and key in norm_grp[layer_grp]:
                    feat_stats = norm_grp[layer_grp][key]
                    return float(feat_stats['mean'][0]), float(feat_stats['std'][0])
            except Exception:
                pass
            return 0.0, 1.0

        # Build vectors for each layer based on schema order
        
        # RT
        rt_means, rt_stds = [], []
        for _, key in RadioLocalizationDataset.RT_SCHEMA:
            m, s = get_stat('rt', key)
            rt_means.append(m)
            rt_stds.append(s)
        stats['rt'] = {'mean': rt_means, 'std': rt_stds}
        
        # PHY
        phy_means, phy_stds = [], []
        for _, key, _ in RadioLocalizationDataset.PHY_SCHEMA:
            m, s = get_stat('phy_fapi', key)
            phy_means.append(m)
            phy_stds.append(s)
        stats['phy'] = {'mean': phy_means, 'std': phy_stds}
        
        # MAC
        mac_means, mac_stds = [], []
        for _, key, _ in RadioLocalizationDataset.MAC_SCHEMA:
            m, s = get_stat('mac_rrc', key)
            mac_means.append(m)
            mac_stds.append(s)
        stats['mac'] = {'mean': mac_means, 'std': mac_stds}
            
        return stats

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

        # Standard Coordinate Handling (Local Bottom-Left [0, W])
        # We assume stored data is in the correct Local frame.
        local_x = ue_x
        local_y = ue_y
        
        # Backward Compatibility / Safety:
        # If values appear to be Global UTM (> 10,000), convert to Local.
        if abs(local_x) > 10000:
             local_x -= x_min
        if abs(local_y) > 10000:
             local_y -= y_min
             
        # If values appear to be Centered-Local (Negative), shift to Bottom-Left [0, W]
        # (This handles datasets generated with the old 'Centered' logic)
        # Note: This assumes valid range is [0, W]. If we see -W/2, we shift.
        if local_x < -100: # Allow small noise/margin
             bbox_width = scene_bbox[2] - scene_bbox[0] if bbox_valid else self.scene_extent
             local_x += (bbox_width / 2.0)
             
        if local_y < -100:
             bbox_height = scene_bbox[3] - scene_bbox[1] if bbox_valid else self.scene_extent
             local_y += (bbox_height / 2.0)

        position = torch.tensor([local_x, local_y], dtype=torch.float32)
        
        # Determine sample-specific extent for grid calculation
        if bbox_valid:
            w = scene_bbox[2] - scene_bbox[0]
            h = scene_bbox[3] - scene_bbox[1]
            sample_extent = float(max(w, h))
        else:
            sample_extent = self.scene_extent
        
        # Convert to grid cell for coarse supervision (grid_size: e.g., 32x32)
        grid_size = 32
        cell_size = sample_extent / grid_size
        
        # Clip positions to valid range [0, sample_extent]
        clamped_x = max(0, min(position[0].item(), sample_extent - 1e-3))
        clamped_y = max(0, min(position[1].item(), sample_extent - 1e-3))
        
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
        
        # Prepare normalized position [0, 1]
        norm_position = position / sample_extent
        
        return {
            'measurements': measurements,
            'radio_map': radio_map,
            'osm_map': osm_map,
            'position': norm_position[:2],  # Normalized [0, 1]
            'cell_grid': cell_grid,
            'sample_extent': torch.tensor(sample_extent, dtype=torch.float32),
        }
    
    
    def _load_measurements(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process temporal measurement sequences for a given index."""
        zarr_idx = idx
        
        # --- Schema Definition for Feature Loading ---
        # Format: (output_list, group_name, feature_name, aggregation_func)
        # default aggregation is scalar mean unless specified otherwise
        
        rt_features = []
        phy_features = []
        mac_features = []
        
        rt_schema = RadioLocalizationDataset.RT_SCHEMA
        phy_schema = RadioLocalizationDataset.PHY_SCHEMA
        mac_schema = RadioLocalizationDataset.MAC_SCHEMA
        
        # --- Helper Functions ---
        def _load_scalar(group, key, default=0.0):
            if key not in group: return None
            val = group[key][zarr_idx]
            if hasattr(val, '__len__'):
                return float(np.mean(val)) if val.size > 0 else default
            return float(val)

        def _load_rt_feature(group, key):
            if key not in group: return None
            val = group[key][zarr_idx]
            # Handle potential multi-dim arrays by taking mean absolute/value
            if key == 'path_gains': return np.mean(np.abs(val))
            if hasattr(val, '__len__'): return np.mean(val)
            return val

        # --- Load RT Features ---
        if 'rt' in self.store:
            group = self.store['rt']
            for name, key in rt_schema:
                val = _load_rt_feature(group, key)
                rt_features.append(float(val) if val is not None else 0.0)
        else:
             rt_features = [0.0] * len(rt_schema)
                    
        rt_data = {}
        if rt_features:
            rt_data['features'] = torch.tensor(
                np.array(rt_features, dtype=np.float32),
                dtype=torch.float32
            ).unsqueeze(0)

        # --- Load PHY Features ---
        if 'phy_fapi' in self.store:
            group = self.store['phy_fapi']
            for name, key, default_val in phy_schema:
                val = _load_scalar(group, key, default=default_val)
                phy_features.append(val if val is not None else default_val)
        else:
             phy_features = [default_val for _, _, default_val in phy_schema]
        
        phy_data = {}
        if phy_features:
            phy_data['features'] = torch.tensor(
                np.array(phy_features, dtype=np.float32),
                dtype=torch.float32
            )
            
        # --- Load MAC Features ---
        if 'mac_rrc' in self.store:
            group = self.store['mac_rrc']
            for name, key, default_val in mac_schema:
                val = _load_scalar(group, key, default=default_val)
                mac_features.append(val if val is not None else default_val)
        else:
             mac_features = [default_val for _, _, default_val in mac_schema]
        
        mac_data = {}
        if mac_features:
            mac_data['features'] = torch.tensor(
                np.array(mac_features, dtype=np.float32),
                dtype=torch.float32
            )

        # Initialize ID/Time tensors
        cell_ids = torch.zeros(1, dtype=torch.long)
        beam_ids = torch.zeros(1, dtype=torch.long)
        timestamps = torch.zeros(1, dtype=torch.float32)
        
        if 'metadata' in self.store:
            meta = self.store['metadata']
            if 'cell_ids' in meta: cell_ids = torch.tensor(meta['cell_ids'][idx], dtype=torch.long)
            if 'beam_ids' in meta: beam_ids = torch.tensor(meta['beam_ids'][idx], dtype=torch.long)
            if 'timestamps' in meta: timestamps = torch.tensor(meta['timestamps'][idx], dtype=torch.float32)

        # Create mask
        seq_len = max(rt_data.get('features', torch.empty(0, 0)).shape[0], 1)
        mask = torch.ones(seq_len, dtype=torch.bool)
        
        # Handle missing (mask/zero/mean)
        if self.handle_missing == 'mask':
            if 'features' in rt_data: mask &= ~torch.isnan(rt_data['features']).any(dim=-1)
            if 'features' in phy_data: phy_data['features'] = torch.nan_to_num(phy_data['features'], nan=0.0)
            if 'features' in mac_data: mac_data['features'] = torch.nan_to_num(mac_data['features'], nan=0.0)
        elif self.handle_missing == 'zero':
             if 'features' in rt_data: rt_data['features'] = torch.nan_to_num(rt_data['features'], nan=0.0)
             if 'features' in phy_data: phy_data['features'] = torch.nan_to_num(phy_data['features'], nan=0.0)
             if 'features' in mac_data: mac_data['features'] = torch.nan_to_num(mac_data['features'], nan=0.0)

        # Normalize features if requested
        if self.normalize and self.norm_stats:
            rt_data = self._normalize_features(rt_data, 'rt')
            phy_data = self._normalize_features(phy_data, 'phy')
            mac_data = self._normalize_features(mac_data, 'mac')
        
        # --- Dynamic Dim Handling ---
        # Instead of fixed 16/8/6, we respect what was loaded, but ensure min padding if needed for model compat
        # Ideally, we should read this from config, but here we just return what we have.
        # The collate_fn handles batch padding.
        # BUT the model expects fixed input size? The FeatureProjection in model does.
        # So we MUST pad to at least the configured model dimension locally or ensure config matches.
        # For now, let's keep the padding logic but clean it up.
        
        # We'll use a dynamic target dim based on the schema length effectively?
        # Or better, just pad to the "known max" for safety, which is effectively the schema size.
        
        rt_target_dim = 16 # Accommodate all possible RT features
        rt_feat = rt_data.get('features', torch.zeros(seq_len, rt_target_dim))
        if rt_feat.shape[-1] < rt_target_dim: 
            rt_feat = torch.cat([rt_feat, torch.zeros(rt_feat.shape[0], rt_target_dim - rt_feat.shape[-1])], dim=-1)
        elif rt_feat.shape[-1] > rt_target_dim:
            rt_feat = rt_feat[..., :rt_target_dim]
            
        phy_feat = phy_data.get('features', torch.zeros(8))
        if len(phy_feat) < 8: phy_feat = torch.cat([phy_feat, torch.zeros(8 - len(phy_feat))], dim=0)
        elif len(phy_feat) > 8: phy_feat = phy_feat[:8]
            
        mac_feat = mac_data.get('features', torch.zeros(6))
        if len(mac_feat) < 6: mac_feat = torch.cat([mac_feat, torch.zeros(6 - len(mac_feat))], dim=0)
        elif len(mac_feat) > 6: mac_feat = mac_feat[:6]

        return {
            'rt_features': rt_feat,
            'phy_features': phy_feat,
            'mac_features': mac_feat,
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
            # Channels: rsrp, rsrq, sinr, cqi, throughput
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
        
        # Pad or slice to ensure 5 channels
        current_channels = radio_map.shape[0]
        if current_channels < 5:
            padding = torch.zeros(5 - current_channels, radio_map.shape[1], radio_map.shape[2], dtype=radio_map.dtype)
            radio_map = torch.cat([radio_map, padding], dim=0)
        
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
        padded_batch.setdefault('sample_extent', []).append(sample.get('sample_extent', torch.tensor(512.0)))
    
    # Stack all tensors
    return {
        'measurements': {
            k: torch.stack(v) for k, v in padded_batch['measurements'].items()
        },
        'radio_map': torch.stack(padded_batch['radio_map']),
        'osm_map': torch.stack(padded_batch['osm_map']),
        'position': torch.stack(padded_batch['position']),
        'cell_grid': torch.stack(padded_batch['cell_grid']),
        'sample_extent': torch.stack(padded_batch['sample_extent']),
    }
