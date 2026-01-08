"""
LMDB-based Radio Localization Dataset

Perfect multiprocessing support - no async event loop issues like Zarr.
Drop-in replacement for RadioLocalizationDataset.

Usage:
    dataset = LMDBRadioLocalizationDataset(
        lmdb_path='data/processed/dataset.lmdb',
        split='train',
        map_resolution=1.0,
        scene_extent=512,
    )
"""

import lmdb
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LMDBRadioLocalizationDataset(torch.utils.data.Dataset):
    """LMDB-based dataset for radio localization with perfect multiprocessing support.
    
    Key differences from Zarr version:
    - No async event loops → multiprocessing works perfectly
    - Lazy initialization → only stores path in __init__
    - Per-worker LMDB environment → opened after fork/spawn
    - Fast random access via memory-mapped files
    """
    
    # Schema definitions (same as RadioLocalizationDataset)
    RT_SCHEMA = [
        ('toa', 'toa'),
        ('path_gains', 'path_gains'),
        ('path_delays', 'path_delays'),
        ('num_paths', 'num_paths'),
        ('rms_delay_spread', 'rms_delay_spread'),
        ('rms_angular_spread', 'rms_angular_spread'),
        # Additional features computed from raw data
        ('mean_path_gain', None),
        ('max_path_gain', None),
        ('total_power', None),
        ('n_significant_paths', None),
        ('delay_range', None),
        ('dominant_path_gain', None),
        ('dominant_path_delay', None),
    ]
    
    PHY_SCHEMA = [
        ('rsrp', 'rsrp', 0.0),
        ('rsrq', 'rsrq', 0.0),
        ('sinr', 'sinr', 0.0),
        ('cqi', 'cqi', 0.0),
        ('ri', 'ri', 0.0),
        ('pmi', 'pmi', 0.0),
        ('l1_rsrp', 'l1_rsrp_beams', 0.0),
        ('best_beam_id', 'best_beam_ids', 0.0),
    ]
    
    MAC_SCHEMA = [
        ('serving_cell_id', 'serving_cell_id', 0.0),
        ('neighbor_cell_id_1', 'neighbor_cell_ids', 0.0),
        ('neighbor_cell_id_2', 'neighbor_cell_ids', 0.0),
        ('timing_advance', 'timing_advance', 0.0),
        ('dl_throughput', 'dl_throughput_mbps', 0.0),
        ('bler', 'bler', 0.0),
    ]
    
    def __init__(
        self,
        lmdb_path: str,
        split: str = 'train',
        map_resolution: float = 1.0,
        scene_extent: int = 512,
        normalize: bool = True,
        handle_missing: str = 'mask',
        augmentation: Optional[Dict] = None,
        osm_channels: Optional[List[int]] = None,
    ):
        """Initialize LMDB dataset.
        
        Args:
            lmdb_path: Path to LMDB database directory
            split: 'train', 'val', 'test', or 'all'
            map_resolution: Meters per pixel
            scene_extent: Scene size in meters
            normalize: Apply normalization
            handle_missing: How to handle missing values ('mask' or 'zero')
            augmentation: Augmentation config (applied on GPU in training loop)
            osm_channels: Which OSM channels to use (None = all 5)
        """
        self.lmdb_path = Path(lmdb_path)
        self.split = split
        self.map_resolution = map_resolution
        self.scene_extent = scene_extent
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.osm_channels = osm_channels
        
        # Critical: Don't open LMDB in __init__!
        # Store only the path - environment opened lazily per worker
        self._env = None
        self._metadata = None
        self._indices = None
        
        logger.info(f"LMDBRadioLocalizationDataset initialized for {split} split (lazy loading)")
    
    def _ensure_initialized(self):
        """Lazy initialization - opens LMDB environment and loads metadata.
        
        Called on first __len__ or __getitem__ access, which happens after
        DataLoader workers are forked/spawned.
        """
        if self._env is not None:
            return  # Already initialized
        
        # Open LMDB environment (read-only, no locking for multiprocessing)
        self._env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,  # Critical for multiprocessing
            readahead=False,  # Let OS handle prefetching
            meminit=False,  # Don't initialize new pages
        )
        
        # Load metadata
        with self._env.begin() as txn:
            metadata_bytes = txn.get(b'__metadata__')
            if metadata_bytes is None:
                raise ValueError(f"No metadata found in LMDB: {self.lmdb_path}")
            self._metadata = pickle.loads(metadata_bytes)
        
        # Get split indices (support both 'n_samples' and 'num_samples' for backward compatibility)
        n_samples = self._metadata.get('n_samples', self._metadata.get('num_samples', 0))
        if n_samples == 0:
            raise ValueError(f"No samples found in LMDB metadata: {self.lmdb_path}")
        self._indices = self._get_split_indices(n_samples)
        
        # Build normalization stats in PyTorch format if needed
        if self.normalize:
            self._build_norm_stats()
        
        logger.debug(f"LMDB initialized in process {os.getpid()}: {len(self._indices)} samples")
    
    def _get_split_indices(self, n_samples: int) -> np.ndarray:
        """Get indices for train/val/test split."""
        indices = np.arange(n_samples)
        
        if self.split == 'all':
            return indices
        
        # Default split: 70% train, 15% val, 15% test (reproducible)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if self.split == 'train':
            return indices[:n_train]
        elif self.split == 'val':
            return indices[n_train:n_train + n_val]
        elif self.split == 'test':
            return indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _build_norm_stats(self):
        """Convert metadata normalization stats to PyTorch tensors."""
        if 'normalization' not in self._metadata:
            self.norm_stats = None
            return
        
        norm_data = self._metadata['normalization']
        self.norm_stats = {}
        
        # RT stats
        if 'rt' in norm_data:
            rt_means, rt_stds = [], []
            for feat_name, _ in self.RT_SCHEMA[:6]:  # Only stored features
                if feat_name in norm_data['rt']:
                    rt_means.append(norm_data['rt'][feat_name]['mean'])
                    rt_stds.append(norm_data['rt'][feat_name]['std'])
                else:
                    rt_means.append(0.0)
                    rt_stds.append(1.0)
            
            self.norm_stats['rt'] = {
                'mean': torch.tensor(rt_means, dtype=torch.float32),
                'std': torch.clamp(torch.tensor(rt_stds, dtype=torch.float32), min=1e-8),
            }
        
        # PHY stats
        if 'phy' in norm_data:
            phy_means, phy_stds = [], []
            for feat_name, _, default in self.PHY_SCHEMA:
                phy_key = feat_name.replace('_', ' ').title().replace(' ', '')
                if phy_key in norm_data['phy']:
                    phy_means.append(norm_data['phy'][phy_key]['mean'])
                    phy_stds.append(norm_data['phy'][phy_key]['std'])
                else:
                    phy_means.append(0.0)
                    phy_stds.append(1.0)
            
            self.norm_stats['phy'] = {
                'mean': torch.tensor(phy_means, dtype=torch.float32),
                'std': torch.clamp(torch.tensor(phy_stds, dtype=torch.float32), min=1e-8),
            }
        
        # MAC stats
        if 'mac' in norm_data:
            mac_means, mac_stds = [], []
            for feat_name, _, default in self.MAC_SCHEMA:
                if feat_name in norm_data['mac']:
                    mac_means.append(norm_data['mac'][feat_name]['mean'])
                    mac_stds.append(norm_data['mac'][feat_name]['std'])
                else:
                    mac_means.append(0.0)
                    mac_stds.append(1.0)
            
            self.norm_stats['mac'] = {
                'mean': torch.tensor(mac_means, dtype=torch.float32),
                'std': torch.clamp(torch.tensor(mac_stds, dtype=torch.float32), min=1e-8),
            }
    
    def __len__(self) -> int:
        self._ensure_initialized()
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample from LMDB.
        
        Returns dictionary with:
        - measurements: Radio features (RT, PHY, MAC)
        - radio_map: Scene radio propagation map [5, H, W]
        - osm_map: Scene building/geometry map [5, H, W]
        - position: Ground truth UE position [2]
        - cell_grid: Coarse grid cell index for position
        """
        self._ensure_initialized()
        
        # Get actual sample index
        sample_idx = self._indices[idx]
        
        # Load from LMDB (support both 6-digit and 8-digit formats)
        key_8 = f'sample_{sample_idx:08d}'.encode()
        key_6 = f'sample_{sample_idx:06d}'.encode()
        with self._env.begin() as txn:
            value = txn.get(key_8)
            if value is None:
                value = txn.get(key_6)
            if value is None:
                raise KeyError(f"Sample {sample_idx} not found in LMDB (tried both 6 and 8-digit keys)")
            sample = pickle.loads(value)
        
        # Process measurements
        measurements = self._process_measurements(sample)
        
        # Get maps for this scene
        scene_idx = sample.get('scene_index', 0)
        radio_map = self._get_radio_map(scene_idx)
        osm_map = self._get_osm_map(scene_idx)
        
        # Ground truth position - handle tuple or numpy array
        # Only use x, y (first 2 elements), ignore z if present
        pos = sample['position']
        if isinstance(pos, (tuple, list)):
            position = torch.tensor(pos[:2], dtype=torch.float32)  # x, y only
        else:
            pos_array = np.asarray(pos).flatten()
            position = torch.from_numpy(pos_array[:2]).float()  # x, y only
        
        # Normalize position to [0, 1] to match model's normalized coordinate system
        scene_extent = sample.get('scene_extent', self.scene_extent)
        position = position / scene_extent  # Normalize to [0, 1]
        
        # Compute coarse grid cell (uses normalized position)
        cell_grid = self._compute_grid_cell(position, 1.0)  # Use normalized extent of 1.0
        
        return {
            'measurements': measurements,
            'radio_map': radio_map,
            'osm_map': osm_map,
            'position': position,
            'cell_grid': cell_grid,
            'scene_extent': scene_extent,  # Keep for denormalization later
        }
    
    def _process_measurements(self, sample: dict) -> Dict[str, torch.Tensor]:
        """Process RT, PHY, MAC features into model input format."""
        max_cells = 20  # Model expects up to 20 cells
        
        # Support both old key names ('rt', 'phy', 'mac') and new ones ('rt_features', 'phy_features', 'mac_features')
        rt_data = sample.get('rt_features', sample.get('rt', {}))
        phy_data = sample.get('phy_features', sample.get('phy', {}))
        mac_data = sample.get('mac_features', sample.get('mac', {}))
        
        # RT features (16 dims per cell)
        rt_features = self._build_rt_features(rt_data, max_cells)
        
        # PHY features (8 dims per cell)
        phy_features = self._build_phy_features(phy_data, max_cells)
        
        # MAC features (6 dims per cell)
        mac_features = self._build_mac_features(mac_data, max_cells)
        
        # Cell/beam IDs
        n_actual_cells = sample.get('actual_num_cells', rt_features.shape[0] if rt_features.sum() > 0 else 16)
        cell_ids = torch.zeros(max_cells, dtype=torch.long)
        cell_ids[:min(n_actual_cells, max_cells)] = torch.arange(min(n_actual_cells, max_cells))
        
        beam_ids = torch.zeros(max_cells, dtype=torch.long)
        if 'best_beam_ids' in phy_data and len(phy_data['best_beam_ids']) > 0:
            beam_data = phy_data['best_beam_ids']
            if beam_data.ndim > 1:
                beam_data = beam_data[:, 0]  # First beam
            n = min(len(beam_data), max_cells)
            beam_ids[:n] = torch.from_numpy(beam_data[:n].astype(np.int64)).long()
        
        # Timestamps (dummy for now)
        timestamps = torch.zeros(max_cells)
        
        # Mask for valid cells
        mask = torch.zeros(max_cells, dtype=torch.bool)
        mask[:min(n_actual_cells, max_cells)] = True
        
        return {
            'rt_features': rt_features,
            'phy_features': phy_features,
            'mac_features': mac_features,
            'cell_ids': cell_ids,
            'beam_ids': beam_ids,
            'timestamps': timestamps,
            'mask': mask,
        }
    
    def _build_rt_features(self, rt_data: dict, max_cells: int) -> torch.Tensor:
        """Build RT feature tensor [max_cells, 16]."""
        features = torch.zeros(max_cells, 16)
        
        if not rt_data:
            return features
        
        # Determine number of cells from data shape
        # LMDB stores per-sample data, so rt_data['toa'] is [n_cells] or [n_cells, ...]
        toa = rt_data.get('toa', None)
        if toa is None or (hasattr(toa, '__len__') and len(toa) == 0):
            return features
        
        toa = np.asarray(toa)
        n_cells = toa.shape[0] if toa.ndim >= 1 else 1
        n_cells = min(n_cells, max_cells)
        
        if n_cells == 0:
            return features
        
        # ToA feature
        if toa.ndim > 1:
            features[:n_cells, 0] = torch.from_numpy(toa[:n_cells, 0]).float()
        elif toa.ndim == 1:
            features[:n_cells, 0] = torch.from_numpy(toa[:n_cells]).float()
        
        # Path gains/delays aggregation
        if 'path_gains' in rt_data and rt_data['path_gains'] is not None:
            path_gains = np.asarray(rt_data['path_gains'])
            if path_gains.size > 0:
                if path_gains.ndim >= 2:
                    n = min(path_gains.shape[0], n_cells)
                    features[:n, 1] = torch.from_numpy(np.mean(path_gains[:n], axis=-1)).float()
                    features[:n, 2] = torch.from_numpy(np.max(path_gains[:n], axis=-1)).float()
                elif path_gains.ndim == 1:
                    features[0, 1] = float(np.mean(path_gains))
                    features[0, 2] = float(np.max(path_gains))
        
        if 'path_delays' in rt_data and rt_data['path_delays'] is not None:
            path_delays = np.asarray(rt_data['path_delays'])
            if path_delays.size > 0:
                if path_delays.ndim >= 2:
                    n = min(path_delays.shape[0], n_cells)
                    features[:n, 3] = torch.from_numpy(np.mean(path_delays[:n], axis=-1)).float()
                elif path_delays.ndim == 1:
                    features[0, 3] = float(np.mean(path_delays))
        
        if 'num_paths' in rt_data and rt_data['num_paths'] is not None:
            num_paths = np.asarray(rt_data['num_paths'])
            if num_paths.size > 0:
                n = min(len(num_paths.flatten()), n_cells)
                features[:n, 4] = torch.from_numpy(num_paths.flatten()[:n]).float()
        
        if 'rms_delay_spread' in rt_data and rt_data['rms_delay_spread'] is not None:
            rms_ds = np.asarray(rt_data['rms_delay_spread'])
            if rms_ds.size > 0:
                n = min(len(rms_ds.flatten()), n_cells)
                features[:n, 5] = torch.from_numpy(rms_ds.flatten()[:n]).float()
        
        if 'rms_angular_spread' in rt_data and rt_data['rms_angular_spread'] is not None:
            rms_as = np.asarray(rt_data['rms_angular_spread'])
            if rms_as.size > 0:
                n = min(len(rms_as.flatten()), n_cells)
                features[:n, 6] = torch.from_numpy(rms_as.flatten()[:n]).float()
        
        # Normalize if enabled
        if self.normalize and self.norm_stats and 'rt' in self.norm_stats:
            mean = self.norm_stats['rt']['mean']
            std = self.norm_stats['rt']['std']
            features[:n_cells, :len(mean)] = (features[:n_cells, :len(mean)] - mean) / std
        
        return features
    
    def _build_phy_features(self, phy_data: dict, max_cells: int) -> torch.Tensor:
        """Build PHY feature tensor [max_cells, 8]."""
        features = torch.zeros(max_cells, 8)
        
        if not phy_data:
            return features
        
        # Determine n_cells from any available feature
        rsrp = phy_data.get('rsrp', None)
        if rsrp is None or (hasattr(rsrp, '__len__') and len(rsrp) == 0):
            # Try other features
            for key in ['sinr', 'cqi', 'rsrq']:
                if key in phy_data and phy_data[key] is not None:
                    rsrp = phy_data[key]
                    break
        
        if rsrp is None:
            return features
        
        rsrp = np.asarray(rsrp)
        n_cells = rsrp.shape[0] if rsrp.ndim >= 1 else 1
        n_cells = min(n_cells, max_cells)
        
        if n_cells == 0:
            return features
        
        def safe_extract(data, feature_idx):
            """Extract feature values, padding/truncating to n_cells."""
            data = np.asarray(data)
            if data.size == 0:
                return
            if data.ndim == 0:
                features[:n_cells, feature_idx] = float(data)
            elif data.ndim == 1:
                n = min(len(data), n_cells)
                features[:n, feature_idx] = torch.from_numpy(data[:n].astype(np.float32))
            else:  # ndim >= 2
                n = min(data.shape[0], n_cells)
                features[:n, feature_idx] = torch.from_numpy(np.mean(data[:n], axis=-1).astype(np.float32))
        
        # Aggregate multi-dimensional features
        if 'rsrp' in phy_data and phy_data['rsrp'] is not None:
            safe_extract(phy_data['rsrp'], 0)
        if 'rsrq' in phy_data and phy_data['rsrq'] is not None:
            safe_extract(phy_data['rsrq'], 1)
        if 'sinr' in phy_data and phy_data['sinr'] is not None:
            safe_extract(phy_data['sinr'], 2)
        if 'cqi' in phy_data and phy_data['cqi'] is not None:
            safe_extract(phy_data['cqi'], 3)
        if 'ri' in phy_data and phy_data['ri'] is not None:
            safe_extract(phy_data['ri'], 4)
        if 'pmi' in phy_data and phy_data['pmi'] is not None:
            safe_extract(phy_data['pmi'], 5)
        if 'l1_rsrp_beams' in phy_data and phy_data['l1_rsrp_beams'] is not None:
            l1_rsrp = np.asarray(phy_data['l1_rsrp_beams'])
            if l1_rsrp.size > 0:
                if l1_rsrp.ndim >= 2:
                    n = min(l1_rsrp.shape[0], n_cells)
                    features[:n, 6] = torch.from_numpy(np.max(l1_rsrp[:n], axis=-1).astype(np.float32))
                elif l1_rsrp.ndim == 1:
                    features[0, 6] = float(np.max(l1_rsrp))
        if 'best_beam_ids' in phy_data and phy_data['best_beam_ids'] is not None:
            beam_ids = np.asarray(phy_data['best_beam_ids'])
            if beam_ids.size > 0:
                if beam_ids.ndim >= 2:
                    n = min(beam_ids.shape[0], n_cells)
                    features[:n, 7] = torch.from_numpy(beam_ids[:n, 0].astype(np.float32))
                elif beam_ids.ndim == 1:
                    n = min(len(beam_ids), n_cells)
                    features[:n, 7] = torch.from_numpy(beam_ids[:n].astype(np.float32))
        
        # Normalize
        if self.normalize and self.norm_stats and 'phy' in self.norm_stats:
            mean = self.norm_stats['phy']['mean']
            std = self.norm_stats['phy']['std']
            features[:n_cells] = (features[:n_cells] - mean) / std
        
        return features
    
    def _build_mac_features(self, mac_data: dict, max_cells: int) -> torch.Tensor:
        """Build MAC feature tensor [max_cells, 6]."""
        features = torch.zeros(max_cells, 6)
        
        if not mac_data:
            return features
        
        # Determine n_cells from any available feature
        serving_cell_id = mac_data.get('serving_cell_id', None)
        if serving_cell_id is None:
            for key in ['timing_advance', 'dl_throughput_mbps']:
                if key in mac_data and mac_data[key] is not None:
                    serving_cell_id = mac_data[key]
                    break
        
        if serving_cell_id is None:
            return features
        
        serving_cell_id = np.asarray(serving_cell_id)
        n_cells = serving_cell_id.shape[0] if serving_cell_id.ndim >= 1 else 1
        n_cells = min(n_cells, max_cells)
        
        if n_cells == 0:
            return features
        
        def safe_extract_1d(data, feature_idx: int):
            """Extract 1D data and place directly into features tensor."""
            data = np.asarray(data)
            if data.size == 0:
                return
            data = data.flatten()
            n = min(len(data), n_cells)
            features[:n, feature_idx] = torch.from_numpy(data[:n]).float()
        
        if 'serving_cell_id' in mac_data and mac_data['serving_cell_id'] is not None:
            safe_extract_1d(mac_data['serving_cell_id'], 0)
        if 'neighbor_cell_ids' in mac_data and mac_data['neighbor_cell_ids'] is not None:
            neighbors = np.asarray(mac_data['neighbor_cell_ids'])
            if neighbors.size > 0:
                if neighbors.ndim >= 2:
                    n = min(neighbors.shape[0], n_cells)
                    features[:n, 1] = torch.from_numpy(neighbors[:n, 0]).float()
                    if neighbors.shape[1] > 1:
                        features[:n, 2] = torch.from_numpy(neighbors[:n, 1]).float()
                elif neighbors.ndim == 1:
                    n = min(len(neighbors), n_cells)
                    features[:n, 1] = torch.from_numpy(neighbors[:n]).float()
        if 'timing_advance' in mac_data and mac_data['timing_advance'] is not None:
            safe_extract_1d(mac_data['timing_advance'], 3)
        if 'dl_throughput_mbps' in mac_data and mac_data['dl_throughput_mbps'] is not None:
            safe_extract_1d(mac_data['dl_throughput_mbps'], 4)
        
        # Normalize
        if self.normalize and self.norm_stats and 'mac' in self.norm_stats:
            mean = self.norm_stats['mac']['mean']
            std = self.norm_stats['mac']['std']
            features[:n_cells, :len(mean)] = (features[:n_cells, :len(mean)] - mean) / std
        
        return features
    
    def _get_radio_map(self, scene_idx: int) -> torch.Tensor:
        """Get radio map for scene [5, 256, 256]."""
        radio_maps = self._metadata.get('radio_maps', None)
        if radio_maps is None:
            return torch.zeros(5, 256, 256)
        
        radio_map = radio_maps[scene_idx]
        return torch.from_numpy(radio_map).float()
    
    def _get_osm_map(self, scene_idx: int) -> torch.Tensor:
        """Get OSM map for scene [5, 256, 256] or selected channels."""
        osm_maps = self._metadata.get('osm_maps', None)
        if osm_maps is None:
            return torch.zeros(5, 256, 256)
        
        osm_map = osm_maps[scene_idx]
        osm_tensor = torch.from_numpy(osm_map).float()
        
        # Channel selection
        if self.osm_channels is not None:
            osm_tensor = osm_tensor[self.osm_channels]
        
        return osm_tensor
    
    def _compute_grid_cell(self, position: torch.Tensor, scene_extent: float) -> torch.Tensor:
        """Compute 32x32 grid cell index for position."""
        grid_size = 32
        
        # Normalize to [0, 1]
        norm_pos = position / scene_extent
        
        # Clamp to valid range
        norm_pos = torch.clamp(norm_pos, 0.0, 0.999)
        
        # Convert to grid indices
        grid_x = int(norm_pos[0] * grid_size)
        grid_y = int(norm_pos[1] * grid_size)
        
        # Flatten to single index
        cell_idx = grid_y * grid_size + grid_x
        
        return torch.tensor(cell_idx, dtype=torch.long)


# Import for convenience
import os
