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
from collections import OrderedDict
import logging

from src.config.feature_schema import RTFeatureIndex, PHYFeatureIndex, MACFeatureIndex

logger = logging.getLogger(__name__)


class LMDBRadioLocalizationDataset(torch.utils.data.Dataset):
    """LMDB-based dataset for radio localization with perfect multiprocessing support.
    
    Key differences from legacy Zarr version:
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
        split_seed: int = 42,
        map_cache_size: int = 0,
        sequence_length: int = 0,
        max_cells: int = 2,
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
            split_seed: Seed for deterministic split shuffling (per instance)
            map_cache_size: Max scenes to cache (0 disables caching)
            sequence_length: Number of time reports per UE trajectory (0 disables sequences)
            max_cells: Maximum cells per report
        """
        self.lmdb_path = Path(lmdb_path)
        self.split = split
        self.map_resolution = map_resolution
        self.scene_extent = scene_extent
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.osm_channels = osm_channels
        self._split_seed = split_seed
        self._split_rng = np.random.default_rng(split_seed)
        self._map_cache_size = map_cache_size
        self._map_cache = OrderedDict()
        self.sequence_length = sequence_length
        self.max_cells = max_cells
        
        # Critical: Don't open LMDB in __init__!
        # Store only the path - environment opened lazily per worker
        self._env = None
        self._metadata = None
        self._indices = None
        self._sequence_indices = None
        
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

        if not self.sequence_length:
            seq_len = self._metadata.get('sequence_length')
            if seq_len:
                self.sequence_length = int(seq_len)
        if not self.max_cells:
            max_cells = self._metadata.get('max_dimensions', {}).get('max_cells')
            if max_cells:
                self.max_cells = int(max_cells)
        
        # Get split indices (support both 'n_samples' and 'num_samples' for backward compatibility)
        n_samples = self._metadata.get('n_samples', self._metadata.get('num_samples', 0))
        if n_samples == 0:
            raise ValueError(f"No samples found in LMDB metadata: {self.lmdb_path}")
        self._indices = self._get_split_indices(n_samples)

        if self.sequence_length and self.sequence_length > 1:
            self._sequence_indices = self._build_sequence_indices(n_samples)
        
        # Build normalization stats in PyTorch format if needed
        if self.normalize:
            self._build_norm_stats()
        
        logger.debug(f"LMDB initialized in process {os.getpid()}: {len(self._indices)} samples")
    
    def _get_split_indices(self, n_samples: int) -> np.ndarray:
        """Get indices for train/val/test split."""
        indices = np.arange(n_samples)
        
        if self.split == 'all':
            return indices

        split_indices = None
        if self._metadata:
            split_indices = self._metadata.get('split_indices')
        if split_indices and self.split in split_indices and split_indices[self.split] is not None:
            return np.array(split_indices[self.split], dtype=np.int64)
        if self._metadata and self._metadata.get('split_name') == self.split:
            return indices
        
        # Default split: 70% train, 15% val, 15% test (reproducible)
        indices = self._split_rng.permutation(indices)
        
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

    def _build_sequence_indices(self, n_samples: int) -> List[List[int]]:
        """Build sequence index lists for trajectory-based samples."""
        sequence_slices = self._metadata.get('sequence_slices') if self._metadata else None
        split_sequence_indices = self._metadata.get('split_sequence_indices') if self._metadata else None

        if sequence_slices:
            if self.split != 'all' and split_sequence_indices and self.split in split_sequence_indices:
                seq_ids = split_sequence_indices[self.split]
            else:
                seq_ids = list(range(len(sequence_slices)))
            sequences = []
            for seq_id in seq_ids:
                start, length = sequence_slices[seq_id]
                if self.sequence_length and length != self.sequence_length:
                    continue
                sequences.append(list(range(start, start + length)))
            return sequences

        if not self.sequence_length or self.sequence_length <= 1:
            return [self._indices.tolist()]

        total_sequences = n_samples // self.sequence_length
        split_set = None
        if self.split != 'all':
            split_set = set(self._indices.tolist())

        sequences = []
        for seq_id in range(total_sequences):
            start = seq_id * self.sequence_length
            seq_indices = list(range(start, start + self.sequence_length))
            if split_set is not None:
                if not all(idx in split_set for idx in seq_indices):
                    continue
            sequences.append(seq_indices)

        if split_set is not None and not sequences:
            logger.warning(
                "Sequence mode enabled but no full sequences found for split. "
                "Check LMDB split indices or regenerate with sequence-aware splits."
            )
        return sequences
    
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
        if self._sequence_indices is not None:
            return len(self._sequence_indices)
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
        
        if self._sequence_indices is None:
            sample = self._load_sample_by_index(self._indices[idx])
            return self._build_sample(sample)

        sequence_indices = self._sequence_indices[idx]
        with self._env.begin() as txn:
            samples = [self._load_sample_by_index(sample_idx, txn=txn) for sample_idx in sequence_indices]
        return self._build_sequence(samples)

    def _load_sample_by_index(self, sample_idx: int, txn: Optional[lmdb.Transaction] = None) -> dict:
        key_8 = f'sample_{sample_idx:08d}'.encode()
        key_6 = f'sample_{sample_idx:06d}'.encode()
        owns_txn = False
        if txn is None:
            txn = self._env.begin()
            owns_txn = True
        try:
            value = txn.get(key_8)
            if value is None:
                value = txn.get(key_6)
            if value is None:
                raise KeyError(f"Sample {sample_idx} not found in LMDB (tried both 6 and 8-digit keys)")
            return pickle.loads(value)
        finally:
            if owns_txn:
                txn.abort()

    def _build_sample(self, sample: dict) -> Dict[str, torch.Tensor]:
        measurements = self._process_measurements(sample, timestamp=sample.get('timestamp'))
        aux_targets = self._extract_aux_from_raw(sample)
        scene_id = sample.get('scene_id', '')
        radio_map, osm_map = self._get_scene_maps(scene_id)
        position, scene_extent = self._extract_position(sample)
        cell_grid = self._compute_grid_cell(position, 1.0)
        scene_idx = sample.get('scene_idx', None)
        if scene_idx is None:
            scene_idx = self._metadata.get('scene_id_to_idx', {}).get(scene_id, -1)
        scene_idx = int(scene_idx) if scene_idx is not None else -1

        return {
            'measurements': measurements,
            'radio_map': radio_map,
            'osm_map': osm_map,
            'position': position,
            'cell_grid': cell_grid,
            'scene_extent': scene_extent,
            'scene_idx': torch.tensor(scene_idx, dtype=torch.long),
            'aux_targets': aux_targets,
        }

    def _build_sequence(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Empty sequence in LMDB dataset")

        scene_id = samples[0].get('scene_id', '')
        radio_map, osm_map = self._get_scene_maps(scene_id)

        per_report = [self._process_measurements(s, timestamp=s.get('timestamp')) for s in samples]
        aux_reports = [self._extract_aux_from_raw(s) for s in samples]
        rt_seq = torch.stack([m['rt_features'] for m in per_report], dim=0)
        phy_seq = torch.stack([m['phy_features'] for m in per_report], dim=0)
        mac_seq = torch.stack([m['mac_features'] for m in per_report], dim=0)
        cell_ids_seq = torch.stack([m['cell_ids'] for m in per_report], dim=0)
        beam_ids_seq = torch.stack([m['beam_ids'] for m in per_report], dim=0)
        timestamps_seq = torch.stack([m['timestamps'] for m in per_report], dim=0)
        mask_seq = torch.stack([m['mask'] for m in per_report], dim=0)

        seq_len, max_cells = cell_ids_seq.shape
        measurements = {
            'rt_features': rt_seq.reshape(seq_len * max_cells, -1),
            'phy_features': phy_seq.reshape(seq_len * max_cells, -1),
            'mac_features': mac_seq.reshape(seq_len * max_cells, -1),
            'cell_ids': cell_ids_seq.reshape(seq_len * max_cells),
            'beam_ids': beam_ids_seq.reshape(seq_len * max_cells),
            'timestamps': timestamps_seq.reshape(seq_len * max_cells),
            'mask': mask_seq.reshape(seq_len * max_cells),
        }

        position, scene_extent = self._extract_position(samples[-1])
        cell_grid = self._compute_grid_cell(position, 1.0)
        scene_idx = samples[-1].get('scene_idx', None)
        if scene_idx is None:
            scene_idx = self._metadata.get('scene_id_to_idx', {}).get(scene_id, -1)
        scene_idx = int(scene_idx) if scene_idx is not None else -1

        aux_targets = {}
        if aux_reports:
            for key in aux_reports[0].keys():
                vals = [r[key].item() if isinstance(r[key], torch.Tensor) else float(r[key]) for r in aux_reports]
                aux_targets[key] = torch.tensor(np.mean(vals), dtype=torch.float32)

        return {
            'measurements': measurements,
            'radio_map': radio_map,
            'osm_map': osm_map,
            'position': position,
            'cell_grid': cell_grid,
            'scene_extent': scene_extent,
            'scene_idx': torch.tensor(scene_idx, dtype=torch.long),
            'aux_targets': aux_targets,
        }

    def _extract_position(self, sample: dict) -> Tuple[torch.Tensor, float]:
        pos = sample['position']
        if isinstance(pos, (tuple, list)):
            position = torch.tensor(pos[:2], dtype=torch.float32)
        else:
            pos_array = np.asarray(pos).flatten()
            position = torch.from_numpy(pos_array[:2]).float()

        scene_metadata = sample.get('scene_metadata', {})
        bbox = scene_metadata.get('bbox', {})
        if 'x_max' in bbox and 'x_min' in bbox:
            actual_width = bbox['x_max'] - bbox['x_min']
            actual_height = bbox['y_max'] - bbox['y_min']
            scene_extent = max(actual_width, actual_height)
        else:
            scene_extent = sample.get('scene_extent', self.scene_extent)

        position = position / scene_extent
        position = torch.clamp(position, 0.0, 1.0)
        return position, scene_extent

    def _extract_aux_from_raw(self, sample: dict) -> Dict[str, torch.Tensor]:
        rt_data = sample.get('rt_features', sample.get('rt', {})) or {}
        mac_data = sample.get('mac_features', sample.get('mac', {})) or {}

        def _mean_first(arr) -> float:
            arr = np.asarray(arr).astype(np.float32).flatten()
            if arr.size == 0:
                return 0.0
            return float(np.mean(arr[:self.max_cells]))

        nlos = _mean_first(rt_data.get('is_nlos', []))
        num_paths = _mean_first(rt_data.get('num_paths', []))
        timing_advance = _mean_first(mac_data.get('timing_advance', []))
        toa = _mean_first(rt_data.get('toa', []))
        ta_unit = 16.0 / (15000.0 * 4096.0)
        ta_residual = timing_advance - (2.0 * toa / ta_unit) if toa else 0.0

        return {
            'nlos': torch.tensor(nlos, dtype=torch.float32),
            'num_paths': torch.tensor(num_paths, dtype=torch.float32),
            'timing_advance': torch.tensor(timing_advance, dtype=torch.float32),
            'ta_residual': torch.tensor(ta_residual, dtype=torch.float32),
        }
    
    def _process_measurements(self, sample: dict, timestamp: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Process RT, PHY, MAC features into model input format."""
        max_cells = self.max_cells
        
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
        n_actual_cells = sample.get('actual_num_cells', rt_features.shape[0] if rt_features.sum() > 0 else 2)
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
        if timestamp is None:
            timestamp = sample.get('timestamp', 0.0)
        timestamps = torch.full((max_cells,), float(timestamp), dtype=torch.float32)
        
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
        """Build RT feature tensor [max_cells, 16].
        
        Feature mapping (aligned with RT_SCHEMA):
            0: toa - Time of Arrival (seconds)
            1: mean_path_gain - Mean path gain (linear)
            2: max_path_gain - Max path gain (linear)
            3: mean_path_delay - Mean path delay (seconds)
            4: num_paths - Number of multipath components
            5: rms_delay_spread - RMS delay spread (seconds)
            6: rms_angular_spread - RMS angular spread (radians)
            7: total_power - Sum of squared path gains (linear power)
            8: n_significant_paths - Paths with gain > 1% of max
            9: delay_range - Max delay - min delay (seconds)
            10: dominant_path_gain - Gain of strongest path
            11: dominant_path_delay - Delay of strongest path
            12: is_nlos - Is non-line-of-sight (0 or 1)
            13-15: Reserved
        """
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
        
        # Feature 0: ToA
        if toa.ndim > 1:
            features[:n_cells, RTFeatureIndex.TOA] = torch.from_numpy(
                toa[:n_cells, 0].astype(np.float32)
            )
        elif toa.ndim == 1:
            features[:n_cells, RTFeatureIndex.TOA] = torch.from_numpy(
                toa[:n_cells].astype(np.float32)
            )
        
        # Process path_gains for features 1, 2, 7, 8, 10
        path_gains = None
        if 'path_gains' in rt_data and rt_data['path_gains'] is not None:
            path_gains = np.asarray(rt_data['path_gains'])
            if path_gains.size > 0:
                if path_gains.ndim >= 2:
                    n = min(path_gains.shape[0], n_cells)
                    # Feature 1: mean path gain
                    features[:n, RTFeatureIndex.MEAN_PATH_GAIN] = torch.from_numpy(
                        np.mean(path_gains[:n], axis=-1).astype(np.float32)
                    )
                    # Feature 2: max path gain
                    max_gains = np.max(np.abs(path_gains[:n]), axis=-1)
                    features[:n, RTFeatureIndex.MAX_PATH_GAIN] = torch.from_numpy(
                        max_gains.astype(np.float32)
                    )
                    # Feature 7: total power (sum of squared gains)
                    total_power = np.sum(path_gains[:n] ** 2, axis=-1)
                    features[:n, RTFeatureIndex.TOTAL_POWER] = torch.from_numpy(
                        total_power.astype(np.float32)
                    )
                    # Feature 8: number of significant paths (> 1% of max)
                    for i in range(n):
                        threshold = 0.01 * max_gains[i] if max_gains[i] > 0 else 0
                        n_sig = np.sum(np.abs(path_gains[i]) > threshold)
                        features[i, RTFeatureIndex.N_SIGNIFICANT_PATHS] = float(n_sig)
                    # Feature 10: dominant path gain (strongest path)
                    dom_idx = np.argmax(np.abs(path_gains[:n]), axis=-1)
                    for i in range(n):
                        features[i, RTFeatureIndex.DOMINANT_PATH_GAIN] = float(
                            np.abs(path_gains[i, dom_idx[i]])
                        )
                elif path_gains.ndim == 1:
                    features[0, RTFeatureIndex.MEAN_PATH_GAIN] = float(np.mean(path_gains))
                    features[0, RTFeatureIndex.MAX_PATH_GAIN] = float(np.max(np.abs(path_gains)))
                    features[0, RTFeatureIndex.TOTAL_POWER] = float(np.sum(path_gains ** 2))
                    max_g = np.max(np.abs(path_gains))
                    features[0, RTFeatureIndex.N_SIGNIFICANT_PATHS] = float(
                        np.sum(np.abs(path_gains) > 0.01 * max_g)
                    )
                    features[0, RTFeatureIndex.DOMINANT_PATH_GAIN] = float(max_g)
        
        # Process path_delays for features 3, 9, 11
        path_delays = None
        if 'path_delays' in rt_data and rt_data['path_delays'] is not None:
            path_delays = np.asarray(rt_data['path_delays'])
            if path_delays.size > 0:
                if path_delays.ndim >= 2:
                    n = min(path_delays.shape[0], n_cells)
                    # Feature 3: mean path delay
                    features[:n, RTFeatureIndex.MEAN_PATH_DELAY] = torch.from_numpy(
                        np.mean(path_delays[:n], axis=-1).astype(np.float32)
                    )
                    # Feature 9: delay range (max - min for non-zero delays)
                    for i in range(n):
                        nonzero_delays = path_delays[i][path_delays[i] > 0]
                        if len(nonzero_delays) > 1:
                            features[i, RTFeatureIndex.DELAY_RANGE] = float(
                                np.max(nonzero_delays) - np.min(nonzero_delays)
                            )
                    # Feature 11: dominant path delay (delay of strongest path)
                    if path_gains is not None and path_gains.ndim >= 2:
                        dom_idx = np.argmax(np.abs(path_gains[:n]), axis=-1)
                        for i in range(n):
                            features[i, RTFeatureIndex.DOMINANT_PATH_DELAY] = float(
                                path_delays[i, dom_idx[i]]
                            )
                elif path_delays.ndim == 1:
                    features[0, RTFeatureIndex.MEAN_PATH_DELAY] = float(np.mean(path_delays))
                    nonzero_d = path_delays[path_delays > 0]
                    if len(nonzero_d) > 1:
                        features[0, RTFeatureIndex.DELAY_RANGE] = float(
                            np.max(nonzero_d) - np.min(nonzero_d)
                        )
        
        # Feature 4: num_paths
        if 'num_paths' in rt_data and rt_data['num_paths'] is not None:
            num_paths = np.asarray(rt_data['num_paths'])
            if num_paths.size > 0:
                n = min(len(num_paths.flatten()), n_cells)
                features[:n, RTFeatureIndex.NUM_PATHS] = torch.from_numpy(
                    num_paths.flatten()[:n].astype(np.float32)
                )
        
        # Feature 5: rms_delay_spread
        if 'rms_delay_spread' in rt_data and rt_data['rms_delay_spread'] is not None:
            rms_ds = np.asarray(rt_data['rms_delay_spread'])
            if rms_ds.size > 0:
                n = min(len(rms_ds.flatten()), n_cells)
                features[:n, RTFeatureIndex.RMS_DELAY_SPREAD] = torch.from_numpy(
                    rms_ds.flatten()[:n].astype(np.float32)
                )
        
        # Feature 6: rms_angular_spread
        if 'rms_angular_spread' in rt_data and rt_data['rms_angular_spread'] is not None:
            rms_as = np.asarray(rt_data['rms_angular_spread'])
            if rms_as.size > 0:
                n = min(len(rms_as.flatten()), n_cells)
                features[:n, RTFeatureIndex.RMS_ANGULAR_SPREAD] = torch.from_numpy(
                    rms_as.flatten()[:n].astype(np.float32)
                )
        
        # Feature 12: is_nlos
        if 'is_nlos' in rt_data and rt_data['is_nlos'] is not None:
            is_nlos = np.asarray(rt_data['is_nlos'])
            if is_nlos.size > 0:
                n = min(len(is_nlos.flatten()), n_cells)
                features[:n, RTFeatureIndex.IS_NLOS] = torch.from_numpy(
                    is_nlos.flatten()[:n].astype(np.float32)
                )
        
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
            safe_extract(phy_data['rsrp'], PHYFeatureIndex.RSRP)
        if 'rsrq' in phy_data and phy_data['rsrq'] is not None:
            safe_extract(phy_data['rsrq'], PHYFeatureIndex.RSRQ)
        if 'sinr' in phy_data and phy_data['sinr'] is not None:
            safe_extract(phy_data['sinr'], PHYFeatureIndex.SINR)
        if 'cqi' in phy_data and phy_data['cqi'] is not None:
            safe_extract(phy_data['cqi'], PHYFeatureIndex.CQI)
        if 'ri' in phy_data and phy_data['ri'] is not None:
            safe_extract(phy_data['ri'], PHYFeatureIndex.RI)
        if 'pmi' in phy_data and phy_data['pmi'] is not None:
            safe_extract(phy_data['pmi'], PHYFeatureIndex.PMI)
        if 'l1_rsrp_beams' in phy_data and phy_data['l1_rsrp_beams'] is not None:
            l1_rsrp = np.asarray(phy_data['l1_rsrp_beams'])
            if l1_rsrp.size > 0:
                if l1_rsrp.ndim >= 2:
                    n = min(l1_rsrp.shape[0], n_cells)
                    features[:n, PHYFeatureIndex.L1_RSRP] = torch.from_numpy(
                        np.max(l1_rsrp[:n], axis=-1).astype(np.float32)
                    )
                elif l1_rsrp.ndim == 1:
                    features[0, PHYFeatureIndex.L1_RSRP] = float(np.max(l1_rsrp))
        if 'best_beam_ids' in phy_data and phy_data['best_beam_ids'] is not None:
            beam_ids = np.asarray(phy_data['best_beam_ids'])
            if beam_ids.size > 0:
                if beam_ids.ndim >= 2:
                    n = min(beam_ids.shape[0], n_cells)
                    features[:n, PHYFeatureIndex.BEST_BEAM_ID] = torch.from_numpy(
                        beam_ids[:n, 0].astype(np.float32)
                    )
                elif beam_ids.ndim == 1:
                    n = min(len(beam_ids), n_cells)
                    features[:n, PHYFeatureIndex.BEST_BEAM_ID] = torch.from_numpy(
                        beam_ids[:n].astype(np.float32)
                    )
        
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
            safe_extract_1d(mac_data['serving_cell_id'], MACFeatureIndex.SERVING_CELL_ID)
        if 'neighbor_cell_ids' in mac_data and mac_data['neighbor_cell_ids'] is not None:
            neighbors = np.asarray(mac_data['neighbor_cell_ids'])
            if neighbors.size > 0:
                if neighbors.ndim >= 2:
                    n = min(neighbors.shape[0], n_cells)
                    features[:n, MACFeatureIndex.NEIGHBOR_CELL_ID_1] = torch.from_numpy(
                        neighbors[:n, 0]
                    ).float()
                    if neighbors.shape[1] > 1:
                        features[:n, MACFeatureIndex.NEIGHBOR_CELL_ID_2] = torch.from_numpy(
                            neighbors[:n, 1]
                        ).float()
                elif neighbors.ndim == 1:
                    n = min(len(neighbors), n_cells)
                    features[:n, MACFeatureIndex.NEIGHBOR_CELL_ID_1] = torch.from_numpy(
                        neighbors[:n]
                    ).float()
        if 'timing_advance' in mac_data and mac_data['timing_advance'] is not None:
            safe_extract_1d(mac_data['timing_advance'], MACFeatureIndex.TIMING_ADVANCE)
        if 'dl_throughput_mbps' in mac_data and mac_data['dl_throughput_mbps'] is not None:
            safe_extract_1d(mac_data['dl_throughput_mbps'], MACFeatureIndex.DL_THROUGHPUT)
        if 'bler' in mac_data and mac_data['bler'] is not None:
            safe_extract_1d(mac_data['bler'], MACFeatureIndex.BLER)
        
        # Normalize
        if self.normalize and self.norm_stats and 'mac' in self.norm_stats:
            mean = self.norm_stats['mac']['mean']
            std = self.norm_stats['mac']['std']
            features[:n_cells, :len(mean)] = (features[:n_cells, :len(mean)] - mean) / std
        
        return features
    
    def _get_scene_maps(self, scene_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get radio and OSM maps for a scene with optional caching."""
        if scene_id in self._map_cache:
            radio_map, osm_map = self._map_cache[scene_id]
            self._map_cache.move_to_end(scene_id)
            return radio_map, osm_map

        radio_map = None
        osm_map = None

        # First try loading from cached metadata
        radio_maps = self._metadata.get('radio_maps', None)
        osm_maps = self._metadata.get('osm_maps', None)
        if radio_maps is not None and isinstance(scene_id, int):
            radio_map = torch.from_numpy(radio_maps[scene_id]).float()
        if osm_maps is not None and isinstance(scene_id, int):
            osm_map = torch.from_numpy(osm_maps[scene_id]).float()

        if radio_map is None or osm_map is None:
            # New format: load from scene-specific key
            scene_key = f'__scene_map_{scene_id}__'.encode()
            with self._env.begin() as txn:
                scene_data = txn.get(scene_key)
                if scene_data is not None:
                    scene_map = pickle.loads(scene_data)
                    if radio_map is None and scene_map.get('radio_map') is not None:
                        radio_map = torch.from_numpy(scene_map['radio_map']).float()
                    if osm_map is None and scene_map.get('osm_map') is not None:
                        osm_map = torch.from_numpy(scene_map['osm_map']).float()

        if radio_map is None:
            logger.warning(f"No radio map found for scene: {scene_id}")
            radio_map = torch.zeros(5, 256, 256)
        if osm_map is None:
            logger.warning(f"No OSM map found for scene: {scene_id}")
            osm_map = torch.zeros(5, 256, 256)

        if self.osm_channels is not None:
            osm_map = osm_map[self.osm_channels]

        if self._map_cache_size > 0:
            self._map_cache[scene_id] = (radio_map, osm_map)
            self._map_cache.move_to_end(scene_id)
            while len(self._map_cache) > self._map_cache_size:
                self._map_cache.popitem(last=False)

        return radio_map, osm_map

    def _get_radio_map(self, scene_id: str) -> torch.Tensor:
        """Get radio map for scene [5, 256, 256]."""
        return self._get_scene_maps(scene_id)[0]

    def _get_osm_map(self, scene_id: str) -> torch.Tensor:
        """Get OSM map for scene [5, 256, 256] or selected channels."""
        return self._get_scene_maps(scene_id)[1]
    
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
