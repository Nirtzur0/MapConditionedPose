"""
LMDB-based Radio Localization Dataset

Perfect multiprocessing support with LMDB-backed storage.

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
    """LMDB-based dataset for radio localization with multiprocessing-friendly access.

    Uses lazy initialization per worker and memory-mapped random access.
    """
    
    # Schema definitions for LMDB dataset
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
        normalize_maps: bool = False,
        map_norm_mode: str = "zscore",
        map_log_throughput: bool = False,
        map_log_epsilon: float = 1e-3,
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
        self.normalize_maps = normalize_maps
        self.map_norm_mode = map_norm_mode
        self.map_log_throughput = map_log_throughput
        self.map_log_epsilon = map_log_epsilon
        
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
        
        # Get split indices
        n_samples = self._metadata.get('num_samples', 0)
        if not n_samples:
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
        norm_data = self._metadata.get('normalization')
        if not norm_data:
            norm_data = self._metadata.get('normalization_stats')
        if not norm_data:
            self.norm_stats = None
            return
        self.norm_stats = {}
        
        # RT stats
        if 'rt' in norm_data:
            rt_stats = norm_data['rt']
            rt_dim = len(RTFeatureIndex)
            rt_means = torch.zeros(rt_dim, dtype=torch.float32)
            rt_stds = torch.ones(rt_dim, dtype=torch.float32)

        def _apply_rt_stat(idx: RTFeatureIndex, key: str) -> None:
            if key in rt_stats:
                    mean_val = rt_stats[key].get('mean')
                    std_val = rt_stats[key].get('std')
                    if (
                        mean_val is None
                        or std_val is None
                        or not np.isfinite(mean_val)
                        or not np.isfinite(std_val)
                        or std_val == 0
                    ):
                        logger.warning(
                            "Non-finite normalization stats for rt/%s; using defaults.", key
                        )
                        return
                    rt_means[idx.value] = float(mean_val)
                    rt_stds[idx.value] = float(std_val)

            _apply_rt_stat(RTFeatureIndex.TOA, 'toa')
            _apply_rt_stat(RTFeatureIndex.RMS_DELAY_SPREAD, 'rms_delay_spread')
            _apply_rt_stat(RTFeatureIndex.RMS_ANGULAR_SPREAD, 'rms_angular_spread')
            _apply_rt_stat(RTFeatureIndex.DOPPLER_SPREAD, 'doppler_spread')
            _apply_rt_stat(RTFeatureIndex.COHERENCE_TIME, 'coherence_time')
            _apply_rt_stat(RTFeatureIndex.TOTAL_POWER, 'path_gains')
            _apply_rt_stat(RTFeatureIndex.N_SIGNIFICANT_PATHS, 'num_paths')
            _apply_rt_stat(RTFeatureIndex.IS_NLOS, 'is_nlos')

            self.norm_stats['rt'] = {
                'mean': rt_means,
                'std': torch.clamp(rt_stds, min=1e-8),
            }
        
        # PHY stats
        if 'phy' in norm_data:
            phy_means, phy_stds = [], []
            for feat_name, key, default in self.PHY_SCHEMA:
                if feat_name == 'l1_rsrp':
                    phy_means.append(0.0)
                    phy_stds.append(1.0)
                    continue
                lookup_key = key or feat_name
                mean_val = 0.0
                std_val = 1.0
                if lookup_key in norm_data['phy']:
                    mean_val = norm_data['phy'][lookup_key].get('mean', 0.0)
                    std_val = norm_data['phy'][lookup_key].get('std', 1.0)
                    if not np.isfinite(mean_val) or not np.isfinite(std_val) or std_val == 0:
                        logger.warning(
                            "Non-finite normalization stats for phy/%s; using defaults.", lookup_key
                        )
                        mean_val = 0.0
                        std_val = 1.0
                phy_means.append(mean_val)
                phy_stds.append(std_val)
            
            self.norm_stats['phy'] = {
                'mean': torch.tensor(phy_means, dtype=torch.float32),
                'std': torch.clamp(torch.tensor(phy_stds, dtype=torch.float32), min=1e-8),
            }
        
        # MAC stats
        if 'mac' in norm_data:
            mac_means, mac_stds = [], []
            for feat_name, key, default in self.MAC_SCHEMA:
                lookup_key = key or feat_name
                mean_val = 0.0
                std_val = 1.0
                if lookup_key in norm_data['mac']:
                    mean_val = norm_data['mac'][lookup_key].get('mean', 0.0)
                    std_val = norm_data['mac'][lookup_key].get('std', 1.0)
                    if not np.isfinite(mean_val) or not np.isfinite(std_val) or std_val == 0:
                        logger.warning(
                            "Non-finite normalization stats for mac/%s; using defaults.", lookup_key
                        )
                        mean_val = 0.0
                        std_val = 1.0
                mac_means.append(mean_val)
                mac_stds.append(std_val)
            
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
        position, scene_extent, scene_size = self._extract_position(sample)
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
            'scene_size': scene_size,
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

        position, scene_extent, scene_size = self._extract_position(samples[-1])
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
            'scene_size': scene_size,
            'scene_idx': torch.tensor(scene_idx, dtype=torch.long),
            'aux_targets': aux_targets,
        }

    def _extract_position(self, sample: dict) -> Tuple[torch.Tensor, float, torch.Tensor]:
        pos = sample['position']
        if isinstance(pos, (tuple, list)):
            position = torch.tensor(pos[:2], dtype=torch.float32)
        else:
            pos_array = np.asarray(pos).flatten()
            position = torch.from_numpy(pos_array[:2]).float()

        scene_metadata = sample.get('scene_metadata', {}) or {}
        bbox = scene_metadata.get('bbox') or {}
        if not bbox and 'scene_bbox' in sample:
            x_min, y_min, x_max, y_max = sample['scene_bbox']
            bbox = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}

        scene_size = None
        if all(k in bbox for k in ('x_min', 'y_min', 'x_max', 'y_max')):
            actual_width = float(bbox['x_max'] - bbox['x_min'])
            actual_height = float(bbox['y_max'] - bbox['y_min'])
            if actual_width > 0 and actual_height > 0:
                scene_size = torch.tensor([actual_width, actual_height], dtype=torch.float32)
                scene_extent = max(actual_width, actual_height)
            else:
                scene_extent = sample.get('scene_extent', self.scene_extent)
        else:
            scene_extent = sample.get('scene_extent', self.scene_extent)

        if scene_size is None:
            scene_size = torch.tensor([scene_extent, scene_extent], dtype=torch.float32)

        position = position / scene_size
        position = torch.clamp(position, 0.0, 1.0)
        return position, scene_extent, scene_size

    def _extract_aux_from_raw(self, sample: dict) -> Dict[str, torch.Tensor]:
        rt_data = sample.get('rt_features', {}) or {}
        mac_data = sample.get('mac_features', {}) or {}

        def _mean_valid(arr, min_val=None, max_val=None) -> float:
            arr = np.asarray(arr).astype(np.float32).flatten()
            if arr.size == 0:
                return 0.0
            arr = arr[:self.max_cells]
            mask = np.isfinite(arr)
            if min_val is not None:
                mask &= arr >= min_val
            if max_val is not None:
                mask &= arr <= max_val
            if not np.any(mask):
                return 0.0
            return float(np.mean(arr[mask]))

        nlos = _mean_valid(rt_data.get('is_nlos', []), min_val=0.0, max_val=1.0)
        num_paths = _mean_valid(rt_data.get('num_paths', []), min_val=0.0)
        timing_advance = _mean_valid(mac_data.get('timing_advance', []), min_val=0.0)

        toa_arr = np.asarray(rt_data.get('toa', []), dtype=np.float32).flatten()
        ta_arr = np.asarray(mac_data.get('timing_advance', []), dtype=np.float32).flatten()
        n = min(self.max_cells, toa_arr.size, ta_arr.size)
        ta_residual = 0.0
        if n > 0:
            toa_arr = toa_arr[:n]
            ta_arr = ta_arr[:n]
            valid = np.isfinite(toa_arr) & np.isfinite(ta_arr) & (toa_arr >= 0.0) & (ta_arr >= 0.0)
            if np.any(valid):
                ta_unit = 16.0 / (15000.0 * 4096.0)
                ta_residual = float(np.mean(ta_arr[valid] - (2.0 * toa_arr[valid] / ta_unit)))

        return {
            'nlos': torch.tensor(nlos, dtype=torch.float32),
            'num_paths': torch.tensor(num_paths, dtype=torch.float32),
            'timing_advance': torch.tensor(timing_advance, dtype=torch.float32),
            'ta_residual': torch.tensor(ta_residual, dtype=torch.float32),
        }
    
    def _process_measurements(self, sample: dict, timestamp: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Process RT, PHY, MAC features into model input format."""
        max_cells = self.max_cells
        
        rt_data = sample.get('rt_features', {})
        phy_data = sample.get('phy_features', {})
        mac_data = sample.get('mac_features', {})
        
        cell_id_lookup, n_cells_meta = self._build_cell_id_lookup(sample, max_cells)

        # RT features (16 dims per cell)
        rt_features = self._build_rt_features(rt_data, max_cells)
        
        # PHY features (8 dims per cell)
        phy_features = self._build_phy_features(phy_data, max_cells)
        
        # MAC features (6 dims per cell)
        n_actual_cells = n_cells_meta if n_cells_meta > 0 else sample.get(
            'actual_num_cells', rt_features.shape[0] if rt_features.sum() > 0 else 2
        )
        n_actual_cells = min(int(n_actual_cells), max_cells)
        mac_features = self._build_mac_features(mac_data, max_cells, cell_id_lookup=cell_id_lookup, n_cells=n_actual_cells)

        # Cell/beam IDs
        cell_ids = torch.zeros(max_cells, dtype=torch.long)
        cell_ids[:n_actual_cells] = torch.arange(n_actual_cells)
        
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

        # Apply missing-value handling
        if self.handle_missing in ("mask", "zero"):
            finite_mask = torch.isfinite(rt_features).all(dim=-1)
            finite_mask &= torch.isfinite(phy_features).all(dim=-1)
            finite_mask &= torch.isfinite(mac_features).all(dim=-1)
            if self.handle_missing == "mask":
                mask = mask & finite_mask

            rt_features = torch.nan_to_num(rt_features, nan=0.0, posinf=0.0, neginf=0.0)
            phy_features = torch.nan_to_num(phy_features, nan=0.0, posinf=0.0, neginf=0.0)
            mac_features = torch.nan_to_num(mac_features, nan=0.0, posinf=0.0, neginf=0.0)
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
            1: unused (mean_path_gain removed)
            2: unused (max_path_gain removed)
            3: unused (mean_path_delay removed)
            4: unused (num_paths removed)
            5: rms_delay_spread - RMS delay spread (seconds)
            6: rms_angular_spread - RMS angular spread (radians)
            7: total_power - Sum of squared path gains (linear power)
            8: n_significant_paths - Paths with gain > 1% of max
            9: unused (delay_range removed)
            10: unused (dominant_path_gain removed)
            11: unused (dominant_path_delay removed)
            12: is_nlos - Is non-line-of-sight (0 or 1)
            13: doppler_spread - Doppler spread (Hz)
            14: coherence_time - Coherence time (seconds)
            15: reserved
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
            toa = np.nan_to_num(toa, nan=0.0, posinf=0.0, neginf=0.0)
            toa = np.where(toa >= 0.0, toa, 0.0)
            features[:n_cells, RTFeatureIndex.TOA] = torch.from_numpy(
                toa[:n_cells, 0].astype(np.float32)
            )
        elif toa.ndim == 1:
            toa = np.nan_to_num(toa, nan=0.0, posinf=0.0, neginf=0.0)
            toa = np.where(toa >= 0.0, toa, 0.0)
            features[:n_cells, RTFeatureIndex.TOA] = torch.from_numpy(
                toa[:n_cells].astype(np.float32)
            )
        
        # Process path_gains for features 7, 8
        path_gains = None
        if 'path_gains' in rt_data and rt_data['path_gains'] is not None:
            path_gains = np.asarray(rt_data['path_gains'])
            path_gains = np.nan_to_num(path_gains, nan=0.0, posinf=0.0, neginf=0.0)
            if path_gains.size > 0:
                if path_gains.ndim >= 2:
                    n = min(path_gains.shape[0], n_cells)
                    # Feature 7: total power (sum of squared gains)
                    total_power = np.sum(path_gains[:n] ** 2, axis=-1)
                    features[:n, RTFeatureIndex.TOTAL_POWER] = torch.from_numpy(
                        total_power.astype(np.float32)
                    )
                    # Feature 8: number of significant paths (> 1% of max)
                    max_gains = np.max(np.abs(path_gains[:n]), axis=-1)
                    for i in range(n):
                        threshold = 0.01 * max_gains[i] if max_gains[i] > 0 else 0
                        n_sig = np.sum(np.abs(path_gains[i]) > threshold)
                        features[i, RTFeatureIndex.N_SIGNIFICANT_PATHS] = float(n_sig)
                elif path_gains.ndim == 1:
                    features[0, RTFeatureIndex.TOTAL_POWER] = float(np.sum(path_gains ** 2))
                    max_g = np.max(np.abs(path_gains))
                    features[0, RTFeatureIndex.N_SIGNIFICANT_PATHS] = float(
                        np.sum(np.abs(path_gains) > 0.01 * max_g)
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

        # Feature 13: doppler_spread
        if 'doppler_spread' in rt_data and rt_data['doppler_spread'] is not None:
            doppler_spread = np.asarray(rt_data['doppler_spread'])
            doppler_spread = np.nan_to_num(doppler_spread, nan=0.0, posinf=0.0, neginf=0.0)
            if doppler_spread.size > 0:
                n = min(len(doppler_spread.flatten()), n_cells)
                features[:n, RTFeatureIndex.DOPPLER_SPREAD] = torch.from_numpy(
                    doppler_spread.flatten()[:n].astype(np.float32)
                )

        # Feature 14: coherence_time
        if 'coherence_time' in rt_data and rt_data['coherence_time'] is not None:
            coherence_time = np.asarray(rt_data['coherence_time'])
            coherence_time = np.nan_to_num(coherence_time, nan=0.0, posinf=0.0, neginf=0.0)
            if coherence_time.size > 0:
                n = min(len(coherence_time.flatten()), n_cells)
                features[:n, RTFeatureIndex.COHERENCE_TIME] = torch.from_numpy(
                    coherence_time.flatten()[:n].astype(np.float32)
                )
        
        # Normalize if enabled
        if self.normalize and self.norm_stats and 'rt' in self.norm_stats:
            mean = self.norm_stats['rt']['mean']
            std = self.norm_stats['rt']['std']
            features[:n_cells, :len(mean)] = (features[:n_cells, :len(mean)] - mean) / std
        
        return features

    def _build_cell_id_lookup(self, sample: dict, max_cells: int) -> Tuple[Dict[int, int], int]:
        """Build lookup from physical cell_id to slot index using scene metadata."""
        scene_metadata = sample.get('scene_metadata', {}) or {}
        sites = scene_metadata.get('sites') or []
        valid_sites = [s for s in sites if s.get('cell_id') is not None]
        cell_ids = []
        for site in valid_sites:
            try:
                cell_ids.append(int(site.get('cell_id')))
            except (TypeError, ValueError):
                continue
        if max_cells > 0:
            cell_ids = cell_ids[:max_cells]
        lookup = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
        return lookup, len(cell_ids)

    def _build_phy_features(self, phy_data: dict, max_cells: int) -> torch.Tensor:
        """Build PHY feature tensor [max_cells, 8].

        Note: l1_rsrp is intentionally excluded from the feature vector.
        """
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
                # Prefer first RX dimension to preserve per-cell values.
                row = data[0]
                if row.ndim == 1:
                    n = min(len(row), n_cells)
                    features[:n, feature_idx] = torch.from_numpy(row[:n].astype(np.float32))
                else:
                    n = min(row.shape[0], n_cells)
                    features[:n, feature_idx] = torch.from_numpy(
                        np.mean(row[:n], axis=-1).astype(np.float32)
                    )
        
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
    
    def _build_mac_features(
        self,
        mac_data: dict,
        max_cells: int,
        cell_id_lookup: Optional[Dict[int, int]] = None,
        n_cells: Optional[int] = None,
    ) -> torch.Tensor:
        """Build MAC feature tensor [max_cells, 6]."""
        features = torch.zeros(max_cells, 6)
        
        if not mac_data:
            return features

        if n_cells is None:
            n_cells = max_cells
        n_cells = min(int(n_cells), max_cells)
        if n_cells <= 0:
            return features
        
        def safe_extract_1d(data, feature_idx: int):
            """Extract 1D data and place directly into features tensor."""
            data = np.asarray(data)
            if data.size == 0:
                return
            data = data.flatten()
            n = min(len(data), n_cells)
            features[:n, feature_idx] = torch.from_numpy(data[:n]).float()

        def _extract_scalar(val) -> Optional[float]:
            if val is None:
                return None
            arr = np.asarray(val).flatten()
            if arr.size == 0:
                return None
            return float(arr[0])

        serving_id = _extract_scalar(mac_data.get('serving_cell_id'))
        serving_slot = None
        if serving_id is not None and cell_id_lookup:
            serving_slot = cell_id_lookup.get(int(serving_id))

        if serving_slot is None:
            # Fall back to raw serving_cell_id data if it looks per-cell
            if mac_data.get('serving_cell_id') is not None:
                safe_extract_1d(mac_data['serving_cell_id'], MACFeatureIndex.SERVING_CELL_ID)
        else:
            features[serving_slot, MACFeatureIndex.SERVING_CELL_ID] = 1.0

        neighbors = mac_data.get('neighbor_cell_ids')
        if neighbors is not None and cell_id_lookup:
            neighbors = np.asarray(neighbors).flatten()
            neighbor_ids = []
            for cid in neighbors:
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    continue
                if cid_int not in neighbor_ids and cid_int in cell_id_lookup:
                    neighbor_ids.append(cid_int)
                if len(neighbor_ids) >= 2:
                    break
            if neighbor_ids:
                features[cell_id_lookup[neighbor_ids[0]], MACFeatureIndex.NEIGHBOR_CELL_ID_1] = 1.0
            if len(neighbor_ids) > 1:
                features[cell_id_lookup[neighbor_ids[1]], MACFeatureIndex.NEIGHBOR_CELL_ID_2] = 1.0
        elif neighbors is not None:
            safe_extract_1d(neighbors, MACFeatureIndex.NEIGHBOR_CELL_ID_1)

        def _apply_serving_scalar(key: str, feature_idx: int):
            val = mac_data.get(key)
            if val is None:
                return
            arr = np.asarray(val)
            if arr.ndim == 1 and arr.size >= n_cells:
                safe_extract_1d(arr, feature_idx)
                return
            scalar = _extract_scalar(arr)
            if scalar is None:
                return
            if serving_slot is not None:
                features[serving_slot, feature_idx] = float(scalar)
            else:
                safe_extract_1d(arr, feature_idx)

        _apply_serving_scalar('timing_advance', MACFeatureIndex.TIMING_ADVANCE)
        _apply_serving_scalar('dl_throughput_mbps', MACFeatureIndex.DL_THROUGHPUT)
        _apply_serving_scalar('bler', MACFeatureIndex.BLER)
        
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

        # Normalize radio map channels if requested.
        if self.normalize_maps:
            radio_map = self._normalize_radio_map(radio_map)

        if self.osm_channels is not None:
            osm_map = osm_map[self.osm_channels]

        if self._map_cache_size > 0:
            self._map_cache[scene_id] = (radio_map, osm_map)
            self._map_cache.move_to_end(scene_id)
            while len(self._map_cache) > self._map_cache_size:
                self._map_cache.popitem(last=False)

        return radio_map, osm_map

    def _normalize_radio_map(self, radio_map: torch.Tensor) -> torch.Tensor:
        """Per-channel normalization with optional log compression for throughput."""
        if radio_map.dim() != 3:
            return radio_map

        radio_map = radio_map.clone()

        # Log-compress throughput channel if present (channel 4).
        if self.map_log_throughput and radio_map.shape[0] >= 5:
            throughput = radio_map[4]
            throughput = torch.clamp(throughput, min=0.0)
            radio_map[4] = torch.log1p(throughput + self.map_log_epsilon)

        if self.map_norm_mode == "zscore":
            flat = radio_map.view(radio_map.shape[0], -1)
            mean = flat.mean(dim=1, keepdim=True)
            std = flat.std(dim=1, keepdim=True)
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            radio_map = (radio_map - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)

        return radio_map

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
        
        # Convert to grid indices (bottom-left origin)
        grid_x = int(norm_pos[0] * grid_size)
        grid_y = int(norm_pos[1] * grid_size)
        
        # Flatten to single index
        cell_idx = grid_y * grid_size + grid_x
        
        return torch.tensor(cell_idx, dtype=torch.long)


# Import for convenience
import os
