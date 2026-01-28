"""
LMDB Dataset Writer for Multi-Layer Features
Efficient key-value storage with perfect multiprocessing support
"""

import numpy as np
import lmdb
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class LMDBDatasetWriter:
    """
    Writes multi-layer features to LMDB dataset.
    
    Each sample is stored as a pickled dictionary with all data.
    Metadata is stored separately for efficient querying.
    
    Sample structure:
        {
            'rt_features': dict with path_gains, path_delays, etc.
            'phy_features': dict with rsrp, rsrq, sinr, etc.
            'mac_features': dict with cell_ids, timing_advance, etc.
            'position': (x, y, z)
            'timestamp': float
            'scene_id': str
            'ue_id': int
            'radio_map': array [C, H, W] or scene_idx
            'osm_map': array [C, H, W] or scene_idx
        }
    
    Metadata structure:
        {
            'num_samples': int
            'scene_ids': list of unique scene IDs
            'scene_maps': dict mapping scene_id -> {'radio_map': array, 'osm_map': array}
            'normalization_stats': dict with mean/std for each feature
            'split_indices': dict with train/val/test indices (if applicable)
            'max_dimensions': dict with max_cells, max_beams, max_paths
        }
    """
    
    def __init__(self, 
                 output_dir: Path,
                 map_size: int = 100 * 1024**3,  # 100GB default
                 split_name: Optional[str] = None,
                 sequence_length: Optional[int] = None):
        """
        Args:
            output_dir: Output directory for LMDB database
            map_size: Maximum database size in bytes (default 100GB)
            split_name: Name of split (train/val/test), None for single dataset
            sequence_length: Reports per UE trajectory (for sequence-aware splits)
        """
        self.output_dir = Path(output_dir)
        self.map_size = map_size
        self.split_name = split_name
        self.sequence_length = sequence_length
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LMDB
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db_name = f"dataset_{timestamp}"
        if split_name:
            self.db_name += f"_{split_name}"
        
        self.db_path = self.output_dir / f"{self.db_name}.lmdb"
        
        logger.info(f"Creating LMDB database at {self.db_path}")
        self.env = lmdb.open(
            str(self.db_path),
            map_size=map_size,
            max_dbs=2,  # Main DB + metadata DB
            meminit=False,
            map_async=True,
        )
        
        # Sample counter
        self.sample_count = 0
        
        # Track dimensions
        self.max_dimensions = {}
        
        # Track scene maps (store once per scene)
        self.scene_maps = {}
        self.scene_id_to_idx = {}

        # Optional split metadata
        self.split_ratios = None
        self.split_seed = 42
        self.kfold_config = None

        # Sequence tracking (optional, if ue_ids provided)
        self._sequence_ids = []
        self._sequence_slices = []
        self._current_sequence_id = None
        self._current_sequence_start = None
        self._trajectory_map = {}
        self._next_sequence_id = 0
        
        # Accumulate data for normalization stats
        self.feature_accumulators = {
            'rt': [],
            'phy': [],
            'mac': []
        }
        
        logger.info(f"LMDB Writer initialized: {self.db_path}")
    
    def set_max_dimensions(self, dimensions: Dict[str, int]):
        """Set maximum dimensions for arrays (max_cells, max_beams, max_paths)."""
        self.max_dimensions.update(dimensions)
        logger.info(f"Max dimensions set: {self.max_dimensions}")

    def set_split_ratios(self, split_ratios: Dict[str, float], split_seed: int = 42) -> None:
        """Set deterministic split ratios for a combined dataset."""
        if not split_ratios:
            return
        total = sum(split_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {total}")
        self.split_ratios = split_ratios
        self.split_seed = split_seed

    def set_kfold(self, num_folds: int, fold_index: int, shuffle: bool = True, seed: int = 42) -> None:
        """Set K-fold configuration for a combined dataset."""
        if num_folds < 2:
            raise ValueError(f"num_folds must be >= 2, got {num_folds}")
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")
        self.kfold_config = {
            "num_folds": num_folds,
            "fold_index": fold_index,
            "shuffle": shuffle,
            "seed": seed,
        }

    def _track_sequence(self, sequence_id: int) -> None:
        if self._current_sequence_id is None:
            self._current_sequence_id = sequence_id
            self._current_sequence_start = self.sample_count
            return
        if sequence_id != self._current_sequence_id:
            self._sequence_slices.append(
                (self._current_sequence_start, self.sample_count - self._current_sequence_start)
            )
            self._sequence_ids.append(self._current_sequence_id)
            self._current_sequence_id = sequence_id
            self._current_sequence_start = self.sample_count
    
    def write_scene_maps(self, scene_id: str, radio_map: np.ndarray, osm_map: np.ndarray):
        """Store maps for a scene (called once per scene)."""
        if scene_id not in self.scene_maps:
            scene_idx = len(self.scene_maps)
            self.scene_maps[scene_id] = {
                'radio_map': radio_map.astype(np.float32),
                'osm_map': osm_map.astype(np.float32),
                'scene_idx': scene_idx
            }
            self.scene_id_to_idx[scene_id] = scene_idx
            logger.debug(f"Stored maps for scene {scene_id} (index {scene_idx})")
    
    def append(self, scene_data: Dict[str, Any], scene_id: str, scene_metadata: Optional[Dict] = None):
        """
        Append all samples from one scene to the dataset.
        
        Args:
            scene_data: Dictionary with batched arrays for all samples in the scene.
                       Keys can be either:
                       - Nested: 'rt' (dict of arrays), 'phy_fapi' (dict), 'mac_rrc' (dict)
                       - Flat: 'rt/path_gains', 'phy_fapi/rsrp', 'mac_rrc/timing_advance', etc.
            scene_id: Scene identifier
            scene_metadata: Optional metadata about the scene
        """
        if not scene_data:
            logger.warning(f"Empty scene_data for {scene_id}, skipping")
            return
        
        # Register scene ID
        if scene_id not in self.scene_id_to_idx:
            self.scene_id_to_idx[scene_id] = len(self.scene_id_to_idx)
        scene_idx = self.scene_id_to_idx[scene_id]
        
        # Determine number of samples in this scene
        positions = scene_data.get('positions')
        if positions is None or len(positions) == 0:
            logger.warning(f"No position data for scene {scene_id}, skipping")
            return
        
        num_samples = len(positions)
        logger.debug(f"Appending {num_samples} samples from scene {scene_id}")
        
        # Convert flat keys (e.g., 'rt/path_gains') to nested dicts if needed
        rt_data = self._extract_layer_data(scene_data, 'rt')
        phy_data = self._extract_layer_data(scene_data, 'phy_fapi')
        mac_data = self._extract_layer_data(scene_data, 'mac_rrc')
        
        # Prepare scene metadata (add tx_local if possible)
        scene_metadata_local = scene_metadata
        if scene_metadata:
            scene_metadata_local = dict(scene_metadata)
            bbox = scene_metadata.get('bbox') or {}
            sites = scene_metadata.get('sites') or []
            if sites and all(k in bbox for k in ('x_min', 'y_min')):
                x_min = float(bbox['x_min'])
                y_min = float(bbox['y_min'])
                tx_local = []
                for site in sites:
                    pos = site.get('position')
                    if not pos or len(pos) < 2:
                        continue
                    local_pos = [
                        float(pos[0]) - x_min,
                        float(pos[1]) - y_min,
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    ]
                    tx_local.append({
                        'site_id': site.get('site_id'),
                        'site_type': site.get('site_type'),
                        'cell_id': site.get('cell_id'),
                        'sector_id': site.get('sector_id'),
                        'position': local_pos,
                    })
                if tx_local:
                    scene_metadata_local['tx_local'] = tx_local

        # Iterate through each sample in the scene
        for i in range(num_samples):
            sample = {
                'scene_id': scene_id,
                'scene_idx': scene_idx,
            }

            # Store per-sample scene extent for correct position normalization.
            if scene_metadata and 'bbox' in scene_metadata:
                bbox = scene_metadata['bbox']
                if all(k in bbox for k in ('x_min', 'y_min', 'x_max', 'y_max')):
                    width = float(bbox['x_max'] - bbox['x_min'])
                    height = float(bbox['y_max'] - bbox['y_min'])
                    sample['scene_extent'] = max(width, height)
                    sample['scene_bbox'] = (
                        float(bbox['x_min']),
                        float(bbox['y_min']),
                        float(bbox['x_max']),
                        float(bbox['y_max']),
                    )
            
            # Extract position for this sample
            pos = positions[i]  # shape: [3] - (x, y, z)
            sample['position'] = (float(pos[0]), float(pos[1]), float(pos[2]))
            
            # Extract timestamp for this sample
            timestamps = scene_data.get('timestamps')
            if timestamps is not None and len(timestamps) > i:
                sample['timestamp'] = float(timestamps[i])
            else:
                sample['timestamp'] = 0.0

            # Extract UE trajectory metadata
            ue_ids = scene_data.get('ue_ids')
            if ue_ids is not None and len(ue_ids) > i:
                ue_id = int(ue_ids[i])
                sample['ue_id'] = ue_id
                seq_key = (scene_id, ue_id)
                sequence_id = self._trajectory_map.get(seq_key)
                if sequence_id is None:
                    sequence_id = self._next_sequence_id
                    self._trajectory_map[seq_key] = sequence_id
                    self._next_sequence_id += 1
                sample['sequence_id'] = sequence_id
                self._track_sequence(sequence_id)

            t_steps = scene_data.get('t_steps')
            if t_steps is not None and len(t_steps) > i:
                sample['t_step'] = int(t_steps[i])
            
            # Extract RT features for this sample
            if rt_data:
                sample['rt_features'] = {}
                for key, value in rt_data.items():
                    if isinstance(value, np.ndarray) and len(value) > i:
                        sample['rt_features'][key] = value[i]
                    else:
                        sample['rt_features'][key] = np.array([], dtype=np.float32)
                self._accumulate_features('rt', sample['rt_features'])
            
            # Extract PHY/FAPI features for this sample
            if phy_data:
                sample['phy_features'] = {}
                for key, value in phy_data.items():
                    if isinstance(value, np.ndarray) and len(value) > i:
                        sample['phy_features'][key] = value[i]
                    else:
                        dtype = np.int32 if key in ['cqi', 'ri', 'pmi', 'best_beam_ids', 'serving_cell_id'] else np.float32
                        sample['phy_features'][key] = np.array([], dtype=dtype)
                self._accumulate_features('phy', sample['phy_features'])
            
            # Extract MAC/RRC features for this sample
            if mac_data:
                sample['mac_features'] = {}
                for key, value in mac_data.items():
                    if isinstance(value, np.ndarray) and len(value) > i:
                        sample['mac_features'][key] = value[i]
                    elif isinstance(value, (int, float)):
                        sample['mac_features'][key] = value
                    else:
                        sample['mac_features'][key] = -1 if key == 'serving_cell_id' else 0
                self._accumulate_features('mac', sample['mac_features'])
            
            # Add scene metadata if provided (same for all samples in scene)
            if scene_metadata_local:
                sample['scene_metadata'] = scene_metadata_local
            
            # Write to LMDB
            with self.env.begin(write=True) as txn:
                key = f'sample_{self.sample_count:08d}'.encode()
                value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key, value)
            
            self.sample_count += 1
            
            if self.sample_count % 1000 == 0:
                logger.info(f"Written {self.sample_count} samples")
    
    def _extract_layer_data(self, scene_data: Dict[str, Any], prefix: str) -> Dict[str, np.ndarray]:
        """
        Extract layer data handling both nested dict and flat key formats.
        
        Args:
            scene_data: The full scene data dictionary
            prefix: Layer prefix ('rt', 'phy_fapi', 'mac_rrc')
            
        Returns:
            Dictionary of feature arrays for this layer
        """
        # Check for nested dict first
        if prefix in scene_data and isinstance(scene_data[prefix], dict):
            return scene_data[prefix]
        
        # Check for flat keys (e.g., 'rt/path_gains')
        layer_data = {}
        prefix_with_slash = f"{prefix}/"
        for key, value in scene_data.items():
            if key.startswith(prefix_with_slash):
                feature_name = key[len(prefix_with_slash):]
                layer_data[feature_name] = value
        
        return layer_data if layer_data else None

    def _accumulate_features(self, layer: str, features: Dict):
        """Accumulate features for computing normalization statistics."""
        # Sample every 10th sample to save memory
        if self.sample_count % 10 == 0:
            self.feature_accumulators[layer].append(features.copy())
    
    def finalize(self) -> Path:
        """
        Finalize the dataset: compute stats, write metadata.
        
        Returns:
            Path to the LMDB database
        """
        logger.info(f"Finalizing dataset with {self.sample_count} samples")
        
        # Compute normalization statistics
        norm_stats = self._compute_normalization_stats()
        
        # Finalize sequence tracking
        if self._current_sequence_id is not None and self._current_sequence_start is not None:
            self._sequence_slices.append(
                (self._current_sequence_start, self.sample_count - self._current_sequence_start)
            )
            self._sequence_ids.append(self._current_sequence_id)

        # Prepare metadata
        split_indices = None
        split_sequence_indices = None
        if self.kfold_config:
            rng = np.random.default_rng(self.kfold_config["seed"])
            if self._sequence_slices:
                num_sequences = len(self._sequence_slices)
                seq_indices = np.arange(num_sequences)
                if self.kfold_config["shuffle"]:
                    rng.shuffle(seq_indices)
                folds = np.array_split(seq_indices, self.kfold_config["num_folds"])
                val_seq_ids = folds[self.kfold_config["fold_index"]]
                train_folds = [f for i, f in enumerate(folds) if i != self.kfold_config["fold_index"]]
                train_seq_ids = np.concatenate(train_folds) if train_folds else np.array([], dtype=int)
                split_sequence_indices = {
                    "train": train_seq_ids.tolist(),
                    "val": val_seq_ids.tolist(),
                }
                split_indices = {}
                for split_name, seq_ids in split_sequence_indices.items():
                    indices = []
                    for seq_id in seq_ids:
                        start, length = self._sequence_slices[seq_id]
                        indices.extend(range(start, start + length))
                    split_indices[split_name] = indices
            else:
                sample_indices = np.arange(self.sample_count)
                if self.kfold_config["shuffle"]:
                    rng.shuffle(sample_indices)
                folds = np.array_split(sample_indices, self.kfold_config["num_folds"])
                val_idx = folds[self.kfold_config["fold_index"]]
                train_folds = [f for i, f in enumerate(folds) if i != self.kfold_config["fold_index"]]
                train_idx = np.concatenate(train_folds) if train_folds else np.array([], dtype=int)
                split_indices = {
                    "train": train_idx.tolist(),
                    "val": val_idx.tolist(),
                }
        elif self.split_ratios:
            rng = np.random.default_rng(self.split_seed)
            if self._sequence_slices:
                num_sequences = len(self._sequence_slices)
                permuted = rng.permutation(num_sequences)
                train_end = int(num_sequences * self.split_ratios.get('train', 0.0))
                val_end = train_end + int(num_sequences * self.split_ratios.get('val', 0.0))
                split_sequence_indices = {
                    'train': permuted[:train_end].tolist(),
                    'val': permuted[train_end:val_end].tolist(),
                    'test': permuted[val_end:].tolist(),
                }
                split_indices = {}
                for split_name, seq_ids in split_sequence_indices.items():
                    indices = []
                    for seq_id in seq_ids:
                        start, length = self._sequence_slices[seq_id]
                        indices.extend(range(start, start + length))
                    split_indices[split_name] = indices
            else:
                permuted = rng.permutation(self.sample_count)
                train_end = int(self.sample_count * self.split_ratios.get('train', 0.0))
                val_end = train_end + int(self.sample_count * self.split_ratios.get('val', 0.0))
                split_indices = {
                    'train': permuted[:train_end].tolist(),
                    'val': permuted[train_end:val_end].tolist(),
                    'test': permuted[val_end:].tolist(),
                }
        elif self.split_name:
            split_indices = {self.split_name: list(range(self.sample_count))}

        metadata = {
            'num_samples': self.sample_count,
            'scene_ids': list(self.scene_maps.keys()),
            'scene_id_to_idx': self.scene_id_to_idx,
            'max_dimensions': self.max_dimensions,
            'normalization_stats': norm_stats,
            'created_at': datetime.now().isoformat(),
            'split_name': self.split_name,
            'split_indices': split_indices,
            'split_sequence_indices': split_sequence_indices,
            'split_seed': self.split_seed,
            'split_ratios': self.split_ratios,
            'kfold': self.kfold_config,
            'sequence_length': self.sequence_length,
            'sequence_slices': self._sequence_slices if self._sequence_slices else None,
        }
        
        # Write metadata to LMDB
        with self.env.begin(write=True) as txn:
            # Store metadata
            txn.put(b'__metadata__', pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))
            
            # Store scene maps
            for scene_id, scene_data in self.scene_maps.items():
                key = f'__scene_map_{scene_id}__'.encode()
                txn.put(key, pickle.dumps(scene_data, protocol=pickle.HIGHEST_PROTOCOL))
        
        # Sync to disk
        self.env.sync()
        self.env.close()
        
        logger.info(f"Dataset finalized: {self.db_path}")
        logger.info(f"Total samples: {self.sample_count}")
        logger.info(f"Total scenes: {len(self.scene_maps)}")
        logger.info(f"Database size: {self.db_path.stat().st_size / 1024**2:.1f} MB")
        
        return self.db_path
    
    def _compute_normalization_stats(self) -> Dict:
        """Compute mean and std for normalization."""
        stats = {}

        def _collect_finite(values: List[np.ndarray], val: Any) -> None:
            if isinstance(val, np.ndarray) and val.size > 0:
                arr = np.asarray(val)
                if np.iscomplexobj(arr):
                    arr = np.abs(arr)
                arr = arr.astype(np.float32, copy=False)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    values.append(arr)
            elif isinstance(val, (int, float)) and val != 0:
                if np.isfinite(val):
                    values.append(np.array([float(val)], dtype=np.float32))

        for layer, samples in self.feature_accumulators.items():
            if not samples:
                continue

            layer_stats = {}
            
            # Get all keys from first sample
            sample_keys = samples[0].keys()
            
            for key in sample_keys:
                # Collect all values for this key
                values = []
                for sample in samples:
                    val = sample[key]
                    _collect_finite(values, val)

                if values:
                    values = np.concatenate(values)
                    layer_stats[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }
            
            stats[layer] = layer_stats
        
        logger.info(f"Computed normalization stats for {len(stats)} layers")
        return stats
