"""
Data stacking and aggregation utilities.
"""

import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

MAX_CELLS = 16  # Fixed size for array padding
MAX_TX = 8


class DataStacker:
    """
    Handles data aggregation and padding for multi-scene datasets.
    
    Responsibilities:
    - Stack feature batches from multiple simulations
    - Pad arrays to fixed dimensions for consistent storage
    - Handle variable-length features
    """
    
    def __init__(self, max_neighbors: int = 8, num_subcarriers: int = 64):
        """
        Initialize data stacker.
        
        Args:
            max_neighbors: Maximum number of neighbor cells to track
            num_subcarriers: CFR subcarrier count for padding/truncation
        """
        self.max_neighbors = max_neighbors
        self.num_subcarriers = num_subcarriers
        
    def stack_scene_data(self, all_data: Dict[str, List]) -> Dict[str, np.ndarray]:
        """
        Stack lists of feature batches into arrays.
        
        Args:
            all_data: Dictionary containing lists of feature batches
            
        Returns:
            Dictionary of stacked and padded arrays
        """
        stacked = {}
        
        # Positions and timestamps
        if len(all_data['positions']) > 0:
            stacked['positions'] = np.concatenate(all_data['positions'], axis=0)
            stacked['timestamps'] = np.concatenate(all_data['timestamps'], axis=0)
            if all_data.get('ue_ids'):
                stacked['ue_ids'] = np.concatenate(all_data['ue_ids'], axis=0)
            if all_data.get('t_steps'):
                stacked['t_steps'] = np.concatenate(all_data['t_steps'], axis=0)
        else:
            return {}

        # Process feature layers
        self._process_feature_list(all_data['rt'], 'rt', stacked)
        self._process_feature_list(all_data['phy_fapi'], 'phy_fapi', stacked)
        self._process_feature_list(all_data['mac_rrc'], 'mac_rrc', stacked)

        return stacked
    
    def _process_feature_list(
        self,
        feat_list: List[Dict],
        key_prefix: str,
        stacked: Dict
    ) -> None:
        """Process and stack a list of feature dictionaries."""
        if not feat_list:
            return
            
        keys = set()
        for entry in feat_list:
            keys.update(entry.keys())
        
        for key in keys:
            # Collect all batches for this key
            batches = [d.get(key) for d in feat_list if d.get(key) is not None]
            if not batches:
                continue
            
            try:
                # Apply padding per batch BEFORE concat
                processed_batches = []
                for b in batches:
                    val = self._pad_feature(key, b)
                    processed_batches.append(val)

                # Ensure all batches have the same shape (except axis 0)
                padded_batches = self._align_batch_shapes(processed_batches, key)
                
                # Concatenate
                concatenated = np.concatenate(padded_batches, axis=0)
                
                # Avoid double prefixing
                if key.startswith(f"{key_prefix}/"):
                    final_key = key
                else:
                    final_key = f"{key_prefix}/{key}"
                    
                stacked[final_key] = concatenated
                
            except ValueError as e:
                logger.warning(f"Failed to stack feature {key}: {e}")
                # Last resort: just take the first batch
                if processed_batches:
                    logger.warning(f"Using only first batch for {key}")
                    final_key = key if key.startswith(f"{key_prefix}/") else f"{key_prefix}/{key}"
                    stacked[final_key] = processed_batches[0]
    
    def _pad_feature(self, key: str, val: np.ndarray) -> np.ndarray:
        """Apply feature-specific padding."""
        # RT Features: [Batch, Sites]
        if key in ['rt/rms_delay_spread', 'rt/num_paths']:
            return self._pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=0)
        
        # PHY Features: [Batch, Sites, Beams]
        elif key in ['phy_fapi/l1_rsrp_beams', 'phy_fapi/best_beam_ids']:
            fill_val = -150.0 if 'rsrp' in key else -1
            return self._pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=fill_val)

        # PHY Features: [Batch, Rx, Cells]
        elif key in ['phy_fapi/rsrp', 'phy_fapi/rsrq', 'phy_fapi/sinr', 
                     'phy_fapi/cqi', 'phy_fapi/ri', 'phy_fapi/pmi']:
            fill_val = -150.0 if 'rsr' in key or 'sinr' in key else 0
            if val.ndim == 2:
                val = np.expand_dims(val, axis=1)
            # Pad last dimension (Cells)
            val = self._pad_cell_dim(val, target_size=MAX_CELLS, fill_value=fill_val)
            # Pad axis 1 (Rx/Sites)
            return self._pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=fill_val)

        # CFR Features: [Batch, Cells, Subcarriers]
        elif key in ['phy_fapi/cfr_magnitude', 'phy_fapi/cfr_phase']:
            val = self._pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=0)
            return self._pad_cell_dim(val, target_size=self.num_subcarriers, fill_value=0)
        
        # MAC Features: [Batch, Rx]
        elif key in ['mac_rrc/serving_cell_id', 'mac_rrc/timing_advance', 
                     'mac_rrc/dl_throughput_mbps', 'mac_rrc/bler']:
            return self._pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=0)

        # MAC Features: [Batch, Rx, Neighbors]
        elif key == 'mac_rrc/neighbor_cell_ids':
            # Pad to max_neighbors (last)
            val = self._pad_cell_dim(val, target_size=self.max_neighbors, fill_value=-1)
            # Pad to MAX_CELLS (axis 1)
            return self._pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=-1)
        
        return val
    
    def _align_batch_shapes(
        self,
        batches: List[np.ndarray],
        key: str
    ) -> List[np.ndarray]:
        """Align shapes of all batches (except axis 0)."""
        if not batches:
            return batches
            
        ndim = batches[0].ndim
        max_shape = list(batches[0].shape)
        
        # Find maximum shape for each dimension
        for b in batches[1:]:
            for i in range(1, ndim):  # Skip axis 0 (batch)
                max_shape[i] = max(max_shape[i], b.shape[i])
        
        # Pad all batches to max_shape
        padded_batches = []
        for b in batches:
            if list(b.shape[1:]) != max_shape[1:]:
                # Need padding
                pad_width = [(0, 0)]  # No padding on axis 0
                for i in range(1, ndim):
                    pad_width.append((0, max_shape[i] - b.shape[i]))
                fill_val = -150.0 if ('rsr' in key or 'sinr' in key) else 0
                b_padded = np.pad(b, pad_width, mode='constant', constant_values=fill_val)
                padded_batches.append(b_padded)
            else:
                padded_batches.append(b)
        
        return padded_batches
    
    def _pad_cell_dim(
        self,
        a: np.ndarray,
        target_size: int = 16,
        fill_value: float = 0
    ) -> np.ndarray:
        """Pad last dimension to target size."""
        if a is None:
            return a
        if a.shape[-1] >= target_size:
            return a[..., :target_size]
        
        padding = [(0, 0)] * (a.ndim - 1) + [(0, target_size - a.shape[-1])]
        return np.pad(a, padding, mode='constant', constant_values=fill_value)

    def _pad_tx_dim(
        self,
        a: np.ndarray,
        target_size: int = 8,
        fill_value: float = 0,
        axis: int = 1
    ) -> np.ndarray:
        """Pad specified axis to target size."""
        if a is None:
            return a
        if axis >= a.ndim:
            return a
        
        if a.shape[axis] >= target_size:
            slices = [slice(None)] * a.ndim
            slices[axis] = slice(0, target_size)
            return a[tuple(slices)]
        
        padding = [(0, 0)] * a.ndim
        padding[axis] = (0, target_size - a.shape[axis])
        return np.pad(a, padding, mode='constant', constant_values=fill_value)
