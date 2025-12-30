"""
MAC/RRC Feature Extractor
Extracts Layer 3 (MAC/RRC) features from system state and measurements.
"""

import numpy as np
import logging
from typing import Optional, Any, Union

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .data_structures import PHYFAPILayerFeatures, MACRRCLayerFeatures
from ..measurement_utils import compute_timing_advance

logger = logging.getLogger(__name__)


class MACRRCFeatureExtractor:
    """
    Extracts Layer 3 (MAC/RRC) features from system state and measurements.
    """
    
    def __init__(self,
                 max_neighbors: int = 8,
                 enable_throughput: bool = True,
                 enable_handover: bool = False):
        self.max_neighbors = max_neighbors
        self.enable_throughput = enable_throughput
        self.enable_handover = enable_handover
        logger.info(f"MACRRCFeatureExtractor initialized: max_neighbors={max_neighbors}")
    
    def extract(self,
                phy_features: PHYFAPILayerFeatures,
                ue_positions: Union[np.ndarray, Any],
                site_positions: np.ndarray,
                cell_ids: np.ndarray) -> MACRRCLayerFeatures:
        
        # Logic often involves looking up which cell is strongest from PHY features.
        # This layer often acts as a bridge to application logic, so converting to numpy here might be acceptable if logic is complex.
        # But let's try to keep it tensor-compatible if possible.
        
        rsrp = phy_features.rsrp # [batch, rx, cells]
        num_cells = len(cell_ids)
        
        # Helper to align dimensions [Batch, Rx, Cells]
        def align_dims(tensor, n_cells, expected_batch_size=None):
            t_shape = tensor.shape
            
            # Helper to check if a dim matches size
            def dim_matches(dim_size, target):
                if target is None: return True
                return dim_size == target

            if len(t_shape) == 3:
                # Ambiguity: [Batch, Rx, Cells] vs [Batch, Cells, 1] vs [Cells, Batch, 1]
                # Prioritize checking Batch dimension if provided
                
                # Case 1: [Batch, Rx, Cells] (Target)
                if dim_matches(t_shape[0], expected_batch_size):
                    # Check if last dim matches cells
                    if t_shape[2] == n_cells:
                        return tensor
                    # Check if mid dim matches cells (and last doesn't)
                    if t_shape[1] == n_cells:
                         if TF_AVAILABLE and tf.is_tensor(tensor):
                             return tf.transpose(tensor, perm=[0, 2, 1])
                         return np.swapaxes(tensor, 1, 2)
                         
                # Case 2: [Cells, Batch, 1] -> Permute to [Batch, 1, Cells]
                if t_shape[0] == n_cells and dim_matches(t_shape[1], expected_batch_size):
                     if TF_AVAILABLE and tf.is_tensor(tensor):
                        return tf.transpose(tensor, perm=[1, 2, 0])
                     return np.transpose(tensor, (1, 2, 0))
                
                # Fallback to previous heuristics if batch size ambiguous or not provided
                if t_shape[2] == n_cells: return tensor
                if t_shape[1] == n_cells: 
                    if TF_AVAILABLE and tf.is_tensor(tensor): return tf.transpose(tensor, perm=[0, 2, 1])
                    return np.swapaxes(tensor, 1, 2)
                    
            elif len(t_shape) == 2:
                # [Batch, Cells] vs [Cells, Batch]
                if dim_matches(t_shape[0], expected_batch_size) and t_shape[1] == n_cells:
                    return tensor
                elif t_shape[0] == n_cells and dim_matches(t_shape[1], expected_batch_size):
                    # [Cells, Batch] -> [Batch, Cells]
                    if TF_AVAILABLE and tf.is_tensor(tensor):
                        return tf.transpose(tensor, perm=[1, 0])
                    return np.transpose(tensor)
                    
                # Fallback
                if t_shape[1] == n_cells: return tensor
                if t_shape[0] == n_cells: 
                     if TF_AVAILABLE and tf.is_tensor(tensor): return tf.transpose(tensor, perm=[1, 0])
                     return np.transpose(tensor)
                     
            return tensor

        # Serving Cell: max RSRP
        # Infer batch size
        inferred_bs = None
        if TF_AVAILABLE and tf.is_tensor(ue_positions): inferred_bs = tf.shape(ue_positions)[0]
        elif isinstance(ue_positions, np.ndarray): inferred_bs = ue_positions.shape[0]

        if TF_AVAILABLE and tf.is_tensor(rsrp):
            # Align dimensions
            rsrp = align_dims(rsrp, num_cells, inferred_bs)
                
            # Now rsrp is [Batch, Rx, Cells]
            best_cell_idx = tf.argmax(rsrp, axis=-1) # [Batch, Rx]
            
            # REMOVED SQUEEZE: Consistent [Batch, Rx] output
            # if len(best_cell_idx.shape) > 1 and best_cell_idx.shape[-1] == 1:
            #     best_cell_idx = tf.squeeze(best_cell_idx, axis=-1)
            
            # Map index to CellID
            cids = tf.convert_to_tensor(cell_ids, dtype=tf.int64)
            # Ensure best_cell_idx is within bounds of cell_ids
            best_cell_idx = tf.clip_by_value(best_cell_idx, 0, num_cells - 1)
            serving_cell_id = tf.gather(cids, best_cell_idx) # [Batch, Rx]
            
            # Neighbors
            k = min(self.max_neighbors, num_cells)
            values, inds = tf.math.top_k(rsrp, k=k)
            inds = tf.clip_by_value(inds, 0, num_cells - 1)
            neighbor_cell_ids = tf.gather(cids, inds) # [Batch, Rx, K]
            
        else:
            # Numpy
            # Robust Shape Fix for mismatch between Transmitters and RSRP dimensions
            # Usually happens if implicit sources or dimensionality issues in Sionna output
            if isinstance(rsrp, np.ndarray) and rsrp.ndim >= 2:
                # Find dimension matching Batch
                batch_dim = -1
                for i, d in enumerate(rsrp.shape):
                    if d == inferred_bs:
                        batch_dim = i
                        break
                
                if batch_dim != -1:
                    # Move batch to 0
                    if batch_dim != 0:
                        rsrp = np.swapaxes(rsrp, 0, batch_dim)
                    
                    # Check if it's already [Batch, Rx, Cells]
                    if rsrp.ndim == 3 and rsrp.shape[2] == num_cells:
                        # Correctly shaped, do not slice dim 1 (which is Rx)
                        pass
                    else:
                        # Now [Batch, Likely_Cells, ...]
                        # If dim 1 does not match num_cells
                        current_cells = rsrp.shape[1] if rsrp.ndim > 1 else 1
                        if current_cells != num_cells:
                            if current_cells > num_cells:
                                 # Slice to expected number of cells (Sionna often returns Transmitters + something?)
                                 # Assuming first N are the valid transmitters
                                 rsrp = rsrp[:, :num_cells]

            # Pass inferred_bs here too!
            rsrp = align_dims(rsrp, num_cells, inferred_bs)
            
            # Handle Rank 1 case [Batch] -> effectively [Batch, 1Cell] where values are RSRP
            # But argmax on [Batch] gives scalar!
            if rsrp.ndim == 1 and inferred_bs is not None and rsrp.shape[0] == inferred_bs:
                 # It's likely [Batch] treating as 1 cell
                 # If we have 1 cell, index is 0.
                 best_cell_idx = np.zeros(inferred_bs, dtype=int)
                 # Expand to [Batch, 1] to respect Rx dim
                 best_cell_idx = best_cell_idx[:, np.newaxis]
            else:
                 best_cell_idx = np.argmax(rsrp, axis=-1)
                 # Ensure Rank 2 [Batch, Rx]
                 if best_cell_idx.ndim == 1:
                     best_cell_idx = best_cell_idx[:, np.newaxis]
            
            # REMOVED SQUEEZE: Consistent [Batch, Rx] output
            # if best_cell_idx.ndim > 1 and best_cell_idx.shape[-1] == 1:
            #    best_cell_idx = np.squeeze(best_cell_idx, axis=-1)
                
            # Clip for safety
            best_cell_idx = np.clip(best_cell_idx, 0, num_cells - 1)
            serving_cell_id = cell_ids[best_cell_idx]
            
            k = min(self.max_neighbors, rsrp.shape[-1]) if rsrp.ndim > 1 else 1
            if rsrp.ndim == 1:
                 # 1 cell available per batch item?
                 inds = np.zeros((inferred_bs, 1), dtype=int)
            else:
                 inds = np.argsort(-rsrp, axis=-1)[..., :k]
            
            inds = np.clip(inds, 0, num_cells - 1)
            neighbor_cell_ids = cell_ids[inds]

        # Timing Advance
        if TF_AVAILABLE and tf.is_tensor(ue_positions):
             ue_pos = tf.cast(ue_positions, tf.float32)
             site_pos = tf.convert_to_tensor(site_positions, dtype=tf.float32)
             
             # best_cell_idx: [Batch, Rx]
             # serving_site_pos = gather(site_pos, best_cell_idx) -> [Batch, Rx, 3]
             # Safety: handle site_pos being smaller than cell_ids (e.g. 1 site, 3 cells/sectors)
             num_sites = tf.shape(site_pos)[0]
             # Use modulo to map cell index to site index safely if mismatch
             site_idx = tf.math.mod(best_cell_idx, tf.cast(num_sites, best_cell_idx.dtype))
             
             serving_site_pos = tf.gather(site_pos, site_idx)
             
             # UE pos: [Batch, 3]. Expand to [Batch, 1, 3] to match Rx dimension
             if len(ue_pos.shape) == 2:
                  ue_pos_exp = tf.expand_dims(ue_pos, 1) # [Batch, 1, 3]
             else:
                  ue_pos_exp = ue_pos
                  
             # dist: [Batch, Rx]
             dist = tf.norm(ue_pos_exp - serving_site_pos, axis=-1)
             timing_advance = compute_timing_advance(dist)
             
             # Throughput
             # cqi: [Batch, Rx, Cells]
             cqi = phy_features.cqi
             # Fix shape of cqi if needed
             cqi = align_dims(cqi, num_cells)
                  
             # Select serving cell CQI: [Batch, Rx] or [Batch]
             # gather from last dim using best_cell_idx
             dl_throughput = self._simulate_throughput(cqi)
             
             # Safety for throughput gathering
             # Ensure indices match the last dimension of dl_throughput
             n_tpt_cells = tf.shape(dl_throughput)[-1]
             safe_idx = tf.clip_by_value(best_cell_idx, 0, tf.cast(n_tpt_cells, best_cell_idx.dtype) - 1)

             if dl_throughput.ndim == 3 and safe_idx.shape.rank == 1:
                 # Should not happen as safe_idx is [Batch, Rx] now
                 # But if it did...
                 idx_exp = tf.expand_dims(safe_idx, -1) # [B, 1]
                 rx = tf.shape(dl_throughput)[1]
                 idx_exp = tf.tile(idx_exp, [1, rx]) # [B, Rx]
                 dl_t_serv = tf.gather(dl_throughput, idx_exp, batch_dims=2, axis=-1)
                 
             elif len(dl_throughput.shape) == len(safe_idx.shape) + 1:
                 bd = len(safe_idx.shape)
                 dl_t_serv = tf.gather(dl_throughput, safe_idx, batch_dims=bd)
             else:
                 dl_t_serv = tf.gather(dl_throughput, safe_idx, batch_dims=1)
             
        else:
             # Numpy path
             num_sites = len(site_positions)
             # Safety modulo
             site_idx = best_cell_idx % num_sites
             serving_pos = site_positions[site_idx] # [Batch, Rx, 3] or [Batch, 3]
             
             u_pos = ue_positions
             if u_pos.ndim == 2:
                 if serving_pos.ndim == 3:
                      u_pos = u_pos[:, np.newaxis, :] 
                  
             dist = np.linalg.norm(u_pos - serving_pos, axis=-1)
             timing_advance = compute_timing_advance(dist)
             
             cqi = phy_features.cqi
             
             # Robust Shape Fix for CQI (Same as RSRP)
             if isinstance(cqi, np.ndarray) and cqi.ndim >= 2:
                 # Find dimension matching Batch
                 batch_dim = -1
                 for i, d in enumerate(cqi.shape):
                     if d == inferred_bs:
                         batch_dim = i
                         break
                 
                 if batch_dim != -1:
                     if batch_dim != 0:
                         cqi = np.swapaxes(cqi, 0, batch_dim)
                     
                     # Dim 1 check
                     # Check if it's already [Batch, Rx, Cells]
                     if cqi.ndim == 3 and cqi.shape[2] == num_cells:
                         pass
                     else:
                         current_cells = cqi.shape[1] if cqi.ndim > 1 else 1
                         if current_cells != num_cells:
                              if current_cells > num_cells:
                                   cqi = cqi[:, :num_cells]

             cqi = align_dims(cqi, num_cells)
                  
             dl_throughput = self._simulate_throughput(cqi)
             
             # Safety for gather
             safe_idx = np.clip(best_cell_idx, 0, dl_throughput.shape[-1] - 1)

             # Robust Dimension Expansion for safe_idx vs dl_throughput
             # safe_idx is [Batch, Rx]
             # dl_throughput should be [Batch, Rx, Cells]
             # If dl_throughput is missing Rx dim (e.g. [Batch, Cells] or [Batch]), expand it.
             while dl_throughput.ndim <= safe_idx.ndim:
                  # Insert axis 1 (Rx) assuming 0=Batch, -1=Cells
                  dl_throughput = np.expand_dims(dl_throughput, axis=1)

             # Align ranks for take_along_axis
             if dl_throughput.ndim > safe_idx.ndim:
                 idx_expanded = safe_idx
                 while idx_expanded.ndim < dl_throughput.ndim:
                     idx_expanded = idx_expanded[..., np.newaxis]
             else:
                 idx_expanded = safe_idx[..., np.newaxis]

             dl_t_serv = np.take_along_axis(dl_throughput, idx_expanded, axis=-1)
             dl_t_serv = dl_t_serv.squeeze(axis=-1)
             
             # REMOVED SQUEEZE: Consistent [Batch, Rx] output
             # if dl_t_serv.ndim > 1 and dl_t_serv.shape[-1] == 1:
             #    dl_t_serv = dl_t_serv.squeeze(axis=-1)
             
        # Squeeze Rx dim if it's 1 for final output [Batch]?
        # The dataclass expects [Batch, NumRx]. Usually we keep it.
        # But old code might expect [Batch]. 
        # features.py usually keeps [Batch, Rx].
        
        return MACRRCLayerFeatures(
            serving_cell_id=serving_cell_id,
            neighbor_cell_ids=neighbor_cell_ids,
            timing_advance=timing_advance,
            dl_throughput_mbps=dl_t_serv if self.enable_throughput else None
        )

    def _simulate_throughput(self, cqi: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Simulate throughput from CQI (Mbps)."""
        # Simple linear model: 10 Mbps per CQI step? Max ~150 Mbps?
        # Supports TF and Numpy
        if TF_AVAILABLE and tf.is_tensor(cqi):
            cqi_f = tf.cast(cqi, tf.float32)
            return cqi_f * 10.0
        else:
            return cqi * 10.0

    def _simulate_bler(self, sinr: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Simulate BLER from SINR (0 to 1)."""
        # Sigmoid function for BLER curve
        # Center at 5 dB? Slope -0.5?
        if TF_AVAILABLE and tf.is_tensor(sinr):
            sinr_f = tf.cast(sinr, tf.float32)
            return tf.sigmoid(-(sinr_f - 5.0) / 2.0)
        else:
            return 1.0 / (1.0 + np.exp((sinr - 5.0) / 2.0))
