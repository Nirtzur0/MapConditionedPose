"""
PHY Feature Extractor
Extracts Layer 2 (PHY/FAPI) features from channel matrices and RT features.
"""

import numpy as np
import logging
from typing import Optional, Any, Union, List

from .tensor_ops import TensorOps, get_ops, _to_numpy
from .data_structures import RTLayerFeatures, PHYFAPILayerFeatures
from ..measurement_utils import (
    compute_rsrp, compute_rsrq, compute_sinr, compute_cqi,
    compute_rank_indicator, compute_timing_advance, compute_pmi,
    compute_beam_rsrp
)

logger = logging.getLogger(__name__)

# Try importing Sionna/TensorFlow
try:
    import tensorflow as tf
    import sionna
    SIONNA_AVAILABLE = True
    TF_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    TF_AVAILABLE = False


class PHYFAPIFeatureExtractor:
    """
    Extracts Layer 2 (PHY/FAPI) features from channel matrices and RT features.
    """
    
    def __init__(self,
                 noise_figure_db: float = 9.0,
                 thermal_noise_density_dbm: float = -174.0,
                 enable_beam_management: bool = True,
                 num_beams: int = 64):
        self.noise_figure_db = noise_figure_db
        self.thermal_noise_density_dbm = thermal_noise_density_dbm
        self.enable_beam_management = enable_beam_management
        self.num_beams = num_beams
        
        logger.info(f"PHYFAPIFeatureExtractor initialized: NF={noise_figure_db} dB, "
                   f"beams={num_beams if enable_beam_management else 'disabled'}")
    
    def extract(self, 
                rt_features: RTLayerFeatures,
                channel_matrix: Optional[Union[np.ndarray, Any]] = None,
                interference_matrices: Optional[List[Union[np.ndarray, Any]]] = None,
                pilot_re_indices: Optional[Union[np.ndarray, Any]] = None) -> PHYFAPILayerFeatures:
        
        # Determine Ops backend based on input
        # check channel_matrix or rt_features.path_gains
        ref_data = channel_matrix if channel_matrix is not None else rt_features.path_gains
        ops = get_ops(ref_data)

        if pilot_re_indices is None:
            if ops.is_tensor(ref_data) and TF_AVAILABLE:
                pilot_re_indices = tf.range(0, 100, 4)
            else:
                pilot_re_indices = np.arange(0, 100, 4)
        
        # Compute noise power
        bandwidth_hz = rt_features.bandwidth_hz
        # N0 (linear)
        n0 = 10**((self.thermal_noise_density_dbm + self.noise_figure_db - 30) / 10)
        # Total Noise Power = N0 * BW
        noise_power_linear = n0 * bandwidth_hz
        noise_power_dbm = 10 * np.log10(noise_power_linear) + 30
        
        # RSRP
        rsrp_dbm = None
        if channel_matrix is not None:
            # Use utility - auto-detects TF internally usually, or we pass ops?
            # compute_rsrp handles both if written generically or has checks.
            # Assuming compute_rsrp handles it.
            rsrp_dbm = compute_rsrp(channel_matrix, 0, pilot_re_indices)
            
            # Helper to unsqueeze if needed
            def _unsqueeze_last(x):
                # ops.expand_dims
                if len(ops.shape(x)) == 2:
                    return ops.expand_dims(x, -1)
                return x
            
            rsrp_dbm = _unsqueeze_last(rsrp_dbm)
            
        else:
            # Approximate from path gains
            gains = rt_features.path_gains
            powers = ops.square(ops.abs(gains))
            total_power = ops.sum(powers, axis=-1)
            rsrp_val = 10.0 * ops.log10(total_power + 1e-10) + 30.0 # ops.log10 is log10
            # log10(x) + 30
            # Wait, ops.log10 implementation for TF: tf.math.log(x) / tf.math.log(10.0) -> Correct.
            
            rsrp_dbm = ops.expand_dims(rsrp_val, -1)
        
        # Use Sionna MIMO Capacity for CQI/RI
        cqi = None
        ri = None
        sinr = None
        
        # Only use Sionna advanced capacity if available and using TF backend
        if SIONNA_AVAILABLE and channel_matrix is not None and ops.is_tensor(channel_matrix):
            from sionna.mimo import capacity_optimal
            
            # channel_matrix: [Targets, RxAnt, Sources, TxAnt, Freq]
            h_serv = channel_matrix[:, :, 0, :, :] # Slice Source 0 -> [Batch, RxAnt, TxAnt, Freq]
            
            # Permute to [Batch, Freq, RxAnt, TxAnt]
            h_perm = ops.transpose(h_serv, perm=[0, 3, 1, 2])
            
            n0_eff = noise_power_linear / float(ops.shape(h_perm)[1])
            
            # Capacity (bits/s/Hz)
            cap_bits = capacity_optimal(h_perm, n0_eff)
            # Result: [Batch, Freq]
            
            # Average SE
            se_avg = ops.mean(cap_bits, axis=-1) # [Batch]
            
            # Map SE to Effective SINR
            # C = log2(1 + SINR) -> SINR = 2^C - 1
            # ops doesn't have pow(2, x). Use generic.
            # TF: 2^x = exp(x * ln(2))
            # Numpy: 2**x
            
            if ops.is_tensor(se_avg):
                sinr_eff_linear = tf.pow(2.0, se_avg) - 1.0
            else:
                sinr_eff_linear = np.power(2.0, se_avg) - 1.0
                
            sinr_eff_db = 10.0 * ops.log10(sinr_eff_linear + 1e-10)
            
            # Expand to [Batch, 1, 1]
            sinr_eff_db = ops.expand_dims(ops.expand_dims(sinr_eff_db, -1), -1) # reshape -1, 1, 1 not in ops
            # ops.expand_dims twice?
            
            cqi = compute_cqi(sinr_eff_db)
            sinr = sinr_eff_db
            
            # RI Estimate
            h_mean = ops.mean(h_perm, axis=1) # [Batch, RxAnt, TxAnt]
            
            if ops.is_tensor(h_mean):
                s = tf.linalg.svd(h_mean, compute_uv=False)
                ri_val = tf.reduce_sum(tf.cast(s > 1e-9, tf.int32), axis=-1, keepdims=True)
                ri = tf.reshape(ri_val, [-1, 1, 1])
            else:
                # Numpy svd
                s = np.linalg.svd(h_mean, compute_uv=False)
                ri_val = np.sum(s > 1e-9, axis=-1, keepdims=True)
                ri = ri_val.reshape(-1, 1, 1)

        else:
             # Fallback Legacy
             pass
             
        # RSSI
        # rssi_linear = 10^((rsrp - 30)/10) + noise
        if ops.is_tensor(rsrp_dbm):
            rssi_linear = tf.pow(10.0, (rsrp_dbm - 30.0)/10.0) + noise_power_linear
        else:
            rssi_linear = 10**((rsrp_dbm - 30) / 10) + noise_power_linear
            
        rssi_dbm = 10.0 * ops.log10(rssi_linear) + 30.0
            
        # RSRQ
        if channel_matrix is None:
             num_cells = ops.shape(rsrp_dbm)[-1]
             
        rsrq = compute_rsrq(rsrp_dbm, rssi_dbm, N=12)
        
        # SINR (if not calculated via Capacity)
        if sinr is None:
            if channel_matrix is not None:
                # compute_sinr might need ops? It uses standard operators usually.
                sinr = compute_sinr(channel_matrix, noise_power_linear, interference_matrices)
                if len(ops.shape(sinr)) == 2:
                    sinr = ops.expand_dims(sinr, -1)
            else:
                sinr = rsrp_dbm - noise_power_dbm

        # CQI (if not calculated)
        if cqi is None:
            cqi = compute_cqi(sinr)
        
        # RI (if not calculated)
        if ri is None:
            if channel_matrix is not None:
                h_spatial = ops.mean(channel_matrix, axis=[-2, -1]) if ops.is_tensor(channel_matrix) else ops.mean(channel_matrix, axis=(-2, -1))
                # My generic ops.mean(axis=Any)
                # But TF axis can be list. Numpy axis tuple.
                # ops.mean implementation passes straight to backend. TF supports list, Numpy supports tuple.
                # generic: use tuple? TF convert tuple to list? TF supports tuple too in newer versions?
                # Safer: if ops.is_tensor -> list, else tuple.
                if ops.is_tensor(channel_matrix):
                     h_spatial = ops.mean(channel_matrix, axis=[-2, -1])
                else:
                     h_spatial = ops.mean(channel_matrix, axis=(-2, -1))
                     
                ri = compute_rank_indicator(h_spatial)
            else:
                # Default 1
                ones = _to_numpy(rsrp_dbm) * 0 + 1 # Hacky creation
                if ops.is_tensor(rsrp_dbm):
                     ri = tf.ones_like(rsrp_dbm, dtype=tf.int32)
                else:
                     ri = np.ones_like(rsrp_dbm, dtype=np.int32)
        
        # PMI
        if channel_matrix is not None:
             if ops.is_tensor(channel_matrix):
                 h_spatial = ops.mean(channel_matrix, axis=[-2, -1])
             else:
                 h_spatial = ops.mean(channel_matrix, axis=(-2, -1))
             pmi = compute_pmi(h_spatial)
        else:
             if ops.is_tensor(rsrp_dbm):
                 pmi = tf.zeros_like(rsrp_dbm, dtype=tf.int32)
             else:
                 pmi = np.zeros_like(rsrp_dbm, dtype=np.int32)
                 
        # Beam Management
        l1_rsrp_beams = None
        best_beam_ids = None
        if self.enable_beam_management:
            l1_rsrp_beams = compute_beam_rsrp(rt_features.path_gains, None, self.num_beams)
            # Top-4
            if ops.is_tensor(l1_rsrp_beams):
                 values, indices = tf.math.top_k(l1_rsrp_beams, k=4)
                 best_beam_ids = indices
            else:
                 best_beam_ids = np.argsort(-l1_rsrp_beams, axis=-1)[..., :4]

        return PHYFAPILayerFeatures(
            rsrp=rsrp_dbm,
            rsrq=rsrq,
            sinr=sinr,
            cqi=cqi,
            ri=ri,
            pmi=pmi,
            l1_rsrp_beams=l1_rsrp_beams,
            best_beam_ids=best_beam_ids,
            channel_matrix=channel_matrix,
        )
