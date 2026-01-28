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
from ..shape_contract import normalize_channel_matrix, normalize_sinr

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

SIONNA_MIMO_AVAILABLE = False
_CAPACITY_OPTIMAL = None
if SIONNA_AVAILABLE:
    try:
        from sionna.mimo import capacity_optimal as _CAPACITY_OPTIMAL
        SIONNA_MIMO_AVAILABLE = True
    except Exception:
        _CAPACITY_OPTIMAL = None
        SIONNA_MIMO_AVAILABLE = False


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
        if SIONNA_AVAILABLE and TF_AVAILABLE and not SIONNA_MIMO_AVAILABLE:
            logger.info("sionna.mimo not available; using Shannon-capacity fallback for CQI/RI.")

    def _compute_capacity(self, h_perm: Any, n0_eff: Any, ops: TensorOps) -> Any:
        """Compute MIMO capacity per subcarrier for h_perm [B, F, Rx, Tx]."""
        if ops.is_tensor(h_perm):
            # SISO/SIMO shortcut
            if int(h_perm.shape[-1]) == 1:
                h_sq = tf.reduce_sum(tf.square(tf.abs(h_perm)), axis=[-2, -1])
                snr = h_sq / n0_eff
                return tf.math.log(1.0 + snr) / tf.math.log(2.0)

            s = tf.linalg.svd(h_perm, compute_uv=False)
            params = tf.square(s) / n0_eff
            cap_per_stream = tf.math.log(1.0 + params) / tf.math.log(2.0)
            return tf.reduce_sum(cap_per_stream, axis=-1)

        # NumPy path
        h_np = ops.to_numpy(h_perm)
        if h_np.shape[-1] == 1:
            h_sq = np.sum(np.abs(h_np) ** 2, axis=(-2, -1))
            snr = h_sq / n0_eff
            return np.log2(1.0 + snr)

        s = np.linalg.svd(h_np, compute_uv=False)
        params = (s ** 2) / n0_eff
        cap_per_stream = np.log2(1.0 + params)
        return np.sum(cap_per_stream, axis=-1)
    
    def extract(self, 
                rt_features: RTLayerFeatures,
                channel_matrix: Optional[Union[np.ndarray, Any]] = None,
                interference_matrices: Optional[List[Union[np.ndarray, Any]]] = None,
                pilot_re_indices: Optional[Union[np.ndarray, Any]] = None) -> PHYFAPILayerFeatures:
        
        # Determine Ops backend based on input
        # check channel_matrix or rt_features.path_gains
        ref_data = channel_matrix if channel_matrix is not None else rt_features.path_gains
        ops = get_ops(ref_data)
        if channel_matrix is not None:
            channel_matrix = normalize_channel_matrix(channel_matrix, strict=False).data

        if pilot_re_indices is None:
            if ops.is_tensor(ref_data) and TF_AVAILABLE:
                num_sc = tf.shape(ref_data)[-1]
                pilot_re_indices = tf.range(0, num_sc, 4)
            else:
                try:
                    num_sc = int(ref_data.shape[-1])
                except Exception:
                    num_sc = 100
                step = 4 if num_sc >= 4 else 1
                pilot_re_indices = np.arange(0, num_sc, step)
        
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
            rsrp_dbm = ops.expand_dims(rsrp_val, -1)
        
        # Use Sionna MIMO Capacity for CQI/RI
        cqi = None
        ri = None
        sinr = None
        
        # Only use Sionna advanced capacity if available and using TF backend
        if SIONNA_AVAILABLE and channel_matrix is not None and ops.is_tensor(channel_matrix):
            # channel_matrix: [Batch, RxAnt, Sources, TxAnt, Freq]
            # or [Batch, Rx, RxAnt, Sources, TxAnt, Freq]
            cm_rank = len(ops.shape(channel_matrix))
            if cm_rank == 6:
                # [B, Rx, RxAnt, C, TxAnt, F] -> serve Rx=0, C=0
                h_serv = channel_matrix[:, 0, :, 0, :, :]
            else:
                # [B, RxAnt, C, TxAnt, F]
                h_serv = channel_matrix[:, :, 0, :, :] # Slice Source 0 -> [Batch, RxAnt, TxAnt, Freq]
            
            # Permute to [Batch, Freq, RxAnt, TxAnt]
            h_perm = ops.transpose(h_serv, perm=[0, 3, 1, 2])
            
            n0_eff = noise_power_linear / float(ops.shape(h_perm)[1])
            
            # Capacity (bits/s/Hz)
            if _CAPACITY_OPTIMAL is not None:
                cap_bits = _CAPACITY_OPTIMAL(h_perm, n0_eff)
            else:
                cap_bits = self._compute_capacity(h_perm, n0_eff, ops)
            # Result: [Batch, Freq]
            se_avg = ops.mean(cap_bits, axis=-1) # [Batch]
            
            # Map SE to Effective SINR
            # C = log2(1 + SINR) -> SINR = 2^C - 1
            # ops doesn't have pow(2, x). Use generic.
            # TF: 2^x = exp(x * ln(2))
            # Numpy: 2**x
            if se_avg is not None:
                if ops.is_tensor(se_avg):
                    sinr_eff_linear = tf.pow(2.0, se_avg) - 1.0
                else:
                    sinr_eff_linear = np.power(2.0, se_avg) - 1.0
                    
                sinr_eff_db = 10.0 * ops.log10(sinr_eff_linear + 1e-10)
                
                # Expand to [Batch, 1, 1]
                sinr_eff_db = ops.expand_dims(ops.expand_dims(sinr_eff_db, -1), -1) # reshape -1, 1, 1 not in ops
                # ops.expand_dims twice?
                
                serving_sinr_db = sinr_eff_db
                serving_cqi = compute_cqi(serving_sinr_db)
                
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

        # RSSI (wideband) approximation: sum over N resource elements + noise.
        # This avoids RSRQ saturation at the upper bound.
        rsrq_n_re = 12.0
        if ops.is_tensor(rsrp_dbm):
            rsrp_linear = tf.pow(10.0, (rsrp_dbm - 30.0) / 10.0)
            rssi_linear = rsrp_linear * rsrq_n_re + noise_power_linear
        else:
            rsrp_linear = np.power(10.0, (rsrp_dbm - 30.0) / 10.0)
            rssi_linear = rsrp_linear * rsrq_n_re + noise_power_linear

        rssi_dbm = 10.0 * ops.log10(rssi_linear + 1e-10) + 30.0

        # RSRQ
        rsrq = compute_rsrq(rsrp_dbm, rssi_dbm, N=int(rsrq_n_re))
        
        # SINR (per-cell when channel matrix is available)
        if channel_matrix is not None:
            sinr_full = compute_sinr(channel_matrix, noise_power_linear, interference_matrices)
            if len(ops.shape(sinr_full)) == 2:
                sinr_full = ops.expand_dims(sinr_full, -1)
            sinr = normalize_sinr(sinr_full)
        else:
            sinr = rsrp_dbm - noise_power_dbm

        # CQI (align with SINR shape; fall back to serving-only if needed)
        if cqi is None or ops.shape(cqi)[-1] != ops.shape(sinr)[-1]:
            cqi = compute_cqi(sinr)
        elif 'serving_cqi' in locals() and ops.shape(cqi)[-1] != ops.shape(rsrp_dbm)[-1]:
            cqi = compute_cqi(sinr)
        
        # RI (if not calculated or mismatched shape)
        if ri is None or (channel_matrix is not None and ops.shape(ri)[-1] != ops.shape(rsrp_dbm)[-1]):
            if channel_matrix is not None:
                sinr_for_ri = normalize_sinr(sinr)
                if sinr_for_ri is not None and len(ops.shape(sinr_for_ri)) == 3:
                    sinr_for_ri = sinr_for_ri[:, 0, :]
                # Average over frequency only to preserve spatial structure
                h_spatial = ops.mean(channel_matrix, axis=-1)
                # [B, Rx, RxAnt, C, TxAnt] -> [B, C, RxAnt, TxAnt] using Rx=0
                h_spatial = h_spatial[:, 0, ...]
                h_spatial = ops.transpose(h_spatial, perm=[0, 2, 1, 3])
                ri = compute_rank_indicator(h_spatial, snr_db=sinr_for_ri)
            else:
                # Default 1
                ones = _to_numpy(rsrp_dbm) * 0 + 1 # Hacky creation
                if ops.is_tensor(rsrp_dbm):
                     ri = tf.ones_like(rsrp_dbm, dtype=tf.int32)
                else:
                     ri = np.ones_like(rsrp_dbm, dtype=np.int32)
        
        # PMI
        if channel_matrix is not None:
             h_spatial = ops.mean(channel_matrix, axis=-1)
             h_spatial = h_spatial[:, 0, ...]
             h_spatial = ops.transpose(h_spatial, perm=[0, 2, 1, 3])
             pmi = compute_pmi(h_spatial, num_beams=self.num_beams)
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
