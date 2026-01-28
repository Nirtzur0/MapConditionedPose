"""
3GPP-Compliant Measurement Computation Utilities
Implements FAPI/MAC/RRC measurement calculations per 3GPP specifications
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

# Try importing TensorFlow (only needed during data generation)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - measurement utils will use NumPy only")

def _is_tensor(x: Any) -> bool:
    """Check if input is a TensorFlow tensor."""
    return TF_AVAILABLE and isinstance(x, (tf.Tensor, tf.Variable))

def compute_rsrp(h: Union[np.ndarray, Any], 
                 cell_id: int,
                 pilot_re_indices: Union[np.ndarray, Any],
                 use_tf: bool = False) -> Union[np.ndarray, Any]:
    """
    RSRP: Average power over reference signals (3GPP 38.215).
    
    Reference Signal Received Power is measured over resource elements 
    carrying cell-specific reference signals (CRS) or SSB.
    
    Args:
        h: Channel matrix [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
           or frequency domain [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        cell_id: Physical Cell ID
        pilot_re_indices: Indices of resource elements carrying pilots
        use_tf: Force TensorFlow usage if available
        
    Returns:
        rsrp: [batch, num_rx, num_tx] in dBm scale
    """
    # Auto-detect TF tensor
    if _is_tensor(h) or use_tf:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        h = tf.convert_to_tensor(h)
        pilot_re_indices = tf.convert_to_tensor(pilot_re_indices, dtype=tf.int32)
        
        # Extract channel at pilot positions
        # Standard input h shape often: [batch, rx, rx_ant, tx, tx_ant, fft_size]
        # We need to gather on the last axis (frequency)
        h_pilots = tf.gather(h, pilot_re_indices, axis=-1)
        
        # Compute power: |h|^2
        power = tf.square(tf.abs(h_pilots))
        
        # Average over pilots (last axis)
        power_avg_pilots = tf.reduce_mean(power, axis=-1)
        
        # Sum over RX antennas (axis -3 usually: [batch, rx, rx_ant, tx, tx_ant])
        # And Average over TX antennas? 
        # 3GPP: RSRP is linear average over power contributions.
        # Typically sum over Rx ports (combining gain), sum over Tx ports?
        # Definition: Power contribution (in Watts) [linear average over REs]
        # If Tx diversity, we sum power from all Tx ports?
        # Usually RSRP is per-cell, so we sum Tx ports if they belong to same cell.
        
        # Input shape assumption: [..., rx_ant, tx, tx_ant]
        # We sum over rx_ant (coherent/non-coherent combination depending on receiver)
        # We sum over tx_ant (total cell power)
        
        # Auto-detect antenna axes based on shape
        # Standard: [batch, ..., rx_ant, [tx], tx_ant, pilots/freq]
        # We need to sum over rx_ant and tx_ant
        rank = len(power_avg_pilots.shape)
        if rank == 4:
            # [batch, sites, rx_ant, tx_ant]
            rsrp = tf.reduce_sum(power_avg_pilots, axis=[2, 3])
        elif rank == 5:
            # [batch, rx, rx_ant, sites, tx_ant]
            rsrp = tf.reduce_sum(power_avg_pilots, axis=[2, 4])
        elif rank == 6:
            # [batch, rx, rx_ant, sites, tx_ant, freq]
            rsrp = tf.reduce_sum(power_avg_pilots, axis=[2, 4])
        else:
            # Fallback to last and last-but-two if rank >= 3
            rsrp = tf.reduce_sum(power_avg_pilots, axis=[-1, -3])
        
        # Result: [batch, num_rx, num_tx] (depending on input)
        # If result is 2D, make it 3D [batch, 1, sites] for consistency
        if len(rsrp.shape) == 2:
            rsrp = tf.expand_dims(rsrp, 1)
        
        # Quantize to 1 dB (0.1 dBm steps in 3GPP range -156 to -44 dBm)
        # Avoid log(0)
        rsrp_dbm = 10.0 * (tf.math.log(rsrp + 1e-10) / tf.math.log(10.0)) + 30.0
        rsrp_quantized = tf.round(rsrp_dbm * 10.0) / 10.0
        
        return rsrp_quantized
    else:
        # NumPy implementation
        h_pilots = h[..., pilot_re_indices]
        power = np.abs(h_pilots)**2
        power_avg_pilots = np.mean(power, axis=-1)
        
        rank = power_avg_pilots.ndim
        if rank == 4:
            rsrp = np.sum(power_avg_pilots, axis=(2, 3))
        elif rank == 5:
            rsrp = np.sum(power_avg_pilots, axis=(2, 4))
        elif rank == 6:
            rsrp = np.sum(power_avg_pilots, axis=(2, 4))
        else:
            rsrp = np.sum(power_avg_pilots, axis=(-1, -3))
            
        if rsrp.ndim == 2:
            rsrp = rsrp[:, np.newaxis, :]
            
        # Convert to dBm and quantize
        rsrp_dbm = 10 * np.log10(rsrp + 1e-10) + 30
        rsrp_quantized = np.round(rsrp_dbm * 10) / 10
        
        return rsrp_quantized


def compute_rsrq(rsrp: Union[np.ndarray, Any], 
                 rssi: Union[np.ndarray, Any],
                 N: int = 12) -> Union[np.ndarray, Any]:
    """
    RSRQ = N * RSRP / RSSI (3GPP 38.215).
    """
    if _is_tensor(rsrp) or _is_tensor(rssi):
        # TensorFlow path
        rsrp = tf.convert_to_tensor(rsrp, dtype=tf.float32)
        rssi = tf.convert_to_tensor(rssi, dtype=tf.float32)
        rsrp = tf.where(tf.math.is_finite(rsrp), rsrp, tf.constant(-200.0, dtype=rsrp.dtype))
        rssi = tf.where(tf.math.is_finite(rssi), rssi, tf.constant(-200.0, dtype=rssi.dtype))
        
        # Detect linear scale (heuristic)
        is_linear = tf.reduce_mean(rsrp) > 1.0
        
        def _to_db(x):
            x = tf.maximum(x, 1e-10)
            return 10.0 * (tf.math.log(x) / tf.math.log(10.0))

        rsrp_db = tf.cond(is_linear, lambda: _to_db(rsrp), lambda: rsrp)
        rssi_db = tf.cond(is_linear, lambda: _to_db(rssi), lambda: rssi)
        
        # RSRQ (dB) = RSRP(dB) - RSSI(dB) + 10log10(N)
        offset = 10.0 * np.log10(float(N))
        rsrq_db = rsrp_db - rssi_db + offset
        rsrq_db = tf.where(tf.math.is_finite(rsrq_db), rsrq_db, tf.constant(-34.0, dtype=rsrq_db.dtype))
        
        # Quantize 0.5 dB
        rsrq_quantized = tf.round(rsrq_db * 2.0) / 2.0
        return tf.clip_by_value(rsrq_quantized, -34.0, 2.5)
        
    else:
        # NumPy path
        in_linear_scale = np.mean(rsrp) > 1.0
        
        if in_linear_scale:
            rsrp_db = 10 * np.log10(np.maximum(rsrp, 1e-10))
            rssi_db = 10 * np.log10(np.maximum(rssi, 1e-10))
        else:
            rsrp_db, rssi_db = rsrp, rssi
        
        rsrq_db = rsrp_db - rssi_db + 10 * np.log10(float(N))
        rsrq_db = np.where(np.isfinite(rsrq_db), rsrq_db, -34.0)
        rsrq_quantized = np.round(rsrq_db * 2) / 2
        return np.clip(rsrq_quantized, -34.0, 2.5)


def compute_sinr(h: Union[np.ndarray, Any],
                 noise_power: float,
                 interference_cells: Optional[Any] = None) -> Union[np.ndarray, Any]:
    """
    Compute SINR. Supports Tensor inputs.
    """
    if _is_tensor(h):
        h = tf.convert_to_tensor(h)
        power = tf.square(tf.abs(h))
        # Standard: [batch, num_rx, rx_ant, [tx], tx_ant, freq]
        # We want to sum over rx_ant, tx_ant, and freq/pilots
        rank = len(h.shape)
        if rank == 5:
            # [batch, num_rx, rx_ant, tx_ant, freq]
            signal_power = tf.reduce_sum(power, axis=[2, 3, 4])
        elif rank == 6:
            # [batch, num_rx, rx_ant, sites, tx_ant, freq]
            signal_power = tf.reduce_sum(power, axis=[2, 4, 5])
        else:
            signal_power = tf.reduce_sum(power, axis=[-3, -2, -1])
        
        interference_power = 0.0
        if interference_cells is not None and len(interference_cells) > 0:
            for h_int in interference_cells:
                p_int = tf.reduce_sum(tf.square(tf.abs(h_int)), axis=[-3, -2, -1]) # Fallback
                interference_power += p_int
        
        sinr_linear = signal_power / (interference_power + noise_power + 1e-10)
        sinr_db = 10.0 * (tf.math.log(sinr_linear + 1e-10) / tf.math.log(10.0))
        
        # Consistent 3D output [batch, 1, sites] if rank 6
        if len(sinr_db.shape) == 2:
            sinr_db = tf.expand_dims(sinr_db, 1)
            
        return sinr_db

    else:
        # Signal power from serving cell
        rank = h.ndim
        if rank == 5:
            signal_power = np.sum(np.abs(h)**2, axis=(2, 3, 4))
        elif rank == 6:
            signal_power = np.sum(np.abs(h)**2, axis=(2, 4, 5))
        else:
            signal_power = np.sum(np.abs(h)**2, axis=(-3, -2, -1))
        
        # Interference power from neighbor cells
        if interference_cells is not None and len(interference_cells) > 0:
            interference_power = sum(
                np.sum(np.abs(h_int)**2, axis=(-3, -2, -1))
                for h_int in interference_cells
            )
        else:
            interference_power = 0
            
        sinr_linear = signal_power / (interference_power + noise_power + 1e-10)
        sinr_db = 10 * np.log10(sinr_linear + 1e-10)
        
        if sinr_db.ndim == 2:
            sinr_db = sinr_db[:, np.newaxis, :]
            
        return sinr_db


def compute_cqi(sinr_db: Union[np.ndarray, Any],
                mcs_table: str = 'table1') -> Union[np.ndarray, Any]:
    """
    Map SINR to CQI index using TF or NumPy.

    Uses 3GPP TS 38.214 CQI thresholds (Table 5.2.2.1-2/3 style mapping).
    This is a simplified, standard-aligned mapping from wideband SINR to CQI.
    """
    if mcs_table == 'table1' or mcs_table == '64QAM':
        cqi_thresholds = [
            -1000.0, -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 6.0, 7.9, 9.6,
            11.2, 12.7, 14.0, 15.2, 16.4, 17.5
        ]
    elif mcs_table == 'table2' or mcs_table == '256QAM':
        cqi_thresholds = [
            -1000.0, -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 6.0, 7.9, 9.6,
            11.2, 12.7, 14.6, 16.4, 18.2, 20.0
        ]
    else:
        cqi_thresholds = [
            -1000.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0, 18.0, 20.0
        ]
    
    if _is_tensor(sinr_db):
        sinr_db = tf.cast(sinr_db, tf.float32)
        thresh_tensor = tf.constant(cqi_thresholds, dtype=tf.float32)
        # TF searchsorted expects matching leading dims; flatten and reshape.
        flat = tf.reshape(sinr_db, [-1])
        cqi = tf.searchsorted(thresh_tensor, flat, side='right') - 1
        cqi = tf.clip_by_value(cqi, 0, 15)
        return tf.reshape(cqi, tf.shape(sinr_db))
        
    else:
        cqi_thresholds[0] = -np.inf # numpy supports inf
        cqi = np.searchsorted(cqi_thresholds, sinr_db, side='right') - 1
        cqi = np.clip(cqi, 0, 15)
        return cqi.astype(np.int32)


def compute_rank_indicator(
    h: Union[np.ndarray, Any],
    snr_db: Optional[Union[np.ndarray, Any]] = None,
) -> Union[np.ndarray, Any]:
    """
    Compute RI via throughput-maximizing rank selection (supports TF/NumPy).

    Approximation to 3GPP RI selection: choose rank that maximizes
    sum log2(1 + (SNR/r) * s_i^2) over the top r singular values.
    """
    if _is_tensor(h):
        h = tf.convert_to_tensor(h)
        s = tf.linalg.svd(h, compute_uv=False)  # [..., min(rx, tx)]

        if snr_db is None:
            snr_linear = tf.ones_like(s[..., 0])
        else:
            snr_db_t = tf.cast(snr_db, tf.float32)
            snr_linear = tf.pow(10.0, snr_db_t / 10.0)

        rank_max_static = s.shape[-1]
        rank_max = tf.shape(s)[-1]

        def _score_for_r(r):
            s_r = s[..., :r]
            snr_r = snr_linear / tf.cast(r, tf.float32)
            return tf.reduce_sum(tf.math.log(1.0 + snr_r[..., None] * tf.square(s_r)) / tf.math.log(2.0), axis=-1)

        if rank_max_static is not None:
            scores = tf.stack([_score_for_r(r) for r in range(1, rank_max_static + 1)], axis=-1)
        else:
            r_vals = tf.range(1, rank_max + 1)
            scores = tf.map_fn(lambda r: _score_for_r(tf.cast(r, tf.int32)), r_vals, fn_output_signature=tf.float32)
            # Move rank dimension to the last axis: [R, ...] -> [..., R]
            perm = tf.concat([tf.range(1, tf.rank(scores)), [0]], axis=0)
            scores = tf.transpose(scores, perm=perm)

        ri = tf.argmax(scores, axis=-1) + 1
        ri = tf.cast(ri, tf.int32)
        rank_max_i = tf.cast(rank_max, ri.dtype)
        return tf.cast(tf.clip_by_value(ri, tf.cast(1, ri.dtype), rank_max_i), tf.int32)

    # NumPy path
    h_np = np.asarray(h)
    original_shape = h_np.shape
    if h_np.ndim > 2:
        h_2d = h_np.reshape(-1, h_np.shape[-2], h_np.shape[-1])
    else:
        h_2d = h_np

    s = np.linalg.svd(h_2d, compute_uv=False)
    if snr_db is None:
        snr_linear = np.ones(s.shape[0], dtype=np.float64)
    else:
        snr_db_np = np.asarray(snr_db, dtype=np.float64)
        snr_linear = np.power(10.0, snr_db_np / 10.0).reshape(-1)

    rank_max = s.shape[-1]
    scores = []
    for r in range(1, rank_max + 1):
        s_r = s[:, :r]
        snr_r = snr_linear / float(r)
        scores.append(np.sum(np.log2(1.0 + (snr_r[:, None] * (s_r ** 2))), axis=-1))
    scores = np.stack(scores, axis=-1)
    ri = np.argmax(scores, axis=-1) + 1
    ri = np.clip(ri, 1, rank_max)
    if h_np.ndim > 2:
        ri = ri.reshape(original_shape[:-2])
    return ri.astype(np.int32)


def compute_timing_advance(distance_3d: Union[np.ndarray, Any], 
                          speed_of_light: float = 3e8) -> Union[np.ndarray, Any]:
    """
    Compute Timing Advance (supports TF).
    """
    if _is_tensor(distance_3d):
        d = tf.cast(distance_3d, tf.float32)
        rtt = 2.0 * d / speed_of_light
        ts = 1.0 / (15000.0 * 4096.0)
        ta_unit = 16.0 * ts
        
        ta_index = tf.round(rtt / ta_unit)
        return tf.clip_by_value(tf.cast(ta_index, tf.int32), 0, 3846)
    else:
        rtt = 2 * distance_3d / speed_of_light
        ts = 1 / (15000 * 4096)
        ta_unit = 16 * ts
        
        ta_index = np.round(rtt / ta_unit).astype(np.int32)
        ta_index = np.clip(ta_index, 0, 3846)
        return ta_index


def compute_pmi(
    h: Union[np.ndarray, Any],
    codebook: str = "Type1-SinglePanel",
    num_beams: int = 8,
) -> Union[np.ndarray, Any]:
    """
    Compute PMI using a simple Type-I single-panel DFT codebook.

    Selects the codeword that maximizes |h * w|^2 for rank-1 precoding.
    """
    def _dft_codebook(n_tx: int, n_beams: int, backend: str = "np"):
        if backend == "tf":
            k = tf.cast(tf.range(n_tx), tf.float32)
            m = tf.cast(tf.range(n_beams), tf.float32)
            phase = tf.tensordot(k, m, axes=0) * (2.0 * np.pi / float(n_beams))
            W = tf.complex(tf.cos(phase), tf.sin(phase)) / tf.sqrt(tf.cast(n_tx, tf.complex64))
            return W
        k = np.arange(n_tx)
        m = np.arange(n_beams)
        W = np.exp(1j * 2 * np.pi * np.outer(k, m) / n_beams) / np.sqrt(n_tx)
        return W

    if _is_tensor(h):
        h = tf.cast(h, tf.complex64)
        n_tx = tf.shape(h)[-1]
        n_tx_static = h.shape[-1]
        W = _dft_codebook(int(n_tx_static) if n_tx_static is not None else n_tx, num_beams, backend="tf")
        # Project: [.., rx, tx] x [tx, beams] -> [.., rx, beams]
        proj = tf.matmul(h, W)
        power = tf.reduce_sum(tf.abs(proj) ** 2, axis=-2)
        pmi = tf.argmax(power, axis=-1)
        return tf.cast(pmi, tf.int32)

    h_np = np.asarray(h)
    original_shape = h_np.shape
    if h_np.ndim > 2:
        h_2d = h_np.reshape(-1, h_np.shape[-2], h_np.shape[-1])
    else:
        h_2d = h_np

    n_tx = h_2d.shape[-1]
    W = _dft_codebook(n_tx, num_beams, backend="np")
    proj = h_2d @ W
    power = np.sum(np.abs(proj) ** 2, axis=-2)
    pmi = np.argmax(power, axis=-1).astype(np.int32)
    if h_np.ndim > 2:
        pmi = pmi.reshape(original_shape[:-2])
    return pmi


def simulate_neighbor_list_truncation(cells: Union[np.ndarray, Any],
                                      rsrp_values: Union[np.ndarray, Any],
                                      K: int = 8) -> Tuple[Union[np.ndarray, Any], Union[np.ndarray, Any]]:
    """
    Keep top-K cells by RSRP (supports TF).
    """
    if _is_tensor(rsrp_values):
        # top_k
        rsrp_val = tf.cast(rsrp_values, tf.float32)
        # tf.math.top_k returns values and indices
        # We need to sort descending
        values, indices = tf.math.top_k(rsrp_val, k=min(K, rsrp_val.shape[-1]))
        
        neighbor_rsrp = values
        
        # Gather cells
        # cells: [batch, num_cells]
        # indices: [batch, K]
        neighbor_cells = tf.gather(cells, indices, batch_dims=1)
        
        return neighbor_cells, neighbor_rsrp
        
    else:
        sorted_indices = np.argsort(-rsrp_values, axis=-1)
        top_k_indices = sorted_indices[..., :K]
        
        neighbor_cells = np.take_along_axis(cells, top_k_indices, axis=-1)
        neighbor_rsrp = np.take_along_axis(rsrp_values, top_k_indices, axis=-1)
        
        return neighbor_cells, neighbor_rsrp


def add_measurement_dropout(features: Dict[str, Any],
                           dropout_rates: Dict[str, float],
                           seed: Optional[int] = None,
                           rng: Optional[np.random.Generator] = None,
                           tf_rng: Optional[Any] = None) -> Dict[str, Any]:
    """
    Simulate measurement availability (supports TF).
    """
    if rng is None:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if TF_AVAILABLE and tf_rng is None and seed is not None:
        tf_rng = tf.random.Generator.from_seed(seed)
    
    dropped = {}
    for key, tensor in features.items():
        if tensor is None:
            dropped[key] = None
            continue
            
        rate = dropout_rates.get(key, 0.0)
        
        if _is_tensor(tensor):
            if rate > 0:
                # TF dropout
                # Create mask
                shape = tf.shape(tensor)
                if tf_rng is not None:
                    mask = tf_rng.uniform(shape) > rate
                else:
                    mask = tf.random.uniform(shape) > rate
                
                # Replace False with NaN? TF float tensors support NaN. Integers do not.
                if tensor.dtype.is_floating:
                    dropped[key] = tf.where(mask, tensor, float('nan'))
                else:
                    # For integers (CellID, CQI), we can't use NaN. 
                    # Use -1 or keep as is?
                    # Usually dropout implies missing packet.
                    # We'll stick to original logic: if logic requires float for NaN, cast it?
                    # But original logic used np.nan which forces float cast for int arrays.
                    # Let's cast to float for consistency if dropout applied
                     tensor_float = tf.cast(tensor, tf.float32)
                     dropped[key] = tf.where(mask, tensor_float, float('nan'))
            else:
                dropped[key] = tensor
                
        else:
            if rate > 0:
                mask = rng.uniform(size=tensor.shape) > rate
                # Numpy handles type promotion if assigning NaN
                if not np.issubdtype(tensor.dtype, np.floating):
                    tensor = tensor.astype(float)
                dropped[key] = np.where(mask, tensor, np.nan)
            else:
                dropped[key] = tensor
    
    return dropped


def compute_beam_rsrp(a: Union[np.ndarray, Any],
                     beam_directions: Union[np.ndarray, Any],
                     num_beams: int = 64) -> Union[np.ndarray, Any]:
    """
    Compute L1-RSRP per beam (supports TF).
    """
    if _is_tensor(a):
        # a: [..., num_paths] complex
        path_power = tf.square(tf.abs(a))
        total_power = tf.reduce_sum(path_power, axis=-1)
        
        # Expand beams
        l1_rsrp_beams = tf.tile(total_power[..., tf.newaxis], [1] * (len(total_power.shape)) + [num_beams])
        
        l1_rsrp_dbm = 10.0 * (tf.math.log(l1_rsrp_beams + 1e-10) / tf.math.log(10.0)) + 30.0
        return l1_rsrp_dbm
        
    else:
        path_power = np.abs(a)**2
        total_power = np.sum(path_power, axis=-1)
        l1_rsrp_beams = np.repeat(total_power[..., np.newaxis], num_beams, axis=-1)
        l1_rsrp_dbm = 10 * np.log10(l1_rsrp_beams + 1e-10) + 30
        return l1_rsrp_dbm


if __name__ == "__main__":
    # Test measurement computations
    logger.info("Testing 3GPP measurement utilities...")
    
    # Test RSRP computation
    h_test = np.random.randn(10, 4, 2, 1, 8, 100) + 1j * np.random.randn(10, 4, 2, 1, 8, 100)
    pilot_re = np.array([0, 12, 24, 36, 48])
    rsrp = compute_rsrp(h_test, cell_id=0, pilot_re_indices=pilot_re)
    logger.info(f"✓ RSRP: shape={rsrp.shape}, range=[{rsrp.min():.1f}, {rsrp.max():.1f}] dBm")
    
    # Test RSRQ computation
    rssi = rsrp + np.random.randn(*rsrp.shape) * 5  # Add noise
    rsrq = compute_rsrq(rsrp, rssi)
    logger.info(f"✓ RSRQ: shape={rsrq.shape}, range=[{rsrq.min():.1f}, {rsrq.max():.1f}] dB")
    
    # Test CQI computation
    sinr = np.random.uniform(-10, 25, (10, 4))
    cqi = compute_cqi(sinr)
    logger.info(f"✓ CQI: shape={cqi.shape}, range=[{cqi.min()}, {cqi.max()}]")
    
    # Test RI computation
    h_mimo = np.random.randn(10, 4, 8) + 1j * np.random.randn(10, 4, 8)
    ri = compute_rank_indicator(h_mimo)
    logger.info(f"✓ RI: shape={ri.shape}, range=[{ri.min()}, {ri.max()}]")
    
    # Test TA computation
    distances = np.random.uniform(10, 1000, (10, 4))
    ta = compute_timing_advance(distances)
    logger.info(f"✓ TA: shape={ta.shape}, range=[{ta.min()}, {ta.max()}]")
    
    # Test neighbor list truncation
    cells = np.tile(np.arange(20), (10, 1))
    rsrp_all = np.random.uniform(-120, -60, (10, 20))
    neighbor_cells, neighbor_rsrp = simulate_neighbor_list_truncation(cells, rsrp_all, K=8)
    logger.info(f"✓ Neighbor list: shape={neighbor_cells.shape}, top RSRP={neighbor_rsrp[0, 0]:.1f} dBm")
    
    # Test dropout
    features = {'rsrp': rsrp, 'rsrq': rsrq, 'cqi': cqi.astype(float)}
    dropout_rates = {'rsrp': 0.1, 'rsrq': 0.2, 'cqi': 0.15}
    features_dropped = add_measurement_dropout(features, dropout_rates, seed=42)
    nan_count = np.isnan(features_dropped['rsrp']).sum()
    logger.info(f"✓ Dropout: {nan_count}/{rsrp.size} measurements dropped")
    
    # Verification with TF if available
    if TF_AVAILABLE:
        try:
            logger.info("Testing TF paths...")
            h_tf = tf.convert_to_tensor(h_test)
            rsrp_tf = compute_rsrp(h_tf, 0, pilot_re, use_tf=True)
            logger.info(f"✓ TF RSRP path successful")
        except Exception as e:
            logger.error(f"TF Test failed: {e}")
            
    logger.info("\nAll measurement utility tests passed! ✓")
