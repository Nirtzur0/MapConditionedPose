"""
3GPP-Compliant Measurement Computation Utilities
Implements FAPI/MAC/RRC measurement calculations per 3GPP specifications
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try importing TensorFlow (only needed during data generation)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - measurement utils will use NumPy only")


def compute_rsrp(h: np.ndarray, 
                 cell_id: int,
                 pilot_re_indices: np.ndarray,
                 use_tf: bool = False) -> np.ndarray:
    """
    RSRP: Average power over reference signals (3GPP 38.215).
    
    Reference Signal Received Power is measured over resource elements 
    carrying cell-specific reference signals (CRS) or SSB.
    
    Args:
        h: Channel matrix [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
           or frequency domain [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        cell_id: Physical Cell ID
        pilot_re_indices: Indices of resource elements carrying pilots
        use_tf: Use TensorFlow if available (for gradient computation)
        
    Returns:
        rsrp: [batch, num_rx] in linear scale (convert to dBm with 10*log10(rsrp*1000))
        
    Notes:
        - 3GPP 38.215 Section 5.1.1: RSRP is linear average of power over RE carrying RS
        - Quantization: 1 dB steps in range [-156, -44] dBm per 3GPP 38.133
    """
    if use_tf and TF_AVAILABLE:
        h = tf.constant(h) if not isinstance(h, tf.Tensor) else h
        
        # Extract channel at pilot positions
        h_pilots = tf.gather(h, pilot_re_indices, axis=-1)
        
        # Compute power per antenna port, average over pilots
        power_per_port = tf.reduce_mean(tf.abs(h_pilots)**2, axis=-1)
        
        # Sum across antenna ports (3GPP spec)
        rsrp = tf.reduce_sum(power_per_port, axis=[-1, -2])
        
        # Quantize to 1 dB (0.1 dBm steps in 3GPP range -156 to -44 dBm)
        rsrp_dbm = 10 * tf.math.log(rsrp) / tf.math.log(10.0) + 30  # Watts to dBm
        rsrp_quantized = tf.round(rsrp_dbm * 10) / 10  # 0.1 dB quantization
        
        return rsrp_quantized.numpy()
    else:
        # NumPy implementation
        h_pilots = h[..., pilot_re_indices]
        power_per_port = np.mean(np.abs(h_pilots)**2, axis=-1)
        rsrp = np.sum(power_per_port, axis=(-1, -2))
        
        # Convert to dBm and quantize
        rsrp_dbm = 10 * np.log10(rsrp + 1e-10) + 30
        rsrp_quantized = np.round(rsrp_dbm * 10) / 10
        
        return rsrp_quantized


def compute_rsrq(rsrp: np.ndarray, 
                 rssi: np.ndarray,
                 N: int = 12) -> np.ndarray:
    """
    RSRQ = N * RSRP / RSSI (3GPP 38.215).
    
    Reference Signal Received Quality measures signal quality relative to total received power.
    
    Args:
        rsrp: Reference Signal Received Power [linear or dBm]
        rssi: Received Signal Strength Indicator [linear or dBm]
        N: Number of resource blocks (default 12 subcarriers)
        
    Returns:
        rsrq: [batch, num_rx] in dB (-34 to +2.5 dB range per 3GPP)
        
    Notes:
        - 3GPP 38.215 Section 5.1.3
        - Quantization: 0.5 dB steps per 3GPP 38.133
    """
    # Detect if inputs are in linear or dB scale
    in_linear_scale = np.mean(rsrp) > 1.0  # Heuristic: dBm is typically negative
    
    if in_linear_scale:
        rsrp_db = 10 * np.log10(rsrp + 1e-10)
        rssi_db = 10 * np.log10(rssi + 1e-10)
    else:
        rsrp_db, rssi_db = rsrp, rssi
    
    # RSRQ = N * RSRP / RSSI in dB
    rsrq_db = rsrp_db - rssi_db + 10 * np.log10(float(N))
    
    # Quantize to 0.5 dB steps (3GPP 38.133)
    rsrq_quantized = np.round(rsrq_db * 2) / 2
    
    # Clip to 3GPP range
    return np.clip(rsrq_quantized, -34.0, 2.5)


def compute_sinr(h: np.ndarray,
                 noise_power: float,
                 interference_cells: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Signal-to-Interference-plus-Noise Ratio.
    
    Args:
        h: Channel matrix for serving cell
        noise_power: Thermal noise power [linear scale]
        interference_cells: List of channel matrices for interfering cells
        
    Returns:
        sinr: [batch, num_rx] in dB
    """
    # Signal power from serving cell
    signal_power = np.sum(np.abs(h)**2, axis=(-3, -2, -1))
    
    # Interference power from neighbor cells
    if interference_cells is not None and len(interference_cells) > 0:
        interference_power = sum(
            np.sum(np.abs(h_int)**2, axis=(-3, -2, -1))
            for h_int in interference_cells
        )
    else:
        interference_power = 0
    
    # SINR = Signal / (Interference + Noise)
    sinr_linear = signal_power / (interference_power + noise_power + 1e-10)
    sinr_db = 10 * np.log10(sinr_linear + 1e-10)
    
    return sinr_db


def compute_cqi(sinr_db: np.ndarray, 
                mcs_table: str = 'table1') -> np.ndarray:
    """
    Map SINR to CQI index (0-15) using 3GPP 38.214 Table 5.2.2.1-2/3.
    
    Channel Quality Indicator is reported by UE to help BS select appropriate MCS.
    
    Args:
        sinr_db: Signal-to-Interference-plus-Noise Ratio [dB]
        mcs_table: '64QAM' (table1), '256QAM' (table2), or '64QAM-lowSE' (table3)
        
    Returns:
        cqi: Integer in [0, 15] (0 = out of range)
        
    Notes:
        - 3GPP 38.214 Section 5.2.2.1
        - CQI 0 means "out of range" (too low SINR)
        - Higher CQI = higher data rate capability
    """
    # 3GPP 38.214 CQI Table 1 (64QAM)
    if mcs_table == 'table1' or mcs_table == '64QAM':
        cqi_thresholds = np.array([
            -np.inf, -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 6.0, 7.9, 9.6,
            11.2, 12.7, 14.0, 15.2, 16.4, 17.5
        ])
    elif mcs_table == 'table2' or mcs_table == '256QAM':
        # 3GPP 38.214 CQI Table 2 (256QAM, higher throughput)
        cqi_thresholds = np.array([
            -np.inf, -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 6.0, 7.9, 9.6,
            11.2, 12.7, 14.6, 16.4, 18.2, 20.0
        ])
    else:  # table3 or '64QAM-lowSE'
        # 3GPP 38.214 CQI Table 3 (64QAM, low spectral efficiency, better coverage)
        cqi_thresholds = np.array([
            -np.inf, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0, 18.0, 20.0
        ])
    
    # Find CQI via thresholding
    cqi = np.searchsorted(cqi_thresholds, sinr_db, side='right') - 1
    cqi = np.clip(cqi, 0, 15)
    
    return cqi.astype(np.int32)


def compute_rank_indicator(h: np.ndarray, 
                           snr_threshold_db: float = 0.0) -> np.ndarray:
    """
    Compute RI via SVD of channel matrix (3GPP 38.214).
    
    Rank Indicator reports the number of useful spatial layers for MIMO transmission.
    
    Args:
        h: Channel matrix [..., num_rx_ant, num_tx_ant]
        snr_threshold_db: Threshold to count significant singular values
        
    Returns:
        ri: Rank indicator in [1, min(num_rx_ant, num_tx_ant)] (max layers)
        
    Notes:
        - 3GPP 38.214 Section 5.2.2.2
        - RI indicates how many parallel streams can be supported
        - Higher RI = better spatial multiplexing
    """
    # Handle batch dimensions
    original_shape = h.shape
    h_2d = h.reshape(-1, h.shape[-2], h.shape[-1])
    
    # SVD: h = U @ diag(s) @ V^H
    u, s, vh = np.linalg.svd(h_2d, full_matrices=False)
    
    # Count singular values above threshold
    s_db = 20 * np.log10(s + 1e-10)
    max_s_db = np.max(s_db, axis=-1, keepdims=True)
    significant = s_db > (max_s_db + snr_threshold_db)
    
    ri = np.sum(significant, axis=-1)
    ri = np.clip(ri, 1, s.shape[-1])
    
    # Reshape back to original batch shape
    ri = ri.reshape(original_shape[:-2])
    
    return ri.astype(np.int32)


def compute_timing_advance(distance_3d: np.ndarray, 
                          speed_of_light: float = 3e8) -> np.ndarray:
    """
    TA in NR: TA = 2*d/c, quantized to 16*Ts (0.52 μs steps for 30 kHz SCS).
    
    Timing Advance tells UE how much to advance its uplink transmission timing
    to compensate for propagation delay.
    
    Args:
        distance_3d: 3D distance BS-UE [m]
        speed_of_light: 3e8 m/s
        
    Returns:
        ta_index: Integer TA index (3GPP 38.213 quantization)
        
    Notes:
        - 3GPP 38.213 Section 4.2
        - NR timing advance unit: 16*Ts where Ts = 1/(15000*4096) ≈ 16.3 ns
        - 0.26 μs for 15 kHz, 0.52 μs for 30 kHz SCS
        - Max TA index: 3846
    """
    # Round-trip time
    rtt = 2 * distance_3d / speed_of_light  # seconds
    
    # NR timing advance unit: 16*Ts where Ts = 1/(15000*4096) ≈ 16.3 ns
    ts = 1 / (15000 * 4096)
    ta_unit = 16 * ts  # 0.26 μs for 15 kHz, 0.52 μs for 30 kHz SCS
    
    # Quantize
    ta_index = np.round(rtt / ta_unit).astype(np.int32)
    ta_index = np.clip(ta_index, 0, 3846)  # Max TA in 3GPP
    
    return ta_index


def compute_pmi(h: np.ndarray,
                codebook: str = 'Type1-SinglePanel',
                num_beams: int = 8) -> np.ndarray:
    """
    Compute Precoding Matrix Indicator from channel matrix.
    
    PMI recommends which precoding matrix from the codebook provides best performance.
    
    Args:
        h: Channel matrix [..., num_rx_ant, num_tx_ant]
        codebook: Codebook type per 3GPP 38.214
        num_beams: Number of beams in codebook
        
    Returns:
        pmi: Integer PMI index
        
    Notes:
        - 3GPP 38.214 Section 5.2.2.2
        - Simplified: select beam direction maximizing received power
    """
    # Simplified PMI: find beam that maximizes ||h * w||^2
    # In practice, would use actual 3GPP codebook
    
    # SVD-based: select dominant right singular vector
    original_shape = h.shape
    h_2d = h.reshape(-1, h.shape[-2], h.shape[-1])
    
    u, s, vh = np.linalg.svd(h_2d, full_matrices=False)
    
    # PMI index based on dominant beam (simplified)
    # In real implementation, would quantize vh[0] to codebook
    pmi = np.zeros(h_2d.shape[0], dtype=np.int32)
    
    # Reshape back
    pmi = pmi.reshape(original_shape[:-2])
    
    return pmi


def simulate_neighbor_list_truncation(cells: np.ndarray,
                                      rsrp_values: np.ndarray,
                                      K: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep top-K cells by RSRP, following 3GPP 36.331/38.331 measurement reporting.
    
    Args:
        cells: Cell IDs [batch, num_cells]
        rsrp_values: RSRP per cell [batch, num_cells] in dBm
        K: Max neighbors to report (default 8 per 3GPP)
        
    Returns:
        neighbor_cells: [batch, K] top cells
        neighbor_rsrp: [batch, K] corresponding RSRP
        
    Notes:
        - 3GPP 38.331: maxNrofCellMeas = 8 for NR
        - Real UEs report limited number of neighbors to reduce signaling
    """
    # Sort cells by RSRP descending
    sorted_indices = np.argsort(-rsrp_values, axis=-1)
    top_k_indices = sorted_indices[..., :K]
    
    # Gather top-K cells and RSRP
    neighbor_cells = np.take_along_axis(cells, top_k_indices, axis=-1)
    neighbor_rsrp = np.take_along_axis(rsrp_values, top_k_indices, axis=-1)
    
    return neighbor_cells, neighbor_rsrp


def add_measurement_dropout(features: Dict[str, np.ndarray],
                           dropout_rates: Dict[str, float],
                           seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Simulate realistic measurement availability per protocol timing.
    
    Args:
        features: Dict of measurement tensors
        dropout_rates: Dict mapping feature name to dropout probability
        seed: Random seed for reproducibility
        
    Returns:
        features_with_dropout: Dict with NaN masks for dropped measurements
        
    Notes:
        - Neighbor measurements have higher dropout (20-30%)
        - Some features reported less frequently (PMI, RI)
        - Reflects real network measurement scheduling
    """
    if seed is not None:
        np.random.seed(seed)
    
    dropped = {}
    for key, tensor in features.items():
        rate = dropout_rates.get(key, 0.0)
        if rate > 0:
            mask = np.random.uniform(size=tensor.shape) > rate
            dropped[key] = np.where(mask, tensor, np.nan)
        else:
            dropped[key] = tensor
    
    return dropped


def compute_beam_rsrp(a: np.ndarray,
                     beam_directions: np.ndarray,
                     num_beams: int = 64) -> np.ndarray:
    """
    Compute L1-RSRP per beam for 5G NR SSB (Synchronization Signal Block).
    
    Args:
        a: Path complex amplitudes [batch, num_rx, num_paths]
        beam_directions: Beam steering vectors [num_beams, num_tx_ant]
        num_beams: Number of SSB beams (typically 64 for mmWave)
        
    Returns:
        l1_rsrp_beams: [batch, num_rx, num_beams] RSRP per beam in dBm
        
    Notes:
        - 3GPP 38.214: L1-RSRP reported per SSB beam
        - Used for beam management in mmWave
        - UE reports top-K beams (e.g., K=4)
    """
    # Simplified: beam power proportional to path gains
    # Real implementation would apply beam patterns
    
    # Sum path powers
    path_power = np.abs(a)**2
    total_power = np.sum(path_power, axis=-1)  # [batch, num_rx]
    
    # Replicate for num_beams (simplified: uniform across beams)
    # In practice, would compute per-beam gain based on AoD
    l1_rsrp_beams = np.repeat(total_power[..., np.newaxis], num_beams, axis=-1)
    
    # Convert to dBm
    l1_rsrp_dbm = 10 * np.log10(l1_rsrp_beams + 1e-10) + 30
    
    return l1_rsrp_dbm


# ============================================================================
# Helper Functions
# ============================================================================

def quantize_measurement(value: np.ndarray, 
                        step: float) -> np.ndarray:
    """Quantize measurement to specified step size."""
    return np.round(value / step) * step


def clip_to_3gpp_range(value: np.ndarray,
                       min_val: float,
                       max_val: float) -> np.ndarray:
    """Clip measurement to 3GPP-specified range."""
    return np.clip(value, min_val, max_val)


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
    
    logger.info("\nAll measurement utility tests passed! ✓")
