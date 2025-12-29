"""
Multi-Layer Feature Extractors for 5G NR Radio Environment
Extracts RT, PHY/FAPI, and MAC/RRC layer features from Sionna simulations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

try:
    from .measurement_utils import (
        compute_rsrp, compute_rsrq, compute_sinr, compute_cqi,
        compute_rank_indicator, compute_timing_advance, compute_pmi,
        compute_beam_rsrp, add_measurement_dropout
    )
except ImportError:
    # Fallback for direct execution
    from measurement_utils import (
        compute_rsrp, compute_rsrq, compute_sinr, compute_cqi,
        compute_rank_indicator, compute_timing_advance, compute_pmi,
        compute_beam_rsrp, add_measurement_dropout
    )

logger = logging.getLogger(__name__)

# Try importing Sionna/TensorFlow (only for data generation)
try:
    import tensorflow as tf
    import sionna
    from sionna.rt import Scene, PlanarArray, Transmitter, Receiver
    SIONNA_AVAILABLE = True
    TF_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    TF_AVAILABLE = False
    logger.warning("Sionna/TF not available - feature extractors will operate in mock mode")

def _to_numpy(v: Any) -> np.ndarray:
    """Helper to convert Any (Tensor or Array) to NumPy array."""
    if v is None:
        return None
    if hasattr(v, 'numpy'):
        return v.numpy()
    if isinstance(v, (list, tuple)):
        return np.array(v)
    return v

@dataclass
class RTLayerFeatures:
    """
    Layer 1: Ray Tracing / Physical Layer Features
    Direct outputs from Sionna RT propagation simulation
    """
    # Path-level features (per ray/path)
    path_gains: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] complex amplitudes
    path_delays: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] time-of-arrival (seconds)
    path_aoa_azimuth: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] angle-of-arrival azimuth (radians)
    path_aoa_elevation: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] angle-of-arrival elevation (radians)
    path_aod_azimuth: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] angle-of-departure azimuth (radians)
    path_aod_elevation: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] angle-of-departure elevation (radians)
    path_doppler: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] Doppler shift (Hz)
    
    # Aggregate statistics (derived from paths)
    rms_delay_spread: Union[np.ndarray, Any]  # [batch, num_rx] RMS-DS (seconds)
    rms_angular_spread: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx] RMS Angular Spread (radians)
    k_factor: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] Rician K-factor (dB)
    num_paths: Union[np.ndarray, Any] = field(default_factory=lambda: np.array([]))  # [batch, num_rx] path count
    
    # Positioning Features
    toa: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx] Time of Arrival (s)
    is_nlos: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx] Non-Line-of-Sight boolean
    
    is_mock: bool = False
    
    # Metadata
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage (converts all to NumPy)."""
        
        # Convert all to numpy first
        rms_ds = _to_numpy(self.rms_delay_spread)
        n_paths = _to_numpy(self.num_paths)
        
        result = {
            'rt/rms_delay_spread': rms_ds,
            'rt/num_paths': n_paths,
        }
        
        # Only include K-factor if computed
        if self.k_factor is not None:
             k_fac = _to_numpy(self.k_factor)
             # Check size or just add if not None (size check tricky if scalar)
             if hasattr(k_fac, 'size') and k_fac.size > 0:
                 result['rt/k_factor'] = k_fac
             elif not hasattr(k_fac, 'size'): # Scalar
                 result['rt/k_factor'] = k_fac

        if self.rms_angular_spread is not None:
             result['rt/rms_angular_spread'] = _to_numpy(self.rms_angular_spread)
        
        if self.toa is not None:
             result['rt/toa'] = _to_numpy(self.toa)
             
        if self.is_nlos is not None:
             # Convert boolean to int for zarr storage? Or keep bool.
             result['rt/is_nlos'] = _to_numpy(self.is_nlos)
             
        # Add path-level features for validation
        if self.path_gains is not None:
             result['rt/path_gains'] = _to_numpy(self.path_gains)
             
        if self.path_delays is not None:
             result['rt/path_delays'] = _to_numpy(self.path_delays)
             
        # logger.debug(f"Skipping path-level arrays for Zarr storage (num_paths: {self.num_paths})")
        return result


@dataclass
class PHYFAPILayerFeatures:
    """
    Layer 2: PHY/FAPI (L1 Measurements)
    Channel quality and link-level measurements per 3GPP FAPI spec
    """
    # FAPI Measurements (3GPP 38.215)
    rsrp: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Reference Signal Received Power (dBm)
    rsrq: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Reference Signal Received Quality (dB)
    sinr: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Signal-to-Interference-plus-Noise (dB)
    
    # Link adaptation indicators (3GPP 38.214)
    cqi: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Channel Quality Indicator [0-15]
    ri: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Rank Indicator [1-8]
    pmi: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Precoding Matrix Indicator
    
    # Beam management (5G NR specific)
    l1_rsrp_beams: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx, num_beams] per-beam RSRP
    best_beam_ids: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx, K] top-K beam indices
    
    # Channel matrix (for research/validation)
    channel_matrix: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ...]
    
    # Advanced KPIs
    capacity_mbps: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx]
    condition_number: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx]
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage."""
        d = {
            'phy_fapi/rsrp': _to_numpy(self.rsrp),
            'phy_fapi/rsrq': _to_numpy(self.rsrq),
            'phy_fapi/sinr': _to_numpy(self.sinr),
            'phy_fapi/cqi': _to_numpy(self.cqi),
            'phy_fapi/ri': _to_numpy(self.ri),
            'phy_fapi/pmi': _to_numpy(self.pmi),
        }
        if self.l1_rsrp_beams is not None:
            d['phy_fapi/l1_rsrp_beams'] = _to_numpy(self.l1_rsrp_beams)
        if self.best_beam_ids is not None:
            d['phy_fapi/best_beam_ids'] = _to_numpy(self.best_beam_ids)
        if self.channel_matrix is not None:
            d['phy_fapi/channel_matrix'] = _to_numpy(self.channel_matrix)
        if self.capacity_mbps is not None:
            d['phy_fapi/capacity_mbps'] = _to_numpy(self.capacity_mbps)
        if self.condition_number is not None:
            d['phy_fapi/condition_number'] = _to_numpy(self.condition_number)
        return d


@dataclass
class MACRRCLayerFeatures:
    """
    Layer 3: MAC/RRC (System-Level)
    Network-level measurements and cell management
    """
    # Cell identification
    serving_cell_id: Union[np.ndarray, Any]  # [batch, num_rx] Physical Cell ID
    neighbor_cell_ids: Union[np.ndarray, Any]  # [batch, num_rx, K] K neighbor cell IDs
    
    # Timing and synchronization
    timing_advance: Union[np.ndarray, Any]  # [batch, num_rx] TA command (TA index)
    
    # Optional fields with defaults
    tracking_area_code: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] TAC
    
    # Throughput and performance (simulated)
    dl_throughput_mbps: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] downlink throughput
    ul_throughput_mbps: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] uplink throughput
    bler: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] block error rate
    
    # Handover and mobility
    handover_events: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] binary indicator
    time_since_handover: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx] seconds since last HO
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage."""
        d = {
            'mac_rrc/serving_cell_id': _to_numpy(self.serving_cell_id),
            'mac_rrc/neighbor_cell_ids': _to_numpy(self.neighbor_cell_ids),
            'mac_rrc/timing_advance': _to_numpy(self.timing_advance),
        }
        
        if self.tracking_area_code is not None:
            d['mac_rrc/tracking_area_code'] = _to_numpy(self.tracking_area_code)
        if self.dl_throughput_mbps is not None:
            d['mac_rrc/dl_throughput_mbps'] = _to_numpy(self.dl_throughput_mbps)
        if self.ul_throughput_mbps is not None:
            d['mac_rrc/ul_throughput_mbps'] = _to_numpy(self.ul_throughput_mbps)
        if self.bler is not None:
            d['mac_rrc/bler'] = _to_numpy(self.bler)
        if self.handover_events is not None:
            d['mac_rrc/handover_events'] = _to_numpy(self.handover_events)
        if self.time_since_handover is not None:
            d['mac_rrc/time_since_handover'] = _to_numpy(self.time_since_handover)
        return d


class RTFeatureExtractor:
    """
    Extracts Layer 1 (RT) features from Sionna propagation paths.
    """
    
    def __init__(self, 
                 carrier_frequency_hz: float = 3.5e9,
                 bandwidth_hz: float = 100e6,
                 compute_k_factor: bool = False):
        self.carrier_frequency_hz = carrier_frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.compute_k_factor = compute_k_factor
        
        logger.info(f"RTFeatureExtractor initialized: fc={carrier_frequency_hz/1e9:.2f} GHz, "
                   f"BW={bandwidth_hz/1e6:.1f} MHz")
    
    def extract(self, paths: Any, batch_size: int = None, num_rx: int = None) -> RTLayerFeatures:
        """
        Extract RT features. Supports GPU tensors natively.
        """
        if not SIONNA_AVAILABLE:
            logger.warning("Sionna not available - returning mock RT features")
            return self._extract_mock(batch_size=batch_size or 10, num_rx=num_rx or 4)
        
        try:
            # --- GPU Acceleration Logic ---
            is_tensor_input = False
            if TF_AVAILABLE and hasattr(paths, 'a') and tf.is_tensor(paths.a):
                is_tensor_input = True
            
            if is_tensor_input:
                return self._extract_tf(paths, batch_size, num_rx)
            else:
                return self._extract_numpy(paths, batch_size, num_rx)
                
        except Exception as e:
            logger.error(f"Failed to extract RT features from Sionna paths: {e}")
            logger.warning("Falling back to mock RT features")
            return self._extract_mock(batch_size=batch_size or 10, num_rx=num_rx or 4)

    def _extract_tf(self, paths: Any, batch_size: int, num_rx: int) -> RTLayerFeatures:
        """Process features using TensorFlow operations on GPU."""
        # paths.a shape: [sources, targets, paths, rx_ant, tx_ant, 1, 1]
        raw_a = paths.a
        raw_tau = paths.tau # [sources, targets, paths]
        
        # Magnitude
        mag_a = tf.abs(raw_a)
        
        # Reduce over antennas (Axes 3, 4) if present
        # Check rank
        rank = len(mag_a.shape)
        if rank >= 5:
            mag_a_red = tf.reduce_mean(mag_a, axis=[3, 4]) # Result: [S, T, P, 1, 1]
        else:
            mag_a_red = mag_a
            
        # Squeeze last dims if they are 1
        # Use reshape to be safe? Or simple squeeze.
        # Targets are usually dimension 1, Sources 0.
        # We want [Targets (Batch), Sources (Rx), Paths]
        
        # Current: [S, T, P]
        
        # Permute to [T, S, P]
        # Check standard Sionna output
        # If [S, T, P, ...], we perform transpose(1, 0, ...)
        
        # Helper to transpose first 2 dims
        def _transpose_st(x):
            # assume [S, T, ...]
            perm = [1, 0] + list(range(2, len(x.shape)))
            return tf.transpose(x, perm=perm)
        
        mag_a_fin = _transpose_st(mag_a_red)
        tau_fin = _transpose_st(raw_tau)
        
        # Squeeze trailing 1s
        # mag_a_fin might be [T, S, P, 1, 1]
        # We want [T, S, P]
        # Safe squeeze:
        shape = tf.shape(mag_a_fin)
        if len(mag_a_fin.shape) > 3:
             mag_a_fin = tf.reshape(mag_a_fin, [shape[0], shape[1], -1]) # [T, S, P]
        
        # Angles
        ang_r_az = _transpose_st(paths.phi_r)
        ang_r_el = _transpose_st(paths.theta_r)
        ang_t_az = _transpose_st(paths.phi_t)
        ang_t_el = _transpose_st(paths.theta_t)
        
        if hasattr(paths, 'doppler') and paths.doppler is not None:
             dop = _transpose_st(paths.doppler)
        else:
             dop = tf.zeros_like(tau_fin)

        # Aggregate Stats (TF)
        # RMS DS
        # Power per path
        powers = tf.square(mag_a_fin)
        total_power = tf.reduce_sum(powers, axis=-1, keepdims=True) + 1e-10
        p = powers / total_power
        mean_delay = tf.reduce_sum(p * tau_fin, axis=-1)
        mean_delay_sq = tf.reduce_sum(p * tf.square(tau_fin), axis=-1)
        rms_ds = tf.sqrt(tf.maximum(mean_delay_sq - tf.square(mean_delay), 0.0))
        
        # K Factor
        k_factor = None
        if self.compute_k_factor:
            los_power = powers[..., 0]
            nlos_power = tf.reduce_sum(powers[..., 1:], axis=-1) + 1e-10
            k_linear = los_power / nlos_power
            k_db = 10.0 * (tf.math.log(k_linear + 1e-10) / tf.math.log(10.0))
            k_factor = k_db

        # Angular Spread
        rms_as = self._compute_angular_spread_tf(mag_a_fin, ang_r_az)

        # Angular Spread
        rms_as = self._compute_angular_spread_tf(mag_a_fin, ang_r_az)

        # Num Paths
        threshold = 1e-13
        num_paths = tf.reduce_sum(tf.cast(mag_a_fin > threshold, tf.int32), axis=-1)
        
        return RTLayerFeatures(
            path_gains=mag_a_fin,
            path_delays=tau_fin,
            path_aoa_azimuth=ang_r_az,
            path_aoa_elevation=ang_r_el,
            path_aod_azimuth=ang_t_az,
            path_aod_elevation=ang_t_el,
            path_doppler=dop,
            rms_delay_spread=rms_ds,
            rms_angular_spread=rms_as,
            k_factor=k_factor,
            num_paths=num_paths,
            carrier_frequency_hz=self.carrier_frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            is_mock=False
        )
        
    def _extract_numpy(self, paths: Any, batch_size: int, num_rx: int) -> RTLayerFeatures:
        """Legacy NumPy extraction."""
        # Convert all to numpy first
        def _to_numpy(x): return x.numpy() if hasattr(x, 'numpy') else np.array(x)
        def _to_complex(x):
             # handle tuples
             if isinstance(x, tuple) and len(x) == 2:
                  return _to_numpy(x[0]) + 1j * _to_numpy(x[1])
             return _to_numpy(x)

        # Basic extraction logic from original code...
        path_gains_complex = _to_complex(paths.a)
        path_delays = _to_numpy(paths.tau)
        
        # Normalize shapes to [Batch(T), Rx(S), Paths, ...]
        # Standard Sionna is [S, T, P, ...]
        # We swap 0 and 1 to get [T, S, P, ...]
        if path_gains_complex.ndim >= 2:
             path_gains_complex = np.swapaxes(path_gains_complex, 0, 1)
             path_delays = np.swapaxes(path_delays, 0, 1)
        
        # Reduce antennas and time
        # We want final shape [Batch, Rx, Paths]
        # Recursively mean over extra dimensions
        while path_gains_complex.ndim > 3:
             path_gains_complex = np.mean(path_gains_complex, axis=3)
        
        path_gains_magnitude = np.abs(path_gains_complex)
        
        # Ensure delays match
        while path_delays.ndim > 3:
             path_delays = np.mean(path_delays, axis=3)
             
        # Angles
        p_az = np.swapaxes(_to_numpy(paths.phi_r), 0, 1)
        p_el = np.swapaxes(_to_numpy(paths.theta_r), 0, 1)
        t_az = np.swapaxes(_to_numpy(paths.phi_t), 0, 1)
        t_el = np.swapaxes(_to_numpy(paths.theta_t), 0, 1)
        
        # Doppler
        if hasattr(paths, 'doppler') and paths.doppler is not None:
             dop = np.swapaxes(_to_numpy(paths.doppler), 0, 1)
        else:
             dop = np.zeros_like(path_delays)
             
        # Stats
        rms_ds = self._compute_rms_ds(path_gains_magnitude, path_delays)
        k_factor = None
        if self.compute_k_factor:
             k_factor = self._compute_k_factor(path_gains_magnitude)
             
        rms_as = self._compute_angular_spread_numpy(path_gains_magnitude, p_az)

        num_paths = np.sum(path_gains_magnitude > 1e-13, axis=-1)
        
        return RTLayerFeatures(
            path_gains=path_gains_magnitude,
            path_delays=path_delays,
            path_aoa_azimuth=p_az,
            path_aoa_elevation=p_el,
            path_aod_azimuth=t_az,
            path_aod_elevation=t_el,
            path_doppler=dop,
            rms_delay_spread=rms_ds,
            rms_angular_spread=rms_as,
            k_factor=k_factor,
            num_paths=num_paths,
            carrier_frequency_hz=self.carrier_frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            is_mock=False
        )

    def _compute_rms_ds(self, gains: np.ndarray, delays: np.ndarray) -> np.ndarray:
        # NumPy version
        powers = np.abs(gains)**2
        total_power = np.sum(powers, axis=-1, keepdims=True) + 1e-10
        p = powers / total_power
        mean_delay = np.sum(p * delays, axis=-1)
        mean_delay_sq = np.sum(p * delays**2, axis=-1)
        rms_ds = np.sqrt(np.maximum(mean_delay_sq - mean_delay**2, 0))
        return rms_ds
    
    def _compute_angular_spread_tf(self, gains: Any, az_angles: Any) -> Any:
        """Compute RMS Angular Spread (Azimuth) using circular statistics."""
        # gains: [Batch, Rx, Paths]
        # az_angles: [Batch, Rx, Paths] (radians)
        
        powers = tf.square(tf.abs(gains))
        total_power = tf.reduce_sum(powers, axis=-1, keepdims=True) + 1e-10
        weights = powers / total_power
        
        # Mean vector
        # R = sum(w * exp(j * phi))
        mean_vec = tf.reduce_sum(weights * tf.complex(tf.cos(az_angles), tf.sin(az_angles)), axis=-1)
        
        # Circular variance = 1 - |R|
        # Circular Std Dev = sqrt(-2 * ln(|R|))
        r_abs = tf.abs(mean_vec)
        # Numerical stability: clamp R to [0, 1]
        r_abs = tf.minimum(r_abs, 1.0 - 1e-7)
        r_abs = tf.maximum(r_abs, 1e-7)
        
        rms_as = tf.sqrt(-2.0 * tf.math.log(r_abs))
        return rms_as

    def _compute_angular_spread_numpy(self, gains: np.ndarray, az_angles: np.ndarray) -> np.ndarray:
        """Compute RMS Angular Spread (Azimuth) using circular statistics."""
        powers = np.abs(gains)**2
        total_power = np.sum(powers, axis=-1, keepdims=True) + 1e-10
        weights = powers / total_power
        
        mean_vec = np.sum(weights * np.exp(1j * az_angles), axis=-1)
        r_abs = np.abs(mean_vec)
        r_abs = np.clip(r_abs, 1e-7, 1.0 - 1e-7)
        
        rms_as = np.sqrt(-2.0 * np.log(r_abs))
        return rms_as

    def _compute_k_factor(self, gains: np.ndarray) -> np.ndarray:
        powers = np.abs(gains)**2
        los_power = powers[..., 0]
        nlos_power = np.sum(powers[..., 1:], axis=-1) + 1e-10
        k_linear = los_power / nlos_power
        k_db = 10 * np.log10(k_linear + 1e-10)
        return k_db
    
    def _extract_mock(self, batch_size: int = 10, num_rx: int = 4) -> RTLayerFeatures:
        """Mock features."""
        num_paths = 50
        return RTLayerFeatures(
            path_gains=np.random.randn(batch_size, num_rx, num_paths) + 
                      1j * np.random.randn(batch_size, num_rx, num_paths),
            path_delays=np.sort(np.random.uniform(0, 1e-6, (batch_size, num_rx, num_paths)), axis=-1),
            path_aoa_azimuth=np.random.uniform(0, 2*np.pi, (batch_size, num_rx, num_paths)),
            path_aoa_elevation=np.random.uniform(-np.pi/2, np.pi/2, (batch_size, num_rx, num_paths)),
            path_aod_azimuth=np.random.uniform(0, 2*np.pi, (batch_size, num_rx, num_paths)),
            path_aod_elevation=np.random.uniform(-np.pi/2, np.pi/2, (batch_size, num_rx, num_paths)),
            path_doppler=np.random.uniform(-100, 100, (batch_size, num_rx, num_paths)),
            rms_delay_spread=np.random.uniform(10e-9, 200e-9, (batch_size, num_rx)),
            rms_angular_spread=np.random.uniform(0.1, 0.5, (batch_size, num_rx)),
            k_factor=np.random.uniform(-5, 15, (batch_size, num_rx)),
            num_paths=np.random.randint(10, num_paths, (batch_size, num_rx)),
            carrier_frequency_hz=self.carrier_frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            is_mock=True,
        )


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
        
        if pilot_re_indices is None:
            pilot_re_indices = np.arange(0, 100, 4)
            if TF_AVAILABLE and hasattr(channel_matrix, 'shape') and tf.is_tensor(channel_matrix):
                pilot_re_indices = tf.range(0, 100, 4)
        
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
            
            # Use utility - auto-detects TF
            rsrp_dbm = compute_rsrp(channel_matrix, 0, pilot_re_indices)
            
            # The utility returns [batch, rx, tx?]. 
            # We want [batch, rx, cells=1?] (since channel_matrix is usually one cell)
            # If utility sums over Tx ants, it returns per-cell-site RSRP.
            # If channel_matrix was just for serving cell, result is [batch, rx].
            # We need to unsqueeze to [batch, rx, 1]
            
            # Helper to unsqueeze if needed (works for TF and NP)
            def _unsqueeze_last(x):
                if hasattr(x, 'shape') and len(x.shape) == 2:
                    if tf.is_tensor(x): return tf.expand_dims(x, -1)
                    return x[..., np.newaxis]
                return x
            
            rsrp_dbm = _unsqueeze_last(rsrp_dbm)
            
        else:
            # Approximate
            # Handle Tensor case for path_gains
            gains = rt_features.path_gains
            if TF_AVAILABLE and tf.is_tensor(gains):
                total_power = tf.reduce_sum(tf.square(tf.abs(gains)), axis=-1)
                rsrp_val = 10.0 * (tf.math.log(total_power + 1e-10) / tf.math.log(10.0)) + 30.0
                rsrp_dbm = tf.expand_dims(rsrp_val, -1)
            else:
                total_power = np.sum(np.abs(gains)**2, axis=-1)
                rsrp_val = 10 * np.log10(total_power + 1e-10) + 30
                rsrp_dbm = rsrp_val[..., np.newaxis]
        
        # Use Sionna MIMO Capacity for CQI/RI
        cqi = None
        ri = None
        sinr = None # Initialize sinr here for the new logic path
        
        if SIONNA_AVAILABLE and channel_matrix is not None and TF_AVAILABLE and tf.is_tensor(channel_matrix):
            from sionna.mimo import capacity_optimal
            
            # channel_matrix: [Targets(Batch), RxAnt, Sources, TxAnt, Freq]
            # capacity_optimal expects H: [batch, num_rx, num_tx, ...]? 
            # It expects [..., num_rx, num_tx] as last dims usually?
            # Or [..., num_rx_ant, num_tx_ant].
            # Let's reshape to [Batch, Freq, RxAnt, TxAnt] (assuming Single Serving Cell source)
            
            # Pick serving cell (Sources=0)
            # h_serv: [Batch, RxAnt, 0, TxAnt, Freq] -> [Batch, RxAnt, TxAnt, Freq]
            # Wait, `channel_matrix` as generated in multi_layer_generator is [Targets, RxAnt, Sources, TxAnt, F]
            # targets=batch.
            
            h_serv = channel_matrix[:, :, 0, :, :] # Slice Source 0
            # [Batch, RxAnt, TxAnt, Freq]
            
            # Sionna usually wants [Batch, ..., Rx, Tx]
            # Permute to [Batch, Freq, RxAnt, TxAnt] allows batching over Freq?
            # capacity_optimal: inputs h defined as [..., num_rx, num_tx]. 
            # It broadcasts over batch dims.
            # So passing [Batch, Freq, RxAnt, TxAnt] works.
            
            h_perm = tf.transpose(h_serv, perm=[0, 3, 1, 2]) # [Batch, Freq, RxAnt, TxAnt]
            
            # Compute Capacity (bits/s/Hz)
            # n0 is spectral density? or noise variance per complex symbol?
            # capacity_optimal(h, n0). n0 is variance of noise.
            # If h is channel transfer function, and we didn't normalize power yet?
            # Assuming h includes path loss.
            # n0 should be noise power per subcarrier? Or density?
            # Usually input is SNR, or H and N0. 
            # N0 per resource element. = Thermal + NF (linear) / FFT_SIZE?
            # Or N0 density * SubcarrierSpacing.
            
            # N0_per_RE = k*T*F * NF * SCS
            n0_eff = noise_power_linear / float(h_perm.shape[1])
            
            cap_bits = capacity_optimal(h_perm, n0_eff)
            # Result: [Batch, Freq] capacity per tone
            
            # Average SE over bandwidth
            se_avg = tf.reduce_mean(cap_bits, axis=-1) # [Batch]
            
            # Map SE to CQI (Approximate: CQI = SE * 2 roughly? Or table lookup)
            # 3GPP SE Mapping Table (Table 1 64QAM)
            # CQI 15 -> 5.5 bits/s/Hz roughly
            # CQI 1 -> 0.15
            # Simple linear scalar for now or utility
            # SE = 0.2 * CQI? -> CQI 15 = 3.0. Max theoretical MIMO is higher.
            # We used `compute_cqi` before based on SINR.
            # Let's derive effective SINR from Capacity: C = log2(1 + SINR_eff) -> SINR_eff = 2^C - 1
            
            sinr_eff_linear = tf.pow(2.0, se_avg) - 1.0
            sinr_eff_db = 10.0 * (tf.math.log(sinr_eff_linear + 1e-10) / tf.math.log(10.0))
            
            # Expand to [Batch, 1, 1] for compat with legacy shape
            sinr_eff_db = tf.reshape(sinr_eff_db, [-1, 1, 1])
            
            cqi = compute_cqi(sinr_eff_db)
            
            # SINR (legacy field)
            sinr = sinr_eff_db
            
            # RI Estimate? 
            # Rank of channel mean?
            h_mean = tf.reduce_mean(h_perm, axis=1) # [Batch, Rx, Tx]
            s = tf.linalg.svd(h_mean, compute_uv=False)
            ri = tf.reduce_sum(tf.cast(s > 1e-9, tf.int32), axis=-1, keepdims=True)
            # broadcast to [batch, 1, 1]
            ri = tf.reshape(ri, [-1, 1, 1])

        else:
             # Fallback Legacy
             pass
             
        # RSSI
        # rssi_linear = 10^((rsrp - 30)/10) + noise
        def _to_linear_dbm(dbm_val):
            return 10.0**((dbm_val - 30.0) / 10.0)
            
        def _to_dbm_linear(lin_val):
            return 10.0 * np.log10(lin_val + 1e-10) + 30.0
            
        if TF_AVAILABLE and tf.is_tensor(rsrp_dbm):
            rssi_linear = tf.pow(10.0, (rsrp_dbm - 30.0)/10.0) + noise_power_linear
            rssi_dbm = 10.0 * (tf.math.log(rssi_linear)/tf.math.log(10.0)) + 30.0
        else:
            rssi_linear = 10**((rsrp_dbm - 30) / 10) + noise_power_linear
            rssi_dbm = 10 * np.log10(rssi_linear) + 30
            
        # RSRQ
        if channel_matrix is None:
             num_cells = rsrp_dbm.shape[-1]
             
        rsrq = compute_rsrq(rsrp_dbm, rssi_dbm, N=12)
        
        # SINR
        if channel_matrix is not None:
            sinr = compute_sinr(channel_matrix, noise_power_linear, interference_matrices)
            # Unsqueeze if needed
            if len(sinr.shape) == 2:
                if TF_AVAILABLE and tf.is_tensor(sinr):
                    sinr = tf.expand_dims(sinr, -1)
                else:
                    sinr = sinr[..., np.newaxis]
        else:
            sinr = rsrp_dbm - noise_power_dbm

        # CQI
        cqi = compute_cqi(sinr)
        
        # RI
        if channel_matrix is not None:
            # Average spatial
            # If TF
            if TF_AVAILABLE and tf.is_tensor(channel_matrix):
                h_spatial = tf.reduce_mean(channel_matrix, axis=[-2, -1])
            else:
                h_spatial = np.mean(channel_matrix, axis=(-2, -1))
            ri = compute_rank_indicator(h_spatial)
        else:
            if TF_AVAILABLE and tf.is_tensor(rsrp_dbm):
                ri = tf.ones(tf.shape(rsrp_dbm), dtype=tf.int32)
            else:
                ri = np.ones(rsrp_dbm.shape, dtype=np.int32)
        
        # PMI
        if channel_matrix is not None:
             if TF_AVAILABLE and tf.is_tensor(channel_matrix):
                 h_spatial = tf.reduce_mean(channel_matrix, axis=[-2, -1])
             else:
                 h_spatial = np.mean(channel_matrix, axis=(-2, -1))
             pmi = compute_pmi(h_spatial)
        else:
             if TF_AVAILABLE and tf.is_tensor(rsrp_dbm):
                 pmi = tf.zeros(tf.shape(rsrp_dbm), dtype=tf.int32)
             else:
                 pmi = np.zeros(rsrp_dbm.shape, dtype=np.int32)
                 
        # Beam Management
        l1_rsrp_beams = None
        best_beam_ids = None
        if self.enable_beam_management:
            l1_rsrp_beams = compute_beam_rsrp(rt_features.path_gains, None, self.num_beams)
            # Top-4
            if TF_AVAILABLE and tf.is_tensor(l1_rsrp_beams):
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
        
        # Serving Cell: max RSRP
        # Logic: Flatten rx/cells dims to find global max
        if TF_AVAILABLE and tf.is_tensor(rsrp):
            # rsrp: [batch, num_sites, num_cells_per_site]
            shape = tf.shape(rsrp)
            batch_size = shape[0]
            num_sites = shape[1]
            # rsrp expected shape: [Batch, Rx, Cells]
            # Check shapes to be robust against [Batch, Cells, 1] vs [Batch, Rx, Cells]
            shape = rsrp.shape
            
            # If shape is [Batch, Cells, 1], we interpret as [Batch, Rx=Cells, Cells=1]?
            # Or is it [Batch, Cells] with 1 expanded?
            # Standard: [Batch, Rx, Cells].
            # If we have [Batch, Cells, 1], it means valid cells are on dim 1.
            # We want to argmax over Cells dimension (last).
            # If last dim is 1, argmax is 0.
            # This implies the input was structured [Batch, Cells, 1] where Rx=Cells??
            # Let's permute to [Batch, Rx=1, Cells]?
            
            # Heuristic: If dim 2 is 1 and dim 1 > 1, assume [Batch, Cells, 1] -> swap to [Batch, 1, Cells]
            if len(shape) == 3 and shape[2] == 1 and shape[1] > 1:
                rsrp = tf.transpose(rsrp, perm=[0, 2, 1]) # [Batch, 1, Cells]
                
            # Now rsrp is [Batch, Rx=1, Cells] (or [Batch, Rx, Cells])
            best_cell_idx = tf.argmax(rsrp, axis=-1) # [Batch, Rx]
            
            # Map index to CellID
            cids = tf.convert_to_tensor(cell_ids, dtype=tf.int64)
            serving_cell_id = tf.gather(cids, best_cell_idx) # [Batch, Rx]
            
            # Neighbors
            k = min(self.max_neighbors, rsrp.shape[-1])
            vals, inds = tf.math.top_k(rsrp, k=k)
            neighbor_cell_ids = tf.gather(cids, inds) # [Batch, Rx, K]
            
        else:
            # Numpy
            if rsrp.ndim == 3 and rsrp.shape[2] == 1 and rsrp.shape[1] > 1:
                 rsrp = np.swapaxes(rsrp, 1, 2)
            
            best_cell_idx = np.argmax(rsrp, axis=-1)
            serving_cell_id = cell_ids[best_cell_idx]
            
            k = min(self.max_neighbors, rsrp.shape[-1])
            # neighbor_cell_ids logic similar
            inds = np.argsort(-rsrp, axis=-1)[..., :k]
            neighbor_cell_ids = cell_ids[inds]

        # Timing Advance
        if TF_AVAILABLE and tf.is_tensor(ue_positions):
             ue_pos = tf.cast(ue_positions, tf.float32)
             site_pos = tf.convert_to_tensor(site_positions, dtype=tf.float32)
             
             # best_cell_idx: [Batch, Rx]
             # serving_site_pos = gather(site_pos, best_cell_idx) -> [Batch, Rx, 3]
             serving_site_pos = tf.gather(site_pos, best_cell_idx)
             
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
             if len(cqi.shape) == 3 and cqi.shape[2] == 1 and cqi.shape[1] > 1:
                  cqi = tf.transpose(cqi, perm=[0, 2, 1])
                  
             # Select serving cell CQI: [Batch, Rx]
             # gather from last dim using best_cell_idx
             dl_throughput = self._simulate_throughput(cqi)
             dl_t_serv = tf.gather(dl_throughput, best_cell_idx, batch_dims=2) # batch_dims=2 handles [B, R]
             
        else:
             # Numpy path
             serving_pos = site_positions[best_cell_idx] # [Batch, Rx, 3]
             
             u_pos = ue_positions
             if u_pos.ndim == 2: u_pos = u_pos[:, np.newaxis, :] # [Batch, 1, 3]
             
             dist = np.linalg.norm(u_pos - serving_pos, axis=-1) # [Batch, Rx]
             timing_advance = compute_timing_advance(dist)
             
             cqi = phy_features.cqi
             if cqi.ndim == 3 and cqi.shape[2] == 1 and cqi.shape[1] > 1:
                  cqi = np.swapaxes(cqi, 1, 2)
                  
             dl_throughput = self._simulate_throughput(cqi)
             dl_t_serv = np.take_along_axis(dl_throughput, best_cell_idx[...,None], axis=-1).squeeze(-1)
             
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


if __name__ == "__main__":
    pass
