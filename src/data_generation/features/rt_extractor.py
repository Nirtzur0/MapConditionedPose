"""
RT Feature Extractor
Extracts Layer 1 (RT) features from Sionna propagation paths.
"""

import numpy as np
import logging
from typing import Optional, Any, Tuple

from .tensor_ops import TensorOps, get_ops
from .data_structures import RTLayerFeatures

logger = logging.getLogger(__name__)

# Try importing Sionna/TensorFlow
try:
    import tensorflow as tf
    import sionna
    from sionna.rt import Scene, PlanarArray, Transmitter, Receiver
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False


class RTFeatureExtractor:
    """
    Extracts Layer 1 (RT) features from Sionna propagation paths.
    """
    
    def __init__(self, 
                 carrier_frequency_hz: float = 3.5e9,
                 bandwidth_hz: float = 100e6,
                 compute_k_factor: bool = False,
                 max_stored_paths: int = 256,
                 max_stored_sites: int = 16,
                 allow_mock_fallback: bool = True):
        self.carrier_frequency_hz = carrier_frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.compute_k_factor = compute_k_factor
        
        self.max_stored_paths = max_stored_paths
        self.max_stored_sites = max_stored_sites
        self.allow_mock_fallback = allow_mock_fallback
        self.logger = logging.getLogger(__name__)
        
        logger.info(f"RTFeatureExtractor initialized: fc={carrier_frequency_hz/1e9:.2f} GHz, "
                   f"BW={bandwidth_hz/1e6:.1f} MHz")
    
    def extract(self, paths: Any, batch_size: int = None, num_rx: int = None) -> RTLayerFeatures:
        """
        Extract RT features using unified TensorOps.
        """
        if not SIONNA_AVAILABLE:
            if not self.allow_mock_fallback:
                raise RuntimeError("Sionna not available and mock fallback is disabled.")
            logger.warning("Sionna not available - returning mock RT features")
            return self._extract_mock(batch_size=batch_size or 10, num_rx=num_rx or 4)
        
        try:
            ops = get_ops(paths)
            return self._extract_generic(ops, paths, batch_size, num_rx)
                
        except Exception as e:
            logger.error(f"Failed to extract RT features from Sionna paths: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if not self.allow_mock_fallback:
                raise
            logger.warning("Falling back to mock RT features")
            return self._extract_mock(batch_size=batch_size or 10, num_rx=num_rx or 4)

    def _extract_generic(self, ops: TensorOps, paths: Any, batch_size: int, num_rx: int) -> RTLayerFeatures:
        """Generic feature extraction using TensorOps."""
        
        # Helper: Convert complex if needed (mostly for NumPy tuples from Sionna)
        def _ensure_complex(x):
            if isinstance(x, tuple) and len(x) == 2:
                # Assuming (real, imag) tuple
                r, i = x
                if hasattr(r, 'tf'):
                    r = r.tf()
                if hasattr(i, 'tf'):
                    i = i.tf()
                return ops.complex(r, i)
            if hasattr(x, 'tf'):
                return x.tf()
            return x

        def _ensure_real(x):
            if hasattr(x, 'tf'):
                return x.tf()
            return x

        # 1. Get Raw Data
        raw_a = _ensure_complex(paths.a)
        raw_tau = _ensure_real(paths.tau)
        
        # Angles helpers
        phi_r = _ensure_real(paths.phi_r)
        theta_r = _ensure_real(paths.theta_r)
        phi_t = _ensure_real(paths.phi_t)
        theta_t = _ensure_real(paths.theta_t)
        
        # Doppler
        doppler = _ensure_real(getattr(paths, 'doppler', None))
        
        # 2. Magnitude
        mag_a = ops.abs(raw_a)
        
        # 3. Reduction (Sionna 4D/5D -> [S, T, P])
        # Sionna format: [sources, targets, paths, rx_ant, tx_ant, 1, 1] usually
        # We want to average over antennas (rx_ant, tx_ant) if present
        # Target shape before transpose: [Sources, Targets, Paths]
        
        def _reduce_to_3d(x):
            # Reduce dims > 3
            curr_x = x
            # Loop safely
            while len(ops.shape(curr_x)) > 3:
                # Reduce the LAST dimension
                # Check for zero size to avoid RuntimeWarning (Mean of empty slice)
                if ops.shape(curr_x)[-1] == 0:
                    curr_x = ops.sum(curr_x, axis=-1)
                else:
                    curr_x = ops.mean(curr_x, axis=-1)
            return curr_x

        mag_a_red = _reduce_to_3d(mag_a)
        raw_tau_red = _reduce_to_3d(raw_tau)
        
        ang_r_az = _reduce_to_3d(phi_r)
        ang_r_el = _reduce_to_3d(theta_r)
        ang_t_az = _reduce_to_3d(phi_t)
        ang_t_el = _reduce_to_3d(theta_t)
        
        if doppler is not None:
             dop_red = _reduce_to_3d(doppler)
        else:
             dop_red = None # Handle later

        # 4. Standardize Layout to [Batch(Target), Rx(Source), Path]
        # Current (Sionna): [Source, Target, Path] usually
        # We check dimensions. If Dim 0 != batch_size and Dim 1 == batch_size, we swap.
        
        def _standardize_layout(x):
             # Assume x is [D0, D1, D2]
             shape = ops.shape(x)
             if len(shape) < 3: return x
             
             # Heuristic: If D0 matches expected batch_size, keep as is
             if batch_size is not None and shape[0] == batch_size:
                 return x

             # Heuristic: If D1 matches expected batch_size, swap
             if batch_size is not None and shape[1] == batch_size:
                  return ops.transpose(x, perm=[1, 0, 2])
             
             # Default fallback: Transpose 0 and 1 anyway (Sionna [Src, Tgt] -> [Tgt, Src])
             return ops.transpose(x, perm=[1, 0, 2])

        mag_a_fin = _standardize_layout(mag_a_red)
        tau_fin = _standardize_layout(raw_tau_red)
        
        ang_r_az = _standardize_layout(ang_r_az)
        ang_r_el = _standardize_layout(ang_r_el)
        ang_t_az = _standardize_layout(ang_t_az)
        ang_t_el = _standardize_layout(ang_t_el)
        
        if dop_red is not None:
             dop_fin = _standardize_layout(dop_red)
        else:
             # Create zeros like tau
             if ops.is_tensor(tau_fin):
                  # TFOps doesn't have zeros_like exposed, use multiply by 0
                  # dop_fin = ops.to_numpy(tau_fin) * 0.0 # Hack: ops mix? No.
                  # Pure generic way: 
                  # Implementation detail: generic ops doesn't support creation yet.
                  # Use multiply by 0
                  try:
                       dop_fin = tau_fin * 0.0
                  except:
                       dop_fin = tau_fin # unsafe
             else:
                  dop_fin = np.zeros_like(tau_fin)

        # 5. Robust Alignment (Pad/Truncate to Max Dimensions)
        def _enforce_shape(t, num_sites=self.max_stored_sites, num_paths=self.max_stored_paths):
            if t is None: return None
            
            shape = ops.shape(t)
            curr_sites = shape[1] if len(shape) > 1 else 1
            curr_paths = shape[2] if len(shape) > 2 else 1
            
            # --- Sites (Dim 1) ---
            if curr_sites > num_sites:
                t = t[:, :num_sites, ...] # Truncate
            elif curr_sites < num_sites:
                # Pad
                pad_amt = num_sites - curr_sites
                paddings = [[0,0]] * len(shape)
                paddings[1] = [0, pad_amt]
                t = ops.pad(t, paddings, mode='CONSTANT', constant_values=0)
                
            # --- Paths (Dim 2) ---
            # Recheck shape after dim 1 change
            shape = ops.shape(t)
            if len(shape) > 2:
                if curr_paths > num_paths:
                    t = t[:, :, :num_paths, ...]
                elif curr_paths < num_paths:
                    pad_amt = num_paths - curr_paths
                    paddings = [[0,0]] * len(shape)
                    paddings[2] = [0, pad_amt]
                    
                    # Special case for delays/angles: pad with 0? 
                    # Yes, 0 gain/delay usually indicates no path.
                    t = ops.pad(t, paddings, mode='CONSTANT', constant_values=0)
                    
            return t

        mag_a_fin = _enforce_shape(mag_a_fin)
        tau_fin = _enforce_shape(tau_fin)
        ang_r_az = _enforce_shape(ang_r_az)
        ang_r_el = _enforce_shape(ang_r_el)
        ang_t_az = _enforce_shape(ang_t_az)
        ang_t_el = _enforce_shape(ang_t_el)
        dop_fin = _enforce_shape(dop_fin)

        # Sanitize NaN/Inf values from RT to avoid propagating invalid stats.
        if ops.is_tensor(mag_a_fin):
            import tensorflow as tf
            mag_a_fin = tf.where(tf.math.is_finite(mag_a_fin), mag_a_fin, tf.zeros_like(mag_a_fin))
            tau_fin = tf.where(tf.math.is_finite(tau_fin), tau_fin, tf.zeros_like(tau_fin))
            ang_r_az = tf.where(tf.math.is_finite(ang_r_az), ang_r_az, tf.zeros_like(ang_r_az))
            ang_r_el = tf.where(tf.math.is_finite(ang_r_el), ang_r_el, tf.zeros_like(ang_r_el))
            ang_t_az = tf.where(tf.math.is_finite(ang_t_az), ang_t_az, tf.zeros_like(ang_t_az))
            ang_t_el = tf.where(tf.math.is_finite(ang_t_el), ang_t_el, tf.zeros_like(ang_t_el))
            dop_fin = tf.where(tf.math.is_finite(dop_fin), dop_fin, tf.zeros_like(dop_fin))
        else:
            mag_a_fin = np.nan_to_num(mag_a_fin, nan=0.0, posinf=0.0, neginf=0.0)
            tau_fin = np.nan_to_num(tau_fin, nan=0.0, posinf=0.0, neginf=0.0)
            ang_r_az = np.nan_to_num(ang_r_az, nan=0.0, posinf=0.0, neginf=0.0)
            ang_r_el = np.nan_to_num(ang_r_el, nan=0.0, posinf=0.0, neginf=0.0)
            ang_t_az = np.nan_to_num(ang_t_az, nan=0.0, posinf=0.0, neginf=0.0)
            ang_t_el = np.nan_to_num(ang_t_el, nan=0.0, posinf=0.0, neginf=0.0)
            dop_fin = np.nan_to_num(dop_fin, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. Aggregate Stats
        
        # RMS Delay Spread
        # Power weighted
        powers = ops.square(mag_a_fin)
        # Add epsilon to total_power to avoid division by zero
        total_power = ops.sum(powers, axis=-1, keepdims=True) + 1e-10
        p = powers / total_power
        
        mean_delay = ops.sum(p * tau_fin, axis=-1)
        mean_delay_sq = ops.sum(p * ops.square(tau_fin), axis=-1)
        
        # Variance = E[X^2] - (E[X])^2
        var_delay = mean_delay_sq - ops.square(mean_delay)
        # Numerical safe sqrt
        var_delay = ops.clip(var_delay, 0.0, 1e9) # clip min 0
        rms_ds = ops.sqrt(var_delay)
        
        # K-Factor
        k_factor = None
        if self.compute_k_factor:
             los_power = powers[..., 0] # Assume 0 is LoS
             nlos_power = ops.sum(powers[..., 1:], axis=-1) + 1e-10
             k_linear = los_power / nlos_power
             k_db = 10.0 * ops.log10(k_linear + 1e-10)
             k_factor = k_db
             
        # Angular Spread
        rms_as = self._compute_angular_spread_generic(ops, mag_a_fin, ang_r_az)
        
        # Num Paths (threshold)
        # Casting boolean to int/float requires specific backend logic usually
        # Ops doesn't have 'cast'.
        # We can implement a clean 'count_above' in ops? Or just handle here.
        if ops.is_tensor(mag_a_fin):
             # TF
             import tensorflow as tf
             num_paths = tf.reduce_sum(tf.cast(mag_a_fin > 1e-13, tf.int32), axis=-1)
        else:
             # NP
             num_paths = np.sum(mag_a_fin > 1e-13, axis=-1)
             
        return RTLayerFeatures(
            path_gains=mag_a_fin,
            path_delays=tau_fin,
            path_aoa_azimuth=ang_r_az,
            path_aoa_elevation=ang_r_el,
            path_aod_azimuth=ang_t_az,
            path_aod_elevation=ang_t_el,
            path_doppler=dop_fin,
            rms_delay_spread=rms_ds,
            rms_angular_spread=rms_as,
            k_factor=k_factor,
            num_paths=num_paths,
            carrier_frequency_hz=self.carrier_frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            is_mock=False
        )

    def _compute_angular_spread_generic(self, ops: TensorOps, gains: Any, az_angles: Any) -> Any:
        """Compute RMS AS using circular statistics with Generic Ops."""
        powers = ops.square(ops.abs(gains))
        total_power = ops.sum(powers, axis=-1, keepdims=True) + 1e-10
        weights = powers / total_power
        
        # Mean vector R = sum(w * exp(j * phi))
        # exp(j*phi) = cos(phi) + j*sin(phi)
        
        # Since ops doesn't fully abstract cos/sin/complex yet perfectly for generic pipeline...
        # We rely on specific backend calls via 'is_tensor' check or expand ops.
        # Let's expand ops quickly in local scope or just use backend-check for math func.
        
        if ops.is_tensor(gains):
             import tensorflow as tf
             c = tf.cos(az_angles)
             s = tf.sin(az_angles)
             phasors = tf.complex(c, s)
             weights = tf.cast(weights, tf.complex64)
        else:
             phasors = np.exp(1j * ops.to_numpy(az_angles))
             
        mean_vec = ops.sum(weights * phasors, axis=-1)
        r_abs = ops.abs(mean_vec)
        
        # Clanp
        r_abs = ops.clip(r_abs, 1e-7, 1.0 - 1e-7)
        
        # -2 ln(R)
        rms_as = ops.sqrt(-2.0 * ops.log10(r_abs) * np.log(10.0)) # ops.log10 is base 10. ln(x) = log10(x) * ln(10)
        
        return rms_as

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
