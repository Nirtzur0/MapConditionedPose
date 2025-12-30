"""
Sionna Native KPI Extractor
Extracts RT and PHY features using strictly TensorFlow operations on GPU.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass

try:
    import sionna
    from sionna.rt import Paths
    SIONNA_AVAILABLE = True
except ImportError as e:
    SIONNA_AVAILABLE = False
    print(f"Sionna Import Failed: {e}")

logger = logging.getLogger(__name__)

class SionnaNativeKPIExtractor:
    """
    Extracts Key Performance Indicators (KPIs) directly from Sionna objects
    using TensorFlow operations to ensure end-to-end GPU execution.
    """
    
    def __init__(self, 
                 carrier_frequency_hz: float = 3.5e9,
                 bandwidth_hz: float = 100e6,
                 subcarrier_spacing: float = 30e3,
                 num_rx_ant: int = 1,
                 num_tx_ant: int = 1,
                 thermal_noise_density_dbm: float = -174.0,
                 noise_figure_db: float = 9.0):
        
        self.fc = carrier_frequency_hz
        self.bw = bandwidth_hz
        self.scs = subcarrier_spacing
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        
        # Noise Power Calculation
        # N0_linear = 10^((Density + NF - 30)/10) * BW
        n0_dbm_hz = thermal_noise_density_dbm + noise_figure_db
        self.n0_linear = (10.0 ** (n0_dbm_hz / 10.0)) * 1e-3 # Watts/Hz
        self.noise_power = self.n0_linear * bandwidth_hz # Total Noise Power (Watts)
        self.noise_power_dbm = 10.0 * np.log10(self.noise_power * 1e3)
        
        
        logger.info(f"SionnaNativeKPIExtractor: BW={bandwidth_hz/1e6:.1f}MHz, Noise Power={self.noise_power_dbm:.1f} dBm")
        
    def _compute_capacity(self, h, n0):
        """
        Compute Capacity of MIMO channel manually (Shannon Capacity).
        C = sum(log2(1 + snr_i))
        h: [..., Rx, Tx]
        n0: scalar or compatible tensor
        """
        # h shape: [..., Rx, Tx]
        rx = h.shape[-2]
        tx = h.shape[-1]
        
        # Ensure complex128 for precision in capacity if needed, or float32
        
        if tx == 1:
            # SIMO / SISO
            # ||h||^2 = sum(|h_i|^2) -> sum over Rx (axis -2)
            h_sq = tf.reduce_sum(tf.square(tf.abs(h)), axis=[-2, -1])
            snr = h_sq / n0
            cap = tf.math.log(1.0 + snr) / tf.math.log(2.0)
            return cap
        
        # MIMO
        # SVD based capacity (Waterfilling assumption or Equal Power)
        # We assume Equal Power allocation (Unknown CSI at Tx) -> params = s^2 / n0
        s = tf.linalg.svd(h, compute_uv=False)
        params = tf.square(s) / n0
        # C = Sum log2(1 + P_i * lambda_i^2 / N0)
        # Using equal power
        cap_per_stream = tf.math.log(1.0 + params) / tf.math.log(2.0)
        cap = tf.reduce_sum(cap_per_stream, axis=-1)
        return cap
        
    def extract_rt(self, paths: Any, batch_size: int) -> Dict[str, Any]:
        """
        Extracts native generic RT metrics from Sionna Paths.
        
        Args:
            paths: sionna.rt.Paths object (outputs from scene.compute_paths or PathSolver)
            batch_size: Explicit batch size for shape validation
        
        Returns:
            Dict of TF tensors.
        """
        if not SIONNA_AVAILABLE:
            return {}
            
        # Paths attributes:
        # a: [sources, targets, paths, rx_ant, tx_ant, 1, 1] complex
        # tau: [sources, targets, paths] float
        # theta_r, phi_r, theta_t, phi_t: [sources, targets, paths]
        
        # We assume standard Downlink: Source=Site, Target=UE.
        # But `batch_size` implies we are looking at specific UEs.
        # Check shapes.
        # If simulation was done for B UEs, then targets dimension should be B.
        # We want outputs structured as [Batch, ...].
        
        # 1. Standardize Shapes: [Targets(Batch), Sources(Cells), Paths, ...]
        # Sionna standard: [S, T, P, ...]
        # We assume T=Batch, S=NumCells.
        
        # Helper for transposing [S, T, ...] -> [T, S, ...]
        # Modified to detect batch dimension
        def _T(x):
            if isinstance(x, tuple):
                # If tuple, it might be wrapped single tensor or list of tensors
                # Sionna Paths properties are Tensors. If we got a tuple, it's unexpected but we can unwrap.
                if len(x) == 1:
                    x = x[0]
                else:
                    # If it's a tuple of values?
                    try:
                        x = tf.stack(x)
                    except:
                        # Fallback for debugging
                        x = x[0] if len(x) > 0 else tf.constant([])

            if hasattr(x, 'shape') and len(x.shape) >= 2:
                # Check Batch Dimension Alignment
                # If Dim 0 is Batch, keep it [Batch, Cells, ...]
                # If Dim 1 is Batch, flip to [Batch, Cells, ...]
                shape = x.shape
                if shape[0] == batch_size:
                    # Already [Batch, ...]
                    return x
                elif shape[1] == batch_size:
                    # [Cells, Batch, ...]
                    perm = [1, 0] + list(range(2, len(x.shape)))
                    return tf.transpose(x, perm)
                else:
                    # Fallback (maybe batch_size=4 and shape 4x4?)
                    # Or shape mismatch. Default to Transpose [S, T] -> [T, S] as per original logic?
                    # Original logic was ALWAYS transpose.
                    perm = [1, 0] + list(range(2, len(x.shape)))
                    return tf.transpose(x, perm)
            return x

        a = paths.a # [S, T, P, RxA, TxA, 1, 1]
        tau = paths.tau # [S, T, P]
        
        # In Sionna 1.2+, these are usually tensors, but let's be safe
        if isinstance(a, (list, tuple)): a = a[0]
        if isinstance(tau, (list, tuple)): tau = tau[0]

        def _align_tensor(tensor, target_shape):
            """Brute-force align tensor to target_shape [S, T, P]"""
            if isinstance(tensor, (list, tuple)): tensor = tensor[0]
            
            # Handle rank 2 [S*T, P] or similar
            if len(tensor.shape) == 2:
                if tensor.shape[0] == target_shape[0] * target_shape[1]:
                    tensor = tf.reshape(tensor, [target_shape[0], target_shape[1], -1])
            
            # Now at least rank 3
            if len(tensor.shape) >= 3:
                # Slice or Pad to match target_shape
                curr = tensor
                for i in range(3):
                    ti = target_shape[i]
                    ci = curr.shape[i]
                    if ci > ti:
                        # Slice
                        if i == 0: curr = curr[:ti, :, :]
                        elif i == 1: curr = curr[:, :ti, :]
                        elif i == 2: curr = curr[:, :, :ti]
                    elif ci < ti:
                        # Pad with zeros
                        paddings = [[0, 0] for _ in range(len(curr.shape))]
                        paddings[i] = [0, ti - ci]
                        curr = tf.pad(curr, paddings)
                return curr
            return tf.broadcast_to(tensor, target_shape) if len(tensor.shape) == 0 else tensor

        target_stp = a.shape[:3] # [S, T, P]
        tau = _align_tensor(tau, target_stp)
        phi_r = _align_tensor(paths.phi_r, target_stp)
        


        # Transpose to [Batch, Cells, Paths, ...]
        a_batch = _T(a)
        tau_batch = _T(tau)
        phi_r_batch = _T(phi_r) # Use this later instead of _T(paths.phi_r)
        

        
        # --- Basic Metrics ---
        
        # Path Power
        # |a|^2. Sum over antennas? 
        # For KPI (ToA etc), we usually look at omni-equivalent or dominant component.
        # Let's average over antennas for "Path Strength" metric.
        # Shape: [B, C, P, RxA, TxA, 1, 1] or [B, C, P, RxA, TxA]
        # We need to robustly identify axes to reduce.
        # We want to reduce RxA and TxA.
        # If input has rank 7: axes 3, 4.
        # If input has rank 5: axes 3, 4.
        # If input has rank 6: ...
        
        # Assuming last dimensions are RxA, TxA, (Frequency/Time potentially)
        # But we know `a_batch` shape is (B, C, P, Rx, Tx) from Debug log.
        # So Rx is axis 3, Tx is axis 4.
        
        path_powers_full = tf.square(tf.abs(a_batch))
        
        # Reduce Rx and Tx dimensions
        # Determine axes based on rank
        rank = len(path_powers_full.shape)
        if rank >= 5:
             # Reduce axis 3 (Rx) and 4 (Tx)
             # Note: indices shift if we don't keepdims?
             # reduce_mean(axis=[3, 4]) works on original indices.
             path_powers_avg = tf.reduce_mean(path_powers_full, axis=[3, 4]) 
        else:
             # Fallback
             path_powers_avg = path_powers_full
             
        # Squeeze trailing dimensions if they exist and are 1
        while len(path_powers_avg.shape) > 3 and path_powers_avg.shape[-1] == 1:
            path_powers_avg = tf.squeeze(path_powers_avg, axis=-1)
            
        path_powers = path_powers_avg # Expected [B, C, P]
        
        # 1. Total Power (Narrowband Channel Gain)
        total_power = tf.reduce_sum(path_powers, axis=-1) # [B, C]
        
        # 2. ToA (First Arrival)
        # Find min delay of paths with significant power.
        # Sionna paths are NOT guaranteed sorted by delay (though usually they are by method).
        # We should not just take index 0.
        # Mask out zero-energy paths (invalid/dummy paths).
        mask = path_powers > 1e-20
        # Set delay of invalid paths to infinity
        inf_delay = tf.fill(tf.shape(tau_batch), float('inf'))
        valid_tau = tf.where(mask, tau_batch, inf_delay)
        
        toa = tf.reduce_min(valid_tau, axis=-1) # [B, C]
        
        # Handle case where all paths are invalid (inf)
        is_valid_link = tf.math.is_finite(toa)
        toa = tf.where(is_valid_link, toa, 0.0) # Replace inf with 0.0
        
        # 3. Delay Spread (RMS)
        # Mean Delay
        # Normalize powers PDF
        sum_p = tf.reduce_sum(path_powers, axis=-1, keepdims=True) + 1e-20
        pdf = path_powers / sum_p
        
        mean_delay = tf.reduce_sum(pdf * tau_batch, axis=-1)
        mean_delay_sq = tf.reduce_sum(pdf * tf.square(tau_batch), axis=-1)
        rms_ds = tf.sqrt(tf.maximum(mean_delay_sq - tf.square(mean_delay), 0.0)) # [B, C]
        
        # 4. K-Factor
        # Ratio of dominant path power (LoS-like) to remaining.
        # Or specifically LoS path if tracked.
        # Approximating: Max path / (Total - Max)
        max_p = tf.reduce_max(path_powers, axis=-1)
        nlos_p = total_power - max_p
        k_factor_linear = max_p / (nlos_p + 1e-20)
        k_factor_db = 10.0 * (tf.math.log(k_factor_linear + 1e-10) / tf.math.log(10.0))
        
        # 5. Angles (Mean AoA/AoD weighted by power)
        # Circular mean
        phi_r = phi_r_batch # Already sanitized and transposed
        
        # Weighted sum of complex phasors
        w_phasors = tf.cast(pdf, tf.complex64) * tf.complex(tf.cos(phi_r), tf.sin(phi_r))
        mean_phasor = tf.reduce_sum(w_phasors, axis=-1)
        mean_aoa_az = tf.math.angle(mean_phasor) # [B, C]
        
        # RMS Angular Spread
        # spread = sqrt(-2 ln(|R|))
        r_abs = tf.abs(mean_phasor)
        r_abs = tf.clip_by_value(r_abs, 1e-9, 1.0 - 1e-9)
        rms_as = tf.sqrt(-2.0 * tf.math.log(r_abs))
        
        # 6. NLoS Detection
        # Heuristic: If ToA is significantly larger than LoS distance / c.
        # But we don't know LoS distance here easily without inputs.
        # Alternative: If K-Factor is low (< 0 dB usually means NLoS).
        is_nlos = k_factor_db < 0.0 # Boolean [B, C]
        
        return {
            'toa': toa,         # [B, C]
            'rms_ds': rms_ds,   # [B, C]
            'k_factor': k_factor_db, # [B, C]
            'aoa_az': mean_aoa_az,   # [B, C]
            'rms_as': rms_as,        # [B, C]
            'is_nlos': is_nlos,      # [B, C] bool
            'path_powers': path_powers, # [B, C, P] (Raw PDP)
            'path_delays': tau_batch,   # [B, C, P]
            'path_gains': a_batch       # [B, C, P, Rx, Tx]
        }

    def extract_phy(self, 
                   channel_response: tf.Tensor, 
                   batch_size: int) -> Dict[str, Any]:
        """
        Extract PHY Metrics (Capacity, SINR, RSRP) from generic channel/CFR.
        
        Args:
            channel_response: Complex channel [Batch, RxAnt, Sources, TxAnt, Freq]
                              (Matches output from cir_to_ofdm_channel we set up)
        """
        # Input shape: [T, RxA, S, TxA, F] corresponds to [Batch, RxAnt, Cells, TxAnt, Freq]
        # We need to process per-cell Metrics.
        
        # 1. RSRP
        # Average Power over Frequency (Reference Signals)
        # Sum Power over RxAnt
        # Sum Power over TxAnt (Cell power)
        
        h_sq = tf.square(tf.abs(channel_response))
        # Avg over Freq (axis 4)
        p_avg_freq = tf.reduce_mean(h_sq, axis=4) # [B, RxA, C, TxA]
        
        # Sum Rx (axis 1)
        p_sum_rx = tf.reduce_sum(p_avg_freq, axis=1) # [B, C, TxA]
        
        # Sum Tx (axis 2) -> RSRP
        rsrp_linear = tf.reduce_sum(p_sum_rx, axis=2) # [B, C]
        rsrp_dbm = 10.0 * (tf.math.log(rsrp_linear + 1e-20) / tf.math.log(10.0)) + 30.0
        
        # 2. SINR
        # For each cell c, signal is c, interference is sum(other cells).
        # We need total power received from each cell. 
        # rsrp_linear is effectively received signal power from cell c.
        
        total_rec_power = tf.reduce_sum(rsrp_linear, axis=-1, keepdims=True) # [B, 1] Sum over cells
        
        interference = total_rec_power - rsrp_linear # [B, C]
        sinr_linear = rsrp_linear / (interference + self.noise_power + 1e-20)
        sinr_db = 10.0 * (tf.math.log(sinr_linear + 1e-20) / tf.math.log(10.0))
        
        # 3. Capacity / SE
        # Requires MIMO processing.
        # h: [B, RxA, C, TxA, F]
        # For each cell C, treat it as serving.
        # Interference should be modeled as noise covariance?
        # For simplicity (and speed), assume matched filter bound or simple MIMO capacity per cell treating others as noise (SINR-based).
        # Advanced: Use `sionna.mimo.capacity_optimal` for single-link bounds.
        
        # Per-Link Capacity (ignoring interference for pure channel potential)
        # Iterate over cells strictly? Or can we batched?
        # capacity_optimal expects [..., Rx, Tx].
        # We have [B, Rx, C, Tx, F].
        # Transpose to [B, C, F, Rx, Tx]
        h_perm = tf.transpose(channel_response, perm=[0, 2, 4, 1, 3]) # [B, C, F, Rx, Tx]
        
        # Reshape to flatten [B, C, F] into one batch dim for Sionna?
        # Or just apply map?
        # Sionna broadcasts.
        # N0 is noise variance.
        # Normalize N0 by F to get per-tone variance? 
        # capacity_optimal input n0 is scalar or compatible shape.
        
        # Use simple log2(1 + SINR) for now if capacity_optimal is too heavy?
        # The user wants "utilize .sys".
        # Let's try to use capacity_optimal on the reshaped tensor.
        
        n0_eff = self.noise_power / h_perm.shape[2] # Per subcarrier noise
        
        # Flatten [B, C, F] -> [BatchEff, Rx, Tx]
        # This might be huge. 
        # batch=4, cells=2, freq=1024 -> 8192 matrices. Doable on GPU.
        
        # Wait, shape [B, C, F, Rx, Tx]
        # capacity_optimal computes C = log2 det(I + H H^H / N0)
        # result shape [B, C, F]
        # cap_per_tone = capacity_optimal(h_perm, n0_eff) # [B, C, F]
        
        # Use custom implementation since capacity_optimal is missing in 1.2.1
        cap_per_tone = self._compute_capacity(h_perm, n0_eff) # [B, C, F]
        
        # Mean SE over Freq
        se = tf.reduce_mean(cap_per_tone, axis=-1) # [B, C] bits/s/Hz
        
        # 4. Condition Number / Rank
        # SVD on H [Rx, Tx].
        # Mean H over freq or SVD per tone? 
        # Usually Rank is reported per band.
        # Let's verify per-tone Rank.
        # s = svd(h_perm) # [B, C, F, min(Rx, Tx)]
        # This is expensive (SVD on 8192 matrices).
        # Fallback: SVD on Mean Channel (Spatial correlation matrix).
        h_mean = tf.reduce_mean(h_perm, axis=2) # [B, C, Rx, Tx]
        s = tf.linalg.svd(h_mean, compute_uv=False)
        cond_num = tf.reduce_max(s, axis=-1) / (tf.reduce_min(s, axis=-1) + 1e-10)
        rank = tf.reduce_sum(tf.cast(s > 1e-6, tf.float32), axis=-1)
        
        return {
            'rsrp': rsrp_dbm,   # [B, C]
            'sinr': sinr_db,    # [B, C]
            'on_se': se,        # [B, C] (Spectral Efficiency)
            'rank': rank,       # [B, C]
            'cond_num': cond_num # [B, C]
        }
