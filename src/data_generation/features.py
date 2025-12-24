"""
Multi-Layer Feature Extractors for 5G NR Radio Environment
Extracts RT, PHY/FAPI, and MAC/RRC layer features from Sionna simulations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
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
    import sionna
    from sionna.rt import Scene, PlanarArray, Transmitter, Receiver
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available - feature extractors will operate in mock mode")


@dataclass
class RTLayerFeatures:
    """
    Layer 1: Ray Tracing / Physical Layer Features
    Direct outputs from Sionna RT propagation simulation
    """
    # Path-level features (per ray/path)
    path_gains: np.ndarray  # [batch, num_rx, num_paths] complex amplitudes
    path_delays: np.ndarray  # [batch, num_rx, num_paths] time-of-arrival (seconds)
    path_aoa_azimuth: np.ndarray  # [batch, num_rx, num_paths] angle-of-arrival azimuth (radians)
    path_aoa_elevation: np.ndarray  # [batch, num_rx, num_paths] angle-of-arrival elevation (radians)
    path_aod_azimuth: np.ndarray  # [batch, num_rx, num_paths] angle-of-departure azimuth (radians)
    path_aod_elevation: np.ndarray  # [batch, num_rx, num_paths] angle-of-departure elevation (radians)
    path_doppler: np.ndarray  # [batch, num_rx, num_paths] Doppler shift (Hz)
    
    # Aggregate statistics (derived from paths)
    rms_delay_spread: np.ndarray  # [batch, num_rx] RMS-DS (seconds)
    k_factor: Optional[np.ndarray] = None  # [batch, num_rx] Rician K-factor (dB)
    num_paths: np.ndarray = field(default_factory=lambda: np.array([]))  # [batch, num_rx] path count
    is_mock: bool = False
    
    # Metadata
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage."""
        result = {
            'rt/rms_delay_spread': self.rms_delay_spread,
            'rt/num_paths': self.num_paths,
        }
        
        # Only include K-factor if computed
        if self.k_factor is not None and self.k_factor.size > 0:
            result['rt/k_factor'] = self.k_factor
        
        # Skip variable-length path arrays for now (they cause Zarr issues)
        # These can be added back later with proper object array handling
        logger.debug(f"Skipping path-level arrays for Zarr storage (num_paths: {self.num_paths})")
        
        return result


@dataclass
class PHYFAPILayerFeatures:
    """
    Layer 2: PHY/FAPI (L1 Measurements)
    Channel quality and link-level measurements per 3GPP FAPI spec
    """
    # FAPI Measurements (3GPP 38.215)
    rsrp: np.ndarray  # [batch, num_rx, num_cells] Reference Signal Received Power (dBm)
    rsrq: np.ndarray  # [batch, num_rx, num_cells] Reference Signal Received Quality (dB)
    sinr: np.ndarray  # [batch, num_rx, num_cells] Signal-to-Interference-plus-Noise (dB)
    
    # Link adaptation indicators (3GPP 38.214)
    cqi: np.ndarray  # [batch, num_rx, num_cells] Channel Quality Indicator [0-15]
    ri: np.ndarray  # [batch, num_rx, num_cells] Rank Indicator [1-8]
    pmi: np.ndarray  # [batch, num_rx, num_cells] Precoding Matrix Indicator
    
    # Beam management (5G NR specific)
    l1_rsrp_beams: Optional[np.ndarray] = None  # [batch, num_rx, num_beams] per-beam RSRP
    best_beam_ids: Optional[np.ndarray] = None  # [batch, num_rx, K] top-K beam indices
    
    # Channel matrix (for research/validation)
    channel_matrix: Optional[np.ndarray] = None  # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ...]
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage."""
        d = {
            'phy_fapi/rsrp': self.rsrp,
            'phy_fapi/rsrq': self.rsrq,
            'phy_fapi/sinr': self.sinr,
            'phy_fapi/cqi': self.cqi,
            'phy_fapi/ri': self.ri,
            'phy_fapi/pmi': self.pmi,
        }
        if self.l1_rsrp_beams is not None:
            d['phy_fapi/l1_rsrp_beams'] = self.l1_rsrp_beams
        if self.best_beam_ids is not None:
            d['phy_fapi/best_beam_ids'] = self.best_beam_ids
        if self.channel_matrix is not None:
            d['phy_fapi/channel_matrix'] = self.channel_matrix
        return d


@dataclass
class MACRRCLayerFeatures:
    """
    Layer 3: MAC/RRC (System-Level)
    Network-level measurements and cell management
    """
    # Cell identification
    serving_cell_id: np.ndarray  # [batch, num_rx] Physical Cell ID
    neighbor_cell_ids: np.ndarray  # [batch, num_rx, K] K neighbor cell IDs
    
    # Timing and synchronization
    timing_advance: np.ndarray  # [batch, num_rx] TA command (TA index)
    
    # Optional fields with defaults
    tracking_area_code: Optional[np.ndarray] = None  # [batch, num_rx] TAC
    
    # Throughput and performance (simulated)
    dl_throughput_mbps: Optional[np.ndarray] = None  # [batch, num_rx] downlink throughput
    ul_throughput_mbps: Optional[np.ndarray] = None  # [batch, num_rx] uplink throughput
    bler: Optional[np.ndarray] = None  # [batch, num_rx] block error rate
    
    # Handover and mobility
    handover_events: Optional[np.ndarray] = None  # [batch, num_rx] binary indicator
    time_since_handover: Optional[np.ndarray] = None  # [batch, num_rx] seconds since last HO
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage."""
        d = {
            'mac_rrc/serving_cell_id': self.serving_cell_id,
            'mac_rrc/neighbor_cell_ids': self.neighbor_cell_ids,
            'mac_rrc/timing_advance': self.timing_advance,
        }
        
        # Debug: check dtypes
        for k, v in d.items():
            if hasattr(v, 'dtype'):
                logger.debug(f"MAC/RRC {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.warning(f"MAC/RRC {k}: not ndarray, type={type(v)}")
        
        if self.tracking_area_code is not None:
            d['mac_rrc/tracking_area_code'] = self.tracking_area_code
        if self.dl_throughput_mbps is not None:
            d['mac_rrc/dl_throughput_mbps'] = self.dl_throughput_mbps
        if self.ul_throughput_mbps is not None:
            d['mac_rrc/ul_throughput_mbps'] = self.ul_throughput_mbps
        if self.bler is not None:
            d['mac_rrc/bler'] = self.bler
        if self.handover_events is not None:
            d['mac_rrc/handover_events'] = self.handover_events
        if self.time_since_handover is not None:
            d['mac_rrc/time_since_handover'] = self.time_since_handover
        return d


class RTFeatureExtractor:
    """
    Extracts Layer 1 (RT) features from Sionna propagation paths.
    """
    
    def __init__(self, 
                 carrier_frequency_hz: float = 3.5e9,
                 bandwidth_hz: float = 100e6,
                 compute_k_factor: bool = False):
        """
        Args:
            carrier_frequency_hz: Carrier frequency (default 3.5 GHz for CBRS)
            bandwidth_hz: System bandwidth (default 100 MHz)
            compute_k_factor: Whether to compute Rician K-factor (expensive)
        """
        self.carrier_frequency_hz = carrier_frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.compute_k_factor = compute_k_factor
        
        logger.info(f"RTFeatureExtractor initialized: fc={carrier_frequency_hz/1e9:.2f} GHz, "
                   f"BW={bandwidth_hz/1e6:.1f} MHz")
    
    def extract(self, paths: Any) -> RTLayerFeatures:
        """
        Extract RT features from Sionna Paths object.
        
        Args:
            paths: Sionna RT Paths object from scene.compute_paths()
            
        Returns:
            RTLayerFeatures with all path-level and aggregate features
        """
        if not SIONNA_AVAILABLE:
            logger.warning("Sionna not available - returning mock RT features")
            return self._extract_mock()
        
        try:
            # Extract path-level features from Sionna Paths
            def _to_numpy(value):
                return value.numpy() if hasattr(value, 'numpy') else np.array(value)

            def _to_complex(value):
                if isinstance(value, tuple) and len(value) == 2:
                    real = _to_numpy(value[0])
                    imag = _to_numpy(value[1])
                    return real + 1j * imag
                return _to_numpy(value)

            def _reduce_ant_time(arr):
                if arr.ndim == 6:  # [rx, rx_ant, tx, tx_ant, paths, time]
                    return np.mean(arr, axis=(1, 3, 5))
                if arr.ndim == 5:  # [rx, rx_ant, tx, tx_ant, paths]
                    return np.mean(arr, axis=(1, 3))
                if arr.ndim == 4:  # [rx, tx, paths, time]
                    return np.mean(arr, axis=3)
                return arr

            def _ensure_batch(arr):
                if arr.ndim == 3:  # [rx, tx, paths]
                    return arr[np.newaxis, ...]
                return arr

            # Complex path gains
            path_gains_complex = _to_complex(paths.a)
            
            # Path delays
            path_delays = _to_numpy(paths.tau)
            
            # Angles (in radians)
            path_aoa_azimuth = _to_numpy(paths.phi_r)
            path_aoa_elevation = _to_numpy(paths.theta_r)
            path_aod_azimuth = _to_numpy(paths.phi_t)
            path_aod_elevation = _to_numpy(paths.theta_t)
            
            # Doppler (if available)
            if hasattr(paths, 'doppler'):
                path_doppler = _to_numpy(paths.doppler)
            else:
                path_doppler = np.zeros_like(path_delays)
            
            # Average over antennas/time to get per-path gains
            path_gains_magnitude = np.abs(path_gains_complex)
            path_gains_avg = _reduce_ant_time(path_gains_magnitude)
            
            path_delays_avg = _reduce_ant_time(path_delays)
            path_aoa_az_avg = _reduce_ant_time(path_aoa_azimuth)
            path_aoa_el_avg = _reduce_ant_time(path_aoa_elevation)
            path_aod_az_avg = _reduce_ant_time(path_aod_azimuth)
            path_aod_el_avg = _reduce_ant_time(path_aod_elevation)
            path_doppler_avg = _reduce_ant_time(path_doppler)

            path_gains_avg = _ensure_batch(path_gains_avg)
            path_delays_avg = _ensure_batch(path_delays_avg)
            path_aoa_az_avg = _ensure_batch(path_aoa_az_avg)
            path_aoa_el_avg = _ensure_batch(path_aoa_el_avg)
            path_aod_az_avg = _ensure_batch(path_aod_az_avg)
            path_aod_el_avg = _ensure_batch(path_aod_el_avg)
            path_doppler_avg = _ensure_batch(path_doppler_avg)
            
            # Ensure we have the right final shape: [batch, num_rx, num_paths]
            # Sometimes need to average over num_tx dimension
            if len(path_gains_avg.shape) > 3:
                path_gains_avg = np.mean(path_gains_avg, axis=2)
                path_delays_avg = np.mean(path_delays_avg, axis=2)
                path_aoa_az_avg = np.mean(path_aoa_az_avg, axis=2)
                path_aoa_el_avg = np.mean(path_aoa_el_avg, axis=2)
                path_aod_az_avg = np.mean(path_aod_az_avg, axis=2)
                path_aod_el_avg = np.mean(path_aod_el_avg, axis=2)
                path_doppler_avg = np.mean(path_doppler_avg, axis=2)
            
            # Compute aggregate statistics
            rms_delay_spread = self._compute_rms_ds(path_gains_avg, path_delays_avg)
            
            k_factor = None
            if self.compute_k_factor:
                k_factor = self._compute_k_factor(path_gains_avg)
            
            # Count valid paths (non-zero gain)
            num_paths = np.sum(np.abs(path_gains_avg) > 1e-10, axis=-1)
            
            return RTLayerFeatures(
                path_gains=path_gains_avg,
                path_delays=path_delays_avg,
                path_aoa_azimuth=path_aoa_az_avg,
                path_aoa_elevation=path_aoa_el_avg,
                path_aod_azimuth=path_aod_az_avg,
                path_aod_elevation=path_aod_el_avg,
                path_doppler=path_doppler_avg,
                rms_delay_spread=rms_delay_spread,
                k_factor=k_factor,
                num_paths=num_paths,
                carrier_frequency_hz=self.carrier_frequency_hz,
                bandwidth_hz=self.bandwidth_hz,
                is_mock=False,
            )
        
        except Exception as e:
            logger.error(f"Failed to extract RT features from Sionna paths: {e}")
            logger.warning("Falling back to mock RT features")
            return self._extract_mock()
    
    def _compute_rms_ds(self, gains: np.ndarray, delays: np.ndarray) -> np.ndarray:
        """
        RMS Delay Spread: sqrt(E[τ²] - E[τ]²).
        
        Args:
            gains: [batch, num_rx, num_paths] complex amplitudes
            delays: [batch, num_rx, num_paths] delays in seconds
            
        Returns:
            rms_ds: [batch, num_rx] RMS delay spread in seconds
        """
        # Power per path
        powers = np.abs(gains)**2
        total_power = np.sum(powers, axis=-1, keepdims=True) + 1e-10
        
        # Normalized power (probability)
        p = powers / total_power
        
        # Mean delay
        mean_delay = np.sum(p * delays, axis=-1)
        
        # Second moment
        mean_delay_sq = np.sum(p * delays**2, axis=-1)
        
        # RMS-DS
        rms_ds = np.sqrt(np.maximum(mean_delay_sq - mean_delay**2, 0))
        
        return rms_ds
    
    def _compute_k_factor(self, gains: np.ndarray) -> np.ndarray:
        """
        Rician K-factor: ratio of LOS to NLOS power (dB).
        
        Args:
            gains: [batch, num_rx, num_paths] complex amplitudes
            
        Returns:
            k_factor: [batch, num_rx] K-factor in dB
        """
        powers = np.abs(gains)**2
        
        # Assume first path is LOS (strongest)
        los_power = powers[..., 0]
        nlos_power = np.sum(powers[..., 1:], axis=-1) + 1e-10
        
        k_linear = los_power / nlos_power
        k_db = 10 * np.log10(k_linear + 1e-10)
        
        return k_db
    
    def _extract_mock(self) -> RTLayerFeatures:
        """Mock features for testing without Sionna."""
        batch_size, num_rx, num_paths = 10, 4, 50
        
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
        """
        Args:
            noise_figure_db: Receiver noise figure (default 9 dB)
            thermal_noise_density_dbm: Thermal noise density (default -174 dBm/Hz)
            enable_beam_management: Compute per-beam RSRP for 5G NR
            num_beams: Number of SSB beams (default 64 for mmWave)
        """
        self.noise_figure_db = noise_figure_db
        self.thermal_noise_density_dbm = thermal_noise_density_dbm
        self.enable_beam_management = enable_beam_management
        self.num_beams = num_beams
        
        logger.info(f"PHYFAPIFeatureExtractor initialized: NF={noise_figure_db} dB, "
                   f"beams={num_beams if enable_beam_management else 'disabled'}")
    
    def extract(self, 
                rt_features: RTLayerFeatures,
                channel_matrix: Optional[np.ndarray] = None,
                interference_matrices: Optional[List[np.ndarray]] = None,
                pilot_re_indices: Optional[np.ndarray] = None) -> PHYFAPILayerFeatures:
        """
        Extract PHY/FAPI features from RT features and channel matrices.
        
        Args:
            rt_features: RT layer features from RTFeatureExtractor
            channel_matrix: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ...] serving cell
            interference_matrices: List of interfering cell channel matrices
            pilot_re_indices: Indices of pilot resource elements
            
        Returns:
            PHYFAPILayerFeatures with all link-level measurements
        """
        if pilot_re_indices is None:
            # Default: every 4th subcarrier (3GPP typical)
            pilot_re_indices = np.arange(0, 100, 4)
        
        # Compute noise power
        bandwidth_hz = rt_features.bandwidth_hz
        noise_power_dbm = (self.thermal_noise_density_dbm + 
                          10 * np.log10(bandwidth_hz) + 
                          self.noise_figure_db)
        noise_power_linear = 10**((noise_power_dbm - 30) / 10)  # dBm to Watts
        
        # RSRP: average power over pilots
        if channel_matrix is not None:
            rsrp_linear = compute_rsrp(channel_matrix, cell_id=0, 
                                      pilot_re_indices=pilot_re_indices)
            rsrp_dbm = 10 * np.log10(rsrp_linear + 1e-10) + 30
            
            # rsrp_linear is [batch, num_tx], reshape to [batch, num_rx, num_tx]
            batch, num_tx = rsrp_linear.shape
            num_rx = 1  # Assuming single RX antenna for now
            rsrp = rsrp_dbm.reshape(batch, num_rx, num_tx)
            num_cells = num_tx
        else:
            # Approximate from path gains
            total_power = np.sum(np.abs(rt_features.path_gains)**2, axis=-1)
            rsrp_dbm = 10 * np.log10(total_power + 1e-10) + 30
            
            # Expand to multi-cell: [batch, num_rx, num_cells]
            num_cells = 1 + (len(interference_matrices) if interference_matrices else 0)
            rsrp = np.repeat(rsrp_dbm[..., np.newaxis], num_cells, axis=-1)
        
        # RSSI: total received power (signal + interference + noise)
        rssi_linear = 10**((rsrp_dbm - 30) / 10) + noise_power_linear
        rssi_dbm = 10 * np.log10(rssi_linear) + 30
        
        # RSRQ
        rsrq = compute_rsrq(rsrp_dbm, rssi_dbm, N=12)
        rsrq = np.repeat(rsrq[..., np.newaxis], num_cells, axis=-1)
        
        # SINR
        if channel_matrix is not None:
            sinr = compute_sinr(channel_matrix, noise_power_linear, interference_matrices)
        else:
            # Approximate: RSRP - noise floor
            sinr = rsrp_dbm - noise_power_dbm
        sinr = np.repeat(sinr[..., np.newaxis], num_cells, axis=-1)
        
        # CQI from SINR
        cqi = compute_cqi(sinr)
        
        # RI from channel matrix
        if channel_matrix is not None:
            # Average over frequency/time dimensions to get spatial channel
            h_spatial = np.mean(channel_matrix, axis=(-2, -1))
            ri = compute_rank_indicator(h_spatial)
        else:
            # Default: rank 1 (SISO)
            ri = np.ones(rsrp.shape, dtype=np.int32)
        
        # PMI (simplified)
        if channel_matrix is not None:
            h_spatial = np.mean(channel_matrix, axis=(-2, -1))
            pmi = compute_pmi(h_spatial)
        else:
            pmi = np.zeros(rsrp.shape, dtype=np.int32)
        
        # Beam management (5G NR)
        l1_rsrp_beams = None
        best_beam_ids = None
        if self.enable_beam_management:
            l1_rsrp_beams = compute_beam_rsrp(
                rt_features.path_gains,
                beam_directions=None,  # Placeholder
                num_beams=self.num_beams
            )
            # Top-4 beams
            best_beam_ids = np.argsort(-l1_rsrp_beams, axis=-1)[..., :4]
        
        return PHYFAPILayerFeatures(
            rsrp=rsrp,
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
        """
        Args:
            max_neighbors: Max neighbor cells to report (3GPP default 8)
            enable_throughput: Simulate throughput from CQI
            enable_handover: Simulate handover events (complex, disabled by default)
        """
        self.max_neighbors = max_neighbors
        self.enable_throughput = enable_throughput
        self.enable_handover = enable_handover
        
        logger.info(f"MACRRCFeatureExtractor initialized: max_neighbors={max_neighbors}")
    
    def extract(self,
                phy_features: PHYFAPILayerFeatures,
                ue_positions: np.ndarray,
                site_positions: np.ndarray,
                cell_ids: np.ndarray) -> MACRRCLayerFeatures:
        """
        Extract MAC/RRC features from PHY measurements and network topology.
        
        Args:
            phy_features: PHY/FAPI layer features
            ue_positions: [batch, num_rx, 3] UE positions (x, y, z) in meters
            site_positions: [num_sites, 3] site positions
            cell_ids: [num_sites] physical cell IDs
            
        Returns:
            MACRRCLayerFeatures with system-level measurements
        """
        batch_size, num_rx = phy_features.rsrp.shape[:2]
        
        # Serving cell: best RSRP
        serving_cell_indices = np.argmax(phy_features.rsrp, axis=-1)  # [batch, num_rx]
        serving_cell_id = cell_ids[serving_cell_indices]
        
        # Neighbor cells: top-K by RSRP (excluding serving)
        rsrp_for_neighbors = phy_features.rsrp.copy()
        rsrp_for_neighbors[np.arange(batch_size)[:, None], 
                          np.arange(num_rx), 
                          serving_cell_indices] = -np.inf
        
        # Get top K neighbors
        neighbor_indices = np.argsort(-rsrp_for_neighbors, axis=-1)[..., :self.max_neighbors]
        neighbor_cell_ids = cell_ids[neighbor_indices]
        
        # Timing Advance: compute from UE-site distances
        serving_site_positions = site_positions[serving_cell_indices]  # [batch, num_rx, 3]
        distances_3d = np.linalg.norm(ue_positions - serving_site_positions, axis=-1)
        timing_advance = compute_timing_advance(distances_3d)
        
        # Throughput simulation (Shannon capacity from CQI)
        dl_throughput_mbps = None
        if self.enable_throughput:
            dl_throughput_mbps = self._simulate_throughput(phy_features.cqi)
        
        # BLER simulation (from SINR)
        bler = self._simulate_bler(phy_features.sinr[..., 0])  # Serving cell SINR
        
        # Handover events (disabled by default)
        handover_events = None
        time_since_handover = None
        if self.enable_handover:
            handover_events = np.zeros((batch_size, num_rx), dtype=np.int32)
            time_since_handover = np.random.uniform(0, 10, (batch_size, num_rx))
        
        return MACRRCLayerFeatures(
            serving_cell_id=serving_cell_id,
            neighbor_cell_ids=neighbor_cell_ids,
            tracking_area_code=None,  # Not simulated
            timing_advance=timing_advance,
            dl_throughput_mbps=dl_throughput_mbps,
            ul_throughput_mbps=None,  # Not simulated
            bler=bler,
            handover_events=handover_events,
            time_since_handover=time_since_handover,
        )
    
    def _simulate_throughput(self, cqi: np.ndarray) -> np.ndarray:
        """
        Estimate throughput from CQI using 3GPP spectral efficiency tables.
        
        Args:
            cqi: [batch, num_rx, num_cells] CQI index [0-15]
            
        Returns:
            throughput: [batch, num_rx] downlink throughput in Mbps
        """
        # 3GPP 38.214 spectral efficiency per CQI (bits/s/Hz)
        # Table 5.2.2.1-2 (64QAM)
        se_table = np.array([
            0.0, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41,
            2.73, 3.32, 3.90, 4.52, 5.12, 5.55
        ])
        
        # Take serving cell CQI
        cqi_serving = cqi[..., 0]
        
        # Spectral efficiency
        se = se_table[cqi_serving]
        
        # Throughput = SE * BW (assuming 100 MHz BW)
        bandwidth_mhz = 100
        throughput_mbps = se * bandwidth_mhz
        
        return throughput_mbps
    
    def _simulate_bler(self, sinr_db: np.ndarray) -> np.ndarray:
        """
        Simulate Block Error Rate from SINR (simplified).
        
        Args:
            sinr_db: [batch, num_rx] SINR in dB
            
        Returns:
            bler: [batch, num_rx] block error rate [0, 1]
        """
        # Simplified BLER model: BLER = 1 / (1 + exp(a*(SINR - threshold)))
        threshold_db = 5.0  # SINR threshold for 10% BLER
        a = 0.5  # Slope parameter
        
        bler = 1 / (1 + np.exp(a * (sinr_db - threshold_db)))
        
        return bler


if __name__ == "__main__":
    # Test feature extractors
    logger.info("Testing Multi-Layer Feature Extractors...")
    
    # Test RT extractor
    rt_extractor = RTFeatureExtractor(carrier_frequency_hz=3.5e9, compute_k_factor=True)
    rt_features = rt_extractor._extract_mock()
    logger.info(f"✓ RT Features: {len(rt_features.to_dict())} arrays")
    logger.info(f"  - Path gains: {rt_features.path_gains.shape}")
    logger.info(f"  - RMS-DS: {rt_features.rms_delay_spread.mean()*1e9:.1f} ns")
    logger.info(f"  - K-factor: {rt_features.k_factor.mean():.1f} dB")
    
    # Test PHY/FAPI extractor
    phy_extractor = PHYFAPIFeatureExtractor(enable_beam_management=True, num_beams=64)
    phy_features = phy_extractor.extract(rt_features)
    logger.info(f"✓ PHY/FAPI Features: {len(phy_features.to_dict())} arrays")
    logger.info(f"  - RSRP: {phy_features.rsrp.mean():.1f} dBm")
    logger.info(f"  - RSRQ: {phy_features.rsrq.mean():.1f} dB")
    logger.info(f"  - CQI: {phy_features.cqi.mean():.1f}")
    logger.info(f"  - Beam RSRP: {phy_features.l1_rsrp_beams.shape}")
    
    # Test MAC/RRC extractor
    mac_extractor = MACRRCFeatureExtractor(max_neighbors=8, enable_throughput=True)
    ue_positions = np.random.uniform(-500, 500, (10, 4, 3))
    site_positions = np.array([[0, 0, 30], [500, 0, 30], [-250, 433, 30]])
    cell_ids = np.array([1, 2, 3])
    # Expand phy_features to multi-cell
    phy_features.rsrp = np.random.uniform(-100, -60, (10, 4, 3))
    phy_features.rsrq = np.random.uniform(-15, -5, (10, 4, 3))
    phy_features.sinr = np.random.uniform(-5, 25, (10, 4, 3))
    phy_features.cqi = np.random.randint(0, 16, (10, 4, 3))
    mac_features = mac_extractor.extract(phy_features, ue_positions, site_positions, cell_ids)
    logger.info(f"✓ MAC/RRC Features: {len(mac_features.to_dict())} arrays")
    logger.info(f"  - Serving cell: {mac_features.serving_cell_id[0, 0]}")
    logger.info(f"  - Neighbors: {mac_features.neighbor_cell_ids[0, 0]}")
    logger.info(f"  - TA: {mac_features.timing_advance[0, 0]}")
    logger.info(f"  - Throughput: {mac_features.dl_throughput_mbps.mean():.1f} Mbps")
    
    logger.info("\nAll feature extractor tests passed! ✓")
