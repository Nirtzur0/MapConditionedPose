"""
Data Structures for Layer Features
Defines dataclasses for storing extracted features from RT, PHY, and MAC layers.
"""

import numpy as np
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass, field
import logging

from .tensor_ops import _to_numpy
from src.config.feature_schema import RT_ZARR_KEYS, PHY_ZARR_KEYS, MAC_ZARR_KEYS

logger = logging.getLogger(__name__)

@dataclass
class RTLayerFeatures:
    """
    Layer 1: Ray Tracing / Physical Layer Features
    Direct outputs from Sionna RT propagation simulation
    """
    # Path-level features (per ray/path)
    path_gains: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] complex amplitudes (canonical shape)
    path_delays: Union[np.ndarray, Any]  # [batch, num_rx, num_paths] time-of-arrival (seconds, canonical shape)
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
            RT_ZARR_KEYS['rms_delay_spread']: rms_ds,
            RT_ZARR_KEYS['num_paths']: n_paths,
        }
        
        # Only include K-factor if computed
        if self.k_factor is not None:
             k_fac = _to_numpy(self.k_factor)
             # Check size or just add if not None (size check tricky if scalar)
             if hasattr(k_fac, 'size') and k_fac.size > 0:
                 result['rt/k_factor'] = k_fac
             elif not hasattr(k_fac, 'size'): # Scalar
                 result[RT_ZARR_KEYS['k_factor']] = k_fac

        if self.rms_angular_spread is not None:
             result[RT_ZARR_KEYS['rms_angular_spread']] = _to_numpy(self.rms_angular_spread)
        
        if self.toa is not None:
             result[RT_ZARR_KEYS['toa']] = _to_numpy(self.toa)
             
        if self.is_nlos is not None:
             # Convert boolean to int for zarr storage? Or keep bool.
             result[RT_ZARR_KEYS['is_nlos']] = _to_numpy(self.is_nlos)
             
        # Add path-level features for validation
        if self.path_gains is not None:
             result[RT_ZARR_KEYS['path_gains']] = _to_numpy(self.path_gains)
             
        if self.path_delays is not None:
             result[RT_ZARR_KEYS['path_delays']] = _to_numpy(self.path_delays)
             
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
    pmi: Union[np.ndarray, Any]  # [batch, num_rx, num_cells] Precoding Matrix Indicator (SVD-based)
    
    # Beam management (5G NR specific)
    l1_rsrp_beams: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx, num_beams] per-beam RSRP
    best_beam_ids: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx, K] top-K beam indices
    
    # Channel matrix (for research/validation)
    channel_matrix: Optional[Union[np.ndarray, Any]] = None  # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ...]
    
    # Channel Frequency Response (CFR) - channel estimation from DMRS
    # This is the key feature for positioning - represents what UE estimates from reference signals
    cfr_magnitude: Optional[Union[np.ndarray, Any]] = None  # [batch, num_cells, num_subcarriers] |H(f)|
    cfr_phase: Optional[Union[np.ndarray, Any]] = None  # [batch, num_cells, num_subcarriers] angle(H(f))
    
    # Advanced KPIs
    capacity_mbps: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx]
    condition_number: Optional[Union[np.ndarray, Any]] = None # [batch, num_rx]
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for Zarr storage."""
        d = {
            PHY_ZARR_KEYS['rsrp']: _to_numpy(self.rsrp),
            PHY_ZARR_KEYS['rsrq']: _to_numpy(self.rsrq),
            PHY_ZARR_KEYS['sinr']: _to_numpy(self.sinr),
            PHY_ZARR_KEYS['cqi']: _to_numpy(self.cqi),
            PHY_ZARR_KEYS['ri']: _to_numpy(self.ri),
            PHY_ZARR_KEYS['pmi']: _to_numpy(self.pmi),
        }
        if self.l1_rsrp_beams is not None:
            d[PHY_ZARR_KEYS['l1_rsrp_beams']] = _to_numpy(self.l1_rsrp_beams)
        if self.best_beam_ids is not None:
            d[PHY_ZARR_KEYS['best_beam_ids']] = _to_numpy(self.best_beam_ids)
        if self.channel_matrix is not None:
            d[PHY_ZARR_KEYS['channel_matrix']] = _to_numpy(self.channel_matrix)
        # CFR features - critical for positioning
        if self.cfr_magnitude is not None:
            d[PHY_ZARR_KEYS['cfr_magnitude']] = _to_numpy(self.cfr_magnitude)
        if self.cfr_phase is not None:
            d[PHY_ZARR_KEYS['cfr_phase']] = _to_numpy(self.cfr_phase)
        if self.capacity_mbps is not None:
            d[PHY_ZARR_KEYS['capacity_mbps']] = _to_numpy(self.capacity_mbps)
        if self.condition_number is not None:
            d[PHY_ZARR_KEYS['condition_number']] = _to_numpy(self.condition_number)
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
            MAC_ZARR_KEYS['serving_cell_id']: _to_numpy(self.serving_cell_id),
            MAC_ZARR_KEYS['neighbor_cell_ids']: _to_numpy(self.neighbor_cell_ids),
            MAC_ZARR_KEYS['timing_advance']: _to_numpy(self.timing_advance),
        }
        
        if self.tracking_area_code is not None:
            d[MAC_ZARR_KEYS['tracking_area_code']] = _to_numpy(self.tracking_area_code)
        if self.dl_throughput_mbps is not None:
            d[MAC_ZARR_KEYS['dl_throughput_mbps']] = _to_numpy(self.dl_throughput_mbps)
        if self.ul_throughput_mbps is not None:
            d[MAC_ZARR_KEYS['ul_throughput_mbps']] = _to_numpy(self.ul_throughput_mbps)
        if self.bler is not None:
            d[MAC_ZARR_KEYS['bler']] = _to_numpy(self.bler)
        if self.handover_events is not None:
            d[MAC_ZARR_KEYS['handover_events']] = _to_numpy(self.handover_events)
        if self.time_since_handover is not None:
            d[MAC_ZARR_KEYS['time_since_handover']] = _to_numpy(self.time_since_handover)
        return d
