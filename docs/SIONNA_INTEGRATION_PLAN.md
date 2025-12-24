# Sionna Integration Implementation Plan

**Created:** December 23, 2025  
**Status:** Ready for Implementation  
**Goal:** Replace mock data generation with real Sionna RT+PHY+SYS simulation

---

## Executive Summary

Currently, the data generator uses **mock mode** with random features. This plan details how to implement **real Sionna ray tracing** across all three layers (RT, PHY, SYS) to generate physics-based training data.

**Key Insight:** Sionna RT (Dr.Jit-based) and Sionna PHY/SYS (TensorFlow-based) are already installed and working. The integration code exists but is stubbed out. We need to complete the implementation.

---

## Current State Analysis

### What Works ✅
- Sionna RT 1.2.1 installed (Dr.Jit + Mitsuba 3)
- TensorFlow 2.16.2 installed (for PHY/SYS layers)
- Scene generation pipeline (geo2sigmap)
- Data structure and Zarr storage
- Feature extraction classes defined
- Mock data generation functional

### What Needs Implementation ❌
- **RT Layer:** Real ray tracing (`extract()` method in RTFeatureExtractor)
- **Simulation:** Sionna scene setup and path computation (`_simulate_rf()` in MultiLayerDataGenerator)
- **PHY Layer:** Real channel matrix computation and link-level features
- **SYS Layer:** Throughput and BLER from PHY-layer abstraction
- **Scene Loading:** Load Mitsuba XML scenes into Sionna RT
- **Transmitter/Receiver Setup:** Antenna arrays, positions, orientations

### Current Stub Code Locations
```
src/data_generation/features.py:
  - Line 188: RTFeatureExtractor.extract() → calls _extract_mock()
  - Line 198-201: Checks SIONNA_AVAILABLE but returns mock
  - Line 360: PHYFAPIFeatureExtractor.extract() → needs channel matrix
  
src/data_generation/multi_layer_generator.py:
  - Line 363-369: _simulate_rf() → hardcoded to use _simulate_mock()
  - Line 236: load_scene() → commented out Sionna integration
```

---

## Implementation Phases

## Phase 1: RT Layer - Ray Tracing (Week 1-2)

### 1.1 Scene Loading and Setup

**File:** `src/data_generation/multi_layer_generator.py`

**Task:** Implement real Sionna scene loading

```python
def _load_sionna_scene(self, scene_path: Path) -> 'sionna.rt.Scene':
    """
    Load Mitsuba XML scene into Sionna RT.
    
    Args:
        scene_path: Path to scene.xml (Mitsuba format)
        
    Returns:
        Sionna RT Scene object
    """
    from sionna.rt import load_scene, Scene
    
    # Load scene from XML
    scene = load_scene(str(scene_path))
    
    # Apply scene-level settings
    scene.frequency = self.config.carrier_frequency_hz  # e.g., 3.5e9
    scene.synthetic_array = True  # Enable array synthesis
    
    return scene
```

**Parameters from config:**
- `carrier_frequency_hz`: 3.5e9 (3.5 GHz) or 28e9 (mmWave)
- `bandwidth_hz`: 100e6 (100 MHz)
- Scene extent from metadata

**Reference:** geo2sigmap generates Mitsuba XML compatible with Sionna RT

### 1.2 Transmitter Setup (Base Stations)

**Task:** Configure transmitters (cell sites) with antenna arrays

```python
def _setup_transmitters(self, scene: 'sionna.rt.Scene', 
                        site_positions: np.ndarray,
                        site_metadata: Dict) -> List['sionna.rt.Transmitter']:
    """
    Setup cell site transmitters with antenna arrays.
    
    Args:
        scene: Sionna RT Scene
        site_positions: [num_sites, 3] positions (x, y, z) in meters
        site_metadata: Site configuration (azimuth, downtilt, etc.)
        
    Returns:
        List of Sionna Transmitter objects
    """
    from sionna.rt import Transmitter, PlanarArray
    
    transmitters = []
    
    for site_idx, pos in enumerate(site_positions):
        # Create antenna array
        # 3GPP typical: 8x8 array for sub-6 GHz, 16x16 for mmWave
        if self.config.carrier_frequency_hz < 10e9:
            # Sub-6 GHz: 8 rows x 8 cols, 0.5λ spacing
            array = PlanarArray(
                num_rows=8,
                num_cols=8,
                vertical_spacing=0.5,    # λ/2
                horizontal_spacing=0.5,   # λ/2
                pattern="iso",            # Isotropic elements (can use 3GPP model)
                polarization="dual"       # Cross-polarized
            )
        else:
            # mmWave: 16x16 array
            array = PlanarArray(
                num_rows=16,
                num_cols=16,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="dual"
            )
        
        # Get site orientation from metadata
        azimuth = site_metadata.get(f'site_{site_idx}_azimuth', 0.0)  # degrees
        downtilt = site_metadata.get(f'site_{site_idx}_downtilt', 10.0)  # degrees
        
        # Create transmitter
        tx = Transmitter(
            name=f"BS_{site_idx}",
            position=pos,  # [x, y, z] in meters
            orientation=[azimuth, downtilt, 0.0],  # [azimuth, elevation, roll] in degrees
            antenna=array
        )
        
        scene.add(tx)
        transmitters.append(tx)
    
    return transmitters
```

**Parameters:**
- **Antenna array:** 8x8 (sub-6 GHz) or 16x16 (mmWave)
- **Spacing:** 0.5λ (typical 3GPP)
- **Polarization:** Dual (X-pol)
- **Pattern:** Isotropic or 3GPP 38.901 sector pattern
- **Downtilt:** 10° (typical macro cell)

**Reference:** 3GPP 38.901 antenna models, Sionna `PlanarArray` docs

### 1.3 Receiver Setup (UE Positions)

**Task:** Configure receivers for UE sampling

```python
def _setup_receiver(self, scene: 'sionna.rt.Scene', 
                   ue_position: np.ndarray) -> 'sionna.rt.Receiver':
    """
    Setup UE receiver at given position.
    
    Args:
        scene: Sionna RT Scene
        ue_position: [3] position (x, y, z) in meters
        
    Returns:
        Sionna Receiver object
    """
    from sionna.rt import Receiver, PlanarArray
    
    # UE antenna: simple 2-element array or single element
    ue_array = PlanarArray(
        num_rows=1,
        num_cols=2,  # Dual antenna (MIMO 2x2 minimum)
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="dual"
    )
    
    # Create receiver
    rx = Receiver(
        name="UE",
        position=ue_position,
        orientation=[0.0, 0.0, 0.0],  # Random orientation can be added
        antenna=ue_array
    )
    
    scene.add(rx)
    return rx
```

**Parameters:**
- **UE array:** 1x2 (dual antenna, typical smartphone)
- **Height:** 1.5m (pedestrian) from config
- **Orientation:** Can randomize for realism

### 1.4 Ray Tracing Computation

**Task:** Run Sionna path solver and extract RT features

```python
def _simulate_rf(self, 
                scene: 'sionna.rt.Scene',
                ue_position: np.ndarray,
                site_positions: np.ndarray,
                cell_ids: np.ndarray) -> Tuple[RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures]:
    """
    Run Sionna RT simulation for one UE position.
    
    Args:
        scene: Loaded Sionna scene with TX/RX
        ue_position: [3] UE position
        site_positions: [num_sites, 3] BS positions
        cell_ids: [num_sites] cell identifiers
        
    Returns:
        (rt_features, phy_features, mac_features)
    """
    # Setup receiver at UE position
    rx = self._setup_receiver(scene, ue_position)
    
    # Compute propagation paths
    # This runs Dr.Jit-accelerated ray tracing
    paths = scene.compute_paths(
        max_depth=5,              # Max reflections (5 is typical)
        num_samples=10_000_000,   # Ray samples (higher = more accurate)
        los=True,                 # Include LoS component
        reflection=True,          # Include reflections
        diffraction=True,         # Include diffraction (slower)
        scattering=False,         # Disable for speed (can enable for urban)
        edge_diffraction=False    # Disable for speed
    )
    
    # Extract RT features using RTFeatureExtractor
    rt_features = self.rt_extractor.extract(paths)
    
    # Generate channel matrices from paths
    channel_freq = scene.compute_channel(paths)  # Frequency domain
    
    # Extract PHY features using channel matrices
    phy_features = self.phy_extractor.extract(
        rt_features=rt_features,
        channel_matrix=channel_freq.numpy(),
        interference_matrices=None  # TODO: multi-cell interference
    )
    
    # Extract SYS features from PHY
    mac_features = self.mac_extractor.extract(
        phy_features=phy_features,
        ue_positions=ue_position[np.newaxis, :],
        site_positions=site_positions,
        cell_ids=cell_ids
    )
    
    # Remove receiver for next iteration
    scene.remove(rx)
    
    return rt_features, phy_features, mac_features
```

**Sionna RT Parameters:**
- `max_depth`: 5 reflections (balance accuracy vs speed)
- `num_samples`: 10M rays (reduce to 1M for faster testing)
- `los`: True (include line-of-sight)
- `reflection`: True (specular reflections from buildings)
- `diffraction`: True (knife-edge diffraction, slower but important for NLOS)
- `scattering`: False (diffuse scattering, very slow)

**Performance Note:** 
- With diffraction: ~1-5 seconds per UE position
- Without diffraction: ~0.1-0.5 seconds per UE position
- GPU acceleration helps significantly

### 1.5 RT Feature Extraction

**File:** `src/data_generation/features.py`

**Task:** Implement `RTFeatureExtractor.extract()` to process Sionna Paths

```python
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
    
    # Import TensorFlow operations for Sionna compatibility
    import tensorflow as tf
    
    # Extract path-level features from Sionna Paths
    # paths.a: Complex path amplitudes [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # paths.tau: Path delays [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    # paths.theta_t, phi_t: AoD angles
    # paths.theta_r, phi_r: AoA angles
    # paths.doppler: Doppler shifts
    
    # Convert to numpy
    path_gains_complex = paths.a.numpy()  # Complex amplitudes
    path_delays = paths.tau.numpy()       # Seconds
    
    # Angles (in radians)
    path_aoa_azimuth = paths.phi_r.numpy()      # AoA azimuth
    path_aoa_elevation = paths.theta_r.numpy()  # AoA elevation
    path_aod_azimuth = paths.phi_t.numpy()      # AoD azimuth
    path_aod_elevation = paths.theta_t.numpy()  # AoD elevation
    
    # Doppler (if available, depends on velocity config)
    if hasattr(paths, 'doppler'):
        path_doppler = paths.doppler.numpy()
    else:
        # Compute from velocity if needed
        path_doppler = np.zeros_like(path_delays)
    
    # Average over antennas to get per-path gains
    # From [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, time_steps]
    # To [batch, num_rx, num_paths]
    path_gains_avg = np.mean(np.abs(path_gains_complex), axis=(2, 4, 6))
    
    # Take first time step if multiple (for static scenes)
    if len(path_delays.shape) > 5:
        path_delays = path_delays[..., 0]
        path_aoa_azimuth = path_aoa_azimuth[..., 0]
        path_aoa_elevation = path_aoa_elevation[..., 0]
        path_aod_azimuth = path_aod_azimuth[..., 0]
        path_aod_elevation = path_aod_elevation[..., 0]
    
    # Average over TX and antennas for delays/angles
    path_delays_avg = np.mean(path_delays, axis=(2, 3, 4))  # [batch, num_rx, num_paths]
    path_aoa_az_avg = np.mean(path_aoa_azimuth, axis=(2, 3, 4))
    path_aoa_el_avg = np.mean(path_aoa_elevation, axis=(2, 3, 4))
    path_aod_az_avg = np.mean(path_aod_azimuth, axis=(2, 3, 4))
    path_aod_el_avg = np.mean(path_aod_elevation, axis=(2, 3, 4))
    path_doppler_avg = np.mean(path_doppler, axis=(2, 3, 4))
    
    # Compute aggregate statistics
    rms_delay_spread = self._compute_rms_delay_spread(path_gains_avg, path_delays_avg)
    
    # K-factor (optional)
    k_factor = None
    if self.enable_k_factor:
        k_factor = self._compute_k_factor(path_gains_avg)
    
    # Count valid paths
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
    )
```

**Key Sionna Objects:**
- `paths.a`: Complex path gains (Dr.Jit tensor)
- `paths.tau`: Path delays in seconds
- `paths.phi_r, theta_r`: AoA angles (spherical coords)
- `paths.phi_t, theta_t`: AoD angles
- All are TensorFlow tensors → convert to numpy

---

## Phase 2: PHY Layer - Link-Level Features (Week 2-3)

### 2.1 Channel Matrix Processing

**Task:** Use Sionna's channel model to compute frequency-domain channel

```python
# In _simulate_rf() after compute_paths():

# Generate channel impulse response (time domain)
channel_time = scene.compute_cir(paths)  
# Shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, delay_bins]

# Or directly in frequency domain (OFDM)
channel_freq = scene.compute_channel(paths)
# Shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_subcarriers]

# Convert to numpy
H = channel_freq.numpy()
```

**Reference:** Sionna RT documentation on `compute_cir()` and `compute_channel()`

### 2.2 RSRP/RSRQ/SINR Computation

**File:** `src/data_generation/measurement_utils.py`

**Task:** Implement 3GPP-compliant measurements from channel matrix

```python
def compute_rsrp_from_channel(H: np.ndarray, 
                              tx_power_dbm: float,
                              num_resource_elements: int = 12) -> np.ndarray:
    """
    Compute RSRP from channel matrix.
    
    RSRP = Reference Signal Received Power (3GPP 38.215)
    Measured on specific reference signal resource elements.
    
    Args:
        H: Channel matrix [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_subcarriers]
        tx_power_dbm: Transmit power in dBm
        num_resource_elements: Number of REs for RSRP measurement (typically 12 in 5G NR)
        
    Returns:
        RSRP in dBm [batch, num_rx, num_cells]
    """
    # Convert tx power to linear
    tx_power_linear = 10 ** (tx_power_dbm / 10) / 1000  # Watts
    
    # Average channel power over RX antennas and subcarriers
    # |H|^2: channel gain
    H_power = np.abs(H) ** 2  # Linear power
    H_avg = np.mean(H_power, axis=(2, 5))  # Average over RX antennas and subcarriers
    # Shape: [batch, num_rx, num_tx, num_tx_ant]
    
    # RSRP = TX power × channel gain × (1 RE bandwidth)
    # Assuming equal power per RE
    rsrp_linear = tx_power_linear * np.mean(H_avg, axis=-1)  # Average over TX antennas
    # Shape: [batch, num_rx, num_tx] where num_tx = num_cells
    
    # Convert to dBm
    rsrp_dbm = 10 * np.log10(rsrp_linear * 1000 + 1e-12)  # Add epsilon to avoid log(0)
    
    # Quantize to 1 dB steps (3GPP realism)
    rsrp_dbm = np.round(rsrp_dbm)
    
    return rsrp_dbm
```

**Similar functions needed:**
- `compute_rsrq_from_channel()`: RSRQ = RSRP / RSSI
- `compute_sinr_from_channel()`: SINR with multi-cell interference
- `compute_cqi_from_sinr()`: Map SINR → CQI table (3GPP 38.214)

**Reference:** 3GPP 38.215 (Physical layer measurements)

### 2.3 Beam Management (5G NR)

**Task:** Compute per-beam RSRP for beam selection

```python
def compute_beam_rsrp(H: np.ndarray,
                     beamforming_vectors: np.ndarray,
                     tx_power_dbm: float) -> np.ndarray:
    """
    Compute per-beam RSRP for beam management.
    
    Args:
        H: Channel matrix [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_subcarriers]
        beamforming_vectors: [num_beams, num_tx_ant] complex beamforming weights
        tx_power_dbm: Transmit power per beam
        
    Returns:
        Beam RSRP [batch, num_rx, num_beams] in dBm
    """
    num_beams = beamforming_vectors.shape[0]
    batch_size, num_rx = H.shape[0], H.shape[1]
    
    beam_rsrp = np.zeros((batch_size, num_rx, num_beams))
    
    for beam_idx in range(num_beams):
        # Apply beamforming: H_eff = H @ w
        w = beamforming_vectors[beam_idx]  # [num_tx_ant]
        H_beam = H @ w  # [batch, num_rx, num_rx_ant, num_tx, num_subcarriers]
        
        # Compute effective channel power
        H_power = np.abs(H_beam) ** 2
        H_avg = np.mean(H_power, axis=(2, 4))  # [batch, num_rx, num_tx]
        
        # Convert to RSRP
        rsrp_linear = (10 ** (tx_power_dbm / 10) / 1000) * H_avg
        beam_rsrp[:, :, beam_idx] = 10 * np.log10(rsrp_linear * 1000 + 1e-12)
    
    return beam_rsrp
```

**Note:** Beamforming vectors can be:
- DFT codebook (3GPP Type I)
- Optimized (SVD-based)
- Random for testing

---

## Phase 3: SYS Layer - System-Level Features (Week 3-4)

### 3.1 Throughput Estimation

**File:** `src/data_generation/features.py` → `MACRRCFeatureExtractor`

**Task:** Use PHY-layer abstraction to estimate throughput

```python
def _simulate_throughput(self, cqi: np.ndarray, sinr_db: np.ndarray) -> np.ndarray:
    """
    Simulate throughput using exponential effective SINR mapping (EESM).
    
    3GPP approach: CQI → MCS → Throughput
    
    Args:
        cqi: [batch, num_rx, num_cells] Channel Quality Indicator [0-15]
        sinr_db: [batch, num_rx, num_cells] SINR in dB
        
    Returns:
        Throughput in Mbps [batch, num_rx, num_cells]
    """
    # 3GPP 38.214 Table 5.2.2.1-2: CQI to MCS mapping
    # Simplified: CQI maps to spectral efficiency
    spectral_efficiency = self._cqi_to_spectral_efficiency(cqi)  # bits/s/Hz
    
    # Throughput = Spectral Efficiency × Bandwidth × Resource Allocation
    # Assume 100% PRB allocation for simplicity (can add scheduler model)
    throughput_bps = spectral_efficiency * self.bandwidth_hz
    throughput_mbps = throughput_bps / 1e6
    
    # Add realistic variations
    # - Scheduler overhead: 10-20% loss
    # - Control channel overhead: ~15%
    # - HARQ retransmissions: depends on BLER
    overhead_factor = 0.7  # 30% overhead
    throughput_mbps *= overhead_factor
    
    return throughput_mbps

def _cqi_to_spectral_efficiency(self, cqi: np.ndarray) -> np.ndarray:
    """Map CQI to spectral efficiency (bits/s/Hz)."""
    # 3GPP 38.214 Table 5.2.2.1-2
    cqi_table = {
        0: 0.0,    # Out of range
        1: 0.1523,
        2: 0.2344,
        3: 0.3770,
        4: 0.6016,
        5: 0.8770,
        6: 1.1758,
        7: 1.4766,
        8: 1.9141,
        9: 2.4063,
        10: 2.7305,
        11: 3.3223,
        12: 3.9023,
        13: 4.5234,
        14: 5.1152,
        15: 5.5547,
    }
    
    # Vectorized lookup
    se = np.zeros_like(cqi, dtype=np.float32)
    for cqi_val, efficiency in cqi_table.items():
        se[cqi == cqi_val] = efficiency
    
    return se
```

### 3.2 BLER Estimation

**Task:** Estimate Block Error Rate from SINR

```python
def _simulate_bler(self, sinr_db: np.ndarray, mcs: int = 10) -> np.ndarray:
    """
    Estimate BLER using link-level abstraction.
    
    Uses AWGN-based lookup tables or analytical models.
    
    Args:
        sinr_db: [batch, num_rx, num_cells] SINR in dB
        mcs: Modulation and Coding Scheme index
        
    Returns:
        BLER [batch, num_rx, num_cells] in range [0, 1]
    """
    # Simplified BLER model (can use lookup tables from link-level sims)
    # Shannon bound approximation
    # BLER ≈ Q(sqrt(2 * SNR_eff * code_rate))
    
    # For demonstration, use logistic function
    # BLER high when SINR < threshold, low when SINR > threshold
    sinr_threshold = self._get_mcs_threshold(mcs)  # dB
    
    # Logistic BLER model
    bler = 1.0 / (1.0 + np.exp(2.0 * (sinr_db - sinr_threshold)))
    
    # Clip to realistic range
    bler = np.clip(bler, 1e-4, 0.5)  # Min 0.01%, max 50%
    
    return bler

def _get_mcs_threshold(self, mcs: int) -> float:
    """Get SINR threshold for target BLER (10%) at given MCS."""
    # Simplified lookup table (should match 3GPP link-level results)
    mcs_thresholds = {
        0: -5.0,   # QPSK 1/4
        5: 0.0,    # QPSK 1/2
        10: 5.0,   # 16-QAM 1/2
        15: 10.0,  # 64-QAM 2/3
        20: 15.0,  # 64-QAM 5/6
        25: 20.0,  # 256-QAM 3/4
    }
    return mcs_thresholds.get(mcs, 5.0)
```

**Reference:** 3GPP 38.214 (Link adaptation), Shannon capacity

---

## Phase 4: Integration & Testing (Week 4-5)

### 4.1 Update Configuration

**File:** `configs/data_generation.yaml`

**Add Sionna-specific parameters:**

```yaml
scene:
  tile_size: 512  # meters
  scene_dir: "data/scenes"
  
sionna_rt:
  enabled: true  # Toggle real vs mock
  max_depth: 5
  num_samples: 10_000_000  # 10M rays (reduce to 1M for testing)
  los: true
  reflection: true
  diffraction: true  # Slower but important
  scattering: false
  edge_diffraction: false
  
  # Performance tuning
  use_gpu: true
  batch_size: 1  # Process multiple UEs together (experimental)

rf_parameters:
  carrier_frequency_hz: 3.5e9
  bandwidth_hz: 100e6
  tx_power_dbm: 43.0
  noise_figure_db: 9.0
  
antenna_arrays:
  bs_sub6:
    num_rows: 8
    num_cols: 8
    vertical_spacing: 0.5
    horizontal_spacing: 0.5
    polarization: "dual"
  
  bs_mmwave:
    num_rows: 16
    num_cols: 16
    vertical_spacing: 0.5
    horizontal_spacing: 0.5
    polarization: "dual"
  
  ue:
    num_rows: 1
    num_cols: 2
    vertical_spacing: 0.5
    horizontal_spacing: 0.5
    polarization: "dual"

sampling:
  num_ue_per_tile: 4000  # Increase from 100
  ue_height: 1.5  # meters
  min_distance_from_bs: 10.0  # meters
  max_distance_from_bs: 1000.0  # meters
  
  # Spatial distribution
  distribution: "uniform"  # or "coverage_weighted"
  exclude_indoor: true  # Use building footprints
```

### 4.2 Refactor MultiLayerDataGenerator

**Update key methods:**

```python
class MultiLayerDataGenerator:
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        
        # Feature extractors
        self.rt_extractor = RTFeatureExtractor(...)
        self.phy_extractor = PHYFAPIFeatureExtractor(...)
        self.mac_extractor = MACRRCFeatureExtractor(...)
        
        # Sionna scene (loaded once per tile)
        self.sionna_scene = None
        self.transmitters = None
    
    def process_scene(self, scene_id: str) -> Dict[str, np.ndarray]:
        """Process one scene/tile."""
        
        # Load scene
        scene_path = self.config.scene_dir / f"{scene_id}/scene.xml"
        self.sionna_scene = self._load_sionna_scene(scene_path)
        
        # Load metadata
        metadata = self._load_scene_metadata(scene_id)
        site_positions = metadata['site_positions']  # [num_sites, 3]
        cell_ids = metadata['cell_ids']
        
        # Setup transmitters (once per scene)
        self.transmitters = self._setup_transmitters(
            self.sionna_scene, site_positions, metadata
        )
        
        # Sample UE positions
        ue_positions = self._sample_ue_positions(
            scene_extent=metadata['bbox'],
            num_samples=self.config.num_ue_per_tile
        )
        
        # Simulate each UE
        all_data = {
            'positions': [],
            'timestamps': [],
            'rt': {},
            'phy_fapi': {},
            'mac_rrc': {},
        }
        
        for ue_idx, ue_pos in enumerate(tqdm(ue_positions, desc=f"Scene {scene_id}")):
            # Run RF simulation
            if self.config.sionna_rt_enabled:
                rt_feat, phy_feat, mac_feat = self._simulate_rf(
                    self.sionna_scene, ue_pos, site_positions, cell_ids
                )
            else:
                # Fallback to mock
                rt_feat, phy_feat, mac_feat = self._simulate_mock(
                    ue_pos, site_positions, cell_ids
                )
            
            # Accumulate
            all_data['positions'].append(ue_pos)
            all_data['timestamps'].append(ue_idx * self.config.report_interval_ms / 1000.0)
            
            # Store features
            self._accumulate_features(all_data, rt_feat, phy_feat, mac_feat)
        
        # Stack into arrays
        stacked_data = self._stack_scene_data(all_data)
        
        return stacked_data
```

### 4.3 Testing Strategy

**Step 1: Unit Tests**

```python
# tests/test_sionna_integration.py

def test_scene_loading():
    """Test loading Mitsuba scene into Sionna."""
    scene_path = "data/scenes/test_scene/scene.xml"
    scene = load_scene(scene_path)
    assert scene is not None
    assert scene.frequency > 0

def test_transmitter_setup():
    """Test BS antenna array creation."""
    scene = load_scene(...)
    gen = MultiLayerDataGenerator(config)
    txs = gen._setup_transmitters(scene, positions, metadata)
    assert len(txs) == 3  # 3 sites
    assert txs[0].antenna.num_rows == 8

def test_ray_tracing():
    """Test single UE ray tracing."""
    scene = load_scene(...)
    # Setup TX/RX
    paths = scene.compute_paths()
    assert paths.a.shape[0] > 0  # Has paths

def test_rt_feature_extraction():
    """Test RTFeatureExtractor with real Sionna paths."""
    paths = ...  # From test_ray_tracing
    extractor = RTFeatureExtractor(...)
    features = extractor.extract(paths)
    assert features.path_gains.shape[2] > 0  # Has paths
    assert features.rms_delay_spread.shape == (1, 1)
```

**Step 2: Integration Test**

```bash
# Generate small test dataset (10 UEs, 1 scene)
python scripts/generate_dataset.py \
    --scene-dir data/scenes/test_scene \
    --output-dir data/test_sionna \
    --num-scenes 1 \
    --num-ue-per-tile 10 \
    --config configs/data_generation_sionna.yaml
```

**Step 3: Validation**

```python
# Validate generated data
import zarr
z = zarr.open('data/test_sionna/dataset.zarr', 'r')

# Check RT features are not mock (not all same)
pg = z['rt/path_gains'][:]
assert np.std(pg) > 0.1  # Real data has variation

# Check RSRP makes physical sense
rsrp = z['phy_fapi/rsrp'][:]
assert np.all(rsrp < 0)  # dBm should be negative
assert np.all(rsrp > -120)  # Not impossibly low

# Check path loss vs distance
positions = z['positions/ue_x'][:], z['positions/ue_y'][:]
distances = compute_distances(positions, bs_positions)
expected_pl = 32.4 + 20*np.log10(distances) + 20*np.log10(3.5e9/1e9)  # Free space
measured_pl = tx_power - rsrp
# Should correlate
correlation = np.corrcoef(expected_pl, measured_pl)[0, 1]
assert correlation > 0.7  # Strong correlation
```

**Step 4: Performance Benchmark**

```python
# Measure throughput
import time

start = time.time()
data = generator.process_scene('test_scene')
elapsed = time.time() - start

num_ues = data['positions'].shape[0]
ues_per_second = num_ues / elapsed

print(f"Generated {num_ues} UEs in {elapsed:.1f}s")
print(f"Throughput: {ues_per_second:.2f} UEs/second")
print(f"Estimated time for 4000 UEs: {4000/ues_per_second/60:.1f} minutes")
```

**Expected Performance:**
- With diffraction: ~2-5 UEs/second → 20-30 minutes for 4000 UEs
- Without diffraction: ~5-10 UEs/second → 7-15 minutes for 4000 UEs
- Can parallelize across multiple tiles

---

## Phase 5: Optimization & Scale-Up (Week 5-6)

### 5.1 GPU Acceleration

Sionna RT uses Dr.Jit which automatically uses GPU if available. Ensure:

```python
# Check GPU availability
import drjit as dr
print(f"Dr.Jit variant: {dr.variant()}")  # Should be 'cuda' or 'llvm'

# Set GPU device
os.environ['DRJIT_DEVICE'] = '0'  # GPU 0
```

### 5.2 Parallelization

**Option A: Multi-process (across scenes)**

```python
from multiprocessing import Pool

def process_scene_wrapper(scene_id):
    gen = MultiLayerDataGenerator(config)
    return gen.process_scene(scene_id)

# Process scenes in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_scene_wrapper, scene_ids)
```

**Option B: Batched UEs (experimental)**

Sionna RT can handle multiple receivers in one call:

```python
# Instead of loop over UEs:
for ue_pos in ue_positions:
    rx = setup_receiver(scene, ue_pos)
    paths = scene.compute_paths()

# Batch version:
rxs = [setup_receiver(scene, pos) for pos in ue_positions[:batch_size]]
paths = scene.compute_paths()  # Computes for all RXs at once
```

This can be 2-5× faster but requires more GPU memory.

### 5.3 Reduce Ray Samples

For initial training, can reduce quality for speed:

```yaml
sionna_rt:
  num_samples: 1_000_000  # 1M instead of 10M (10× faster)
  diffraction: false      # Disable for 2× speedup
```

Later, regenerate with full quality for final model.

---

## Phase 6: Radio Map Generation (Week 6)

### 6.1 Sionna RadioMapSolver

For physics loss, generate precomputed radio maps:

```python
from sionna.rt import RadioMapSolver

def generate_radio_map(scene: Scene, 
                       resolution: float = 1.0,
                       features: List[str] = ['path_gain', 'toa', 'sinr']) -> Dict[str, np.ndarray]:
    """
    Generate radio maps for all features at given resolution.
    
    Args:
        scene: Loaded Sionna scene with TX configured
        resolution: Meters per pixel
        features: List of features to compute
        
    Returns:
        Dictionary of radio maps [H, W] for each feature
    """
    # Define grid
    extent = scene.bounding_box  # [x_min, y_min, z_min, x_max, y_max, z_max]
    x_range = (extent[0], extent[3])
    y_range = (extent[1], extent[4])
    z_height = 1.5  # UE height
    
    # Create solver
    solver = RadioMapSolver(scene)
    
    # Compute maps
    radio_maps = {}
    
    if 'path_gain' in features:
        radio_maps['path_gain'] = solver.compute_path_gain_map(
            x_range=x_range,
            y_range=y_range,
            z=z_height,
            resolution=resolution
        )
    
    if 'toa' in features:
        radio_maps['toa'] = solver.compute_toa_map(...)
    
    if 'sinr' in features:
        radio_maps['sinr'] = solver.compute_sinr_map(...)
    
    # Add more features as needed
    
    return radio_maps
```

**Note:** Radio map generation is slower (10-30 minutes per scene) but only done once. Store with scene.

---

## Implementation Checklist

### Week 1-2: RT Layer
- [ ] Implement `_load_sionna_scene()`
- [ ] Implement `_setup_transmitters()` with antenna arrays
- [ ] Implement `_setup_receiver()` for UE
- [ ] Update `_simulate_rf()` to call Sionna ray tracing
- [ ] Complete `RTFeatureExtractor.extract()` for real paths
- [ ] Test: Single UE ray tracing works
- [ ] Test: RT features have physical values

### Week 2-3: PHY Layer
- [ ] Implement channel matrix extraction from Sionna
- [ ] Complete `compute_rsrp_from_channel()`
- [ ] Complete `compute_rsrq_from_channel()`
- [ ] Complete `compute_sinr_from_channel()`
- [ ] Complete `compute_cqi_from_sinr()`
- [ ] Implement beam management (`compute_beam_rsrp()`)
- [ ] Test: PHY features match link budget calculations

### Week 3-4: SYS Layer
- [ ] Implement `_simulate_throughput()` with EESM
- [ ] Implement `_simulate_bler()` with link abstraction
- [ ] Add CQI-to-spectral-efficiency table
- [ ] Test: Throughput increases with SINR
- [ ] Test: BLER decreases with SINR

### Week 4-5: Integration
- [ ] Create `configs/data_generation_sionna.yaml`
- [ ] Refactor `MultiLayerDataGenerator.process_scene()`
- [ ] Add toggle for real vs mock mode
- [ ] Write unit tests for all components
- [ ] Integration test: Generate 10-UE dataset
- [ ] Validation: Check physical consistency
- [ ] Benchmark: Measure UEs/second throughput

### Week 5-6: Optimization
- [ ] Enable GPU acceleration (verify Dr.Jit uses CUDA)
- [ ] Implement multi-process scene parallelization
- [ ] Test batched UE processing (optional)
- [ ] Tune `num_samples` for speed/accuracy tradeoff
- [ ] Document performance on different hardware

### Week 6: Radio Maps
- [ ] Implement `generate_radio_map()` using RadioMapSolver
- [ ] Generate maps for all test scenes
- [ ] Store maps with scene metadata
- [ ] Update physics loss to load precomputed maps
- [ ] Test: Physics loss works with real maps

---

## Success Criteria

### Functional
✅ Generate 4000 UEs per scene with real Sionna RT  
✅ All 3 layers (RT, PHY, SYS) computed from physics  
✅ Features pass physical validation (path loss, SINR ranges)  
✅ Data structure matches current Zarr format (no breaking changes)  
✅ Toggle between real and mock mode for testing

### Performance
✅ Generate 4000 UEs in < 2 hours per scene (with GPU)  
✅ Process 10 scenes (40K samples) in < 1 day  
✅ Radio map generation < 30 minutes per scene

### Quality
✅ Path loss correlates with distance (r > 0.9)  
✅ SINR decreases with distance  
✅ Throughput increases with SINR  
✅ No negative delays or impossible angles  
✅ Features have realistic distributions (not uniform noise)

---

## Risk Mitigation

### Technical Risks

**1. Sionna RT too slow**
- **Mitigation:** Reduce `num_samples`, disable diffraction, parallelize
- **Fallback:** Use hybrid (ray tracing for key scenarios, mock for bulk)

**2. GPU memory exhaustion**
- **Mitigation:** Process UEs one at a time, reduce scene complexity
- **Fallback:** Use CPU mode (slower but works)

**3. Channel matrix dimensions mismatch**
- **Mitigation:** Careful shape inspection, add asserts, test with simple scene
- **Fallback:** Use RT features directly, skip PHY matrix processing

**4. Quantization/measurement errors**
- **Mitigation:** Validate against theoretical models (Friis equation, etc.)
- **Fallback:** Use idealized measurements, add noise later

---

## Next Steps

1. **Immediate (This Week):**
   - Create `SIONNA_INTEGRATION_PLAN.md` (this document)
   - Review with team
   - Setup development branch (`git checkout -b feature/sionna-integration`)

2. **Week 1:**
   - Start Phase 1: RT Layer implementation
   - Daily commits with unit tests
   - Document any Sionna API issues

3. **Week 2:**
   - Complete RT Layer
   - Begin Phase 2: PHY Layer
   - Generate first test dataset (10 UEs)

4. **Week 3-4:**
   - Complete PHY and SYS layers
   - Integration testing
   - Generate validation dataset (100 UEs)

5. **Week 5-6:**
   - Optimization and scale-up
   - Generate full training dataset (40K+ UEs)
   - Update pipeline documentation

---

## References

### Sionna Documentation
- [Sionna RT Overview](https://nvlabs.github.io/sionna/api/rt.html)
- [Scene Configuration](https://nvlabs.github.io/sionna/api/rt.html#scene)
- [Antenna Arrays](https://nvlabs.github.io/sionna/api/rt.html#antenna)
- [Channel Models](https://nvlabs.github.io/sionna/api/channel.html)

### 3GPP Standards
- **38.211:** Physical channels and modulation
- **38.214:** Physical layer procedures for data
- **38.215:** Physical layer measurements
- **38.901:** Channel models for calibration

### Project Documentation
- `IMPLEMENTATION_GUIDE.md`: Overall architecture and milestones
- `M2_COMPLETE.md`: Current data generation status
- `geo2sigmap/`: Scene generation codebase

---

**Status:** Ready for implementation  
**Owner:** Development Team  
**Timeline:** 6 weeks (can compress to 4 with dedicated focus)  
**Priority:** High (blocks training on real physics data)
