# Cellular Positioning Data Guide

This document outlines cellular network data for UE positioning, with emphasis on what can be **directly extracted from Sionna RT** (ray tracing) simulation.

> **Key**: ✅ = Directly extracted from Sionna | 🔶 = Derived from Sionna outputs | ❌ = Not available in Sionna

---

## 🎯 Most Useful for Positioning

### ✅ Time of Arrival (ToA) / Propagation Delay
**Purpose:** Fundamental distance estimation metric - propagation time from TX to UE.

**Sionna Source:**
```python
paths.tau  # [sources, targets, paths] - propagation delay per path (seconds)
```

**Extraction Method:**
```python
# From SionnaNativeKPIExtractor.extract_rt()
toa = tf.reduce_min(tau_batch, axis=-1)  # First arrival time [B, C]
```

**Used in our model:** `rt/toa`, `rt/path_delays`

---

### ✅ Path Gain / Path Loss
**Purpose:** Signal attenuation - correlates with distance and environment.

**Sionna Source:**
```python
paths.a  # [sources, targets, paths, rx_ant, tx_ant] - complex path coefficients
```

**Extraction Method:**
```python
# From SionnaNativeKPIExtractor.extract_rt()
path_powers = tf.square(tf.abs(paths.a))  # Power per path
total_power = tf.reduce_sum(path_powers, axis=-1)  # Total received power
```

**Used in our model:** `rt/path_gains`, `phy_fapi/rsrp`

---

### ✅ Reference Signal Received Power (RSRP)
**Purpose:** Signal strength from reference signals - primary cell selection metric.

**Sionna Source:** Derived from channel matrix or path gains.

**Extraction Method:**
```python
# From PHYFAPIFeatureExtractor.extract() and measurement_utils.py
h_sq = tf.square(tf.abs(channel_matrix))
rsrp_linear = tf.reduce_mean(h_sq, axis=-1)  # Avg over pilots/freq
rsrp_dbm = 10.0 * tf.math.log(rsrp_linear) / tf.math.log(10.0) + 30.0
```

**Used in our model:** `phy_fapi/rsrp`

---

### ✅ Angle of Arrival (AoA) - Azimuth & Elevation
**Purpose:** Direction from which signal arrives at UE - enables triangulation.

**Sionna Source:**
```python
paths.phi_r    # [sources, targets, paths] - AoA azimuth (radians)
paths.theta_r  # [sources, targets, paths] - AoA elevation (radians)
```

**Extraction Method:**
```python
# From RTFeatureExtractor._extract_generic()
ang_r_az = paths.phi_r   # Azimuth angle at receiver
ang_r_el = paths.theta_r  # Elevation angle at receiver

# Power-weighted mean AoA (from native_extractor.py)
w_phasors = pdf * tf.complex(tf.cos(phi_r), tf.sin(phi_r))
mean_aoa_az = tf.math.angle(tf.reduce_sum(w_phasors, axis=-1))
```

**Used in our model:** `rt/path_aoa_azimuth`, `rt/path_aoa_elevation`

---

### ✅ Angle of Departure (AoD) - Azimuth & Elevation
**Purpose:** Direction of signal leaving TX - useful for beam-based positioning.

**Sionna Source:**
```python
paths.phi_t    # [sources, targets, paths] - AoD azimuth (radians)
paths.theta_t  # [sources, targets, paths] - AoD elevation (radians)
```

**Extraction Method:**
```python
# From RTFeatureExtractor._extract_generic()
ang_t_az = paths.phi_t   # Azimuth angle at transmitter
ang_t_el = paths.theta_t  # Elevation angle at transmitter
```

**Used in our model:** `rt/path_aod_azimuth`, `rt/path_aod_elevation`

---

### ✅ Doppler Shift
**Purpose:** Velocity estimation - movement direction relative to cell.

**Sionna Source:**
```python
paths.doppler  # [sources, targets, paths] - Doppler shift per path (Hz)
```

**Extraction Method:**
```python
# From RTFeatureExtractor._extract_generic()
doppler = paths.doppler  # Direct from Sionna paths object
```

**Used in our model:** `rt/path_doppler`

---

### ✅ Physical Cell ID (PCI) / Serving Cell
**Purpose:** Cell identification for positioning context.

**Sionna Source:** Scene TX configuration (transmitter indices).

**Extraction Method:**
```python
# From MACRRCFeatureExtractor.extract()
# Cell ID derived from TX index in scene configuration
serving_cell_id = np.argmax(rsrp_per_cell, axis=-1)  # Strongest cell
```

**Used in our model:** `mac_rrc/serving_cell_id`, `mac_rrc/neighbor_cell_ids`

---

## 🔬 Derived Metrics (Computed from Sionna Outputs)

### 🔶 Signal-to-Interference-plus-Noise Ratio (SINR)
**Purpose:** Channel quality accounting for interference.

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_phy()
total_power = tf.reduce_sum(rsrp_linear, axis=-1, keepdims=True)
interference = total_power - rsrp_linear  # Power from other cells
sinr_linear = rsrp_linear / (interference + noise_power)
sinr_db = 10.0 * tf.math.log(sinr_linear) / tf.math.log(10.0)
```

**Used in our model:** `phy_fapi/sinr`

---

### 🔶 RMS Delay Spread
**Purpose:** Multipath severity indicator - higher in complex environments.

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_rt()
pdf = path_powers / tf.reduce_sum(path_powers, axis=-1, keepdims=True)
mean_delay = tf.reduce_sum(pdf * tau, axis=-1)
mean_delay_sq = tf.reduce_sum(pdf * tf.square(tau), axis=-1)
rms_ds = tf.sqrt(mean_delay_sq - tf.square(mean_delay))
```

**Used in our model:** `rt/rms_delay_spread`

---

### 🔶 RMS Angular Spread
**Purpose:** Scattering indicator - quantifies angular distribution of paths.

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_rt()
# Circular dispersion of AoA weighted by power
r_abs = tf.abs(mean_phasor)  # Resultant vector length
rms_as = tf.sqrt(-2.0 * tf.math.log(r_abs))
```

**Used in our model:** `rt/rms_angular_spread`

---

### 🔶 Rician K-Factor
**Purpose:** LoS/NLoS indicator - ratio of dominant to scattered power.

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_rt()
max_power = tf.reduce_max(path_powers, axis=-1)
nlos_power = total_power - max_power
k_factor_linear = max_power / (nlos_power + 1e-20)
k_factor_db = 10.0 * tf.math.log(k_factor_linear) / tf.math.log(10.0)
```

**Used in our model:** `rt/k_factor`

---

### 🔶 NLoS Detection Flag
**Purpose:** Binary indicator of Non-Line-of-Sight condition.

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_rt()
is_nlos = k_factor_db < 0.0  # K-factor < 0 dB indicates NLoS
```

**Used in our model:** `rt/is_nlos`

---

### 🔶 Channel Quality Indicator (CQI)
**Purpose:** Link adaptation metric (0-15 scale).

**Derivation:**
```python
# From measurement_utils.compute_cqi()
# Maps SINR to CQI per 3GPP 38.214 Table
cqi = np.clip(np.round((sinr_db + 6.7) / 1.9), 0, 15)
```

**Used in our model:** `phy_fapi/cqi`

---

### 🔶 Rank Indicator (RI)
**Purpose:** MIMO spatial multiplexing capability (1-8).

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_phy()
s = tf.linalg.svd(channel_matrix, compute_uv=False)  # Singular values
rank = tf.reduce_sum(tf.cast(s > 1e-6, tf.float32), axis=-1)
```

**Used in our model:** `phy_fapi/ri`

---

### 🔶 Timing Advance (TA)
**Purpose:** Round-trip time estimate - distance proxy.

**Derivation:**
```python
# From measurement_utils.compute_timing_advance()
# TA = 2 * distance / c, quantized to TA index
ta_us = 2.0 * toa  # Round-trip delay
ta_index = np.round(ta_us / TA_STEP_US)  # 3GPP TA granularity
```

**Used in our model:** `mac_rrc/timing_advance`

---

### 🔶 Channel Capacity / Spectral Efficiency
**Purpose:** Theoretical throughput bound.

**Derivation:**
```python
# From SionnaNativeKPIExtractor._compute_capacity()
s = tf.linalg.svd(channel_matrix, compute_uv=False)
capacity = tf.reduce_sum(tf.math.log(1.0 + tf.square(s) / n0) / tf.math.log(2.0))
```

**Used in our model:** `phy_fapi/capacity_mbps`

---

### 🔶 RSRQ (Reference Signal Received Quality)
**Purpose:** Signal quality relative to total received power.

**Derivation:**
```python
# From measurement_utils.compute_rsrq()
rsrq_db = 10.0 * np.log10(N * rsrp_linear / rssi_linear)
```

**Used in our model:** `phy_fapi/rsrq`

---

### 🔶 RSSI (Received Signal Strength Indicator)
**Purpose:** Total received power including noise.

**Derivation:**
```python
# From PHYFAPIFeatureExtractor.extract()
rssi_linear = 10**((rsrp_dbm - 30) / 10) + noise_power_linear
rssi_dbm = 10.0 * np.log10(rssi_linear) + 30.0
```

---

### 🔶 Condition Number
**Purpose:** Channel matrix conditioning - affects MIMO performance.

**Derivation:**
```python
# From SionnaNativeKPIExtractor.extract_phy()
s = tf.linalg.svd(channel_matrix, compute_uv=False)
cond_num = tf.reduce_max(s, axis=-1) / tf.reduce_min(s, axis=-1)
```

**Used in our model:** `phy_fapi/condition_number`

---

## ❌ Not Available in Sionna RT

The following parameters require real network infrastructure or protocol simulation and are **NOT available** in Sionna ray tracing:

| Parameter | Reason |
|-----------|--------|
| **HARQ Feedback** | Requires MAC layer protocol simulation |
| **SFN/Slot Synchronization** | Network timing protocol - not simulated |
| **Power Headroom Report (PHR)** | UL power control - requires protocol stack |
| **RACH Preamble Detection** | Random access protocol - not simulated |
| **CSI-RS/SSB Configuration** | Network scheduling - not in RT |
| **PMI (full codebook)** | Requires precoder codebook implementation |
| **Beamforming Weights** | Can be set but not derived from RT |
| **MU-MIMO Grouping** | Scheduler decision - not simulated |
| **Throughput/BLER** | Requires full PHY+MAC simulation |

---

## 📊 Summary: What We Use in Our Model

### RT Layer (Direct from Sionna `paths` object)
| Feature | Sionna Source | Shape |
|---------|---------------|-------|
| `path_gains` | `paths.a` | `[B, C, P]` |
| `path_delays` | `paths.tau` | `[B, C, P]` |
| `path_aoa_azimuth` | `paths.phi_r` | `[B, C, P]` |
| `path_aoa_elevation` | `paths.theta_r` | `[B, C, P]` |
| `path_aod_azimuth` | `paths.phi_t` | `[B, C, P]` |
| `path_aod_elevation` | `paths.theta_t` | `[B, C, P]` |
| `path_doppler` | `paths.doppler` | `[B, C, P]` |
| `toa` | `min(paths.tau)` | `[B, C]` |
| `rms_delay_spread` | derived | `[B, C]` |
| `rms_angular_spread` | derived | `[B, C]` |
| `k_factor` | derived | `[B, C]` |
| `is_nlos` | derived | `[B, C]` |

### PHY Layer (Derived from channel response)
| Feature | Derivation | Shape |
|---------|------------|-------|
| `rsrp` | `mean(\|h\|²)` | `[B, C]` |
| `rsrq` | `N*RSRP/RSSI` | `[B, C]` |
| `sinr` | `signal/(interference+noise)` | `[B, C]` |
| `cqi` | SINR→CQI mapping | `[B, C]` |
| `ri` | `rank(H)` | `[B, C]` |
| `capacity_mbps` | Shannon capacity | `[B, C]` |
| `condition_number` | `σ_max/σ_min` | `[B, C]` |

### MAC Layer (Simulated/Derived)
| Feature | Derivation | Shape |
|---------|------------|-------|
| `serving_cell_id` | `argmax(RSRP)` | `[B]` |
| `neighbor_cell_ids` | sorted by RSRP | `[B, K]` |
| `timing_advance` | `2*ToA` quantized | `[B, C]` |

---

## 🔗 Code References

- **RT Extraction:** `src/data_generation/features/rt_extractor.py`
- **Native TF Extraction:** `src/data_generation/features/native_extractor.py`
- **PHY Extraction:** `src/data_generation/features/phy_extractor.py`
- **MAC Extraction:** `src/data_generation/features/mac_extractor.py`
- **Measurement Utils:** `src/data_generation/measurement_utils.py`
- **Data Structures:** `src/data_generation/features/data_structures.py`
