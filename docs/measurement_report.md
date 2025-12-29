# Field Measurement Plan for Sionna Simulation Validation

## 1. Objective
To validate the fidelity of Sionna ray-tracing simulations against real-world 4G/5G deployments. This data will be used to calibrate the simulation parameters and verify the inputs to the training algorithms.

## 2. Site & Environment Configuration (Simulation Inputs)
*Accurate simulation requires precise inputs. Any error here directly biases the output.*

### 2.1. Site Topology (Base Station)
For every sector/cell at the site, record:
- **Location:** Precise Latitude, Longitude (GPS/RTK), and Ground Elevation (AMSL).
- **Antenna Height:** Center of radiation height relative to ground (AGL).
- **Orientation:**
  - **Azimuth:** Physical heading (degrees True North).
  - **Mechanical Uptilt/Downtilt:** Physical tilt of the bracket.
  - **Electrical Downtilt:** Internal tilt setting (RET).
- **Antenna Hardware:**
  - Manufacturer & Model Number (to retrieve 3D antenna patterns).
  - Number of ports/elements (MIMO configuration, e.g., 2x2, 4x4, 64x64 Massive MIMO).
- **Transmission Parameters:**
  - **Carrier Frequency (f_c):** Exact center frequency (e.g., 3.5 GHz).
  - **Bandwidth:** Channel bandwidth (e.g., 100 MHz).
  - **Tx Power:** Configured reference signal power (RS Power) per port (dBm).

### 2.2. Environment Model
To verify the "Digital Twin":
- **3D Map:** If available, obtain Lidar or high-resolution photogrammetry of the clutter options.
- **Material Properties:** Note dominant materials (glass, concrete, brick, foliage) to calibrate dielectric properties in Sionna.

---

## 3. UE / Scanner Measurements (Validation Ground Truth)
*Data to be collected during Drive Test or Walk Test.*

### 3.1. Positioning (Critical)
- **High-Precision GNSS:** RTK-GPS is recommended (<1m accuracy).
- **Timestamps:** GPS-aligned timestamps for all logs.

### 3.2. Layer 1: RF & Channel Measurements (PHY)
*Compare against [PHYFAPILayerFeatures](file:///Users/nirtzur/Documents/projects/CellularPositioningResearch/src/data_generation/features.py#80-120) and [RTLayerFeatures](file:///Users/nirtzur/Documents/projects/CellularPositioningResearch/src/data_generation/features.py#37-78).*

| Measurement | Simulation Equivalent | Purpose |
| :--- | :--- | :--- |
| **RSRP (dBm)** | `phy_fapi/rsrp` | Primary coverage validation. Checks path loss model calibration. |
| **SINR (dB)** | `phy_fapi/sinr` | Validates interference and noise modeling. |
| **RSRQ (dB)** | `phy_fapi/rsrq` | Load and interference validation. |
| **PCI (Cell ID)** | `mac_rrc/serving_cell_id` | Verifies correct cell selection/handover logic. |
| **Power Delay Profile (PDP) *** | `path_gains`, `path_delays` | **Critical for Ray Tracing.** Shows multipath components (MPC) energy and timing. |
| **RMS Delay Spread (ns) *** | `rt/rms_delay_spread` | Derived from PDP. Validates multipath richness. |
| **K-Factor (dB) *** | `rt/k_factor` | Ratio of LOS to NLOS power. Validates fading distribution. |

**(*) Note:** PDP, Delay Spread, and K-Factor often require a specialized **Channel Sounder** or high-end Scanner, not just a standard UE. If unavailable, RSRP/SINR are the minimum viable set.

### 3.3. Layer 3: Performance Measurements (MAC/App)
*Compare against [MACRRCLayerFeatures](file:///Users/nirtzur/Documents/projects/CellularPositioningResearch/src/data_generation/features.py#122-175).*

| Measurement | Simulation Equivalent | Purpose |
| :--- | :--- | :--- |
| **Throughput (Mbps)** | `mac_rrc/dl_throughput_mbps` | UDP/TCP tests. Validates link adaptation (MCS/CQI) models. |
| **BLER (%)** | `mac_rrc/bler` | Block Error Rate. Validates channel quality reliability. |
| **Timing Advance (TA)** | `mac_rrc/timing_advance` | Coarse distance validation. |

---

## 4. Measurement Procedure (Test Scenarios)

### 4.1. Static Points (Calibration)
Select 5-10 diverse locations:
- **LOS (Line of Sight):** Direct visibility to antenna.
- **NLOS (Non-Line of Sight):** Behind a building/obstruction.
- **Deep Indoor:** If relevant to study.
- **Cell Edge:** Where handover is expected.
*Action:* Record 1-minute stationary logs at each point to average out fast fading.

### 4.2. Mobility Routes (Validation)
- Drive/Walk looping routes that transition between sectors.
- Validation: Does the simulation predict the signal drop at the same corner? Does it handover at the same street crossing?

## 5. Output Data Format
Ask the equipment vendor/operator for logs in **CSV** or **JSON** format containing:
```csv
timestamp, lat, lon, height_agl, pci, earfcn, rsrp, sinr, rsrq, delay_spread_rms, best_beam_index
```
Ensure `best_beam_index` is logged if testing 5G FR2 (mmWave) or Massive MIMO.
